"""
大学生心理分析数字人代理 - 主服务器
=====================================

本模块提供基于 Flask 的 Web 服务，集成以下功能：
1. DeepSeek API 心理咨询服务
2. DeepFace 面部表情识别
3. 对话历史管理
4. RESTful API 接口
5. 数字人形象选择功能
6. 语音输入识别功能

作者: SRTP 项目组
版本: 2.2
"""

import os
import base64
import logging
import datetime
import subprocess
import uuid
import glob
import io
import wave
import tempfile
from typing import Dict, Any, Optional

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS

# 语音识别相关导入
try:
    import speech_recognition as sr
    import pydub
    from pydub import AudioSegment
    SPEECH_RECOGNITION_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("语音识别库加载成功")
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("语音识别库未安装，语音输入功能不可用。请运行: pip install SpeechRecognition pydub")

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== DeepFace 导入 ====================
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace 库加载成功")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace 库未安装，表情识别功能不可用。请运行: pip install deepface")

# ==================== Flask 应用初始化 ====================
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # 启用跨域支持

# ==================== 配置常量 ====================
# DeepSeek API 配置（建议使用环境变量存储密钥）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-215440b00f1d426fb21a2f11eef6cf02")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# TTS API 配置 (SiliconFlow)
TTS_API_URL = "https://api.siliconflow.cn/v1/audio/speech"
TTS_API_TOKEN = "sk-lvtuhfndddcmdyvnjtbzjuobfoewylsnqaqwfsnuznpilhkp"

# SadTalker 配置
SADTALKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SadTalker")
SADTALKER_IMAGE = os.path.join(SADTALKER_DIR, "my_photo.png")  # 数字人图片（默认）
SADTALKER_OUTPUT_DIR = os.path.join(SADTALKER_DIR, "results")
AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_output")
AVATARS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatars")
SPEECH_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_input")

# 确保输出目录存在
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(SADTALKER_OUTPUT_DIR, exist_ok=True)
os.makedirs(AVATARS_DIR, exist_ok=True)
os.makedirs(SPEECH_INPUT_DIR, exist_ok=True)

# 检查 avatars 目录下是否有默认图片，如果没有则创建
default_avatar_path = os.path.join(AVATARS_DIR, "avatar1.png")
if not os.path.exists(default_avatar_path):
    # 将 SadTalker 的默认图片复制到 avatars 目录作为 avatar1
    if os.path.exists(SADTALKER_IMAGE):
        import shutil
        shutil.copy2(SADTALKER_IMAGE, default_avatar_path)
        logger.info(f"已将默认数字人图片复制到: {default_avatar_path}")
    else:
        # 创建三个示例图片路径
        for i in range(1, 4):
            avatar_path = os.path.join(AVATARS_DIR, f"avatar{i}.png")
            if not os.path.exists(avatar_path):
                logger.warning(f"数字人图片不存在: {avatar_path}，请放置相应图片文件")

# 情绪映射表
EMOTION_MAP = {
    'angry':    {'icon': '😠', 'name': '生气', 'context': '有些生气'},
    'disgust':  {'icon': '🤢', 'name': '厌恶', 'context': '有些反感'},
    'fear':     {'icon': '😨', 'name': '恐惧', 'context': '感到紧张'},
    'happy':    {'icon': '😊', 'name': '开心', 'context': '看起来心情不错'},
    'sad':      {'icon': '😢', 'name': '悲伤', 'context': '情绪有些低落'},
    'surprise': {'icon': '😲', 'name': '惊讶', 'context': '有些惊讶'},
    'neutral':  {'icon': '😐', 'name': '平静', 'context': '情绪平稳'}
}

# 默认情绪分数（当检测失败时使用）
DEFAULT_EMOTION_SCORES = {
    'angry': 0.0, 'disgust': 0.0, 'fear': 0.0,
    'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 100.0
}

# 对话历史（全局变量）
conversation_history: list = []


# ==================== 语音识别模块 ====================
def recognize_speech_from_audio(audio_data: bytes, audio_format: str = "webm") -> Dict[str, Any]:
    """
    从音频数据中识别语音
    
    Args:
        audio_data: 音频字节数据
        audio_format: 音频格式（webm, wav, mp3等）
        
    Returns:
        包含识别结果的字典
    """
    if not SPEECH_RECOGNITION_AVAILABLE:
        return {
            "success": False,
            "error": "语音识别库未安装",
            "text": ""
        }
    
    try:
        recognizer = sr.Recognizer()
        
        # 创建临时文件保存音频
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        try:
            # 使用 pydub 加载音频文件
            if audio_format == "webm":
                audio = AudioSegment.from_file(tmp_file_path, format="webm")
            elif audio_format == "mp3":
                audio = AudioSegment.from_mp3(tmp_file_path)
            elif audio_format == "wav":
                audio = AudioSegment.from_wav(tmp_file_path)
            else:
                # 尝试自动检测格式
                audio = AudioSegment.from_file(tmp_file_path)
            
            # 转换为 wav 格式（SpeechRecognition 需要）
            wav_data = io.BytesIO()
            audio.export(wav_data, format="wav")
            wav_data.seek(0)
            
            # 使用 SpeechRecognition 识别
            with sr.AudioFile(wav_data) as source:
                # 调整环境噪声
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                # 识别语音
                text = recognizer.recognize_google(audio_data, language="zh-CN")
                
                logger.info(f"语音识别成功: {text}")
                
                return {
                    "success": True,
                    "text": text,
                    "confidence": 0.9  # 暂时使用固定值
                }
                
        except sr.UnknownValueError:
            return {
                "success": False,
                "error": "无法理解音频内容",
                "text": ""
            }
        except sr.RequestError as e:
            logger.error(f"语音识别服务错误: {e}")
            return {
                "success": False,
                "error": f"语音识别服务错误: {e}",
                "text": ""
            }
        except Exception as e:
            logger.error(f"语音识别处理错误: {e}")
            return {
                "success": False,
                "error": f"处理错误: {str(e)}",
                "text": ""
            }
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"语音识别失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "text": ""
        }


def save_audio_file(audio_data: bytes, filename: str) -> str:
    """
    保存音频文件到本地
    
    Args:
        audio_data: 音频字节数据
        filename: 文件名
        
    Returns:
        保存的文件路径
    """
    file_path = os.path.join(SPEECH_INPUT_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(audio_data)
    return file_path


# ==================== 心理分析代理类 ====================
class PsychologicalAgent:
    """
    心理分析代理类

    负责与 DeepSeek API 交互，提供心理咨询服务。
    支持多轮对话，结合用户情绪状态生成个性化回复。
    """

    def __init__(self, api_key: str):
        """
        初始化心理分析代理

        Args:
            api_key: DeepSeek API 密钥
        """
        self.api_key = api_key
        self.api_url = DEEPSEEK_API_URL

    def _build_system_prompt(self, emotion: str) -> str:
        """
        构建系统提示词

        Args:
            emotion: 检测到的情绪类型

        Returns:
            包含情绪上下文的系统提示词
        """
        emotion_info = EMOTION_MAP.get(emotion, EMOTION_MAP['neutral'])
        emotion_context = emotion_info['context']

        return f"""你是一名专业的大学心理健康顾问，专门帮助大学生解决心理问题。

重要上下文信息：
- 系统检测到用户当前的情绪状态为：{emotion} ({emotion_context})
- 这个情绪信息来自实时面部表情分析
- 请结合用户描述的文字内容和检测到的情绪状态，提供更精准的心理分析

你的职责：
1. 分析学生的心理状态和情绪问题
2. 提供专业、温暖的心理支持和建议
3. 识别危机情况并给出适当建议
4. 用同理心和理解来回应用户

请以温暖、专业、支持性的语气回应，避免使用专业术语，用通俗易懂的语言提供建议。"""

    def analyze(self, user_input: str, emotion: str = "neutral") -> Dict[str, Any]:
        """
        分析用户输入并生成心理咨询回复

        Args:
            user_input: 用户输入的文本
            emotion: 检测到的情绪类型，默认为 neutral

        Returns:
            包含分析结果的字典，包括 success、response、model_source 等字段
        """
        try:
            # 构建消息列表
            messages = [{"role": "system", "content": self._build_system_prompt(emotion)}]

            # 添加最近的对话历史（最多 3 轮，即 6 条消息）
            if conversation_history:
                messages.extend(conversation_history[-6:])

            messages.append({"role": "user", "content": user_input})

            logger.info(f"调用 DeepSeek API - 情绪: {emotion}, 输入: {user_input[:50]}...")

            # 调用 API
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.7,  # 控制回复的随机性
                    "max_tokens": 800,   # 限制回复长度
                    "stream": False
                },
                timeout=30
            )

            # 处理响应
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['choices'][0]['message']['content']

                # 更新对话历史
                self._update_history(user_input, assistant_response)

                return {
                    "success": True,
                    "response": assistant_response,
                    "model_source": "deepseek_api"
                }
            else:
                logger.error(f"API 调用失败: {response.status_code} - {response.text[:200]}")
                return {
                    "success": False,
                    "error": f"API 错误: {response.status_code}",
                    "response": "抱歉，AI 服务暂时不可用，请稍后再试。"
                }

        except requests.exceptions.Timeout:
            logger.error("API 请求超时")
            return {
                "success": False,
                "error": "请求超时",
                "response": "请求超时，请检查网络连接后重试。"
            }
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "系统暂时出现问题，请稍后再试。"
            }

    def _update_history(self, user_input: str, assistant_response: str) -> None:
        """
        更新对话历史

        Args:
            user_input: 用户输入
            assistant_response: AI 回复
        """
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_response})

        # 限制历史记录长度（保留最近 10 条）
        if len(conversation_history) > 10:
            conversation_history[:] = conversation_history[-10:]


# ==================== 表情识别模块 ====================
def analyze_emotion_from_image(image_data: str) -> Dict[str, Any]:
    """
    从 Base64 编码的图像中分析面部表情

    Args:
        image_data: Base64 编码的图像数据

    Returns:
        包含情绪分析结果的字典，包括 dominant_emotion、emotion_scores、face_detected
    """
    # 检查 DeepFace 是否可用
    if not DEEPFACE_AVAILABLE:
        logger.warning("DeepFace 库不可用，返回默认情绪")
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": DEFAULT_EMOTION_SCORES.copy(),
            "face_detected": False
        }

    logger.info("开始 DeepFace 表情分析...")

    try:
        # 解码 Base64 图像
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("图像解码失败")
            return {
                "dominant_emotion": "neutral",
                "emotion_scores": DEFAULT_EMOTION_SCORES.copy(),
                "face_detected": False
            }

        # 使用 DeepFace 分析表情
        analysis = DeepFace.analyze(
            img,
            actions=['emotion'],
            detector_backend='opencv',  # 使用 OpenCV 检测器（速度快）
            enforce_detection=False,    # 不强制检测到人脸
            silent=True                 # 静默模式
        )

        if not analysis:
            logger.warning("DeepFace 返回空结果")
            return {
                "dominant_emotion": "neutral",
                "emotion_scores": DEFAULT_EMOTION_SCORES.copy(),
                "face_detected": False
            }

        # 提取结果
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_scores = analysis[0]['emotion']

        # 检查是否真正检测到人脸（通过检查 face_confidence 或 region）
        face_region = analysis[0].get('region', {})
        face_confidence = analysis[0].get('face_confidence', 0)

        # 记录详细的分析结果用于调试
        logger.info(f"表情分析结果: dominant={dominant_emotion}, scores={emotion_scores}")
        logger.info(f"人脸区域: {face_region}, 置信度: {face_confidence}")

        # 转换为 Python float 类型（避免 JSON 序列化问题）
        converted_scores = {k: float(v) for k, v in emotion_scores.items()}

        # 判断是否真正检测到人脸
        # 如果人脸区域太小或置信度太低，可能是误检
        face_detected = True
        if face_region:
            w = face_region.get('w', 0)
            h = face_region.get('h', 0)
            # 如果检测到的人脸区域太小（小于50x50像素），认为没有检测到有效人脸
            if w < 50 or h < 50:
                logger.warning(f"检测到的人脸区域太小: {w}x{h}")
                face_detected = False

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": converted_scores,
            "face_detected": face_detected
        }

    except Exception as e:
        logger.warning(f"表情分析失败: {str(e)[:100]}")
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": DEFAULT_EMOTION_SCORES.copy(),
            "face_detected": False
        }


def generate_fallback_response(user_input: str, emotion: str) -> Dict[str, Any]:
    """
    生成备选回复（当 API 不可用时使用）

    Args:
        user_input: 用户输入
        emotion: 检测到的情绪

    Returns:
        包含备选回复的字典
    """
    emotion_info = EMOTION_MAP.get(emotion, EMOTION_MAP['neutral'])
    emotion_prefix = f"{emotion_info['context']}，" if emotion != 'neutral' else ""

    # 关键词匹配回复
    keyword_responses = {
        '压力': f"{emotion_prefix}对于压力问题，建议：<br>1. 深呼吸放松练习<br>2. 合理安排时间和优先级<br>3. 适量运动释放压力<br>4. 与朋友或家人倾诉",
        '焦虑': f"{emotion_prefix}应对焦虑的方法：<br>1. 正念冥想练习<br>2. 写下担忧事项<br>3. 渐进式肌肉放松<br>4. 保持规律作息",
        '失眠': f"{emotion_prefix}改善睡眠的建议：<br>1. 睡前1小时不使用电子设备<br>2. 创造舒适的睡眠环境<br>3. 保持规律的作息时间<br>4. 避免睡前摄入咖啡因",
        '抑郁': f"{emotion_prefix}如果持续情绪低落：<br>1. 寻求专业心理咨询<br>2. 保持适度的社交活动<br>3. 坚持适量运动<br>4. 给自己一些时间和耐心",
        '学习': f"{emotion_prefix}学习压力管理：<br>1. 制定合理的学习计划<br>2. 使用番茄工作法提高效率<br>3. 保证充足的休息时间<br>4. 与同学交流学习心得"
    }

    # 查找匹配的关键词
    for keyword, response in keyword_responses.items():
        if keyword in user_input.lower():
            return {
                "success": True,
                "response": response,
                "detected_emotion": emotion,
                "model_source": "fallback_system",
                "timestamp": datetime.datetime.now().isoformat()
            }

    # 通用回复
    return {
        "success": True,
        "response": f"{emotion_prefix}我理解您的困扰。作为心理助手，我建议您可以更详细地描述具体情况和感受，这样我能提供更有针对性的帮助。",
        "detected_emotion": emotion,
        "model_source": "fallback_system",
        "timestamp": datetime.datetime.now().isoformat()
    }


def test_deepseek_api() -> Dict[str, Any]:
    """
    测试 DeepSeek API 连接状态

    Returns:
        包含测试结果的字典
    """
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            },
            timeout=10
        )

        if response.status_code == 200:
            return {"success": True, "message": "API 连接正常"}
        elif response.status_code == 401:
            return {"success": False, "message": "API 密钥无效"}
        else:
            return {"success": False, "message": f"API 返回错误: {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"success": False, "message": "API 连接超时"}
    except Exception as e:
        return {"success": False, "message": f"API 连接失败: {str(e)}"}


# ==================== TTS 文字转语音模块 ====================
def text_to_speech(text: str) -> Dict[str, Any]:
    """
    将文字转换为语音 MP3 文件

    Args:
        text: 要转换的文字内容

    Returns:
        包含音频文件路径的字典
    """
    try:
        # 生成唯一文件名
        audio_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)

        logger.info(f"开始 TTS 转换，文本长度: {len(text)}")

        # 调用 SiliconFlow TTS API
        request_data = {
            "model": "IndexTeam/IndexTTS-2",
            "voice": "IndexTeam/IndexTTS-2:claire",
            "stream": True,
            "input": text,
            "max_tokens": 1600,
            "response_format": "mp3",
            "speed": 1,
            "gain": 0
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {TTS_API_TOKEN}"
        }

        response = requests.post(
            url=TTS_API_URL,
            json=request_data,
            headers=headers,
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"TTS API 错误: {response.status_code} - {response.text[:200]}")
            return {"success": False, "error": f"TTS API 错误: {response.status_code}"}

        # 保存音频文件
        with open(audio_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"TTS 转换成功，保存到: {audio_path}")
        return {
            "success": True,
            "audio_path": audio_path,
            "audio_filename": audio_filename
        }

    except requests.exceptions.Timeout:
        logger.error("TTS API 请求超时")
        return {"success": False, "error": "TTS 请求超时"}
    except Exception as e:
        logger.error(f"TTS 转换失败: {str(e)}")
        return {"success": False, "error": str(e)}


# ==================== SadTalker 视频生成模块 ====================
def generate_talking_video(audio_path: str) -> Dict[str, Any]:
    """
    使用 SadTalker 生成数字人说话视频

    Args:
        audio_path: 音频文件的绝对路径

    Returns:
        包含视频文件路径的字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(SADTALKER_IMAGE):
            logger.error(f"数字人图片不存在: {SADTALKER_IMAGE}")
            return {"success": False, "error": "数字人图片不存在"}

        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return {"success": False, "error": "音频文件不存在"}

        logger.info("开始生成数字人视频...")
        logger.info(f"图片: {SADTALKER_IMAGE}")
        logger.info(f"音频: {audio_path}")

        # 检测 SadTalker 目录下的虚拟环境
        sadtalker_venv_python = os.path.join(SADTALKER_DIR, ".venv", "Scripts", "python.exe")

        if os.path.exists(sadtalker_venv_python):
            python_exec = sadtalker_venv_python
            logger.info(f"使用 SadTalker 虚拟环境: {sadtalker_venv_python}")
        else:
            python_exec = "python"
            logger.warning("未检测到 SadTalker/.venv，使用默认 Python")

        # 构建 SadTalker 命令
        cmd = [
            python_exec, "inference.py",
            "--driven_audio", audio_path,
            "--source_image", SADTALKER_IMAGE,
            "--result_dir", SADTALKER_OUTPUT_DIR,
            "--still",
            "--preprocess", "crop",
            # "--enhancer", "gfpgan",
            "--batch_size", "4"
        ]

        # 在 SadTalker 目录下执行
        process = subprocess.Popen(
            cmd,
            cwd=SADTALKER_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # 读取输出
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            logger.info(f"SadTalker: {line.strip()}")

        process.wait()

        if process.returncode != 0:
            logger.error(f"SadTalker 执行失败，返回码: {process.returncode}")
            return {"success": False, "error": "视频生成失败"}

        # 查找生成的视频文件（最新的 mp4 文件）
        video_pattern = os.path.join(SADTALKER_OUTPUT_DIR, "**", "*.mp4")
        video_files = glob.glob(video_pattern, recursive=True)

        if not video_files:
            logger.error("未找到生成的视频文件")
            return {"success": False, "error": "未找到生成的视频"}

        # 获取最新的视频文件
        latest_video = max(video_files, key=os.path.getmtime)
        video_filename = os.path.basename(latest_video)

        logger.info(f"视频生成成功: {latest_video}")
        return {
            "success": True,
            "video_path": latest_video,
            "video_filename": video_filename
        }

    except Exception as e:
        logger.error(f"视频生成失败: {str(e)}")
        return {"success": False, "error": str(e)}


# ==================== 初始化代理实例 ====================
agent = PsychologicalAgent(DEEPSEEK_API_KEY)


# ==================== API 路由 ====================

@app.route('/')
def root_redirect():
    """
    根路径 - 重定向到选择页面
    """
    return redirect('/select')

@app.route('/index')
def main_page():
    """
    主页面路由 - 提供主页面
    """
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>大学生心理分析数字人代理</title></head>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1 style="color: #4a90e2;">大学生心理分析数字人代理</h1>
            <p>请先 <a href="/select">选择数字人形象</a></p>
        </body>
        </html>
        """


@app.route('/select')
def select_page():
    """选择数字人入口页"""
    try:
        with open('select.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>选择数字人形象</title></head>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1 style="color: #4a90e2;">选择数字人形象</h1>
            <p>select.html 文件未找到，请确保文件存在。</p>
        </body>
        </html>
        """, 404


@app.route('/avatars/<filename>')
def serve_avatar(filename):
    """
    提供数字人图片服务
    """
    try:
        return send_from_directory('avatars', filename)
    except FileNotFoundError:
        logger.error(f"数字人图片未找到: {filename}")
        return jsonify({"error": f"图片 {filename} 未找到"}), 404


@app.route('/api/set_avatar', methods=['POST'])
def set_sadtalker_image():
    """
    设置 SadTalker 使用的数字人图片
    """
    try:
        data = request.get_json()
        avatar_id = data.get('avatar_id', '1')
        
        # 根据 avatar_id 设置对应的图片
        avatar_images = {
            '1': 'avatar1.png',
            '2': 'avatar2.png', 
            '3': 'avatar3.png'
        }
        
        avatar_filename = avatar_images.get(avatar_id, 'avatar1.png')
        new_image_path = os.path.join('avatars', avatar_filename)
        
        # 检查文件是否存在
        if os.path.exists(new_image_path):
            # 更新 SadTalker 配置中的图片路径
            global SADTALKER_IMAGE
            SADTALKER_IMAGE = new_image_path
            logger.info(f"已更新 SadTalker 图片为: {new_image_path}")
            
            # 如果需要，也复制到 SadTalker 目录
            sadtalker_dest = os.path.join(SADTALKER_DIR, "my_photo.png")
            try:
                import shutil
                shutil.copy2(new_image_path, sadtalker_dest)
                logger.info(f"已复制到 SadTalker 目录: {sadtalker_dest}")
            except Exception as e:
                logger.warning(f"复制到 SadTalker 目录失败: {str(e)}")
            
            return jsonify({"success": True, "message": f"已切换为形象 {avatar_id}"})
        else:
            logger.error(f"数字人图片不存在: {new_image_path}")
            return jsonify({"success": False, "error": "图片文件不存在"}), 404
            
    except Exception as e:
        logger.error(f"设置数字人图片失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 语音识别API ====================
@app.route('/api/recognize_speech', methods=['POST'])
def recognize_speech():
    """
    语音识别接口
    
    支持上传音频文件进行语音识别
    """
    try:
        logger.info("收到语音识别请求")
        
        # 检查是否安装了语音识别库
        if not SPEECH_RECOGNITION_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "语音识别功能未启用，请安装依赖: pip install SpeechRecognition pydub"
            }), 501
        
        # 检查请求数据
        if 'audio' not in request.files and 'audio_data' not in request.form:
            return jsonify({"success": False, "error": "没有提供音频数据"}), 400
        
        audio_format = request.form.get('format', 'webm')
        
        if 'audio' in request.files:
            # 处理文件上传
            audio_file = request.files['audio']
            audio_data = audio_file.read()
        else:
            # 处理Base64编码的音频数据
            audio_data_str = request.form.get('audio_data', '')
            if ',' in audio_data_str:
                audio_data_str = audio_data_str.split(',')[1]
            audio_data = base64.b64decode(audio_data_str)
        
        # 识别语音
        result = recognize_speech_from_audio(audio_data, audio_format)
        
        if result["success"]:
            # 保存音频文件（可选）
            filename = f"speech_{uuid.uuid4().hex[:8]}.{audio_format}"
            save_audio_file(audio_data, filename)
            
            result["audio_url"] = f"/api/speech/{filename}"
            result["timestamp"] = datetime.datetime.now().isoformat()
            
            logger.info(f"语音识别成功: {result['text'][:50]}...")
        else:
            logger.warning(f"语音识别失败: {result.get('error', '未知错误')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"语音识别接口错误: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"服务器错误: {str(e)}",
            "text": ""
        }), 500


@app.route('/api/speech/<filename>')
def serve_speech(filename):
    """
    提供语音文件服务
    """
    try:
        return send_from_directory(SPEECH_INPUT_DIR, filename)
    except FileNotFoundError:
        logger.error(f"语音文件未找到: {filename}")
        return jsonify({"error": "语音文件未找到"}), 404


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    心理分析主接口（集成 TTS 和数字人视频生成）

    请求体:
        - message: 用户输入的文本
        - detected_emotion: 检测到的情绪（可选）
        - generate_video: 是否生成数字人视频（可选，默认 True）

    Returns:
        JSON 格式的分析结果，包含视频 URL
    """
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        detected_emotion = data.get('detected_emotion', 'neutral')
        generate_video = data.get('generate_video', True)

        logger.info(f"收到分析请求 - 情绪: {detected_emotion}, 生成视频: {generate_video}")

        if not user_input:
            return jsonify({"success": False, "error": "输入不能为空"}), 400

        # 调用心理分析代理
        result = agent.analyze(user_input, detected_emotion)

        if result["success"]:
            result["detected_emotion"] = detected_emotion
            result["timestamp"] = datetime.datetime.now().isoformat()

            # 如果需要生成视频
            if generate_video:
                response_text = result.get("response", "")

                # 清理 HTML 标签，只保留纯文本用于 TTS
                import re
                clean_text = re.sub(r'<[^>]+>', '', response_text)
                clean_text = clean_text.replace('<br>', '。').replace('&nbsp;', ' ')

                # 步骤1: TTS 文字转语音
                logger.info("开始 TTS 转换...")
                tts_result = text_to_speech(clean_text)

                if tts_result["success"]:
                    audio_path = tts_result["audio_path"]
                    result["audio_url"] = f"/api/audio/{tts_result['audio_filename']}"

                    # 步骤2: SadTalker 生成视频
                    logger.info("开始生成数字人视频...")
                    video_result = generate_talking_video(audio_path)

                    if video_result["success"]:
                        result["video_url"] = f"/api/video/{video_result['video_filename']}"
                        result["video_generated"] = True
                        logger.info(f"视频生成成功: {video_result['video_filename']}")
                    else:
                        result["video_generated"] = False
                        result["video_error"] = video_result.get("error", "视频生成失败")
                        logger.warning(f"视频生成失败: {video_result.get('error')}")
                else:
                    result["video_generated"] = False
                    result["tts_error"] = tts_result.get("error", "TTS 转换失败")
                    logger.warning(f"TTS 转换失败: {tts_result.get('error')}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"分析接口错误: {str(e)}")
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500


@app.route('/api/video/<filename>')
def serve_video(filename):
    """
    提供视频文件服务

    Args:
        filename: 视频文件名

    Returns:
        视频文件
    """
    # 在 SadTalker 输出目录中查找视频
    video_pattern = os.path.join(SADTALKER_OUTPUT_DIR, "**", filename)
    video_files = glob.glob(video_pattern, recursive=True)

    if video_files:
        video_path = video_files[0]
        directory = os.path.dirname(video_path)
        return send_from_directory(directory, filename, mimetype='video/mp4')

    logger.error(f"视频文件未找到: {filename}")
    return jsonify({"error": "视频文件未找到"}), 404


@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """
    提供音频文件服务

    Args:
        filename: 音频文件名

    Returns:
        音频文件
    """
    audio_path = os.path.join(AUDIO_OUTPUT_DIR, filename)

    if os.path.exists(audio_path):
        return send_from_directory(AUDIO_OUTPUT_DIR, filename, mimetype='audio/mpeg')

    logger.error(f"音频文件未找到: {filename}")
    return jsonify({"error": "音频文件未找到"}), 404


@app.route('/api/analyze_local', methods=['POST'])
def analyze_local():
    """
    本地模型分析接口（当前使用 DeepSeek API 作为后备）

    请求体:
        - message: 用户输入的文本
        - detected_emotion: 检测到的情绪（可选）

    Returns:
        JSON 格式的分析结果
    """
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        detected_emotion = data.get('detected_emotion', 'neutral')

        logger.info(f"收到本地分析请求 - 情绪: {detected_emotion}")

        if not user_input:
            return jsonify({"success": False, "error": "输入不能为空"}), 400

        # 尝试使用 DeepSeek API
        result = agent.analyze(user_input, detected_emotion)

        if result["success"]:
            result["detected_emotion"] = detected_emotion
            result["timestamp"] = datetime.datetime.now().isoformat()
        else:
            # API 失败时使用备选回复
            result = generate_fallback_response(user_input, detected_emotion)

        return jsonify(result)

    except Exception as e:
        logger.error(f"本地分析接口错误: {str(e)}")
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500


@app.route('/api/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """
    表情识别接口

    请求体:
        - image: Base64 编码的图像数据

    Returns:
        JSON 格式的情绪分析结果
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "没有提供图片数据"}), 400

        logger.info("收到表情分析请求")

        # 分析表情
        result = analyze_emotion_from_image(data['image'])

        return jsonify({
            "success": True,
            "dominant_emotion": result["dominant_emotion"],
            "emotion_scores": result["emotion_scores"],
            "face_detected": result["face_detected"],
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"表情分析接口错误: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "dominant_emotion": "neutral",
            "emotion_scores": DEFAULT_EMOTION_SCORES.copy(),
            "face_detected": False
        }), 500


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """
    模型状态查询接口

    Returns:
        JSON 格式的模型状态信息
    """
    return jsonify({
        "local_model_loaded": False,
        "model_loading": False,
        "deepface_available": DEEPFACE_AVAILABLE,
        "deepseek_api_available": True,
        "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
def api_status():
    """
    API 状态检查接口

    Returns:
        JSON 格式的 API 状态信息
    """
    api_test = test_deepseek_api()

    return jsonify({
        "status": "healthy" if api_test.get("success") else "warning",
        "deepseek_api": api_test,
        "deepface_available": DEEPFACE_AVAILABLE,
        "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查接口

    Returns:
        JSON 格式的服务健康状态
    """
    return jsonify({
        "status": "healthy",
        "service": "大学生心理分析数字人代理",
        "version": "2.2",
        "timestamp": datetime.datetime.now().isoformat(),
        "features": {
            "psychological_analysis": True,
            "emotion_recognition": DEEPFACE_AVAILABLE,
            "real_time_camera": True,
            "deepseek_api": True,
            "avatar_selection": True,
            "speech_input": SPEECH_RECOGNITION_AVAILABLE
        }
    })


@app.route('/api/conversation/summary', methods=['GET'])
def get_conversation_summary():
    """
    获取对话摘要

    Returns:
        JSON 格式的对话摘要信息
    """
    user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']

    return jsonify({
        "total_conversations": len(conversation_history) // 2,
        "recent_topics": user_messages[-3:] if user_messages else [],
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/api/conversation/reset', methods=['POST'])
def reset_conversation():
    """
    重置对话历史

    Returns:
        JSON 格式的操作结果
    """
    global conversation_history
    conversation_history = []
    logger.info("对话历史已重置")
    return jsonify({"success": True, "message": "对话已重置"})


@app.route('/api/debug', methods=['GET'])
def debug_info():
    """
    调试信息接口（仅用于开发）

    Returns:
        JSON 格式的调试信息
    """
    return jsonify({
        "routes": [str(rule) for rule in app.url_map.iter_rules()],
        "conversation_length": len(conversation_history),
        "deepface_available": DEEPFACE_AVAILABLE,
        "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
        "api_key_set": bool(DEEPSEEK_API_KEY),
        "sadtalker_image": SADTALKER_IMAGE,
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/favicon.ico')
def favicon():
    """处理 favicon 请求，避免 404 错误"""
    return '', 204


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 打印启动信息
    print("=" * 60)
    print("大学生心理分析数字人代理 v2.2")
    print("=" * 60)
    print(f"📱 服务地址: http://localhost:5000")
    print(f"👤 形象选择: http://localhost:5000/select")
    print(f"💬 主页面: http://localhost:5000/index")
    print(f"🎤 语音输入: 支持（需要浏览器支持语音识别）")
    print(f"❤️  健康检查: http://localhost:5000/api/health")
    print(f"📊 模型状态: http://localhost:5000/api/model/status")
    print(f"🔍 调试信息: http://localhost:5000/api/debug")
    print("=" * 60)
    print("可用 API 端点:")
    print("  POST /api/analyze        - 心理分析")
    print("  POST /api/analyze_local  - 本地模型分析")
    print("  POST /api/analyze_emotion - 表情识别")
    print("  POST /api/set_avatar     - 设置数字人形象")
    print("  POST /api/recognize_speech - 语音识别")
    print("  GET  /api/health         - 健康检查")
    print("  GET  /api/model/status   - 模型状态")
    print("  GET  /api/conversation/summary - 对话摘要")
    print("  POST /api/conversation/reset   - 重置对话")
    print("=" * 60)
    
    # 检查语音识别功能
    if SPEECH_RECOGNITION_AVAILABLE:
        print("✅ 语音识别功能已启用")
    else:
        print("⚠️  语音识别功能未启用，请运行: pip install SpeechRecognition pydub")
    
    print("🚀 服务启动中...")
    print("=" * 60)

    # 启动 Flask 服务
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # 禁用自动重载，避免模型重复加载
    )