"""
大学生心理分析数字人代理 - 主服务器（本地模型版）
==============================================

本模块提供基于 Flask 的 Web 服务，集成以下功能：
1. 本地心理大模型心理咨询服务
2. DeepFace 面部表情识别
3. 对话历史管理
4. RESTful API 接口
5. 数字人形象选择功能
6. 语音输入识别功能
7. 用户上传自定义数字人形象功能

作者: SRTP 项目组
版本: 2.4
"""

import os
import base64
import logging
import datetime
import subprocess
import uuid
import glob
import io
import time 
import wave
import tempfile
import shutil
import threading
import json
import asyncio
from typing import Dict, Any, Optional

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ==================== 本地模型相关导入 ====================
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TORCH_AVAILABLE = True
    print("✅ PyTorch 和 Transformers 库加载成功")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"❌ PyTorch 相关库导入失败: {e}")
    print("请运行: pip install torch transformers peft")

# 语音识别相关导入
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    SPEECH_RECOGNITION_AVAILABLE = True
    print("✅ 语音识别库加载成功")
except ImportError as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    print(f"❌ 语音识别库未安装: {e}")
    print("请运行: pip install SpeechRecognition pydub")

# Edge TTS 导入
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    print("✅ Edge TTS 库加载成功")
except ImportError as e:
    EDGE_TTS_AVAILABLE = False
    print(f"❌ Edge TTS 库未安装: {e}")
    print("请运行: pip install edge-tts")

# ==================== 日志配置 ====================
import logging
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
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    logger.warning(f"DeepFace 库未安装: {e}")
    logger.warning("请运行: pip install deepface")

# ==================== Flask 应用初始化 ====================
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # 启用跨域支持

# ==================== 配置常量 ====================
# 本地模型配置 - 修改为相对路径
LOCAL_MODEL_PATH = "models/Qwen1.5-0.5B"
LOCAL_ADAPTER_PATH = "outputs/psychology_trained_model"
MODEL_LOADED = False

# SadTalker 配置
SADTALKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SadTalker")
SADTALKER_IMAGE = os.path.join(SADTALKER_DIR, "my_photo.png")  # 数字人图片（默认）
SADTALKER_OUTPUT_DIR = os.path.join(SADTALKER_DIR, "results")
AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_output")
AVATARS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatars")
SPEECH_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_input")
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")  # 新增：用户上传目录

# 待机视频配置
IDLE_VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idle_videos")
SPEAKING_VIDEOS_DIR = os.path.join(app.static_folder, "speaking_videos")

# 确保输出目录存在
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(SADTALKER_OUTPUT_DIR, exist_ok=True)
os.makedirs(AVATARS_DIR, exist_ok=True)
os.makedirs(SPEECH_INPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)  # 新增：创建上传目录

# 允许上传的图片格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# 检查 avatars 目录下是否有默认图片，如果没有则创建
default_avatar_path = os.path.join(AVATARS_DIR, "avatar1.png")
if not os.path.exists(default_avatar_path):
    # 将 SadTalker 的默认图片复制到 avatars 目录作为 avatar1
    if os.path.exists(SADTALKER_IMAGE):
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

# 当前使用的数字人图片（全局变量，默认为 SadTalker 默认图片）
current_avatar_image = SADTALKER_IMAGE

# 本地模型全局变量
local_tokenizer = None
local_model = None

# ==================== 本地模型加载函数 ====================
def load_local_model():
    """在后台线程中加载本地模型"""
    global local_tokenizer, local_model, MODEL_LOADED

    if not TORCH_AVAILABLE:
        logger.error("PyTorch 未安装，无法加载本地模型")
        return

    try:
        logger.info("开始加载本地心理大模型...")
        logger.info(f"模型路径: {LOCAL_MODEL_PATH}")
        logger.info(f"适配器路径: {LOCAL_ADAPTER_PATH}")

        # 检查模型文件是否存在
        if not os.path.exists(LOCAL_MODEL_PATH):
            logger.error(f"模型路径不存在: {LOCAL_MODEL_PATH}")
            return

        # 加载分词器
        local_tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True
        )
        logger.info("分词器加载完成")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("基础模型加载完成")

        # 加载 PEFT 适配器（心理领域微调）
        if os.path.exists(LOCAL_ADAPTER_PATH):
            local_model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_PATH)
            logger.info("PEFT 适配器加载完成")
        else:
            local_model = base_model
            logger.warning(f"适配器路径不存在，使用基础模型: {LOCAL_ADAPTER_PATH}")

        local_model.eval()  # 设置为评估模式
        MODEL_LOADED = True
        logger.info("✅ 本地心理大模型加载完成！")

    except FileNotFoundError as e:
        logger.error(f"模型文件未找到: {e}")
        MODEL_LOADED = False
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        MODEL_LOADED = False

# ==================== 辅助函数 ====================
def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_image(file_path: str, target_path: str) -> bool:
    """
    处理上传的图片，确保适合 SadTalker 使用

    Args:
        file_path: 原始文件路径
        target_path: 目标文件路径

    Returns:
        处理是否成功
    """
    try:
        # 读取图片
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"无法读取图片: {file_path}")
            return False

        # 检查图片尺寸，如果太大则调整
        height, width = img.shape[:2]
        max_size = 1024

        if height > max_size or width > max_size:
            # 计算缩放比例
            scale = max_size / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)

            # 调整尺寸
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"调整图片尺寸: {width}x{height} -> {new_width}x{new_height}")

        # 保存为PNG格式（SadTalker 推荐格式）
        cv2.imwrite(target_path, img)
        logger.info(f"图片已保存为: {target_path}")

        return True

    except Exception as e:
        logger.error(f"处理图片失败: {str(e)}")
        return False

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


# ==================== 本地模型心理分析函数 ====================
def generate_local_model_response(user_input: str, emotion: str = "neutral") -> str:
    """
    使用本地模型生成心理咨询响应

    Args:
        user_input: 用户输入的文本
        emotion: 检测到的情绪类型

    Returns:
        生成的心理咨询响应文本
    """
    # 检查模型是否已加载
    if not MODEL_LOADED or local_model is None or local_tokenizer is None:
        return "本地模型正在加载中，请稍候..."

    try:
        # 获取情绪上下文描述
        emotion_context = EMOTION_MAP.get(emotion, EMOTION_MAP['neutral'])['context']

        # 构建提示词
        prompt = f"""【心理助手】指令：请以专业心理助手的身份回应用户的心理问题
输入：用户说：{user_input}
情绪状态：{emotion_context}
回答："""

        # 编码输入
        inputs = local_tokenizer(prompt, return_tensors="pt").to(local_model.device)

        # 生成回答
        with torch.no_grad():
            outputs = local_model.generate(
                **inputs,
                max_new_tokens=300,      # 最大生成 token 数
                do_sample=True,          # 启用采样
                temperature=0.7,         # 温度参数（控制随机性）
                top_p=0.9,               # 核采样参数
                repetition_penalty=1.1,  # 重复惩罚
                pad_token_id=local_tokenizer.eos_token_id
            )

        # 解码输出
        response = local_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的回答部分（去掉提示词）
        generated_response = response[len(prompt):].strip()

        # 清理响应（移除可能的后续对话）
        if "用户：" in generated_response:
            generated_response = generated_response.split("用户：")[0].strip()
        if "【心理助手】" in generated_response:
            generated_response = generated_response.split("【心理助手】")[0].strip()

        return generated_response if generated_response else "我理解您的感受，请继续告诉我更多。"

    except Exception as e:
        logger.error(f"本地模型生成响应时出错: {e}")
        return "抱歉，我在生成回复时遇到了问题。请再试一次。"


# ==================== 心理分析代理类（修改版） ====================
class PsychologicalAgent:
    """
    心理分析代理类

    负责使用本地模型，提供心理咨询服务。
    支持多轮对话，结合用户情绪状态生成个性化回复。
    """

    def __init__(self):
        """
        初始化心理分析代理
        """
        self.model_loaded = MODEL_LOADED

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
            # 首先尝试使用本地模型
            if self.model_loaded and MODEL_LOADED:
                logger.info(f"使用本地模型分析 - 情绪: {emotion}, 输入: {user_input[:50]}...")

                # 构建完整的提示词
                full_prompt = self._build_system_prompt(emotion) + f"\n\n用户说: {user_input}\n\n请回复:"

                # 使用本地模型生成响应
                response_text = generate_local_model_response(user_input, emotion)

                if response_text and len(response_text) > 20:  # 确保响应有效
                    # 更新对话历史
                    self._update_history(user_input, response_text)

                    return {
                        "success": True,
                        "response": response_text,
                        "model_source": "local_psychology_model"
                    }
                else:
                    # 本地模型生成失败，使用备选回复
                    logger.warning("本地模型生成响应无效，使用备选回复")

            # 使用备选回复系统
            logger.info(f"使用备选回复系统 - 情绪: {emotion}, 输入: {user_input[:50]}...")
            return generate_fallback_response(user_input, emotion)

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
    生成备选回复（当本地模型不可用时使用）

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


# ==================== Edge TTS 文字转语音模块 ====================
def text_to_speech(text: str) -> Dict[str, Any]:
    """
    使用 Edge TTS 将文字转换为语音
    
    支持的中文音色：
    - zh-CN-XiaoxiaoNeural (晓晓，女声，推荐)
    - zh-CN-YunxiNeural (云希，男声)
    - zh-CN-YunxiaNeural (云夏，男声)
    - zh-CN-YunyangNeural (云扬，男声)
    """
    try:
        # 生成唯一文件名
        audio_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)

        logger.info(f"开始 Edge TTS 转换，文本长度: {len(text)}")
        
        # 清理文本，移除 HTML 标签
        import re
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."
            logger.warning(f"文本过长，截断为1000字符")
        
        # 异步生成语音
        async def generate_speech():
            communicate = edge_tts.Communicate(
                clean_text,
                "zh-CN-XiaoxiaoNeural",  # 中文女声音色
                rate="+0%",              # 语速
                volume="+0%"             # 音量
            )
            await communicate.save(audio_path)
        
        # 运行异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_speech())
        loop.close()
        
        # 验证文件
        file_size = os.path.getsize(audio_path)
        logger.info(f"音频文件生成成功: {audio_path}, 大小: {file_size} bytes")
        
        if file_size < 100:  # 文件太小可能是错误
            logger.error(f"生成的音频文件太小，可能失败: {file_size} bytes")
            return create_silent_audio(audio_filename, audio_path)
        
        return {
            "success": True,
            "audio_path": audio_path,
            "audio_filename": audio_filename,
            "tts_engine": "edge-tts",
            "voice": "zh-CN-XiaoxiaoNeural"
        }

    except Exception as e:
        logger.error(f"Edge TTS 转换失败: {str(e)}", exc_info=True)
        return create_silent_audio(audio_filename, audio_path)

def create_silent_audio(filename: str, path: str) -> Dict[str, Any]:
    """
    创建静音音频（备用方案）
    """
    try:
        # 使用 pydub 创建静音音频
        silent_audio = AudioSegment.silent(duration=1000)  # 1秒静音
        silent_audio.export(path, format="mp3")
        
        logger.warning(f"创建静音音频: {path}")
        return {
            "success": True,
            "audio_path": path,
            "audio_filename": filename,
            "tts_engine": "silent_audio",
            "is_silent": True,
            "warning": "TTS失败，使用静音音频"
        }
        
    except Exception as e:
        logger.error(f"创建静音音频失败: {str(e)}")
        return {
            "success": False, 
            "error": f"所有TTS方案都失败了: {str(e)}"
        }

# ==================== SadTalker 视频生成模块 ====================
def generate_talking_video(audio_path: str, image_path: str = None) -> Dict[str, Any]:
    """
    使用 SadTalker 生成数字人说话视频

    Args:
        audio_path: 音频文件的绝对路径
        image_path: 数字人图片路径（可选，默认使用当前设置的图片）

    Returns:
        包含视频文件路径的字典
    """
    try:
        # 使用指定的图片或默认图片
        if image_path is None:
            image_path = current_avatar_image

        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"数字人图片不存在: {image_path}")
            return {"success": False, "error": "数字人图片不存在"}

        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return {"success": False, "error": "音频文件不存在"}

        logger.info("开始生成数字人视频...")
        logger.info(f"图片: {image_path}")
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
            "--source_image", image_path,
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
agent = PsychologicalAgent()

# 在后台线程中加载本地模型
if TORCH_AVAILABLE:
    model_thread = threading.Thread(target=load_local_model, daemon=True)
    model_thread.start()
    logger.info("已启动本地模型加载线程")


# ==================== API 路由 ====================

@app.route('/')
def root_redirect():
    return redirect('/select')

@app.route('/index')
def main_page():
    # 使用 render_template
    return render_template('index.html')

@app.route('/select')
def select_page():
    # 使用 render_template
    return render_template('select.html')


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


@app.route('/uploads/<filename>')
def serve_uploaded_avatar(filename):
    """
    提供用户上传的数字人图片服务
    """
    try:
        return send_from_directory('uploads', filename)
    except FileNotFoundError:
        logger.error(f"上传的图片未找到: {filename}")
        return jsonify({"error": f"图片 {filename} 未找到"}), 404


@app.route('/api/set_avatar', methods=['POST'])
def set_sadtalker_image():
    """
    设置 SadTalker 使用的数字人图片
    """
    global current_avatar_image

    try:
        data = request.get_json()
        avatar_id = data.get('avatar_id', '1')

        if avatar_id.startswith('upload_'):
            # 使用用户上传的图片
            uploaded_filename = avatar_id.replace('upload_', '')
            avatar_path = os.path.join(UPLOADS_DIR, uploaded_filename)

            if os.path.exists(avatar_path):
                # 将上传的图片复制到 SadTalker 目录
                sadtalker_dest = os.path.join(SADTALKER_DIR, "my_photo.png")
                try:
                    shutil.copy2(avatar_path, sadtalker_dest)
                    current_avatar_image = sadtalker_dest
                    logger.info(f"已设置上传的图片为数字人形象: {uploaded_filename}")

                    return jsonify({
                        "success": True,
                        "message": f"已使用您上传的图片",
                        "image_url": f"/uploads/{uploaded_filename}"
                    })
                except Exception as e:
                    logger.warning(f"复制上传图片失败: {str(e)}")
                    return jsonify({"success": False, "error": "设置上传图片失败"}), 500
            else:
                logger.error(f"上传的图片不存在: {avatar_path}")
                return jsonify({"success": False, "error": "上传的图片不存在"}), 404
        else:
            # 使用预置的数字人图片
            avatar_images = {
                '1': 'avatar1.png',
                '2': 'avatar2.png',
                '3': 'avatar3.png'
            }

            avatar_filename = avatar_images.get(avatar_id, 'avatar1.png')
            new_image_path = os.path.join('avatars', avatar_filename)

            # 检查文件是否存在
            if os.path.exists(new_image_path):
                # 将预置图片复制到 SadTalker 目录
                sadtalker_dest = os.path.join(SADTALKER_DIR, "my_photo.png")
                try:
                    shutil.copy2(new_image_path, sadtalker_dest)
                    current_avatar_image = sadtalker_dest
                    logger.info(f"已更新 SadTalker 图片为: {new_image_path}")

                    return jsonify({
                        "success": True,
                        "message": f"已切换为预置形象 {avatar_id}",
                        "image_url": f"/avatars/{avatar_filename}"
                    })
                except Exception as e:
                    logger.warning(f"复制图片失败: {str(e)}")
                    return jsonify({"success": False, "error": "设置图片失败"}), 500
            else:
                logger.error(f"数字人图片不存在: {new_image_path}")
                return jsonify({"success": False, "error": "图片文件不存在"}), 404

    except Exception as e:
        logger.error(f"设置数字人图片失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/upload_avatar', methods=['POST'])
def upload_avatar():
    """
    上传用户自定义数字人图片
    """
    try:
        # 检查是否有文件上传
        if 'avatar' not in request.files:
            return jsonify({"success": False, "error": "没有上传文件"}), 400

        file = request.files['avatar']

        # 检查文件名
        if file.filename == '':
            return jsonify({"success": False, "error": "没有选择文件"}), 400

        # 检查文件格式
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "不支持的文件格式。请上传图片文件 (png, jpg, jpeg, gif, bmp, webp)"}), 400

        # 生成安全的文件名
        filename = secure_filename(file.filename)
        # 添加时间戳和随机字符串避免重名
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)

        # 保存原始文件
        file.save(file_path)
        logger.info(f"用户上传图片保存到: {file_path}")

        # 处理图片（调整尺寸等）
        processed_filename = f"processed_{unique_filename}"
        processed_path = os.path.join(UPLOADS_DIR, processed_filename)

        if process_uploaded_image(file_path, processed_path):
            # 处理成功，使用处理后的文件
            final_path = processed_path
            final_filename = processed_filename
            # 删除原始文件
            try:
                os.remove(file_path)
            except:
                pass
        else:
            # 处理失败，使用原始文件
            final_path = file_path
            final_filename = unique_filename
            logger.warning(f"图片处理失败，使用原始文件: {final_path}")

        # 生成上传成功的响应
        response_data = {
            "success": True,
            "message": "图片上传成功",
            "filename": final_filename,
            "image_url": f"/uploads/{final_filename}",
            "avatar_id": f"upload_{final_filename}"
        }

        logger.info(f"用户图片上传成功: {final_filename}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"上传图片失败: {str(e)}")
        return jsonify({"success": False, "error": f"上传失败: {str(e)}"}), 500


@app.route('/api/get_avatars', methods=['GET'])
def get_available_avatars():
    """
    获取可用的数字人形象列表
    """
    try:
        avatars = []

        # 添加预置形象
        for i in range(1, 4):
            avatar_file = f"avatar{i}.png"
            avatar_path = os.path.join(AVATARS_DIR, avatar_file)
            if os.path.exists(avatar_path):
                avatars.append({
                    "id": str(i),
                    "name": f"预置形象 {i}",
                    "type": "preset",
                    "image_url": f"/avatars/{avatar_file}"
                })

        # 添加上传的形象
        upload_files = os.listdir(UPLOADS_DIR)
        for filename in upload_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                # 跳过已处理的文件（如果有processed_前缀）
                if not filename.startswith('processed_'):
                    avatars.append({
                        "id": f"upload_{filename}",
                        "name": f"我的形象: {filename[:20]}...",
                        "type": "uploaded",
                        "image_url": f"/uploads/{filename}"
                    })

        return jsonify({
            "success": True,
            "avatars": avatars,
            "total": len(avatars)
        })

    except Exception as e:
        logger.error(f"获取形象列表失败: {str(e)}")
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
    修改后的分析接口：仅生成 TTS 音频，并返回预置视频路径
    """
    try:
        data = request.get_json()
        logger.info(f"收到分析请求: {data}")
        
        user_input = data.get('message', '').strip()
        detected_emotion = data.get('detected_emotion', 'neutral')
        avatar_id = data.get('avatar_id', '1')  # 从请求中获取 avatar_id
        
        if not user_input:
            return jsonify({"success": False, "error": "输入不能为空"}), 400

        # 1. 调用心理分析 (LLM)
        logger.info(f"开始心理分析 - 输入: {user_input[:50]}..., 情绪: {detected_emotion}, 头像: {avatar_id}")
        result = agent.analyze(user_input, detected_emotion)

        if result["success"]:
            # 2. 生成 TTS 音频
            response_text = result.get("response", "")
            logger.info(f"心理分析成功，生成 TTS，文本长度: {len(response_text)}")
            
            # 清理文本
            import re
            clean_text = re.sub(r'<[^>]+>', '', response_text).replace('&nbsp;', ' ')
            
            tts_result = text_to_speech(clean_text)

            if tts_result["success"]:
                # 3. 返回音频 URL 和对应的预置说话视频 URL
                audio_url = f"/api/audio/{tts_result['audio_filename']}"
                
                # 4. 根据 avatar_id 返回对应的预置视频
                video_filename = f"avatar{avatar_id}_talking.mp4"
                video_path = os.path.join(app.static_folder, "speaking_videos", video_filename)
                
                logger.info(f"检查视频文件: {video_path}")
                
                if os.path.exists(video_path):
                    video_url = f"/static/speaking_videos/{video_filename}"
                    logger.info(f"使用预置视频: {video_url}")
                else:
                    # 如果特定 avatar 的视频不存在，使用 avatar1 作为默认
                    video_filename = "avatar1_talking.mp4"
                    video_url = f"/static/speaking_videos/{video_filename}"
                    logger.warning(f"预置视频不存在，使用默认: {video_filename}")
                
                result["audio_url"] = audio_url
                result["video_url"] = video_url
                result["video_generated"] = True
                result["is_preset"] = True
                result["avatar_id"] = avatar_id
                result["tts_engine"] = tts_result.get("tts_engine", "edge-tts")
                
                logger.info(f"返回结果: audio={audio_url}, video={video_url}")
            else:
                logger.error(f"TTS 生成失败: {tts_result.get('error')}")
                result["video_generated"] = False
                result["error"] = tts_result.get('error', 'TTS 生成失败')

        return jsonify(result)

    except Exception as e:
        logger.error(f"分析接口错误: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

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
    本地模型分析接口（与主接口相同，保持兼容性）

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

        # 调用心理分析代理
        result = agent.analyze(user_input, detected_emotion)

        if result["success"]:
            result["detected_emotion"] = detected_emotion
            result["timestamp"] = datetime.datetime.now().isoformat()

        return jsonify(result)

    except Exception as e:
        logger.error(f"本地分析接口错误: {str(e)}")
        return jsonify({"success": False, "error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/idle_videos/<filename>')
def serve_idle_video(filename):
    """
    提供待机视频文件服务
    """
    try:
        return send_from_directory(IDLE_VIDEOS_DIR, filename)
    except FileNotFoundError:
        logger.error(f"待机视频未找到: {filename}")
        return jsonify({"error": f"待机视频 {filename} 未找到"}), 404

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
        "local_model_loaded": MODEL_LOADED,
        "torch_available": TORCH_AVAILABLE,
        "model_loading": TORCH_AVAILABLE and not MODEL_LOADED,
        "deepface_available": DEEPFACE_AVAILABLE,
        "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
        "edge_tts_available": EDGE_TTS_AVAILABLE,
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
def api_status():
    """
    API 状态检查接口

    Returns:
        JSON 格式的 API 状态信息
    """
    model_status_info = {
        "local_model_loaded": MODEL_LOADED,
        "torch_available": TORCH_AVAILABLE,
        "model_loading": TORCH_AVAILABLE and not MODEL_LOADED
    }

    return jsonify({
        "status": "healthy" if (MODEL_LOADED or not TORCH_AVAILABLE) else "loading",
        "model_status": model_status_info,
        "deepface_available": DEEPFACE_AVAILABLE,
        "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
        "edge_tts_available": EDGE_TTS_AVAILABLE,
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
        "service": "大学生心理分析数字人代理（本地模型版）",
        "version": "2.4",
        "timestamp": datetime.datetime.now().isoformat(),
        "features": {
            "local_psychological_analysis": MODEL_LOADED,
            "fallback_system": True,
            "emotion_recognition": DEEPFACE_AVAILABLE,
            "real_time_camera": True,
            "avatar_selection": True,
            "avatar_upload": True,
            "speech_input": SPEECH_RECOGNITION_AVAILABLE,
            "edge_tts": EDGE_TTS_AVAILABLE,  # 新增 Edge TTS 功能
            "idle_animation": True
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


def get_idle_videos() -> list:
    """
    获取所有可用的待机视频
    """
    try:
        videos = []

        # 获取目录下所有视频文件
        video_extensions = ('.mp4', '.webm', '.mov', '.avi')

        for filename in os.listdir(IDLE_VIDEOS_DIR):
            if filename.lower().endswith(video_extensions):
                videos.append({
                    "filename": filename,
                    "url": f"/api/idle_videos/{filename}",  # 修改为正确的 API URL
                    "size": os.path.getsize(os.path.join(IDLE_VIDEOS_DIR, filename)),
                    "modified": os.path.getmtime(os.path.join(IDLE_VIDEOS_DIR, filename))
                })

        # 按修改时间排序（最新的在前）
        videos.sort(key=lambda x: x["modified"], reverse=True)

        logger.info(f"找到 {len(videos)} 个待机视频")
        return videos

    except Exception as e:
        logger.error(f"获取待机视频列表失败: {str(e)}")
        return []

def get_idle_video_for_avatar(avatar_id: str) -> Optional[str]:
    try:
        # 先尝试查找特定于该数字人的待机视频
        if avatar_id in ['1', '2', '3']:
            possible_filenames = [
                f"idle_avatar{avatar_id}.mp4",
            ]
        elif avatar_id.startswith('upload_'):
            possible_filenames = [
                "idle_default.mp4",
                "idle_default.webm",
                "idle_breathing.mp4",
                "idle_breathing.webm"
            ]
        else:
            possible_filenames = [
                "idle_default.mp4",
                "idle_default.webm"
            ]

        # 检查文件是否存在
        for filename in possible_filenames:
            video_path = os.path.join(IDLE_VIDEOS_DIR, filename)
            if os.path.exists(video_path):
                logger.info(f"为数字人 {avatar_id} 找到待机视频: {filename}")
                return f"/api/idle_videos/{filename}"  # 修改为正确的 API URL

        # 如果没有找到特定视频，返回第一个可用视频
        all_videos = get_idle_videos()
        if all_videos:
            return all_videos[0]['url']

        return None

    except Exception as e:
        logger.error(f"获取数字人待机视频失败: {str(e)}")
        return None


@app.route('/api/idle_videos')
def get_idle_videos_list():
    """
    获取可用的待机视频列表
    """
    try:
        avatar_id = request.args.get('avatar_id')

        if avatar_id:
            # 获取特定于该数字人的待机视频
            specific_video = get_idle_video_for_avatar(avatar_id)

            if specific_video:
                # 返回特定视频+其他通用视频
                all_videos = get_idle_videos()
                specific_filename = specific_video.split('/')[-1]

                # 将特定视频放在前面
                videos = [specific_video]
                for video in all_videos:
                    if video['filename'] != specific_filename:
                        videos.append(video['url'])

                return jsonify({
                    "success": True,
                    "videos": videos,
                    "specific_video": specific_video,
                    "avatar_id": avatar_id,
                    "total": len(videos),
                    "timestamp": datetime.datetime.now().isoformat()
                })

        # 默认返回所有视频
        videos = get_idle_videos()
        video_urls = [video['url'] for video in videos]

        return jsonify({
            "success": True,
            "videos": video_urls,
            "total": len(video_urls),
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"获取待机视频列表失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "videos": []
        }), 500


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
        "torch_available": TORCH_AVAILABLE,
        "local_model_loaded": MODEL_LOADED,
        "edge_tts_available": EDGE_TTS_AVAILABLE,
        "sadtalker_image": current_avatar_image,
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
    print("大学生心理分析数字人代理 v2.4（本地模型版）")
    print("=" * 60)
    print(f"📱 服务地址: http://localhost:5000")
    print(f"👤 形象选择: http://localhost:5000/select")
    print(f"💬 主页面: http://localhost:5000/index")
    print(f"📤 新增功能: 用户可上传自定义数字人形象")
    print(f"🎤 语音输入: 支持（需要浏览器支持语音识别）")
    print(f"🔊 TTS引擎: Edge TTS（离线免费）")
    print(f"🔄 待机动画: 数字人空闲时会有呼吸和摆动动画")
    print(f"🧠 本地模型: {'已加载' if MODEL_LOADED else '加载中' if TORCH_AVAILABLE else '未安装'}")
    print(f"❤️  健康检查: http://localhost:5000/api/health")
    print(f"📊 模型状态: http://localhost:5000/api/model/status")
    print(f"🔍 调试信息: http://localhost:5000/api/debug")
    print("=" * 60)
    print("可用 API 端点:")
    print("  POST /api/analyze        - 心理分析（使用本地模型）")
    print("  POST /api/analyze_local  - 本地模型分析（兼容接口）")
    print("  POST /api/analyze_emotion - 表情识别")
    print("  POST /api/set_avatar     - 设置数字人形象")
    print("  POST /api/upload_avatar  - 上传自定义形象")
    print("  GET  /api/get_avatars    - 获取可用形象列表")
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

    # 检查 Edge TTS 功能
    if EDGE_TTS_AVAILABLE:
        print("✅ Edge TTS 功能已启用")
    else:
        print("⚠️  Edge TTS 未启用，请运行: pip install edge-tts")

    # 检查本地模型状态
    if TORCH_AVAILABLE:
        if MODEL_LOADED:
            print("✅ 本地心理大模型已加载")
        else:
            print("🔄 本地心理大模型正在后台加载中...")
    else:
        print("⚠️  PyTorch 未安装，将使用备选回复系统")
        print("    如需使用本地模型，请运行: pip install torch transformers peft")
    
    print("🚀 服务启动中...")
    print("=" * 60)

    # 启动 Flask 服务
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # 禁用自动重载，避免模型重复加载
    )