# app.py - 修复API路由版本
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import requests
import cv2
import numpy as np
import base64
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入deepface
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("警告: deepface 库未安装。请运行: pip install deepface")

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# DeepSeek API配置 - 请确保这是有效的API密钥
DEEPSEEK_API_KEY = "sk-215440b00f1d426fb21a2f11eef6cf02"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 情绪映射
EMOTION_ICONS = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
    'sad': '😢', 'surprise': '😲', 'neutral': '😐'
}
EMOTION_DESCRIPTIONS = {
    'angry': '生气', 'disgust': '厌恶', 'fear': '恐惧', 'happy': '开心',
    'sad': '悲伤', 'surprise': '惊讶', 'neutral': '平静'
}

# 对话历史
conversation_history = []

class PsychologicalAgent:
    """心理分析代理"""
    def __init__(self, api_key):
        self.api_key = api_key
        
    def analyze_with_deepseek(self, user_input, emotion="neutral"):
        """使用DeepSeek API分析"""
        try:
            # 构建情绪上下文
            emotion_context = {
                'happy': '看起来心情不错',
                'sad': '情绪有些低落',
                'angry': '有些生气',
                'fear': '感到紧张',
                'neutral': '情绪平稳',
                'surprise': '有些惊讶',
                'disgust': '有些反感'
            }.get(emotion, '情绪平稳')
            
            # 构建系统提示
            system_prompt = f"""你是一名专业的大学心理健康顾问，专门帮助大学生解决心理问题。
            
重要上下文信息：
- 系统检测到用户当前的情绪状态为：{emotion} ({emotion_context})
- 这个情绪信息来自实时面部表情分析
- 请结合用户描述的文字内容和检测到的情绪状态，提供更精准的心理分析

请以温暖、专业、支持性的语气回应，避免使用专业术语，用通俗易懂的语言提供建议。"""
            
            # 构建消息
            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加最近的对话历史（最多3轮）
            if conversation_history:
                recent_history = conversation_history[-6:]  # 最近3轮对话
                messages.extend(recent_history)
            
            messages.append({"role": "user", "content": user_input})
            
            logger.info(f"调用DeepSeek API，情绪: {emotion}, 输入: {user_input[:50]}...")
            
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 800,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['choices'][0]['message']['content']
                
                # 更新对话历史
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": assistant_response})
                
                # 限制历史记录长度
                if len(conversation_history) > 10:
                    conversation_history[:] = conversation_history[-10:]
                
                return {
                    "success": True,
                    "response": assistant_response,
                    "model_source": "deepseek_api"
                }
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.text[:200]}")
                return {
                    "success": False,
                    "error": f"API错误: {response.status_code}",
                    "response": "抱歉，AI服务暂时不可用，请稍后再试。"
                }
                
        except requests.exceptions.Timeout:
            logger.error("API请求超时")
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

# 初始化Agent
agent = PsychologicalAgent(DEEPSEEK_API_KEY)

@app.route('/')
def index():
    """提供前端页面"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>大学生心理分析数字人代理</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 40px; text-align: center; }
                h1 { color: #4a90e2; }
                .status { padding: 20px; margin: 20px auto; max-width: 600px; border-radius: 10px; }
                .success { background: #d4edda; color: #155724; }
                .warning { background: #fff3cd; color: #856404; }
                .error { background: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <h1>大学生心理分析数字人代理</h1>
            <div class="status success">
                <h3>系统正在运行</h3>
                <p>API服务正常，但index.html文件未找到</p>
                <p>请确保index.html文件与app.py在同一目录</p>
                <p>API测试：<a href="/api/health">/api/health</a></p>
            </div>
        </body>
        </html>
        """

# 前端需要的API端点 - 必须与index.html中的调用匹配
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """通用分析接口 - 前端主要调用这个"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        detected_emotion = data.get('detected_emotion', 'neutral')
        
        logger.info(f"收到分析请求 - 情绪: {detected_emotion}")
        
        if not user_input:
            return jsonify({
                "success": False,
                "error": "输入不能为空"
            }), 400
        
        # 使用DeepSeek API
        result = agent.analyze_with_deepseek(user_input, detected_emotion)
        
        if result["success"]:
            result["detected_emotion"] = detected_emotion
            result["model_source"] = result.get("model_source", "deepseek_api")
            result["timestamp"] = datetime.datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"分析接口错误: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"服务器错误: {str(e)}"
        }), 500

@app.route('/api/analyze_local', methods=['POST'])
def analyze_local():
    """本地分析接口 - 前端会调用这个"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        detected_emotion = data.get('detected_emotion', 'neutral')
        
        logger.info(f"收到本地分析请求 - 情绪: {detected_emotion}")
        
        if not user_input:
            return jsonify({
                "success": False,
                "error": "输入不能为空"
            }), 400
        
        # 直接使用DeepSeek API（本地模型不可用时）
        result = agent.analyze_with_deepseek(user_input, detected_emotion)
        
        if result["success"]:
            result["detected_emotion"] = detected_emotion
            result["model_source"] = "deepseek_api"  # 标记为API
            result["timestamp"] = datetime.datetime.now().isoformat()
        else:
            # 如果API失败，使用备选回复
            result = generate_fallback_response_data(user_input, detected_emotion)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"本地分析接口错误: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"服务器错误: {str(e)}"
        }), 500

@app.route('/api/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """分析表情接口"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "没有提供图片数据"
            }), 400
        
        logger.info("收到表情分析请求")
        
        emotion_result = analyze_emotion_from_image(data['image'])
        
        return jsonify({
            "success": True,
            "dominant_emotion": emotion_result["dominant_emotion"],
            "emotion_scores": emotion_result["emotion_scores"],
            "face_detected": emotion_result.get("face_detected", False),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"表情分析接口错误: {str(e)}")
        # 确保返回的值都是JSON可序列化的
        return jsonify({
            "success": False,
            "error": str(e),
            "dominant_emotion": "neutral",
            "emotion_scores": {
                "angry": 0.0, "disgust": 0.0, "fear": 0.0,
                "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 100.0
            },
            "face_detected": False
        }), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """模型状态接口 - 前端会调用这个"""
    return jsonify({
        "local_model_loaded": False,  # 暂时设为False
        "model_loading": False,
        "deepface_available": DEEPFACE_AVAILABLE,
        "deepseek_api_available": True,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/api/model_status', methods=['GET'])
def model_status_alt():
    """模型状态接口的另一种路由"""
    return model_status()

@app.route('/api/status', methods=['GET'])
def api_status():
    """API状态检查"""
    try:
        # 测试API连接
        api_test_result = test_deepseek_api()
        
        return jsonify({
            "status": "healthy" if api_test_result.get("success") else "warning",
            "deepseek_api": api_test_result,
            "deepface_available": DEEPFACE_AVAILABLE,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"状态检查失败: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "大学生心理分析数字人代理",
        "version": "2.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "features": {
            "psychological_analysis": True,
            "emotion_recognition": DEEPFACE_AVAILABLE,
            "real_time_camera": True,
            "deepseek_api": True
        }
    })

# 辅助函数
def analyze_emotion_from_image(image_data):
    """分析图片中的情绪"""
    if not DEEPFACE_AVAILABLE:
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {
                "angry": 0.0, "disgust": 0.0, "fear": 0.0,
                "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 100.0
            },
            "face_detected": False
        }
    
    try:
        # 解码Base64图片
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "dominant_emotion": "neutral",
                "emotion_scores": {
                    "angry": 0.0, "disgust": 0.0, "fear": 0.0,
                    "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 100.0
                },
                "face_detected": False
            }
        
        # 使用DeepFace分析
        try:
            analysis = DeepFace.analyze(
                img, 
                actions=['emotion'], 
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            
            if analysis is None or len(analysis) == 0:
                return {
                    "dominant_emotion": "neutral",
                    "emotion_scores": {
                        "angry": 0.0, "disgust": 0.0, "fear": 0.0,
                        "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 100.0
                    },
                    "face_detected": False
                }
            
            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_scores = analysis[0]['emotion']
            
            # 关键修复：将float32转换为Python float
            converted_scores = {}
            for emotion, score in emotion_scores.items():
                # 确保所有值都是Python float类型
                converted_scores[emotion] = float(score)
            
            return {
                "dominant_emotion": dominant_emotion,
                "emotion_scores": converted_scores,
                "face_detected": True
            }
            
        except Exception as e:
            logger.warning(f"DeepFace分析失败: {str(e)[:100]}")
            return {
                "dominant_emotion": "neutral",
                "emotion_scores": {
                    "angry": 0.0, "disgust": 0.0, "fear": 0.0,
                    "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 100.0
                },
                "face_detected": False
            }
        
    except Exception as e:
        logger.error(f"情绪分析过程出错: {str(e)}")
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {
                "angry": 0.0, "disgust": 0.0, "fear": 0.0,
                "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 100.0
            },
            "face_detected": False
        }

def generate_fallback_response_data(user_input, emotion):
    """生成备选回复数据"""
    emotion_context = {
        'happy': '看起来您心情不错',
        'sad': '感受到您的低落情绪',
        'angry': '理解您的烦躁',
        'fear': '感受到您的紧张',
        'neutral': ''
    }.get(emotion, '')
    
    if emotion_context:
        emotion_context += "，"
    
    # 关键词匹配
    keyword_responses = {
        '压力': f"{emotion_context}对于压力问题，建议：<br>1. 深呼吸放松练习<br>2. 合理安排时间和优先级<br>3. 适量运动释放压力<br>4. 与朋友或家人倾诉",
        '焦虑': f"{emotion_context}应对焦虑的方法：<br>1. 正念冥想练习<br>2. 写下担忧事项<br>3. 渐进式肌肉放松<br>4. 保持规律作息",
        '失眠': f"{emotion_context}改善睡眠的建议：<br>1. 睡前1小时不使用电子设备<br>2. 创造舒适的睡眠环境<br>3. 保持规律的作息时间<br>4. 避免睡前摄入咖啡因"
    }
    
    lower_input = user_input.lower()
    for keyword, response in keyword_responses.items():
        if keyword in lower_input:
            return {
                "success": True,
                "response": response,
                "detected_emotion": emotion,
                "model_source": "fallback_system",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    generic_response = f"{emotion_context}我理解您的困扰。作为心理助手，我建议您可以更详细地描述具体情况和感受，这样我能提供更有针对性的帮助。"
    
    return {
        "success": True,
        "response": generic_response,
        "detected_emotion": emotion,
        "model_source": "fallback_system",
        "timestamp": datetime.datetime.now().isoformat()
    }

def test_deepseek_api():
    """测试DeepSeek API连接"""
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
            return {"success": True, "message": "API连接正常"}
        elif response.status_code == 401:
            return {"success": False, "message": "API密钥无效"}
        else:
            return {"success": False, "message": f"API返回错误: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"success": False, "message": "API连接超时"}
    except Exception as e:
        return {"success": False, "message": f"API连接失败: {str(e)}"}

@app.route('/api/conversation/summary', methods=['GET'])
def get_conversation_summary():
    """获取对话摘要"""
    user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    
    return jsonify({
        "total_conversations": len(conversation_history) // 2,
        "recent_topics": user_messages[-3:] if user_messages else [],
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/api/conversation/reset', methods=['POST'])
def reset_conversation():
    """重置对话"""
    global conversation_history
    conversation_history = []
    return jsonify({"success": True, "message": "对话已重置"})

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """调试信息"""
    return jsonify({
        "routes": [str(rule) for rule in app.url_map.iter_rules()],
        "conversation_length": len(conversation_history),
        "deepface_available": DEEPFACE_AVAILABLE,
        "api_key_set": bool(DEEPSEEK_API_KEY),
        "timestamp": datetime.datetime.now().isoformat()
    })

# 添加favicon.ico路由避免404错误
@app.route('/favicon.ico')
def favicon():
    return '', 404

if __name__ == '__main__':
    print("=" * 60)
    print("大学生心理分析数字人代理 - API路由修复版")
    print("=" * 60)
    print(f"📱 服务地址: http://localhost:5000")
    print(f"🔍 调试信息: http://localhost:5000/api/debug")
    print(f"❤️  健康检查: http://localhost:5000/api/health")
    print(f"📊 模型状态: http://localhost:5000/api/model/status")
    print("=" * 60)
    print("🚀 服务启动中...")
    print("=" * 60)
    
    # 显示所有可用路由
    print("可用API端点:")
    for rule in app.url_map.iter_rules():
        if rule.rule.startswith('/api') or rule.rule == '/':
            print(f"  {rule.rule}")
    
    print("=" * 60)
    
    # 启动服务
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)