"""
大学生心理分析数字人代理 - 集成服务器（本地模型版）
====================================================

本模块提供支持本地心理大模型的 Flask 服务，功能包括：
1. 本地 Qwen 模型 + PEFT 适配器加载
2. 心理咨询响应生成
3. 备选回复系统

适用场景：
- 需要离线运行的环境
- 对数据隐私有要求的场景
- 需要自定义微调模型的场景

作者: SRTP 项目组
版本: 1.0
"""

import datetime
import logging
import threading
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PyTorch 相关导入 ====================
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TORCH_AVAILABLE = True
    logger.info("PyTorch 和 Transformers 库加载成功")
except ImportError as e:
    TORCH_AVAILABLE = False
    logger.warning(f"PyTorch 相关库未安装: {e}")
    logger.warning("本地模型功能不可用，请安装: pip install torch transformers peft")

# ==================== Flask 应用初始化 ====================
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# ==================== 全局模型变量 ====================
model = None           # 加载的模型实例
tokenizer = None       # 分词器实例
model_loaded = False   # 模型加载状态标志

# ==================== 配置常量 ====================
# 基础模型路径（Qwen1.5-0.5B）
BASE_MODEL_PATH = "C:\\Users\\legion\\.cache\\huggingface\\hub\\models--Qwen--Qwen1.5-0.5B\\snapshots\\8f445e3628f3500ee69f24e1303c9f10f5342a39"

# 微调适配器路径
ADAPTER_PATH = "outputs/psychology_trained_model"

# 情绪上下文映射
EMOTION_CONTEXT_MAP = {
    'happy':   '用户看起来心情愉快',
    'sad':     '用户情绪低落',
    'angry':   '用户有些生气',
    'fear':    '用户感到紧张',
    'surprise': '用户感到惊讶',
    'disgust': '用户有些反感',
    'neutral': '用户情绪平稳'
}


# ==================== 模型加载模块 ====================
def load_psychology_model() -> None:
    """
    加载心理大模型

    该函数在后台线程中执行，加载基础模型和 PEFT 适配器。
    加载完成后设置 model_loaded 标志为 True。
    """
    global model, tokenizer, model_loaded

    if not TORCH_AVAILABLE:
        logger.error("PyTorch 未安装，无法加载模型")
        return

    try:
        logger.info("正在加载心理大模型...")
        logger.info(f"基础模型路径: {BASE_MODEL_PATH}")
        logger.info(f"适配器路径: {ADAPTER_PATH}")

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )
        logger.info("分词器加载完成")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,  # 使用半精度以节省显存
            device_map="auto",          # 自动分配设备
            trust_remote_code=True
        )
        logger.info("基础模型加载完成")

        # 加载 PEFT 适配器（心理领域微调）
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()  # 设置为评估模式
        logger.info("PEFT 适配器加载完成")

        model_loaded = True
        logger.info("✅ 心理大模型加载完成！")

    except FileNotFoundError as e:
        logger.error(f"模型文件未找到: {e}")
        logger.error("请确保模型文件存在于指定路径")
        model_loaded = False
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        model_loaded = False


# ==================== 响应生成模块 ====================
def generate_psychology_response(user_input: str, emotion: str = "neutral") -> str:
    """
    使用本地模型生成心理咨询响应

    Args:
        user_input: 用户输入的文本
        emotion: 检测到的情绪类型

    Returns:
        生成的心理咨询响应文本
    """
    # 检查模型是否已加载
    if not model_loaded or model is None or tokenizer is None:
        return "模型正在加载中，请稍候..."

    try:
        # 获取情绪上下文描述
        emotion_context = EMOTION_CONTEXT_MAP.get(emotion, EMOTION_CONTEXT_MAP['neutral'])

        # 构建提示词
        prompt = f"""【心理助手】指令：请以专业心理助手的身份回应用户的心理问题
输入：用户说：{user_input}
情绪状态：{emotion_context}
回答："""

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,      # 最大生成 token 数
                do_sample=True,          # 启用采样
                temperature=0.7,         # 温度参数（控制随机性）
                top_p=0.9,               # 核采样参数
                repetition_penalty=1.1,  # 重复惩罚
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的回答部分（去掉提示词）
        generated_response = response[len(prompt):].strip()

        # 清理响应（移除可能的后续对话）
        if "用户：" in generated_response:
            generated_response = generated_response.split("用户：")[0].strip()
        if "【心理助手】" in generated_response:
            generated_response = generated_response.split("【心理助手】")[0].strip()

        return generated_response if generated_response else "我理解您的感受，请继续告诉我更多。"

    except Exception as e:
        logger.error(f"生成响应时出错: {e}")
        return "抱歉，我在生成回复时遇到了问题。请再试一次。"


def generate_fallback_response(user_input: str, emotion: str = "neutral") -> str:
    """
    生成备选回复（当模型不可用时使用）

    Args:
        user_input: 用户输入的文本
        emotion: 检测到的情绪类型

    Returns:
        基于关键词匹配的备选回复
    """
    # 情绪前缀映射
    emotion_prefix_map = {
        'happy':   '虽然您情绪还好，但',
        'sad':     '在情绪低落时，',
        'angry':   '在烦躁的时候，',
        'fear':    '感到不安时，',
        'surprise': '感到意外时，',
        'disgust': '感到不适时，',
        'neutral': ''
    }

    # 关键词响应映射
    keyword_responses = {
        '压力': "面对压力时，建议：1.制定合理计划 2.学会说'不' 3.适当放松 4.保持运动",
        '焦虑': "应对焦虑：1.深呼吸练习 2.正念冥想 3.与朋友倾诉 4.寻求专业帮助",
        '失眠': "改善睡眠：1.规律作息 2.睡前放松 3.避免咖啡因 4.舒适环境",
        '情绪': "情绪管理：1.识别情绪 2.接纳感受 3.健康表达 4.转移注意力",
        '抑郁': "情绪低落时：1.寻求支持 2.保持活动 3.专业咨询 4.耐心对待自己",
        '学习': "学习困扰：1.制定计划 2.劳逸结合 3.寻求帮助 4.保持信心",
        '人际': "人际关系：1.真诚沟通 2.换位思考 3.保持边界 4.寻求共识"
    }

    # 获取情绪前缀
    emotion_prefix = emotion_prefix_map.get(emotion, '')

    # 查找匹配的关键词
    for keyword, response in keyword_responses.items():
        if keyword in user_input:
            return f"{emotion_prefix}{response}"

    # 通用回复
    return "我理解您的困扰。作为心理助手，我建议您可以更详细地描述具体情况，这样我能提供更有针对性的帮助。"


# ==================== API 路由 ====================

@app.route('/')
def index():
    """
    首页路由 - 提供前端页面

    Returns:
        渲染的 HTML 模板
    """
    return render_template('index.html')


@app.route('/api/analyze_psychology', methods=['POST'])
def analyze_psychology():
    """
    心理分析 API 接口

    请求体:
        - message: 用户输入的文本
        - detected_emotion: 检测到的情绪（可选，默认 neutral）

    Returns:
        JSON 格式的分析结果
    """
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        detected_emotion = data.get('detected_emotion', 'neutral')

        logger.info(f"收到心理分析请求 - 情绪: {detected_emotion}, 输入: {user_input[:50]}...")

        # 验证输入
        if not user_input:
            return jsonify({
                "success": False,
                "error": "输入不能为空"
            }), 400

        # 生成响应
        if model_loaded:
            response = generate_psychology_response(user_input, detected_emotion)
            model_source = "local_psychology_model"
        else:
            response = generate_fallback_response(user_input, detected_emotion)
            model_source = "fallback_system"

        # 如果响应太短，使用备选回复
        if len(response) < 20:
            response = generate_fallback_response(user_input, detected_emotion)
            model_source = "fallback_system"

        return jsonify({
            "success": True,
            "response": response,
            "detected_emotion": detected_emotion,
            "model_source": model_source,
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"心理分析接口错误: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "心理分析失败"
        }), 500


@app.route('/api/model_status', methods=['GET'])
def get_model_status():
    """
    模型状态查询接口

    Returns:
        JSON 格式的模型状态信息
    """
    return jsonify({
        "model_loaded": model_loaded,
        "torch_available": TORCH_AVAILABLE,
        "status": "ready" if model_loaded else ("loading" if TORCH_AVAILABLE else "unavailable"),
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
        "service": "大学生心理分析数字人代理 - 本地模型版",
        "version": "1.0",
        "model_loaded": model_loaded,
        "timestamp": datetime.datetime.now().isoformat()
    })


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 打印启动信息
    print("=" * 60)
    print("大学生心理分析数字人代理 - 本地模型版 v1.0")
    print("=" * 60)
    print(f"📱 服务地址: http://localhost:5000")
    print(f"❤️  健康检查: http://localhost:5000/api/health")
    print(f"📊 模型状态: http://localhost:5000/api/model_status")
    print("=" * 60)

    if TORCH_AVAILABLE:
        # 在后台线程中加载模型，避免阻塞服务启动
        print("🔄 正在后台加载心理大模型...")
        model_thread = threading.Thread(target=load_psychology_model, daemon=True)
        model_thread.start()
    else:
        print("⚠️  PyTorch 未安装，将使用备选回复系统")

    print("=" * 60)
    print("🚀 服务启动中...")
    print("=" * 60)

    # 启动 Flask 服务
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # 禁用自动重载，避免模型重复加载
    )
