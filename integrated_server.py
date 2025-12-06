from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import datetime
import json
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)
CORS(app)

# 全局模型变量
model = None
tokenizer = None
model_loaded = False

def load_psychology_model():
    """加载心理大模型"""
    global model, tokenizer, model_loaded
    
    try:
        print("正在加载心理大模型...")
        
        # 基础模型路径
        base_model_path = "C:\\Users\\legion\\.cache\\huggingface\\hub\\models--Qwen--Qwen1.5-0.5B\\snapshots\\8f445e3628f3500ee69f24e1303c9f10f5342a39"
        
        # 加载tokenizer和基础模型
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载训练好的心理领域适配器
        model = PeftModel.from_pretrained(base_model, "outputs/psychology_trained_model")
        model.eval()  # 设置为评估模式
        
        print("✅ 心理大模型加载完成！")
        model_loaded = True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        model_loaded = False

def generate_psychology_response(user_input, emotion):
    """生成心理助手响应"""
    if not model_loaded:
        return "模型正在加载中，请稍候..."
    
    try:
        # 构建提示词
        emotion_context = {
            'happy': '用户看起来心情愉快',
            'sad': '用户情绪低落',
            'angry': '用户有些生气',
            'fear': '用户感到紧张',
            'neutral': '用户情绪平稳'
        }.get(emotion, '')
        
        prompt = f"【心理助手】指令：请以专业心理助手的身份回应用户的心理问题\n输入：用户说：{user_input}\n情绪状态：{emotion_context}\n回答："
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回答部分（去掉提示）
        generated_response = response[len(prompt):].strip()
        
        # 清理响应
        if "用户：" in generated_response:
            generated_response = generated_response.split("用户：")[0].strip()
        
        return generated_response
        
    except Exception as e:
        print(f"❌ 生成响应时出错: {e}")
        return "抱歉，我在生成回复时遇到了问题。请再试一次。"

@app.route('/')
def index():
    """提供前端页面"""
    return render_template('index.html')

@app.route('/api/analyze_psychology', methods=['POST'])
def analyze_psychology():
    """心理分析API接口"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        detected_emotion = data.get('detected_emotion', 'neutral')
        
        if not user_input:
            return jsonify({
                "success": False,
                "error": "输入不能为空"
            }), 400
        
        # 生成响应
        response = generate_psychology_response(user_input, detected_emotion)
        
        # 如果响应太短，使用备选回复
        if len(response) < 20:
            response = fallback_response(user_input, detected_emotion)
        
        return jsonify({
            "success": True,
            "response": response,
            "detected_emotion": detected_emotion,
            "model_source": "local_psychology_model",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "心理分析失败"
        }), 500

def fallback_response(user_input, emotion):
    """备选回复"""
    responses = {
        '压力': f"面对压力时，建议：1.制定合理计划 2.学会说'不' 3.适当放松 4.保持运动",
        '焦虑': f"应对焦虑：1.深呼吸练习 2.正念冥想 3.与朋友倾诉 4.寻求专业帮助",
        '失眠': f"改善睡眠：1.规律作息 2.睡前放松 3.避免咖啡因 4.舒适环境",
        '情绪': f"情绪管理：1.识别情绪 2.接纳感受 3.健康表达 4.转移注意力",
        '抑郁': f"情绪低落时：1.寻求支持 2.保持活动 3.专业咨询 4.耐心对待自己"
    }
    
    for keyword, resp in responses.items():
        if keyword in user_input:
            emotion_context = {
                'happy': '虽然您情绪还好，但',
                'sad': '在情绪低落时，',
                'angry': '在烦躁的时候，',
                'fear': '感到不安时，',
                'neutral': ''
            }.get(emotion, '')
            return f"{emotion_context}{resp}"
    
    return "我理解您的困扰。作为心理助手，我建议您可以更详细地描述具体情况，这样我能提供更有针对性的帮助。"

@app.route('/api/model_status', methods=['GET'])
def model_status():
    """检查模型状态"""
    return jsonify({
        "model_loaded": model_loaded,
        "status": "ready" if model_loaded else "loading",
        "timestamp": datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    # 在后台线程中加载模型，避免阻塞启动
    model_thread = threading.Thread(target=load_psychology_model)
    model_thread.start()
    
    print("启动集成服务器...")
    print(f"服务地址: http://localhost:5000")
    print(f"模型状态: {'正在加载' if not model_loaded else '已加载'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)