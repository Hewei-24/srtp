import requests
import json
import logging
from typing import Dict, List, Any
import datetime

class DeepSeekPsychologicalAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.conversation_history = []
        self.system_prompt = """你是一名专业的大学心理健康顾问，专门帮助大学生解决心理问题。
        
你的职责包括：
1. 分析学生的心理状态和情绪问题
2. 提供专业、温暖的心理支持和建议
3. 识别危机情况并给出适当建议
4. 用同理心和理解来回应用户

请以温暖、专业、支持性的语气回应，避免使用专业术语，用通俗易懂的语言提供建议。"""

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 用量统计
        self.total_tokens_used = 0
        self.api_calls_count = 0

    def analyze_psychological_state(self, user_input: str) -> Dict[str, Any]:
        """分析用户心理状态"""
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history[-6:],  # 最近3轮对话
                {"role": "user", "content": user_input}
            ]

            # 调用DeepSeek API
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_response = result['choices'][0]['message']['content']
                
                # 更新对话历史
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                
                # 限制历史记录长度
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                # 统计用量
                usage = result.get('usage', {})
                self.total_tokens_used += usage.get('total_tokens', 0)
                self.api_calls_count += 1
                
                return {
                    "success": True,
                    "response": assistant_response,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "api_used": True
                }
            else:
                self.logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API错误: {response.status_code}",
                    "response": "抱歉，服务暂时不可用，请稍后再试。"
                }

        except Exception as e:
            self.logger.error(f"分析过程中出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "系统暂时出现问题，请稍后再试。"
            }

    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        if not self.conversation_history:
            return {"message": "暂无对话记录"}

        user_messages = [msg['content'] for msg in self.conversation_history if msg['role'] == 'user']
        
        return {
            "total_conversations": len(self.conversation_history) // 2,
            "recent_topics": user_messages[-3:] if user_messages else [],
            "last_active": datetime.datetime.now().isoformat()
        }

    def reset_conversation(self):
        """重置对话"""
        self.conversation_history = []
        self.logger.info("对话历史已重置")