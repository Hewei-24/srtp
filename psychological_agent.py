"""
心理分析代理模块
================

本模块提供独立的心理分析代理类，可被其他模块复用。
主要功能：
1. 与 DeepSeek API 交互
2. 管理对话历史
3. 生成心理咨询回复
4. 统计 API 使用量

使用示例:
    ```python
    from psychological_agent import DeepSeekPsychologicalAgent

    agent = DeepSeekPsychologicalAgent(api_key="your-api-key")
    result = agent.analyze_psychological_state("我最近压力很大")
    print(result['response'])
    ```

作者: SRTP 项目组
版本: 1.0
"""

import datetime
import logging
from typing import Dict, List, Any, Optional

import requests

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DeepSeekPsychologicalAgent:
    """
    DeepSeek 心理分析代理类

    该类封装了与 DeepSeek API 的交互逻辑，提供心理咨询服务。
    支持多轮对话，自动管理对话历史，并统计 API 使用量。

    Attributes:
        api_key: DeepSeek API 密钥
        api_url: API 端点 URL
        conversation_history: 对话历史列表
        system_prompt: 系统提示词
        total_tokens_used: 累计使用的 token 数
        api_calls_count: API 调用次数
    """

    # API 配置常量
    DEFAULT_API_URL = "https://api.deepseek.com/v1/chat/completions"
    DEFAULT_MODEL = "deepseek-chat"
    MAX_HISTORY_LENGTH = 20  # 最大对话历史长度
    HISTORY_CONTEXT_SIZE = 6  # 发送给 API 的历史消息数量

    def __init__(self, api_key: str, api_url: Optional[str] = None):
        """
        初始化心理分析代理

        Args:
            api_key: DeepSeek API 密钥
            api_url: API 端点 URL（可选，默认使用官方端点）
        """
        self.api_key = api_key
        self.api_url = api_url or self.DEFAULT_API_URL
        self.conversation_history: List[Dict[str, str]] = []
        self.logger = logging.getLogger(self.__class__.__name__)

        # 系统提示词 - 定义 AI 的角色和行为
        self.system_prompt = """你是一名专业的大学心理健康顾问，专门帮助大学生解决心理问题。

你的职责包括：
1. 分析学生的心理状态和情绪问题
2. 提供专业、温暖的心理支持和建议
3. 识别危机情况并给出适当建议
4. 用同理心和理解来回应用户

请以温暖、专业、支持性的语气回应，避免使用专业术语，用通俗易懂的语言提供建议。"""

        # 使用统计
        self.total_tokens_used = 0
        self.api_calls_count = 0

        self.logger.info("心理分析代理初始化完成")

    def analyze_psychological_state(self, user_input: str) -> Dict[str, Any]:
        """
        分析用户心理状态并生成回复

        该方法是主要的对外接口，接收用户输入，调用 API 生成回复，
        并自动管理对话历史。

        Args:
            user_input: 用户输入的文本

        Returns:
            包含以下字段的字典：
            - success: 是否成功
            - response: AI 回复内容
            - timestamp: 时间戳
            - api_used: 是否使用了 API
            - error: 错误信息（仅在失败时）
        """
        try:
            # 构建消息列表
            messages = self._build_messages(user_input)

            self.logger.info(f"调用 API，输入: {user_input[:50]}...")

            # 调用 DeepSeek API
            response = self._call_api(messages)

            if response.status_code == 200:
                return self._handle_success_response(response, user_input)
            else:
                return self._handle_error_response(response)

        except requests.exceptions.Timeout:
            self.logger.error("API 请求超时")
            return {
                "success": False,
                "error": "请求超时",
                "response": "请求超时，请检查网络连接后重试。"
            }
        except requests.exceptions.ConnectionError:
            self.logger.error("网络连接错误")
            return {
                "success": False,
                "error": "网络连接错误",
                "response": "无法连接到服务器，请检查网络连接。"
            }
        except Exception as e:
            self.logger.error(f"分析过程中出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "系统暂时出现问题，请稍后再试。"
            }

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        构建发送给 API 的消息列表

        Args:
            user_input: 用户输入

        Returns:
            消息列表，包含系统提示、历史对话和当前输入
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加最近的对话历史
        if self.conversation_history:
            recent_history = self.conversation_history[-self.HISTORY_CONTEXT_SIZE:]
            messages.extend(recent_history)

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def _call_api(self, messages: List[Dict[str, str]]) -> requests.Response:
        """
        调用 DeepSeek API

        Args:
            messages: 消息列表

        Returns:
            API 响应对象
        """
        return requests.post(
            self.api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": self.DEFAULT_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            },
            timeout=30
        )

    def _handle_success_response(
        self,
        response: requests.Response,
        user_input: str
    ) -> Dict[str, Any]:
        """
        处理成功的 API 响应

        Args:
            response: API 响应对象
            user_input: 原始用户输入

        Returns:
            格式化的成功响应字典
        """
        result = response.json()
        assistant_response = result['choices'][0]['message']['content']

        # 更新对话历史
        self._update_conversation_history(user_input, assistant_response)

        # 更新使用统计
        usage = result.get('usage', {})
        self.total_tokens_used += usage.get('total_tokens', 0)
        self.api_calls_count += 1

        self.logger.info(f"API 调用成功，使用 {usage.get('total_tokens', 0)} tokens")

        return {
            "success": True,
            "response": assistant_response,
            "timestamp": datetime.datetime.now().isoformat(),
            "api_used": True
        }

    def _handle_error_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        处理失败的 API 响应

        Args:
            response: API 响应对象

        Returns:
            格式化的错误响应字典
        """
        self.logger.error(f"API 调用失败: {response.status_code} - {response.text}")

        error_messages = {
            401: "API 密钥无效，请检查配置。",
            429: "请求过于频繁，请稍后再试。",
            500: "服务器内部错误，请稍后再试。",
            503: "服务暂时不可用，请稍后再试。"
        }

        error_msg = error_messages.get(
            response.status_code,
            "抱歉，服务暂时不可用，请稍后再试。"
        )

        return {
            "success": False,
            "error": f"API 错误: {response.status_code}",
            "response": error_msg
        }

    def _update_conversation_history(
        self,
        user_input: str,
        assistant_response: str
    ) -> None:
        """
        更新对话历史

        Args:
            user_input: 用户输入
            assistant_response: AI 回复
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })

        # 限制历史记录长度
        if len(self.conversation_history) > self.MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY_LENGTH:]

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        获取对话摘要

        Returns:
            包含对话统计信息的字典
        """
        if not self.conversation_history:
            return {"message": "暂无对话记录"}

        user_messages = [
            msg['content']
            for msg in self.conversation_history
            if msg['role'] == 'user'
        ]

        return {
            "total_conversations": len(self.conversation_history) // 2,
            "recent_topics": user_messages[-3:] if user_messages else [],
            "total_tokens_used": self.total_tokens_used,
            "api_calls_count": self.api_calls_count,
            "last_active": datetime.datetime.now().isoformat()
        }

    def reset_conversation(self) -> None:
        """
        重置对话历史

        清空所有对话记录，但保留使用统计。
        """
        self.conversation_history = []
        self.logger.info("对话历史已重置")

    def get_usage_stats(self) -> Dict[str, int]:
        """
        获取 API 使用统计

        Returns:
            包含使用统计的字典
        """
        return {
            "total_tokens_used": self.total_tokens_used,
            "api_calls_count": self.api_calls_count,
            "average_tokens_per_call": (
                self.total_tokens_used // self.api_calls_count
                if self.api_calls_count > 0 else 0
            )
        }


# ==================== 模块测试 ====================
if __name__ == "__main__":
    # 简单的模块测试
    print("=" * 50)
    print("心理分析代理模块测试")
    print("=" * 50)

    # 注意：实际使用时需要提供有效的 API 密钥
    test_api_key = "your-api-key-here"

    agent = DeepSeekPsychologicalAgent(api_key=test_api_key)

    print(f"代理初始化完成")
    print(f"API URL: {agent.api_url}")
    print(f"对话历史长度: {len(agent.conversation_history)}")

    # 获取摘要
    summary = agent.get_conversation_summary()
    print(f"对话摘要: {summary}")

    print("=" * 50)
    print("测试完成")
