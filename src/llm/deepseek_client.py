"""
DeepSeek API 客户端

使用 OpenAI 兼容接口调用 DeepSeek，支持流式输出。
"""

import time
from typing import Optional, Generator

from openai import OpenAI

from src.config import DEEPSEEK_AUTH_TOKEN, DEEPSEEK_MODEL


class DeepSeekClient:
    """DeepSeek API 客户端 (OpenAI 兼容)"""

    def __init__(
        self,
        api_key: str = DEEPSEEK_AUTH_TOKEN,
        model: str = DEEPSEEK_MODEL,
    ):
        if not api_key:
            raise ValueError("DEEPSEEK_AUTH_TOKEN 未配置")

        # 尝试标准 DeepSeek OpenAI 兼容端点
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 8192,
        stream: bool = False,
    ) -> str:
        """同步对话（非流式）

        注意：deepseek-v4-pro 为推理模型，reasoning + 回答都计入 max_tokens。
        若 max_tokens 过小，推理阶段会耗尽预算，导致 content 为空
        (finish_reason=length)。故默认提高到 8192。
        """
        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            elapsed = time.time() - t0
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            elapsed = time.time() - t0
            raise RuntimeError(f"DeepSeek API 调用失败 ({elapsed:.1f}s): {e}")

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> Generator[str, None, None]:
        """流式对话"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n[API 调用失败: {e}]"


# 全局单例
_client: Optional[DeepSeekClient] = None


def get_client() -> DeepSeekClient:
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client
