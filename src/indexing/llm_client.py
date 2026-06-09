"""LLM抽象接口

支持Anthropic/OpenAI/Ollama三种后端，通过工厂函数创建。
主要用于KV提取、内容分类等需要LLM辅助的管线步骤。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """生成文本回复"""
        ...

    @abstractmethod
    def extract_structured(self, prompt: str, schema: dict[str, Any], system: str = "") -> dict:
        """提取结构化JSON输出"""
        ...

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system: str = "",
        image_mime: str = "image/jpeg",
    ) -> str:
        """基于图片和文本生成回复。默认客户端不支持Vision。"""
        raise NotImplementedError("当前LLM客户端不支持图片输入")


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude客户端"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def extract_structured(self, prompt: str, schema: dict[str, Any], system: str = "") -> dict:
        extraction_prompt = (
            f"{prompt}\n\n"
            f"请严格按照以下JSON Schema输出，不要包含其他内容：\n"
            f"```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"
        )
        raw = self.generate(extraction_prompt, system=system)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            end_idx = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
            text = "\n".join(lines[:end_idx])
        return json.loads(text)


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT客户端"""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None):
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system: str = "",
        image_mime: str = "image/jpeg",
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime};base64,{image_base64}"},
                },
            ],
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def extract_structured(self, prompt: str, schema: dict[str, Any], system: str = "") -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        extraction_prompt = (
            f"{prompt}\n\n请严格输出JSON格式，符合以下schema：\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        messages.append({"role": "user", "content": extraction_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")


class OllamaClient(BaseLLMClient):
    """Ollama本地模型客户端"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:14b"):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http = httpx.Client(timeout=120.0)

    def generate(self, prompt: str, system: str = "") -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        resp = self.http.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    def extract_structured(self, prompt: str, schema: dict[str, Any], system: str = "") -> dict:
        extraction_prompt = (
            f"{prompt}\n\n"
            f"请严格按照以下JSON格式输出，不要包含其他文字：\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        raw = self.generate(extraction_prompt, system=system)
        return AnthropicClient._parse_json(raw)


class DoubaoClient(BaseLLMClient):
    """豆包API客户端（兼容OpenAI格式）"""

    def __init__(self, api_key: str, model: str = "", base_url: str = "https://ark.cn-beijing.volces.com/api/v3"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or "doubao-pro-32k"

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system: str = "",
        image_mime: str = "image/jpeg",
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime};base64,{image_base64}"},
                },
            ],
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def extract_structured(self, prompt: str, schema: dict[str, Any], system: str = "") -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        extraction_prompt = (
            f"{prompt}\n\n请严格输出JSON格式，符合以下schema：\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        messages.append({"role": "user", "content": extraction_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")


def create_llm_client(
    provider: str = "anthropic",
    model: str = "",
    api_key: str = "",
    base_url: str = "",
) -> BaseLLMClient:
    """工厂函数：根据配置创建LLM客户端"""
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    elif provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o", base_url=base_url or None)
    elif provider == "ollama":
        return OllamaClient(base_url=base_url or "http://localhost:11434", model=model or "qwen2.5:14b")
    elif provider == "doubao":
        return DoubaoClient(api_key=api_key, model=model, base_url=base_url or "https://ark.cn-beijing.volces.com/api/v3")
    else:
        raise ValueError(f"不支持的LLM provider: {provider}")


def get_default_client() -> BaseLLMClient:
    """从环境配置创建默认LLM客户端"""
    from config.settings import (
        LLM_PROVIDER, LLM_MODEL,
        ANTHROPIC_API_KEY, OPENAI_API_KEY, DOUBAO_API_KEY, OLLAMA_BASE_URL,
    )

    api_key = ""
    base_url = ""
    if LLM_PROVIDER == "anthropic":
        api_key = ANTHROPIC_API_KEY
    elif LLM_PROVIDER == "openai":
        api_key = OPENAI_API_KEY
    elif LLM_PROVIDER == "ollama":
        base_url = OLLAMA_BASE_URL
    elif LLM_PROVIDER == "doubao":
        api_key = DOUBAO_API_KEY
        base_url = "https://ark.cn-beijing.volces.com/api/v3"

    return create_llm_client(
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        api_key=api_key,
        base_url=base_url,
    )
