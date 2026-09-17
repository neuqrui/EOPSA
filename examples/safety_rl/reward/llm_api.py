import os
import time
import random
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime

# Reduce noisy httpx/httpcore connection-close traces in Ray reward workers.
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.WARNING)

_openai_clients: Dict[str, "AsyncOpenAI"] = {}


# ==========================================
# 1. API Pool 定义 (保持不变，支持多实例)
# ==========================================
class APIPool:
    def __init__(self, api_keys: List[str], pool_name: str = "default", max_error_count: int = 20, log_dir: str = None,
                 retry_interval_minutes: int = 5):
        self.pool_name = pool_name
        self.api_keys = api_keys
        self.max_error_count = max_error_count
        self.error_counts: Dict[str, int] = {key: 0 for key in api_keys}
        self.available_keys = set(api_keys)
        self.unavailable_timestamps: Dict[str, float] = {}
        self.retry_interval_seconds = retry_interval_minutes * 60

        if log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), 'api_logs')
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, f'{pool_name}_pool_{datetime.now().strftime("%Y%m%d")}.log')

    def _mask_api_key(self, api_key: str) -> str:
        if len(api_key) <= 8: return api_key
        return f"{api_key[:4]}...{api_key[-4:]}"

    def _log(self, message: str, is_error: bool = False):
        if is_error:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] [{self.pool_name}] {message}\n")

    def get_api_key(self) -> Optional[str]:
        current_time = time.time()
        keys_to_check = [k for k, t in self.unavailable_timestamps.items() if
                         current_time - t >= self.retry_interval_seconds]
        for k in keys_to_check:
            self.error_counts[k] = 0
            self.available_keys.add(k)
            self.unavailable_timestamps.pop(k, None)

        if not self.available_keys:
            self._log("Warning: No available API keys", is_error=True)
            return None
        return random.choice(list(self.available_keys))

    def mark_error(self, api_key: str, error_message: str):
        if api_key not in self.api_keys: return
        self.error_counts[api_key] += 1
        current_count = self.error_counts[api_key]
        self._log(
            f"API key {self._mask_api_key(api_key)} error ({current_count}/{self.max_error_count}): {error_message}",
            is_error=True)
        if current_count >= self.max_error_count and api_key in self.available_keys:
            self.available_keys.remove(api_key)
            self.unavailable_timestamps[api_key] = time.time()


# ==========================================
# 2. 实例化各个 API 池
# ==========================================
# API keys MUST come from the environment. Do not hardcode secrets.
_env_openai = (os.environ.get("OPENAI_API_KEYS") or os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_KEYS = [k.strip() for k in _env_openai.split(",") if k.strip()]
openai_pool = APIPool(api_keys=OPENAI_KEYS, pool_name="openai")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

_env_gemini = (os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or "").strip()
GEMINI_KEYS = [k.strip() for k in _env_gemini.split(",") if k.strip()]
gemini_pool = APIPool(api_keys=GEMINI_KEYS, pool_name="gemini")


def _install_asyncio_exception_filter() -> None:
    """Suppress benign httpx aclose errors that asyncio logs as unretrieved tasks."""

    def _handler(loop, context):
        exc = context.get("exception")
        if exc is not None and "TCPTransport closed" in str(exc):
            return
        loop.default_exception_handler(context)

    try:
        asyncio.get_running_loop().set_exception_handler(_handler)
    except RuntimeError:
        pass


async def _get_openai_client(api_key: str):
    from openai import AsyncOpenAI

    client = _openai_clients.get(api_key)
    if client is None:
        # Long CoT judge prompts need a generous timeout; default is too aggressive.
        client = AsyncOpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, timeout=300.0)
        _openai_clients[api_key] = client
    return client


# ==========================================
# 3. 具体的异步 API 调用实现
# ==========================================
async def _call_openai_async(prompt: str, model_name: str, temperature: float) -> str:
    _install_asyncio_exception_filter()

    max_retries = min(3, len(openai_pool.available_keys) or 1)
    retry_count = 0

    while retry_count < max_retries:
        api_key = openai_pool.get_api_key()
        if api_key is None:
            return ""

        try:
            client = await _get_openai_client(api_key)
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content

        except Exception as e:
            error_message = str(e)
            openai_pool.mark_error(api_key, error_message)
            if "insufficient" in error_message.lower() or "rate_limit" in error_message.lower():
                await asyncio.sleep(2)
            retry_count += 1

    return ""


async def call_openai_chat_async(
    messages: List[Dict[str, str]],
    model_name: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
) -> str:
    """OpenAI-compatible chat with APIPool key rotation (ChatAnywhere)."""
    _install_asyncio_exception_filter()

    max_retries = min(3, len(openai_pool.available_keys) or 1)
    retry_count = 0

    while retry_count < max_retries:
        api_key = openai_pool.get_api_key()
        if api_key is None:
            return ""

        try:
            client = await _get_openai_client(api_key)
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""

        except Exception as e:
            error_message = str(e)
            openai_pool.mark_error(api_key, error_message)
            if "insufficient" in error_message.lower() or "rate_limit" in error_message.lower():
                await asyncio.sleep(2)
            retry_count += 1

    return ""


async def _call_gemini_async(prompt: str, model_name: str, temperature: float, thinking_level: str = "low") -> str:
    from google import genai
    from google.genai import types

    max_retries = min(3, len(gemini_pool.available_keys) or 1)
    retry_count = 0

    # 判断是否需要开启 thinking 配置
    thinking_config = None
    if "gemini-3" in model_name.lower() or "gemini-2.0-pro-exp" in model_name.lower():
        thinking_config = types.ThinkingConfig(thinking_level=thinking_level)

    while retry_count < max_retries:
        api_key = gemini_pool.get_api_key()
        if api_key is None:
            return ""

        try:
            client = genai.Client(api_key=api_key)

            # 关键：使用 client.aio 进行异步调用以适配 verl 的高并发
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=thinking_config
                )
            )
            return response.text

        except Exception as e:
            error_message = str(e)
            gemini_pool.mark_error(api_key, error_message)
            # Gemini 通常的报错会包含 429 或 quota
            if "429" in error_message or "quota" in error_message.lower():
                await asyncio.sleep(2 * (retry_count + 1))  # 退避策略
            retry_count += 1

    return ""


# ==========================================
# 4. 统一的路由接口 (对外暴露)
# ==========================================
async def call_llm_judge_async(
        prompt: str,
        api_type: str = "gemini",
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1
) -> str:
    """
    统一的异步 LLM 评判接口
    :param api_type: 'gemini' 或 'openai'
    """
    if api_type == "openai":
        return await _call_openai_async(prompt, model_name, temperature)
    elif api_type == "gemini":
        return await _call_gemini_async(prompt, model_name, temperature)
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")

if __name__ == "__main__":
    raise SystemExit(
        "Set OPENAI_API_KEY / GEMINI_API_KEY in the environment; this module no longer ships keys."
    )