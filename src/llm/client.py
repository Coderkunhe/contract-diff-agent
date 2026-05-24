"""Auto-fallback LLM client — auto-switches models AND providers on failure.

Uses thread-local httpx clients to bypass macOS system proxy issues
and avoid thread-safety problems under concurrent streaming load.
"""

import threading
from typing import Any

import httpx
from openai import OpenAI

from src.config import AppConfig
from .pool import (
    get_model_pool, get_next_available, is_model_available,
    mark_model_failure, mark_model_success,
    MODEL_POOL,
)


class AutoFallbackClient:
    """OpenAI-compatible client that auto-switches models AND providers on failure."""

    def __init__(self, primary_model: str | None = None, timeout: float = 300.0):
        self._pool = get_model_pool(primary_model)
        self._used_model: str | None = None
        self._used_provider: str | None = None
        self._timeout = timeout
        self._local = threading.local()

    def _get_client(self, provider: str) -> OpenAI:
        if not hasattr(self._local, "clients"):
            self._local.clients = {}
        if provider not in self._local.clients:
            api_key = AppConfig.get_api_key_for(provider)
            base_url = AppConfig.get_base_url_for(provider)
            if not api_key:
                raise RuntimeError(
                    f"Provider '{provider}' 的 API Key 未设置 "
                    f"(请设置 LLM_API_KEY 环境变量)"
                )
            http_client = httpx.Client(trust_env=False, timeout=self._timeout)
            self._local.clients[provider] = OpenAI(
                api_key=api_key, base_url=base_url, http_client=http_client,
            )
        return self._local.clients[provider]

    @property
    def used_model(self) -> str | None:
        return self._used_model

    @property
    def used_provider(self) -> str | None:
        return self._used_provider

    def create(self, **kwargs) -> Any:
        kwargs.pop("model", None)
        last_error = None

        for attempt, model_id in enumerate(self._pool):
            if not is_model_available(model_id):
                continue

            model_info = next((m for m in MODEL_POOL if m["id"] == model_id), None)
            provider = model_info["provider"] if model_info else "gmi"

            try:
                client = self._get_client(provider)
                response = client.chat.completions.create(
                    model=model_id, **kwargs,
                )
                mark_model_success(model_id)
                self._used_model = model_id
                self._used_provider = provider
                return response
            except Exception as e:
                last_error = e
                mark_model_failure(model_id)
                err_str = str(e)[:100]
                if attempt < len(self._pool) - 1:
                    print(f"  ⚠️ 模型 {model_id} 失败: {err_str} → 切换到下一模型...")
                continue

        raise RuntimeError(
            f"所有模型均不可用。最后错误: {last_error}"
        )

    def create_race(self, race_count: int = 3, **kwargs) -> Any:
        """Fire prompt to N models concurrently, return the first success.

        Losing requests are cancelled. If all fail, raises RuntimeError.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        kwargs.pop("model", None)

        candidates = [m for m in self._pool if is_model_available(m)][:race_count]
        if not candidates:
            raise RuntimeError("无可用模型参与竞速 (所有模型均在冷却中)")

        def _call_one(model_id: str):
            model_info = next((m for m in MODEL_POOL if m["id"] == model_id), None)
            provider = model_info["provider"] if model_info else "gmi"
            label = model_info["label"] if model_info else model_id
            try:
                client = self._get_client(provider)
                response = client.chat.completions.create(
                    model=model_id, **kwargs,
                )
                mark_model_success(model_id)
                return (model_id, provider, label, response, None)
            except Exception as exc:
                mark_model_failure(model_id)
                return (model_id, provider, label, None, exc)

        errors: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = {executor.submit(_call_one, m): m for m in candidates}
            for future in as_completed(futures):
                model_id, provider, label, response, error = future.result()
                if response is not None:
                    self._used_model = model_id
                    self._used_provider = provider
                    for f in futures:
                        f.cancel()
                    print(f"  🏎️ 竞速胜出: {label} ({model_id})")
                    return response
                if error:
                    errors.append((label, str(error)[:80]))

        raise RuntimeError(
            f"竞速全部失败 ({len(candidates)} 模型): {errors}"
        )
