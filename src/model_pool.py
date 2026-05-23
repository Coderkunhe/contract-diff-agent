"""Dynamic model pool with fallback — auto-switches models AND providers."""

import os
import time
from dataclasses import dataclass, field
from typing import Any

# ── Model priority tiers ──────────────────────────────────────────
# Each model has an optional "provider" field. If not set, uses the default
# GMI proxy. Different providers can have different API keys and base URLs.

MODEL_POOL: list[dict[str, Any]] = [
    # Tier 1: Claude via GMI
    {"id": "anthropic/claude-sonnet-4.6", "tier": 1, "label": "Claude Sonnet 4.6",
     "provider": "gmi", "tags": ["claude", "primary"]},
    {"id": "anthropic/claude-opus-4.7", "tier": 1, "label": "Claude Opus 4.7",
     "provider": "gmi", "tags": ["claude", "best"]},
    # Tier 1: DeepSeek direct API (Chinese-native, strong reasoning)
    {"id": "deepseek-chat", "tier": 1, "label": "DeepSeek V3",
     "provider": "deepseek", "tags": ["chinese", "reasoning", "primary"]},
    {"id": "deepseek-reasoner", "tier": 1, "label": "DeepSeek R1",
     "provider": "deepseek", "tags": ["chinese", "reasoning", "best"]},
    # Tier 2: GMI fallbacks
    {"id": "anthropic/claude-sonnet-4.5", "tier": 2, "label": "Claude Sonnet 4.5",
     "provider": "gmi", "tags": ["claude"]},
    {"id": "Qwen/Qwen3.7-Max", "tier": 2, "label": "Qwen 3.7 Max",
     "provider": "gmi", "tags": ["chinese", "quality"]},
    # Tier 3
    {"id": "anthropic/claude-opus-4.6", "tier": 3, "label": "Claude Opus 4.6",
     "provider": "gmi", "tags": ["claude"]},
    {"id": "openai/gpt-5.4", "tier": 3, "label": "GPT 5.4",
     "provider": "gmi", "tags": ["global"]},
    # Tier 4: Fast/fallback
    {"id": "anthropic/claude-haiku-4.5", "tier": 4, "label": "Claude Haiku 4.5",
     "provider": "gmi", "tags": ["claude", "fast"]},
    {"id": "openai/gpt-4o", "tier": 4, "label": "GPT-4o",
     "provider": "gmi", "tags": ["fallback"]},
]

# ── Provider configs ──────────────────────────────────────────────
PROVIDER_CONFIGS = {
    "gmi": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "GMI_BASE_URL",
        "default_base_url": "https://api.gmi-serving.com/v1",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
    },
}


def get_primary_model() -> str:
    """Get the configured primary model from env or default."""
    configured = os.environ.get("CLAUDE_MODEL", "").strip()
    if configured:
        return configured
    return MODEL_POOL[0]["id"]


def get_model_pool(primary: str | None = None) -> list[str]:
    """Get ordered model list with primary first, then pool in tier order.

    Args:
        primary: If set, this model is tried first (even if not in pool).
    """
    primary = primary or get_primary_model()
    seen = {primary}
    ordered = [primary]

    for m in MODEL_POOL:
        if m["id"] not in seen:
            seen.add(m["id"])
            ordered.append(m["id"])

    return ordered


# ── Per-model failure tracking ────────────────────────────────────
@dataclass
class ModelStats:
    failures: int = 0
    last_failure: float = 0.0
    cooldown_until: float = 0.0


_model_stats: dict[str, ModelStats] = {}


def _get_stats(model_id: str) -> ModelStats:
    if model_id not in _model_stats:
        _model_stats[model_id] = ModelStats()
    return _model_stats[model_id]


def mark_model_failure(model_id: str):
    """Mark a model as failed. Repeated failures increase cooldown."""
    s = _get_stats(model_id)
    s.failures += 1
    s.last_failure = time.time()
    cooldown = min(30 * (2 ** (s.failures - 1)), 300)  # 30s → 60s → 120s → ... → max 5min
    s.cooldown_until = time.time() + cooldown


def mark_model_success(model_id: str):
    """Reset failure count on success."""
    s = _get_stats(model_id)
    s.failures = 0
    s.cooldown_until = 0.0


def is_model_available(model_id: str) -> bool:
    """Check if model is not in cooldown."""
    s = _model_stats.get(model_id)
    if s and s.cooldown_until > time.time():
        return False
    return True


def get_next_available(pool: list[str]) -> str | None:
    """Get the first available model from the pool."""
    for m in pool:
        if is_model_available(m):
            return m
    # All in cooldown — wait and retry the first one
    return pool[0] if pool else None


def get_pool_status() -> list[dict]:
    """Get current status of all models for monitoring."""
    status = []
    for m in MODEL_POOL:
        s = _model_stats.get(m["id"])
        status.append({
            "model": m["id"],
            "label": m["label"],
            "tier": m["tier"],
            "failures": s.failures if s else 0,
            "available": is_model_available(m["id"]),
            "cooldown_remaining": max(0, (s.cooldown_until - time.time()) if s else 0),
        })
    return status


# ── Auto-fallback LLM client ──────────────────────────────────────
from openai import OpenAI


class AutoFallbackClient:
    """OpenAI-compatible client that auto-switches models AND providers on failure.

    Supports multiple API providers (GMI, DeepSeek, etc.) each with
    their own API key and base URL.
    """

    def __init__(self, primary_model: str | None = None, timeout: float = 300.0):
        self._pool = get_model_pool(primary_model)
        self._used_model: str | None = None
        self._used_provider: str | None = None
        self._timeout = timeout
        self._clients: dict[str, OpenAI] = {}

    def _get_client(self, provider: str) -> OpenAI:
        """Get or create an OpenAI client for a provider."""
        if provider not in self._clients:
            cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["gmi"])
            api_key = os.environ.get(cfg["api_key_env"], "")
            base_url = os.environ.get(cfg["base_url_env"], cfg["default_base_url"])
            if not api_key:
                raise RuntimeError(
                    f"Provider '{provider}' 的 API Key 未设置 "
                    f"(环境变量: {cfg['api_key_env']})"
                )
            self._clients[provider] = OpenAI(
                api_key=api_key, base_url=base_url, timeout=self._timeout,
            )
        return self._clients[provider]

    @property
    def used_model(self) -> str | None:
        """The model that succeeded in the last call."""
        return self._used_model

    @property
    def used_provider(self) -> str | None:
        """The provider that succeeded in the last call."""
        return self._used_provider

    def create(self, **kwargs) -> Any:
        """Call the API with auto-fallback across models and providers."""
        kwargs.pop("model", None)
        last_error = None

        for attempt, model_id in enumerate(self._pool):
            if not is_model_available(model_id):
                continue

            # Find this model's provider
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
