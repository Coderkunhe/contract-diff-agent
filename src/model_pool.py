"""Dynamic model pool with fallback — auto-switches on failure.

Priority: Claude (structured output king) → Chinese-native → Global fallbacks.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

# ── Model priority tiers ──────────────────────────────────────────
# Tier 1: Claude — best at structured JSON + legal reasoning
# Tier 2: Chinese-native — excellent Chinese quality
# Tier 3: Global strong — reliable all-rounders
# Tier 4: Fast/cheap — last resort

MODEL_POOL: list[dict[str, Any]] = [
    # Tier 1: Claude
    {"id": "anthropic/claude-sonnet-4.6", "tier": 1, "label": "Claude Sonnet 4.6", "tags": ["claude", "primary"]},
    {"id": "anthropic/claude-opus-4.7", "tier": 1, "label": "Claude Opus 4.7", "tags": ["claude", "best"]},
    {"id": "anthropic/claude-sonnet-4.5", "tier": 1, "label": "Claude Sonnet 4.5", "tags": ["claude"]},
    {"id": "anthropic/claude-opus-4.6", "tier": 1, "label": "Claude Opus 4.6", "tags": ["claude"]},
    # Tier 2: Chinese-native
    {"id": "Qwen/Qwen3.7-Max", "tier": 2, "label": "Qwen 3.7 Max", "tags": ["chinese", "quality"]},
    {"id": "deepseek-ai/DeepSeek-V4-Pro", "tier": 2, "label": "DeepSeek V4 Pro", "tags": ["chinese", "reasoning"]},
    {"id": "zai-org/GLM-5.1-FP8", "tier": 2, "label": "GLM 5.1", "tags": ["chinese"]},
    # Tier 3: Global
    {"id": "openai/gpt-5.4", "tier": 3, "label": "GPT 5.4", "tags": ["global"]},
    {"id": "anthropic/claude-haiku-4.5", "tier": 3, "label": "Claude Haiku 4.5", "tags": ["claude", "fast"]},
    # Tier 4: Fallback
    {"id": "deepseek-ai/DeepSeek-V3.2", "tier": 4, "label": "DeepSeek V3.2", "tags": ["fallback"]},
    {"id": "openai/gpt-4o", "tier": 4, "label": "GPT-4o", "tags": ["fallback"]},
]


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
    """OpenAI-compatible client that auto-switches models on failure."""

    def __init__(self, api_key: str, base_url: str, primary_model: str | None = None,
                 timeout: float = 300.0):
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._pool = get_model_pool(primary_model)
        self._used_model: str | None = None

    @property
    def used_model(self) -> str | None:
        """The model that succeeded in the last call."""
        return self._used_model

    def create(self, **kwargs) -> Any:
        """Call the API with auto-fallback across models.

        Accepts the same kwargs as openai.chat.completions.create,
        but 'model' is overridden by the pool's fallback logic.
        """
        kwargs.pop("model", None)  # We manage model selection
        last_error = None

        for attempt, model_id in enumerate(self._pool):
            if not is_model_available(model_id):
                continue

            try:
                response = self._client.chat.completions.create(
                    model=model_id, **kwargs,
                )
                mark_model_success(model_id)
                self._used_model = model_id
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
