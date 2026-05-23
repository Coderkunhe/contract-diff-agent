"""Model pool configuration and statistics.

Defines available models, provider configs, and per-model failure tracking.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

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
    configured = os.environ.get("CLAUDE_MODEL", "").strip()
    if configured:
        return configured
    return MODEL_POOL[0]["id"]


def get_model_pool(primary: str | None = None) -> list[str]:
    primary = primary or get_primary_model()
    seen = {primary}
    ordered = [primary]
    for m in MODEL_POOL:
        if m["id"] not in seen:
            seen.add(m["id"])
            ordered.append(m["id"])
    return ordered


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
    s = _get_stats(model_id)
    s.failures += 1
    s.last_failure = time.time()
    cooldown = min(30 * (2 ** (s.failures - 1)), 300)
    s.cooldown_until = time.time() + cooldown


def mark_model_success(model_id: str):
    s = _get_stats(model_id)
    s.failures = 0
    s.cooldown_until = 0.0


def is_model_available(model_id: str) -> bool:
    s = _model_stats.get(model_id)
    if s and s.cooldown_until > time.time():
        return False
    return True


def get_next_available(pool: list[str]) -> str | None:
    for m in pool:
        if is_model_available(m):
            return m
    return pool[0] if pool else None


def get_pool_status() -> list[dict]:
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
