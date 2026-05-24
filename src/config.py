"""Unified configuration — single source of truth for all settings.

All configurable values live here with sensible defaults.
Every value can be overridden via environment variable.

Usage:
    from src.config import get_config
    cfg = get_config()
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# Provider auto-detection from model ID
# ═══════════════════════════════════════════════════════════════════════

def _guess_provider(model: str) -> str:
    """Map model ID to provider (gmi/deepseek)."""
    if not model:
        return "gmi"
    model_lower = model.lower()
    if model_lower.startswith("deepseek"):
        return "deepseek"
    # anthropic/*, openai/*, Qwen/*, etc. all go through GMI proxy
    return "gmi"


def _default_base_url(provider: str) -> str:
    return {
        "deepseek": "https://api.deepseek.com/v1",
        "gmi": "https://api.gmi-serving.com/v1",
    }.get(provider, "https://api.gmi-serving.com/v1")


# ═══════════════════════════════════════════════════════════════════════
# Unified env-var resolution — new names first, old names as fallback
# ═══════════════════════════════════════════════════════════════════════

_DEPRECATED_WARNED: set[str] = set()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_unified(new_key: str, *old_keys: str, default: str = "") -> str:
    """Read env var: try new_key first, then old_keys (with deprecation warning)."""
    val = os.environ.get(new_key, "")
    if val:
        return val
    for old in old_keys:
        val = os.environ.get(old, "")
        if val:
            if old not in _DEPRECATED_WARNED:
                _DEPRECATED_WARNED.add(old)
                warnings.warn(
                    f"环境变量 {old} 已废弃，请改用 {new_key}。"
                    f"旧变量名将在后续版本移除。",
                    DeprecationWarning, stacklevel=3,
                )
            return val
    return default


# ═══════════════════════════════════════════════════════════════════════
# Config dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AppConfig:
    """All application settings with env-var override support."""

    # ── LLM Provider ─────────────────────────────────────────────────
    api_key: str = ""
    base_url: str = ""
    model: str = "deepseek-chat"

    @property
    def provider(self) -> str:
        return _guess_provider(self.model)

    # ── LLM behaviour ─────────────────────────────────────────────────
    llm_max_tokens: int = 3000
    llm_timeout: int = 300
    llm_batch_size: int = 25
    classify_max_tokens: int = 2500
    enhance_max_tokens: int = 2000
    validate_max_tokens: int = 500

    # ── Retry / Resilience ────────────────────────────────────────────
    max_retries: int = 3
    chapter_retry_limit: int = 5
    model_cooldown_base: int = 30        # seconds, doubled per failure
    model_cooldown_max: int = 300        # seconds cap

    # ── Validation ────────────────────────────────────────────────────
    confidence_threshold: float = 0.6

    # ── Concurrency ───────────────────────────────────────────────────
    enhance_workers: int = 5
    validate_workers: int = 8

    # ── Web ───────────────────────────────────────────────────────────
    max_jobs: int = 20
    web_workers: int = 2

    # ── Paths ─────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_file: str = "data/diff_result.json"

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str | None = None

    # ── Per-provider key/base_url lookup (for model pool) ───────────
    @staticmethod
    def get_api_key_for(provider: str) -> str:
        """Get API key for a specific provider, respecting overrides."""
        if provider == "deepseek":
            # DeepSeek-specific override OR unified key
            return _env("DEEPSEEK_API_KEY", "") or _env_unified("LLM_API_KEY", "ANTHROPIC_API_KEY")
        # GMI (and everything else): unified key OR old ANTHROPIC_API_KEY
        return _env_unified("LLM_API_KEY", "ANTHROPIC_API_KEY")

    @staticmethod
    def get_base_url_for(provider: str) -> str:
        """Get base URL for a specific provider."""
        if provider == "deepseek":
            return _env("DEEPSEEK_BASE_URL") or _env_unified(
                "LLM_BASE_URL", "GMI_BASE_URL",
                default="https://api.deepseek.com/v1",
            )
        return _env_unified(
            "LLM_BASE_URL", "GMI_BASE_URL",
            default="https://api.gmi-serving.com/v1",
        )

    def __post_init__(self):
        # Resolve paths
        self.data_dir = Path(self.data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment, auto-loading .env file."""
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        return cls(
            # LLM Provider
            api_key=_env_unified("LLM_API_KEY", "ANTHROPIC_API_KEY"),
            base_url=_env_unified("LLM_BASE_URL", "GMI_BASE_URL", "DEEPSEEK_BASE_URL",
                                  default="https://api.gmi-serving.com/v1"),
            model=_env_unified("LLM_MODEL", "CLAUDE_MODEL", default="deepseek-chat"),

            # LLM behaviour
            llm_max_tokens=int(_env("LLM_MAX_TOKENS", "3000")),
            llm_timeout=int(_env("LLM_TIMEOUT", "300")),
            llm_batch_size=int(_env("LLM_BATCH_SIZE", "25")),
            classify_max_tokens=int(_env("CLASSIFY_MAX_TOKENS", "2500")),
            enhance_max_tokens=int(_env("ENHANCE_MAX_TOKENS", "2000")),
            validate_max_tokens=int(_env("VALIDATE_MAX_TOKENS", "500")),

            # Retry
            max_retries=int(_env("MAX_RETRIES", "3")),
            chapter_retry_limit=int(_env("CHAPTER_RETRY_LIMIT", "5")),
            model_cooldown_base=int(_env("MODEL_COOLDOWN_BASE", "30")),
            model_cooldown_max=int(_env("MODEL_COOLDOWN_MAX", "300")),

            # Validation
            confidence_threshold=float(_env("CONFIDENCE_THRESHOLD", "0.6")),

            # Concurrency
            enhance_workers=int(_env("ENHANCE_WORKERS", "5")),
            validate_workers=int(_env("VALIDATE_WORKERS", "8")),

            # Web
            max_jobs=int(_env("MAX_JOBS", "20")),
            web_workers=int(_env("WEB_WORKERS", "2")),

            # Paths
            data_dir=Path(_env("DATA_DIR", "data")),
        )

    def resolve_output(self, output: str | None) -> Path:
        """Resolve output path, creating parent dirs."""
        path = Path(output or self.output_file)
        if not path.is_absolute():
            path = self.project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# ═══════════════════════════════════════════════════════════════════════
# Singleton — 避免重复加载 .env
# ═══════════════════════════════════════════════════════════════════════

_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the singleton AppConfig, loading .env on first call."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset singleton (mainly for tests)."""
    global _config
    _config = None
