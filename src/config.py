"""Centralized configuration management.

Loads from .env, environment variables, and provides defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Application configuration loaded from environment."""

    # API
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get(
        "GMI_BASE_URL", "https://api.gmi-serving.com/v1"
    ))
    model: str = field(default_factory=lambda: os.environ.get(
        "CLAUDE_MODEL", "anthropic/claude-sonnet-4.6"
    ))

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("DIFF_DATA_DIR", "data")
    ))
    output_file: str = "data/diff_result.json"

    # LLM limits
    llm_max_tokens: int = 3000
    llm_timeout: int = 300
    llm_batch_size: int = 25

    # Validation
    max_retries: int = 3
    chapter_retry_limit: int = 5
    confidence_threshold: float = 0.6

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment, auto-loading .env."""
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        return cls()

    def resolve_output(self, output: str | None) -> Path:
        """Resolve output path, creating parent dirs."""
        path = Path(output or self.output_file)
        if not path.is_absolute():
            path = self.project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
