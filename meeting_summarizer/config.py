"""Environment-driven configuration.

The pipeline never reads ``os.environ`` directly. Everything funnels through
:class:`Settings` so that mock mode, live mode and tests all follow one code
path and secrets stay out of function signatures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

#: Providers that need no credentials, used for local runs and CI.
MOCK_PROVIDERS = {"mock"}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_list(name: str) -> List[str]:
    raw = _env(name)
    return [item.strip() for item in raw.split(",") if item.strip()]


class ConfigError(RuntimeError):
    """Raised when a live provider is selected without its credentials."""


@dataclass
class Settings:
    """Resolved runtime settings for one pipeline invocation."""

    # --- provider selection -------------------------------------------------
    transcriber: str = field(default_factory=lambda: _env("TRANSCRIBER", "mock"))
    extractor: str = field(default_factory=lambda: _env("EXTRACTOR", "mock"))

    # --- Sarvam AI (speech-to-text) ----------------------------------------
    sarvam_api_key: str = field(default_factory=lambda: _env("SARVAM_API_KEY"))
    sarvam_base_url: str = field(
        default_factory=lambda: _env("SARVAM_BASE_URL", "https://api.sarvam.ai")
    )
    sarvam_model: str = field(
        default_factory=lambda: _env("SARVAM_MODEL", "saaras:v2.5")
    )

    # --- LLM (extraction) ---------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2"))
    )

    # --- delivery: email ----------------------------------------------------
    smtp_host: str = field(default_factory=lambda: _env("SMTP_HOST"))
    smtp_port: int = field(default_factory=lambda: int(_env("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: _env("SMTP_USER"))
    smtp_password: str = field(default_factory=lambda: _env("SMTP_PASSWORD"))
    smtp_use_tls: bool = field(default_factory=lambda: _env_bool("SMTP_USE_TLS", True))
    email_from: str = field(default_factory=lambda: _env("EMAIL_FROM"))
    email_to: List[str] = field(default_factory=lambda: _env_list("EMAIL_TO"))

    # --- delivery: WhatsApp (Twilio) ---------------------------------------
    twilio_account_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    twilio_auth_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    twilio_whatsapp_from: str = field(
        default_factory=lambda: _env("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    )
    whatsapp_to: List[str] = field(default_factory=lambda: _env_list("WHATSAPP_TO"))

    # --- behaviour ----------------------------------------------------------
    chunk_seconds: int = field(default_factory=lambda: int(_env("CHUNK_SECONDS", "600")))
    request_timeout: float = field(
        default_factory=lambda: float(_env("REQUEST_TIMEOUT", "120"))
    )
    max_retries: int = field(default_factory=lambda: int(_env("MAX_RETRIES", "3")))
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", "output"))

    @property
    def is_mock(self) -> bool:
        """True when neither model-backed stage needs a credential."""
        return (
            self.transcriber in MOCK_PROVIDERS and self.extractor in MOCK_PROVIDERS
        )

    def require(self, value: str, name: str, provider: str) -> str:
        if not value:
            raise ConfigError(
                f"{name} is required for the '{provider}' provider. "
                f"Set it in your .env file or switch to the 'mock' provider."
            )
        return value

    def describe(self) -> str:
        mode = "mock (no credentials needed)" if self.is_mock else "live"
        return (
            f"mode={mode} transcriber={self.transcriber} extractor={self.extractor}"
        )


_settings: Optional[Settings] = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide settings, re-reading the environment if asked."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
