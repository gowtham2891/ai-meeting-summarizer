"""Live credential checks.

Each check makes the smallest authenticated request a provider supports and
classifies the response, so someone pasting a key into the UI finds out whether
it works *before* spending a full transcription run discovering that it doesn't.

Checks are single-attempt with a short timeout: this is a "does this key work
right now" probe, not a resilient production call.
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

#: Short on purpose -- a credential probe should fail fast, not hang a UI.
CHECK_TIMEOUT = 20.0

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class CheckResult:
    """The verdict on one credential."""

    provider: str
    ok: bool
    message: str

    def __str__(self) -> str:
        return f"{'OK' if self.ok else 'FAILED'} {self.provider}: {self.message}"


def _missing(provider: str) -> CheckResult:
    return CheckResult(provider, False, "No key provided.")


def _network_error(provider: str, exc: Exception) -> CheckResult:
    return CheckResult(provider, False, f"Could not reach the API: {exc}")


def _error_message(response: httpx.Response) -> str:
    """Pull a human-readable reason out of a JSON error body."""
    try:
        payload: Dict[str, Any] = response.json()
    except ValueError:
        return response.text[:200].strip() or f"HTTP {response.status_code}"

    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "").strip() or f"HTTP {response.status_code}"
    if isinstance(error, str):
        return error
    for key in ("message", "detail", "error_message"):
        if payload.get(key):
            return str(payload[key])
    return f"HTTP {response.status_code}"


def tiny_wav(seconds: float = 0.25, sample_rate: int = 16000) -> bytes:
    """A minimal valid WAV, used as the smallest possible STT probe."""
    frames = struct.pack("<h", 0) * int(sample_rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def check_sarvam(
    api_key: str,
    base_url: str = "https://api.sarvam.ai",
    model: str = "saaras:v2.5",
    timeout: float = CHECK_TIMEOUT,
) -> CheckResult:
    """Verify a Sarvam key by transcribing a quarter-second of silence.

    Sarvam exposes no unauthenticated ping, so the probe is a real request --
    kept to the smallest valid audio payload so it costs effectively nothing.
    """
    provider = "Sarvam AI"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/speech-to-text-translate",
            headers={"api-subscription-key": api_key.strip()},
            files={"file": ("probe.wav", io.BytesIO(tiny_wav()), "audio/wav")},
            data={"model": model},
            timeout=timeout,
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code in (200, 201):
        return CheckResult(provider, True, "Key works.")
    if response.status_code in (401, 403):
        return CheckResult(
            provider, False, f"Key rejected: {_error_message(response)}"
        )
    if response.status_code == 429:
        return CheckResult(
            provider, False, "Rate limited — the key is valid but throttled."
        )
    if response.status_code == 422:
        # The probe audio was understood but rejected as too short/empty,
        # which still proves the credential was accepted.
        return CheckResult(provider, True, "Key works (probe audio rejected as empty).")
    return CheckResult(
        provider, False, f"HTTP {response.status_code}: {_error_message(response)}"
    )


def check_openai(
    api_key: str, model: str = "", timeout: float = CHECK_TIMEOUT
) -> CheckResult:
    """Verify an OpenAI key by listing models — consumes no tokens."""
    provider = "OpenAI"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.get(
            OPENAI_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code == 200:
        names = _openai_model_names(response)
        if model and model not in names:
            return CheckResult(
                provider, True,
                f"Key works, but '{model}' is not in the {len(names)} models this "
                f"key can reach. Pick another model or check your plan.",
            )
        return CheckResult(provider, True, f"Key works — {len(names)} models available.")
    if response.status_code == 401:
        return CheckResult(provider, False, f"Key rejected: {_error_message(response)}")
    if response.status_code == 429:
        return CheckResult(
            provider, False,
            "Rate limited or out of quota — check billing on the OpenAI dashboard.",
        )
    return CheckResult(
        provider, False, f"HTTP {response.status_code}: {_error_message(response)}"
    )


def _openai_model_names(response: httpx.Response) -> List[str]:
    try:
        payload = response.json()
    except ValueError:
        return []
    return [
        str(entry.get("id", ""))
        for entry in payload.get("data", [])
        if isinstance(entry, dict)
    ]


def check_gemini(
    api_key: str, model: str = "", timeout: float = CHECK_TIMEOUT
) -> CheckResult:
    """Verify a Gemini key by listing models — consumes no tokens."""
    provider = "Gemini"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.get(
            GEMINI_MODELS_URL, params={"key": api_key.strip()}, timeout=timeout
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code == 200:
        names = _gemini_model_names(response)
        if model and not any(model.split("/")[-1].lower() == n.lower() for n in names):
            return CheckResult(
                provider, True,
                f"Key works, but '{model}' was not among the {len(names)} available "
                f"models.",
            )
        return CheckResult(provider, True, f"Key works — {len(names)} models available.")
    if response.status_code in (400, 401, 403):
        return CheckResult(provider, False, f"Key rejected: {_error_message(response)}")
    if response.status_code == 429:
        return CheckResult(
            provider, False, "Rate limited — the key is valid but throttled."
        )
    return CheckResult(
        provider, False, f"HTTP {response.status_code}: {_error_message(response)}"
    )


def _gemini_model_names(response: httpx.Response) -> List[str]:
    try:
        payload = response.json()
    except ValueError:
        return []
    return [
        str(entry.get("name", "")).split("/")[-1]
        for entry in payload.get("models", [])
        if isinstance(entry, dict)
    ]


def check_settings(settings, only: Optional[str] = None) -> List[CheckResult]:
    """Check every credential the current provider selection actually needs.

    Mock providers need nothing, so they report OK without any network call --
    which is what keeps the offline demo honest rather than merely quiet.
    """
    results: List[CheckResult] = []
    timeout = min(settings.request_timeout, CHECK_TIMEOUT)

    if only in (None, "transcriber"):
        if settings.transcriber == "mock":
            results.append(
                CheckResult("Transcriber (mock)", True, "No credentials required.")
            )
        elif settings.transcriber == "sarvam":
            results.append(
                check_sarvam(
                    settings.sarvam_api_key,
                    settings.sarvam_base_url,
                    settings.sarvam_model,
                    timeout=timeout,
                )
            )

    if only in (None, "extractor"):
        if settings.extractor == "mock":
            results.append(
                CheckResult("Extractor (mock)", True, "No credentials required.")
            )
        elif settings.extractor == "openai":
            results.append(
                check_openai(
                    settings.openai_api_key, settings.openai_model, timeout=timeout
                )
            )
        elif settings.extractor == "gemini":
            results.append(
                check_gemini(
                    settings.gemini_api_key, settings.gemini_model, timeout=timeout
                )
            )

    return results

# --- what is missing, before anything is attempted -----------------------


@dataclass
class MissingCredential:
    """A credential the current provider selection needs but does not have."""

    label: str
    env_var: str
    needed_for: str
    get_it_at: str = ""

    def __str__(self) -> str:
        return f"{self.label} ({self.env_var}) — needed for {self.needed_for}"


def missing_credentials(settings) -> List[MissingCredential]:
    """Credentials required by the selected providers that are not set.

    Pure inspection: no network, no exceptions. Safe to call on every UI
    rerun, which is what lets the app ask for a key up front instead of
    failing deep inside the pipeline.
    """
    missing: List[MissingCredential] = []
    if settings.transcriber == "sarvam" and not settings.sarvam_api_key.strip():
        missing.append(MissingCredential(
            "Sarvam AI key", "SARVAM_API_KEY",
            "the 'sarvam' transcriber", "sarvam.ai"))

    if settings.extractor == "openai" and not settings.openai_api_key.strip():
        missing.append(MissingCredential(
            "OpenAI API key", "OPENAI_API_KEY",
            "the 'openai' extractor", "platform.openai.com/api-keys"))

    if settings.extractor == "gemini" and not settings.gemini_api_key.strip():
        missing.append(MissingCredential(
            "Gemini API key", "GEMINI_API_KEY",
            "the 'gemini' extractor", "aistudio.google.com/apikey"))

    return missing


def credentials_ready(settings) -> bool:
    """True when every provider the user selected has the key it needs."""
    return not missing_credentials(settings)
