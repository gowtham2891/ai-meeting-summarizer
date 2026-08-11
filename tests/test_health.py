"""Tests for the credential-check module. No network is ever touched."""

from __future__ import annotations

import httpx
import pytest

from meeting_summarizer.config import Settings
from meeting_summarizer.health import (
    CheckResult,
    check_gemini,
    check_openai,
    check_sarvam,
    check_settings,
    tiny_wav,
)


def fake_response(status: int, payload=None, text: str = "") -> httpx.Response:
    request = httpx.Request("POST", "https://example.com")
    if payload is not None:
        return httpx.Response(status, json=payload, request=request)
    return httpx.Response(status, text=text, request=request)


@pytest.fixture
def no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(httpx, "get", guard)
    monkeypatch.setattr(httpx, "post", guard)


class TestCheckResult:
    def test_str_ok(self):
        assert str(CheckResult("X", True, "fine")) == "OK X: fine"

    def test_str_failed(self):
        assert str(CheckResult("X", False, "bad")) == "FAILED X: bad"


class TestTinyWav:
    def test_is_a_valid_wav(self):
        import io
        import wave

        with wave.open(io.BytesIO(tiny_wav())) as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert handle.getnframes() > 0

    def test_is_small(self):
        assert len(tiny_wav()) < 20_000


class TestSarvamCheck:
    def test_blank_key_makes_no_request(self, no_network):
        result = check_sarvam("  ")
        assert not result.ok
        assert "No key provided" in result.message

    def test_success(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(200, {"transcript": ""}))
        assert check_sarvam("valid-key").ok

    def test_sends_the_key_in_the_header(self, monkeypatch):
        seen = {}

        def capture(url, headers=None, files=None, data=None, timeout=None):
            seen.update(headers or {})
            return fake_response(200, {})

        monkeypatch.setattr(httpx, "post", capture)
        check_sarvam("  my-key  ")
        assert seen["api-subscription-key"] == "my-key"

    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_key(self, monkeypatch, status):
        monkeypatch.setattr(
            httpx, "post",
            lambda *a, **k: fake_response(status, {"error": {"message": "invalid key"}}),
        )
        result = check_sarvam("bad-key")
        assert not result.ok
        assert "rejected" in result.message.lower()

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(429, {}))
        result = check_sarvam("valid-key")
        assert not result.ok
        assert "throttled" in result.message

    def test_empty_audio_rejection_still_proves_the_key(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(422, {}))
        result = check_sarvam("valid-key")
        assert result.ok

    def test_network_failure_is_reported_not_raised(self, monkeypatch):
        def boom(*args, **kwargs):
            raise httpx.ConnectError("no route")

        monkeypatch.setattr(httpx, "post", boom)
        result = check_sarvam("valid-key")
        assert not result.ok
        assert "Could not reach" in result.message


class TestOpenAICheck:
    def test_blank_key(self, no_network):
        assert not check_openai("").ok

    def test_success_counts_models(self, monkeypatch):
        payload = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        result = check_openai("valid-key")
        assert result.ok
        assert "2 models" in result.message

    def test_uses_a_bearer_header(self, monkeypatch):
        seen = {}

        def capture(url, headers=None, timeout=None):
            seen.update(headers or {})
            return fake_response(200, {"data": []})

        monkeypatch.setattr(httpx, "get", capture)
        check_openai("secret")
        assert seen["Authorization"] == "Bearer secret"

    def test_missing_model_warns_but_passes(self, monkeypatch):
        payload = {"data": [{"id": "gpt-4o-mini"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        result = check_openai("valid-key", model="gpt-5-turbo")
        assert result.ok
        assert "gpt-5-turbo" in result.message

    def test_unauthorized(self, monkeypatch):
        payload = {"error": {"message": "Incorrect API key provided"}}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(401, payload))
        result = check_openai("bad-key")
        assert not result.ok
        assert "Incorrect API key" in result.message

    def test_quota_exhausted(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(429, {}))
        result = check_openai("valid-key")
        assert not result.ok
        assert "billing" in result.message


class TestGeminiCheck:
    def test_blank_key(self, no_network):
        assert not check_gemini("").ok

    def test_success(self, monkeypatch):
        payload = {"models": [{"name": "models/gemini-2.0-flash"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        assert check_gemini("valid-key").ok

    def test_rejected(self, monkeypatch):
        payload = {"error": {"message": "API key not valid"}}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(400, payload))
        result = check_gemini("bad-key")
        assert not result.ok
        assert "rejected" in result.message.lower()

    def test_non_json_body_survives(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(500, text="oops"))
        assert not check_gemini("valid-key").ok


class TestCheckSettings:
    def test_mock_mode_needs_no_network(self, no_network):
        results = check_settings(Settings(transcriber="mock", extractor="mock"))
        assert len(results) == 2
        assert all(result.ok for result in results)

    def test_live_transcriber_is_checked(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(200, {}))
        results = check_settings(
            Settings(transcriber="sarvam", extractor="mock", sarvam_api_key="k")
        )
        assert any(r.provider == "Sarvam AI" for r in results)

    def test_live_extractor_is_checked(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, {"data": []}))
        results = check_settings(
            Settings(transcriber="mock", extractor="openai", openai_api_key="k")
        )
        assert any(r.provider == "OpenAI" for r in results)

    def test_gemini_extractor_is_checked(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, {"models": []}))
        results = check_settings(
            Settings(transcriber="mock", extractor="gemini", gemini_api_key="k")
        )
        assert any(r.provider == "Gemini" for r in results)

    def test_only_filter(self, no_network):
        results = check_settings(
            Settings(transcriber="mock", extractor="mock"), only="extractor"
        )
        assert len(results) == 1

    def test_missing_key_fails(self, no_network):
        results = check_settings(
            Settings(transcriber="sarvam", extractor="mock", sarvam_api_key="")
        )
        assert not results[0].ok
