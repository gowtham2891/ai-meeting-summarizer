"""Tests for the missing-credential detector. Never touches the network."""

from __future__ import annotations

import httpx
import pytest

from meeting_summarizer.config import Settings
from meeting_summarizer.health import (
    MissingCredential,
    credentials_ready,
    missing_credentials,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The detector must be pure inspection: any request here is a bug."""

    def guard(*args, **kwargs):
        raise AssertionError("missing_credentials must not make network calls")

    monkeypatch.setattr(httpx, "get", guard)
    monkeypatch.setattr(httpx, "post", guard)


class TestMissingCredential:
    def test_str_includes_the_variable_name(self):
        item = MissingCredential("A Key", "A_KEY", "the thing")
        assert "A_KEY" in str(item)
        assert "the thing" in str(item)


class TestDetector:
    def test_mock_mode_needs_nothing(self):
        settings = Settings(transcriber="mock", extractor="mock")
        assert missing_credentials(settings) == []
        assert credentials_ready(settings)

    def test_sarvam_without_a_key_is_reported(self):
        settings = Settings(transcriber="sarvam", extractor="mock", sarvam_api_key="")
        gaps = missing_credentials(settings)
        assert [item.env_var for item in gaps] == ["SARVAM_API_KEY"]
        assert not credentials_ready(settings)

    def test_sarvam_with_a_key_is_satisfied(self):
        settings = Settings(transcriber="sarvam", extractor="mock", sarvam_api_key="k")
        assert missing_credentials(settings) == []

    def test_whitespace_only_key_does_not_count(self):
        settings = Settings(transcriber="sarvam", extractor="mock", sarvam_api_key="   ")
        assert not credentials_ready(settings)

    @pytest.mark.parametrize("extractor,var", [("openai", "OPENAI_API_KEY"),
                                               ("gemini", "GEMINI_API_KEY")])
    def test_each_extractor_reports_its_own_key(self, extractor, var):
        settings = Settings(transcriber="mock", extractor=extractor)
        assert [i.env_var for i in missing_credentials(settings)] == [var]

    def test_both_stages_reported_together(self):
        settings = Settings(transcriber="sarvam", extractor="openai",
                            sarvam_api_key="", openai_api_key="")
        assert {i.env_var for i in missing_credentials(settings)} == {
            "SARVAM_API_KEY", "OPENAI_API_KEY"}

    def test_every_gap_says_where_to_get_the_key(self):
        settings = Settings(transcriber="sarvam", extractor="gemini")
        assert all(item.get_it_at for item in missing_credentials(settings))
