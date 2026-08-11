"""Guards the session-isolation property of the Streamlit app.

A deployed Streamlit app serves every visitor from a single process, so
``os.environ`` is shared global state. A key typed by one visitor must never
land there, or the next visitor's session would silently inherit it.

These tests drive the real app through Streamlit's AppTest harness.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

pytest.importorskip("streamlit", reason="UI extra not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
SECRET = "sk-visitor-key-must-not-leak-42"


@pytest.fixture(autouse=True)
def _app_on_path():
    sys.path.insert(0, str(APP.parent))
    yield
    sys.path.remove(str(APP.parent))


def _input_by_key(at: AppTest, key: str):
    return next((widget for widget in at.text_input if widget.key == key), None)


class TestApiKeyPanel:
    def test_app_starts_clean(self):
        at = AppTest.from_file(str(APP), default_timeout=90)
        at.run()
        assert not at.exception

    def test_key_inputs_exist_and_are_masked(self):
        # proto.type is the TextInput.Type enum: 0 = DEFAULT, 1 = PASSWORD.
        from streamlit.proto.TextInput_pb2 import TextInput

        at = AppTest.from_file(str(APP), default_timeout=90)
        at.run()
        for attr in ("sarvam_api_key", "openai_api_key", "gemini_api_key"):
            widget = _input_by_key(at, f"user_key_{attr}")
            assert widget is not None, f"missing input for {attr}"
            assert widget.proto.type == TextInput.PASSWORD, (
                f"{attr} input must be masked"
            )

    def test_entered_key_never_reaches_os_environ(self):
        """The core security property."""
        at = AppTest.from_file(str(APP), default_timeout=90)
        at.run()

        before = dict(os.environ)
        widget = _input_by_key(at, "user_key_openai_api_key")
        widget.set_value(SECRET).run()

        assert not at.exception
        assert os.environ.get("OPENAI_API_KEY") != SECRET
        assert SECRET not in os.environ.values()
        assert dict(os.environ) == before, "the app mutated the process environment"

    def test_entered_key_is_actually_applied_to_settings(self, monkeypatch):
        """Isolation is worthless if the key never gets used."""
        captured = {}

        import meeting_summarizer.health as health

        def fake_check(settings, only=None):
            captured["probe"] = settings.openai_api_key
            return []

        monkeypatch.setattr(health, "check_settings", fake_check)

        at = AppTest.from_file(str(APP), default_timeout=90)
        at.run()
        _input_by_key(at, "user_key_openai_api_key").set_value(SECRET).run()

        test_button = next(
            (b for b in at.button if "Test connections" in (b.label or "")), None
        )
        assert test_button is not None, "Test connections button missing"
        test_button.click().run()

        assert not at.exception
        assert captured.get("probe") == SECRET

    def test_blank_input_does_not_override_configured_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key-from-environment")

        at = AppTest.from_file(str(APP), default_timeout=90)
        at.run()
        assert not at.exception

        widget = _input_by_key(at, "user_key_openai_api_key")
        assert widget.value in ("", None)
