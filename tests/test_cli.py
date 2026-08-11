"""CLI tests via Click's runner."""

from __future__ import annotations

import wave

import pytest
from click.testing import CliRunner

from meeting_summarizer import __version__
from meeting_summarizer.cli import SYMBOLS, cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "meeting.wav"
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return path


class TestCli:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Transcribe meetings" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_providers_lists_each_stage(self, runner):
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        assert "transcriber" in result.output
        assert "sarvam" in result.output

    def test_demo_runs_offline(self, runner, tmp_path):
        result = runner.invoke(cli, ["demo", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert list(tmp_path.glob("*.md"))
        assert list(tmp_path.glob("*.json"))

    def test_run_produces_a_summary(self, runner, audio_file, tmp_path):
        result = runner.invoke(
            cli,
            [
                "run",
                str(audio_file),
                "--title",
                "Weekly Sync",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Weekly Sync" in result.output
        assert (tmp_path / "out" / "weekly-sync.md").exists()

    def test_run_on_a_missing_file_exits_1(self, runner, tmp_path):
        result = runner.invoke(cli, ["run", str(tmp_path / "nope.wav")])
        assert result.exit_code == 1
        assert "File not found" in result.output

    def test_run_with_a_bad_provider_exits_1(self, runner, audio_file):
        result = runner.invoke(
            cli, ["run", str(audio_file), "--transcriber", "nonexistent"]
        )
        assert result.exit_code == 1
        assert "Unknown transcriber" in result.output

    def test_failed_delivery_exits_3(self, runner, audio_file, tmp_path):
        result = runner.invoke(
            cli,
            [
                "run",
                str(audio_file),
                "--channel",
                "email",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 3
        assert "SMTP_HOST" in result.output


class TestSymbols:
    def test_every_symbol_is_encodable_by_the_active_stdout(self):
        """Guards the Windows cp1252 crash: symbols must survive the console."""
        import sys

        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        for value in SYMBOLS.values():
            value.encode(encoding)  # must not raise

    def test_symbols_are_non_empty(self):
        assert all(SYMBOLS.values())
