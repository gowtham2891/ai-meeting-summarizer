"""End-to-end pipeline tests, plus transcription, delivery and config coverage."""

from __future__ import annotations

import json
import wave

import httpx
import pytest

from meeting_summarizer.audio import AudioError, validate_audio_file
from meeting_summarizer.config import ConfigError, Settings
from meeting_summarizer.delivery.base import available_channels, get_channel
from meeting_summarizer.delivery.channels import (
    ConsoleChannel,
    FileChannel,
    WhatsAppChannel,
    _slugify,
)
from meeting_summarizer.models import MeetingSummary, Transcript
from meeting_summarizer.pipeline import MeetingSummarizerPipeline, summarize
from meeting_summarizer.transcription.base import available_transcribers, get_transcriber
from meeting_summarizer.transcription.mock import MockTranscriber
from meeting_summarizer.transcription.sarvam import (
    SarvamTranscriber,
    SarvamTranscriptionError,
)


@pytest.fixture
def audio_file(tmp_path):
    """A real, minimal WAV file so validation passes."""
    path = tmp_path / "meeting.wav"
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return path


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        transcriber="mock", extractor="mock", output_dir=str(tmp_path / "out")
    )


class TestAudioValidation:
    def test_accepts_a_real_file(self, audio_file):
        assert validate_audio_file(audio_file) == audio_file

    def test_missing_file(self, tmp_path):
        with pytest.raises(AudioError, match="File not found"):
            validate_audio_file(tmp_path / "nope.wav")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.wav"
        path.touch()
        with pytest.raises(AudioError, match="File is empty"):
            validate_audio_file(path)

    def test_wrong_extension(self, tmp_path):
        path = tmp_path / "notes.pdf"
        path.write_bytes(b"data")
        with pytest.raises(AudioError, match="Unsupported file type"):
            validate_audio_file(path)

    def test_directory(self, tmp_path):
        with pytest.raises(AudioError, match="Not a file"):
            validate_audio_file(tmp_path)


class TestMockTranscriber:
    def test_returns_the_builtin_sample(self, settings, audio_file):
        transcript = MockTranscriber(settings).transcribe(audio_file)
        assert len(transcript.segments) == 13
        assert transcript.speakers() == ["Priya", "Arjun", "Meera"]

    def test_prefers_a_json_sidecar(self, settings, audio_file):
        audio_file.with_suffix(".json").write_text(
            json.dumps(
                {
                    "language": "te-IN",
                    "segments": [
                        {"start": 0, "end": 3, "text": "Custom line", "speaker": "Ravi"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        transcript = MockTranscriber(settings).transcribe(audio_file)
        assert transcript.language == "te-IN"
        assert transcript.text == "Custom line"

    def test_reads_a_text_sidecar_with_speakers(self, settings, audio_file):
        audio_file.with_suffix(".txt").write_text(
            "Ravi: We should postpone the launch.\nSita: Agreed, next month works.\n",
            encoding="utf-8",
        )
        transcript = MockTranscriber(settings).transcribe(audio_file)
        assert transcript.speakers() == ["Ravi", "Sita"]
        assert transcript.segments[0].start == 0.0
        assert transcript.segments[1].start == transcript.segments[0].end

    def test_registry_exposes_it(self, settings):
        assert "mock" in available_transcribers()
        assert isinstance(get_transcriber(settings), MockTranscriber)


class TestSarvamTranscriber:
    def test_requires_an_api_key(self):
        with pytest.raises(ConfigError, match="SARVAM_API_KEY"):
            SarvamTranscriber(Settings(transcriber="sarvam", sarvam_api_key=""))

    def test_parses_a_diarized_response(self):
        transcriber = SarvamTranscriber(Settings(sarvam_api_key="test-key"))
        segments = transcriber._parse_segments(
            {
                "diarized_transcript": {
                    "entries": [
                        {
                            "transcript": "Hello everyone",
                            "start_time_seconds": 0.0,
                            "end_time_seconds": 2.5,
                            "speaker_id": 0,
                        },
                        {
                            "transcript": "  ",
                            "start_time_seconds": 2.5,
                            "end_time_seconds": 3.0,
                            "speaker_id": 1,
                        },
                    ]
                }
            },
            offset=10.0,
        )
        assert len(segments) == 1  # blank entry dropped
        assert segments[0].start == 10.0
        assert segments[0].speaker == "Speaker 0"

    def test_falls_back_to_the_flat_transcript(self):
        transcriber = SarvamTranscriber(Settings(sarvam_api_key="test-key"))
        segments = transcriber._parse_segments(
            {"transcript": "One long block of speech", "duration_seconds": 42.0},
            offset=0.0,
        )
        assert len(segments) == 1
        assert segments[0].end == 42.0

    def test_empty_payload_yields_nothing(self):
        transcriber = SarvamTranscriber(Settings(sarvam_api_key="test-key"))
        assert transcriber._parse_segments({}, offset=0.0) == []

    def test_missing_audio_file_raises(self, tmp_path):
        transcriber = SarvamTranscriber(Settings(sarvam_api_key="test-key"))
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(tmp_path / "nope.wav")

    def test_oversized_chunk_is_rejected(self, tmp_path, monkeypatch):
        from meeting_summarizer.transcription import sarvam as sarvam_module

        transcriber = SarvamTranscriber(Settings(sarvam_api_key="test-key"))
        chunk = tmp_path / "big.wav"
        chunk.write_bytes(b"\x00")
        monkeypatch.setattr(sarvam_module, "MAX_UPLOAD_BYTES", 0)
        with pytest.raises(SarvamTranscriptionError, match="above Sarvam's"):
            transcriber._post_chunk(chunk)

    def test_retries_then_gives_up(self, tmp_path, monkeypatch):
        transcriber = SarvamTranscriber(
            Settings(sarvam_api_key="test-key", max_retries=2)
        )
        chunk = tmp_path / "chunk.wav"
        chunk.write_bytes(b"\x00")

        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise httpx.ConnectError("network down")

        monkeypatch.setattr(httpx, "post", boom)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(SarvamTranscriptionError, match="after 2 attempts"):
            transcriber._post_chunk(chunk)
        assert calls["n"] == 2


class TestDelivery:
    @pytest.fixture
    def summary(self) -> MeetingSummary:
        return MeetingSummary(title="Launch Sync", overview="We ship Thursday.")

    def test_registry_lists_every_channel(self):
        assert set(available_channels()) == {"file", "console", "email", "whatsapp"}

    def test_unknown_channel_raises(self, settings):
        with pytest.raises(ValueError, match="Unknown delivery channel"):
            get_channel("carrier-pigeon", settings)

    def test_file_channel_writes_both_formats(self, settings, summary, tmp_path):
        result = FileChannel(settings).send(summary)
        assert result.success
        out_dir = tmp_path / "out"
        assert (out_dir / "launch-sync.md").exists()
        assert (out_dir / "launch-sync.json").exists()
        restored = json.loads((out_dir / "launch-sync.json").read_text(encoding="utf-8"))
        assert restored["title"] == "Launch Sync"

    def test_console_channel_prints(self, settings, summary, capsys):
        result = ConsoleChannel(settings).send(summary)
        assert result.success
        assert "Launch Sync" in capsys.readouterr().out

    def test_email_reports_missing_config(self, settings, summary):
        result = get_channel("email", settings).send(summary)
        assert not result.success
        assert "SMTP_HOST" in result.detail

    def test_whatsapp_reports_missing_config(self, settings, summary):
        result = get_channel("whatsapp", settings).send(summary)
        assert not result.success
        assert "TWILIO_ACCOUNT_SID" in result.detail

    def test_whatsapp_chunking_respects_the_cap(self, settings):
        channel = WhatsAppChannel(settings)
        parts = channel._chunk("\n".join(f"line {i} " * 20 for i in range(60)))
        assert len(parts) > 1
        assert all(len(part) <= channel.MAX_BODY for part in parts)

    def test_whatsapp_chunking_splits_one_giant_line(self, settings):
        channel = WhatsAppChannel(settings)
        parts = channel._chunk("x" * (channel.MAX_BODY * 3))
        assert len(parts) >= 3
        assert all(len(part) <= channel.MAX_BODY for part in parts)

    def test_short_text_is_one_part(self, settings):
        assert WhatsAppChannel(settings)._chunk("short") == ["short"]

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Launch Sync", "launch-sync"),
            ("Q3  Review!! 2026", "q3-review-2026"),
            ("///", "meeting-summary"),
        ],
    )
    def test_slugify(self, title, expected):
        assert _slugify(title) == expected


class TestPipeline:
    def test_runs_end_to_end(self, settings, audio_file):
        result = MeetingSummarizerPipeline(settings=settings).run(
            audio_file, channels=["file"]
        )
        assert isinstance(result.transcript, Transcript)
        assert result.summary.action_items
        assert result.delivered_ok
        assert result.elapsed_seconds >= 0

    def test_reports_progress_for_each_stage(self, settings, audio_file):
        seen = []
        MeetingSummarizerPipeline(
            settings=settings, progress=lambda stage, msg: seen.append(stage)
        ).run(audio_file, channels=["file"])
        assert {"transcribe", "extract", "deliver", "done"} <= set(seen)

    def test_title_override_reaches_the_summary(self, settings, audio_file):
        result = MeetingSummarizerPipeline(settings=settings).run(
            audio_file, title="Weekly Sync", channels=["file"]
        )
        assert result.summary.title == "Weekly Sync"

    def test_a_failing_channel_does_not_abort_the_run(self, settings, audio_file):
        result = MeetingSummarizerPipeline(settings=settings).run(
            audio_file, channels=["file", "email"]
        )
        assert not result.delivered_ok
        assert len(result.deliveries) == 2
        assert result.deliveries[0].success  # file still went out
        assert not result.deliveries[1].success

    def test_unknown_channel_is_reported_not_raised(self, settings, audio_file):
        result = MeetingSummarizerPipeline(settings=settings).run(
            audio_file, channels=["nope"]
        )
        assert not result.delivered_ok
        assert "Unknown delivery channel" in result.deliveries[0].detail

    def test_bad_provider_fails_at_construction(self, settings):
        settings.transcriber = "nonexistent"
        with pytest.raises(ValueError, match="Unknown transcriber"):
            MeetingSummarizerPipeline(settings=settings)

    def test_convenience_wrapper(self, settings, audio_file):
        result = summarize(audio_file, channels=["file"], settings=settings)
        assert result.summary.overview


class TestSettings:
    def test_mock_mode_detection(self):
        assert Settings(transcriber="mock", extractor="mock").is_mock
        assert not Settings(transcriber="sarvam", extractor="mock").is_mock

    def test_require_raises_for_blanks(self):
        with pytest.raises(ConfigError, match="MY_KEY is required"):
            Settings().require("", "MY_KEY", "someprovider")

    def test_require_passes_values_through(self):
        assert Settings().require("abc", "MY_KEY", "p") == "abc"

    def test_describe_mentions_the_mode(self):
        assert "mock" in Settings(transcriber="mock", extractor="mock").describe()
