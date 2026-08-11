"""Tests for the extraction stage: heuristics, parsing helpers and schema widening."""

from __future__ import annotations

from datetime import date

import pytest

from meeting_summarizer.config import Settings
from meeting_summarizer.extraction.base import available_extractors, get_extractor
from meeting_summarizer.extraction.llm import (
    ExtractionError,
    loads_lenient,
    parse_due_date,
    parse_priority,
    to_meeting_summary,
)
from meeting_summarizer.extraction.mock import HeuristicExtractor
from meeting_summarizer.models import Priority, Transcript, TranscriptSegment
from meeting_summarizer.transcription.mock import MockTranscriber


@pytest.fixture
def settings() -> Settings:
    return Settings(transcriber="mock", extractor="mock")


@pytest.fixture
def sample_transcript(settings, tmp_path) -> Transcript:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"\x00")
    return MockTranscriber(settings).transcribe(audio)


class TestRegistry:
    def test_known_providers_are_registered(self):
        assert {"mock", "openai", "gemini"} <= set(available_extractors())

    def test_mock_resolves(self, settings):
        assert isinstance(get_extractor(settings), HeuristicExtractor)

    def test_unknown_provider_raises(self, settings):
        settings.extractor = "does-not-exist"
        with pytest.raises(ValueError, match="Unknown extractor"):
            get_extractor(settings)


class TestHeuristicExtractor:
    def test_produces_a_complete_summary(self, settings, sample_transcript):
        summary = HeuristicExtractor(settings).extract(sample_transcript)
        assert summary.overview
        assert summary.participants == ["Priya", "Arjun", "Meera"]
        assert summary.duration_seconds == sample_transcript.duration

    def test_finds_the_shipping_decision(self, settings, sample_transcript):
        summary = HeuristicExtractor(settings).extract(sample_transcript)
        decisions = " ".join(d.decision.lower() for d in summary.decisions)
        assert "ship thursday" in decisions

    def test_finds_action_items_with_owners(self, settings, sample_transcript):
        summary = HeuristicExtractor(settings).extract(sample_transcript)
        assert summary.action_items
        assert all(item.owner for item in summary.action_items)
        tasks = " ".join(item.task.lower() for item in summary.action_items)
        assert "runbook" in tasks

    def test_action_items_are_deduplicated(self, settings):
        transcript = Transcript(
            segments=[
                TranscriptSegment(start=0, end=5, text="I'll send the report today.", speaker="A"),
                TranscriptSegment(start=5, end=10, text="I'll send the report today.", speaker="B"),
            ]
        )
        summary = HeuristicExtractor(settings).extract(transcript)
        assert len(summary.action_items) == 1

    def test_priority_reacts_to_urgency_words(self, settings):
        transcript = Transcript(
            segments=[
                TranscriptSegment(
                    start=0, end=8, text="I'll fix the failing build before Friday.", speaker="A"
                )
            ]
        )
        summary = HeuristicExtractor(settings).extract(transcript)
        assert summary.action_items[0].priority is Priority.HIGH

    def test_collects_open_questions(self, settings, sample_transcript):
        summary = HeuristicExtractor(settings).extract(sample_transcript)
        assert any("?" in question or "open question" in question.lower()
                   for question in summary.open_questions)

    def test_is_deterministic(self, settings, sample_transcript):
        first = HeuristicExtractor(settings).extract(sample_transcript)
        second = HeuristicExtractor(settings).extract(sample_transcript)
        assert first == second

    def test_empty_transcript_raises(self, settings):
        with pytest.raises(ValueError, match="empty transcript"):
            HeuristicExtractor(settings).extract(Transcript())

    def test_explicit_title_wins(self, settings, sample_transcript):
        summary = HeuristicExtractor(settings).extract(sample_transcript, title="Q3 Review")
        assert summary.title == "Q3 Review"


class TestDueDateParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2026-03-05", date(2026, 3, 5)),
            ("05-03-2026", date(2026, 3, 5)),
            ("05/03/2026", date(2026, 3, 5)),
            ("2026/03/05", date(2026, 3, 5)),
        ],
    )
    def test_parses_known_formats(self, raw, expected):
        assert parse_due_date(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "null", "N/A", "TBD", "next Thursday", "-"])
    def test_returns_none_for_unparseable(self, raw):
        assert parse_due_date(raw) is None


class TestPriorityParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("high", Priority.HIGH),
            ("HIGH", Priority.HIGH),
            ("urgent", Priority.HIGH),
            ("p0", Priority.HIGH),
            ("low", Priority.LOW),
            ("nice to have", Priority.LOW),
            ("medium", Priority.MEDIUM),
            ("banana", Priority.MEDIUM),
            (None, Priority.MEDIUM),
        ],
    )
    def test_normalises(self, raw, expected):
        assert parse_priority(raw) is expected


class TestLenientJson:
    def test_plain_json(self):
        assert loads_lenient('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}

    def test_bare_fence(self):
        assert loads_lenient('```\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self):
        assert loads_lenient('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_empty_response_raises(self):
        with pytest.raises(ExtractionError, match="empty response"):
            loads_lenient("   ")

    def test_response_without_json_raises(self):
        with pytest.raises(ExtractionError, match="no JSON object"):
            loads_lenient("I cannot help with that.")

    def test_malformed_json_raises(self):
        with pytest.raises(ExtractionError, match="malformed JSON"):
            loads_lenient('{"a": 1,,,}')


class TestSchemaWidening:
    def test_builds_a_summary_from_a_raw_payload(self, sample_transcript):
        payload = {
            "title": "Launch Sync",
            "overview": "Shipping Thursday.",
            "key_points": ["Refunds are manual for now", ""],
            "decisions": [{"decision": "Ship Thursday", "made_by": "Priya"}, {"decision": "  "}],
            "action_items": [
                {"task": "Write runbook", "owner": "Meera", "due_date": "2026-03-05", "priority": "high"},
                {"task": "   ", "owner": "Nobody"},
            ],
            "open_questions": ["Budget for a hire?"],
            "participants": ["Priya", "Meera"],
        }
        summary = to_meeting_summary(payload, sample_transcript)

        assert summary.title == "Launch Sync"
        assert summary.key_points == ["Refunds are manual for now"]  # blank dropped
        assert len(summary.decisions) == 1  # blank dropped
        assert len(summary.action_items) == 1  # blank dropped
        assert summary.action_items[0].due_date == date(2026, 3, 5)
        assert summary.action_items[0].priority is Priority.HIGH
        assert summary.duration_seconds == sample_transcript.duration

    def test_falls_back_to_transcript_speakers(self, sample_transcript):
        summary = to_meeting_summary(
            {"overview": "Something happened."}, sample_transcript
        )
        assert summary.participants == sample_transcript.speakers()

    def test_missing_overview_gets_a_placeholder(self, sample_transcript):
        summary = to_meeting_summary({"title": "X"}, sample_transcript)
        assert summary.overview

    def test_bad_payload_raises(self, sample_transcript):
        with pytest.raises(ExtractionError, match="did not match the schema"):
            to_meeting_summary({"key_points": "not-a-list"}, sample_transcript)
