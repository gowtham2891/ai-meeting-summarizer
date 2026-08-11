"""Tests for the domain models and their rendering."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from meeting_summarizer.models import (
    ActionItem,
    Decision,
    MeetingSummary,
    Priority,
    Transcript,
    TranscriptSegment,
)


class TestTranscriptSegment:
    def test_timestamp_under_an_hour(self):
        segment = TranscriptSegment(start=75.4, end=80.0, text="hello")
        assert segment.timestamp() == "01:15"

    def test_timestamp_past_an_hour(self):
        segment = TranscriptSegment(start=3725.0, end=3730.0, text="hello")
        assert segment.timestamp() == "1:02:05"

    def test_duration(self):
        assert TranscriptSegment(start=10.0, end=25.5, text="x").duration == 15.5

    def test_end_before_start_is_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(start=10.0, end=5.0, text="x")

    def test_empty_text_is_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(start=0.0, end=1.0, text="")

    def test_negative_start_is_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(start=-1.0, end=5.0, text="x")


class TestTranscript:
    @pytest.fixture
    def transcript(self) -> Transcript:
        return Transcript(
            segments=[
                TranscriptSegment(start=0.0, end=5.0, text="Hello team.", speaker="Priya"),
                TranscriptSegment(start=5.0, end=12.0, text="Morning.", speaker="Arjun"),
                TranscriptSegment(start=12.0, end=20.0, text="Let's start.", speaker="Priya"),
            ],
            language="en-IN",
        )

    def test_text_joins_segments(self, transcript):
        assert transcript.text == "Hello team. Morning. Let's start."

    def test_duration_is_last_end(self, transcript):
        assert transcript.duration == 20.0

    def test_speakers_are_unique_and_ordered(self, transcript):
        assert transcript.speakers() == ["Priya", "Arjun"]

    def test_dialogue_rendering(self, transcript):
        dialogue = transcript.to_dialogue()
        assert dialogue.splitlines()[0] == "[00:00] Priya: Hello team."
        assert len(dialogue.splitlines()) == 3

    def test_empty_transcript_is_safe(self):
        empty = Transcript()
        assert empty.text == ""
        assert empty.duration == 0.0
        assert empty.speakers() == []


class TestActionItem:
    def test_line_includes_due_date(self):
        item = ActionItem(
            task="Write the runbook",
            owner="Meera",
            due_date=date(2026, 3, 5),
            priority=Priority.HIGH,
        )
        assert item.to_line() == "[HIGH] Meera: Write the runbook (due 2026-03-05)"

    def test_line_without_due_date(self):
        item = ActionItem(task="Ship it", owner="Arjun")
        assert item.to_line() == "[MEDIUM] Arjun: Ship it"

    def test_owner_defaults_to_unassigned(self):
        assert ActionItem(task="Do a thing").owner == "Unassigned"


class TestMeetingSummary:
    @pytest.fixture
    def summary(self) -> MeetingSummary:
        return MeetingSummary(
            title="Launch Sync",
            overview="The team agreed to ship on Thursday.",
            key_points=["Payments refunds still failing"],
            decisions=[
                Decision(
                    decision="Ship Thursday without automated refunds",
                    rationale="Support can absorb manual handling",
                    made_by="Priya",
                )
            ],
            action_items=[
                ActionItem(
                    task="Write the manual refund runbook",
                    owner="Meera",
                    due_date=date(2026, 3, 5),
                    priority=Priority.HIGH,
                )
            ],
            open_questions=["Does the budget allow another hire?"],
            participants=["Priya", "Arjun", "Meera"],
            duration_seconds=1320.0,
        )

    def test_markdown_contains_every_section(self, summary):
        markdown = summary.to_markdown()
        assert "# Launch Sync" in markdown
        assert "## Overview" in markdown
        assert "## Key Points" in markdown
        assert "## Decisions" in markdown
        assert "## Action Items" in markdown
        assert "## Open Questions" in markdown

    def test_markdown_action_table_has_the_row(self, summary):
        markdown = summary.to_markdown()
        assert "| Meera | Write the manual refund runbook | 2026-03-05 | high |" in markdown

    def test_markdown_shows_duration_in_minutes(self, summary):
        assert "**Duration:** 22 min" in summary.to_markdown()

    def test_markdown_omits_empty_sections(self):
        minimal = MeetingSummary(overview="Short standup, nothing decided.")
        markdown = minimal.to_markdown()
        assert "## Overview" in markdown
        assert "## Decisions" not in markdown
        assert "## Action Items" not in markdown

    def test_plain_text_is_whatsapp_friendly(self, summary):
        text = summary.to_plain_text()
        assert "DECISIONS" in text
        assert "ACTION ITEMS" in text
        assert "|" not in text  # no markdown tables

    def test_overview_is_required(self):
        with pytest.raises(ValidationError):
            MeetingSummary(overview="")

    def test_round_trips_through_json(self, summary):
        restored = MeetingSummary.model_validate_json(summary.model_dump_json())
        assert restored == summary
