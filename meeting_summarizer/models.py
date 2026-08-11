"""Core domain models for the meeting summarizer pipeline.

Everything that crosses a module boundary is a Pydantic model, so the LLM
extraction step can be validated with the same schema the delivery step reads.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TranscriptSegment(BaseModel):
    """A single timestamped chunk of speech."""

    start: float = Field(..., ge=0, description="Start offset in seconds")
    end: float = Field(..., ge=0, description="End offset in seconds")
    text: str = Field(..., min_length=1)
    speaker: Optional[str] = Field(
        default=None, description="Speaker label when diarization is available"
    )

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError("segment end must not precede start")
        return v

    @property
    def duration(self) -> float:
        return self.end - self.start

    def timestamp(self) -> str:
        """Render the start offset as ``MM:SS`` (or ``HH:MM:SS`` past an hour)."""
        total = int(self.start)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


class Transcript(BaseModel):
    """The full transcription result for one recording."""

    segments: List[TranscriptSegment] = Field(default_factory=list)
    language: str = Field(default="unknown", description="Detected language code")
    source: str = Field(default="", description="Path or URL of the source audio")

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    @property
    def duration(self) -> float:
        return max((segment.end for segment in self.segments), default=0.0)

    def speakers(self) -> List[str]:
        seen: List[str] = []
        for segment in self.segments:
            if segment.speaker and segment.speaker not in seen:
                seen.append(segment.speaker)
        return seen

    def to_dialogue(self) -> str:
        """Render as ``[MM:SS] Speaker: text`` lines for the LLM prompt."""
        lines = []
        for segment in self.segments:
            speaker = segment.speaker or "Speaker"
            lines.append(f"[{segment.timestamp()}] {speaker}: {segment.text.strip()}")
        return "\n".join(lines)


class ActionItem(BaseModel):
    task: str = Field(..., min_length=1, description="What needs to be done")
    owner: str = Field(default="Unassigned", description="Who owns the task")
    due_date: Optional[date] = Field(default=None)
    priority: Priority = Field(default=Priority.MEDIUM)
    context: str = Field(default="", description="Why this came up in the meeting")

    def to_line(self) -> str:
        due = f" (due {self.due_date.isoformat()})" if self.due_date else ""
        return f"[{self.priority.value.upper()}] {self.owner}: {self.task}{due}"


class Decision(BaseModel):
    decision: str = Field(..., min_length=1)
    rationale: str = Field(default="")
    made_by: str = Field(default="Team")


class MeetingSummary(BaseModel):
    """The structured artifact the whole pipeline exists to produce."""

    title: str = Field(default="Meeting Summary")
    overview: str = Field(..., min_length=1, description="Two to four sentence recap")
    key_points: List[str] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0, ge=0)
    language: str = Field(default="unknown")

    def to_markdown(self) -> str:
        minutes = int(self.duration_seconds // 60)
        lines = [f"# {self.title}", ""]

        meta = []
        if minutes:
            meta.append(f"**Duration:** {minutes} min")
        if self.participants:
            meta.append(f"**Participants:** {', '.join(self.participants)}")
        if self.language and self.language != "unknown":
            meta.append(f"**Language:** {self.language}")
        if meta:
            lines += ["  |  ".join(meta), ""]

        lines += ["## Overview", "", self.overview, ""]

        if self.key_points:
            lines += ["## Key Points", ""]
            lines += [f"- {point}" for point in self.key_points]
            lines.append("")

        if self.decisions:
            lines += ["## Decisions", ""]
            for item in self.decisions:
                lines.append(f"- **{item.decision}**")
                if item.rationale:
                    lines.append(f"  - Rationale: {item.rationale}")
                if item.made_by:
                    lines.append(f"  - Owner: {item.made_by}")
            lines.append("")

        if self.action_items:
            lines += ["## Action Items", "", "| Owner | Task | Due | Priority |", "| --- | --- | --- | --- |"]
            for item in self.action_items:
                due = item.due_date.isoformat() if item.due_date else "-"
                lines.append(
                    f"| {item.owner} | {item.task} | {due} | {item.priority.value} |"
                )
            lines.append("")

        if self.open_questions:
            lines += ["## Open Questions", ""]
            lines += [f"- {question}" for question in self.open_questions]
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def to_plain_text(self) -> str:
        """Compact rendering for channels without markdown (WhatsApp, SMS)."""
        lines = [self.title, "", self.overview]

        if self.decisions:
            lines += ["", "DECISIONS"]
            lines += [f"- {item.decision}" for item in self.decisions]

        if self.action_items:
            lines += ["", "ACTION ITEMS"]
            lines += [f"- {item.to_line()}" for item in self.action_items]

        if self.open_questions:
            lines += ["", "OPEN QUESTIONS"]
            lines += [f"- {question}" for question in self.open_questions]

        return "\n".join(lines)
