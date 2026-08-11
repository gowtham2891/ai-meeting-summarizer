"""Credential-free extractor backed by rule-based heuristics.

This is not pretending to be an LLM -- it is a deterministic baseline that
keeps the pipeline runnable offline, gives the tests something real to assert
on, and doubles as the quality floor the LLM extractor should beat.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..config import Settings
from ..models import ActionItem, Decision, MeetingSummary, Priority, Transcript
from .base import Extractor, register_extractor

DECISION_MARKERS = (
    "it's decided",
    "its decided",
    "we decided",
    "let's go with",
    "lets go with",
    "we'll ship",
    "we will ship",
    "we ship",
    "agreed",
    "final call",
    "we're going with",
    "so we roll out",
    "then it's",
)

ACTION_PATTERNS = (
    re.compile(r"\bI'?ll\s+(?P<task>[^.?!]{6,120})", re.IGNORECASE),
    re.compile(r"\bI will\s+(?P<task>[^.?!]{6,120})", re.IGNORECASE),
    re.compile(r"\bcan you\s+(?P<task>[^.?!]{6,120})", re.IGNORECASE),
    re.compile(r"\bI'?d want\s+(?P<task>[^.?!]{6,120})", re.IGNORECASE),
    re.compile(r"\bneed(?:s)? to\s+(?P<task>[^.?!]{6,120})", re.IGNORECASE),
)

HIGH_PRIORITY_HINTS = ("before", "by ", "asap", "urgent", "blocker", "failing", "this week")
LOW_PRIORITY_HINTS = ("eventually", "nice to have", "someday", "at some point")

OPEN_QUESTION_MARKERS = ("open question", "not sure", "unclear", "tbd", "to be decided")

WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

_STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "are", "was", "have", "has",
    "but", "not", "you", "your", "our", "can", "will", "would", "should",
    "there", "their", "then", "than", "from", "what", "when", "were", "into",
    "about", "which", "them", "they", "been", "just", "like", "okay", "alright",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.-")


def _priority_for(text: str) -> Priority:
    lowered = text.lower()
    if any(hint in lowered for hint in LOW_PRIORITY_HINTS):
        return Priority.LOW
    if any(hint in lowered for hint in HIGH_PRIORITY_HINTS):
        return Priority.HIGH
    return Priority.MEDIUM


def _mentioned_deadline(text: str) -> str:
    lowered = text.lower()
    for day in WEEKDAYS:
        if day in lowered:
            return day.capitalize()
    for phrase in ("month end", "end of month", "next week", "this week", "friday"):
        if phrase in lowered:
            return phrase
    return ""


class HeuristicExtractor(Extractor):
    """Rule-based summary extraction. Deterministic, offline, no API key."""

    name = "mock"

    def extract(self, transcript: Transcript, title: str = "") -> MeetingSummary:
        if not transcript.segments:
            raise ValueError("Cannot extract a summary from an empty transcript.")

        decisions = self._decisions(transcript)
        action_items = self._action_items(transcript)
        questions = self._open_questions(transcript)
        key_points = self._key_points(transcript)

        return MeetingSummary(
            title=title or self._title(transcript),
            overview=self._overview(transcript, decisions, action_items),
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            open_questions=questions,
            participants=transcript.speakers(),
            duration_seconds=transcript.duration,
            language=transcript.language,
        )

    # -- components ---------------------------------------------------------

    def _title(self, transcript: Transcript) -> str:
        keywords = self._keywords(transcript, limit=3)
        if not keywords:
            return "Meeting Summary"
        return "Meeting: " + ", ".join(word.capitalize() for word in keywords)

    def _keywords(self, transcript: Transcript, limit: int = 6) -> List[str]:
        counts: dict[str, int] = {}
        for word in re.findall(r"[a-zA-Z]{4,}", transcript.text.lower()):
            if word in _STOPWORDS:
                continue
            counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [word for word, count in ranked[:limit] if count > 1]

    def _overview(
        self,
        transcript: Transcript,
        decisions: List[Decision],
        actions: List[ActionItem],
    ) -> str:
        minutes = max(int(transcript.duration // 60), 1)
        speakers = transcript.speakers()
        who = f"{len(speakers)} participants" if speakers else "the team"
        topics = self._keywords(transcript, limit=4)
        topic_text = f" covering {', '.join(topics)}" if topics else ""
        return (
            f"A {minutes}-minute discussion between {who}{topic_text}. "
            f"{len(decisions)} decision(s) were reached and "
            f"{len(actions)} action item(s) were assigned."
        )

    def _key_points(self, transcript: Transcript) -> List[str]:
        """Pick the longest utterance per speaker, in meeting order."""
        best: dict[str, tuple[int, str]] = {}
        for segment in transcript.segments:
            speaker = segment.speaker or "Speaker"
            words = len(segment.text.split())
            if words < 8:
                continue
            if words > best.get(speaker, (0, ""))[0]:
                best[speaker] = (words, _clean(segment.text))
        points = [text for _, (_, text) in best.items()]
        return points[:7]

    def _decisions(self, transcript: Transcript) -> List[Decision]:
        found: List[Decision] = []
        seen: set[str] = set()
        for segment in transcript.segments:
            lowered = segment.text.lower()
            if not any(marker in lowered for marker in DECISION_MARKERS):
                continue
            # "Can we ship without refunds?" proposes a decision, it isn't one.
            if segment.text.rstrip().endswith("?"):
                continue
            text = _clean(segment.text)
            key = text.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            found.append(
                Decision(
                    decision=text,
                    rationale="Stated during the meeting.",
                    made_by=segment.speaker or "Team",
                )
            )
        return found

    def _action_items(self, transcript: Transcript) -> List[ActionItem]:
        found: List[ActionItem] = []
        seen: set[str] = set()

        for segment in transcript.segments:
            for pattern in ACTION_PATTERNS:
                match = pattern.search(segment.text)
                if not match:
                    continue
                task = _clean(match.group("task"))
                if len(task.split()) < 2:
                    continue
                key = task.lower()[:50]
                if key in seen:
                    continue
                seen.add(key)

                owner = self._owner_for(segment.text, segment.speaker, pattern)
                deadline = _mentioned_deadline(segment.text)
                context = f"Mentioned at {segment.timestamp()}"
                if deadline:
                    context += f"; deadline referenced as '{deadline}'"

                found.append(
                    ActionItem(
                        task=task[0].upper() + task[1:] if task else task,
                        owner=owner,
                        due_date=None,
                        priority=_priority_for(segment.text),
                        context=context,
                    )
                )
                break  # one action per utterance keeps output readable
        return found

    def _owner_for(self, text: str, speaker: Optional[str], pattern) -> str:
        """`can you ...` targets the addressee; `I'll ...` targets the speaker."""
        if "can you" in pattern.pattern:
            return "Addressee"
        return speaker or "Unassigned"

    def _open_questions(self, transcript: Transcript) -> List[str]:
        found: List[str] = []
        for segment in transcript.segments:
            lowered = segment.text.lower()
            is_marked = any(marker in lowered for marker in OPEN_QUESTION_MARKERS)
            is_short_question = segment.text.strip().endswith("?") and len(
                segment.text.split()
            ) <= 12
            if is_marked or is_short_question:
                cleaned = _clean(segment.text)
                if cleaned and cleaned not in found:
                    found.append(cleaned)
        return found[:5]


def _factory(settings: Settings) -> Extractor:
    return HeuristicExtractor(settings)


register_extractor("mock", _factory)
