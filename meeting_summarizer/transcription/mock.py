"""Deterministic transcriber used for local runs, demos and CI.

It reads a sidecar ``.json``/``.txt`` transcript next to the audio file when one
exists, and otherwise emits a built-in sample meeting. No credentials, no
network, identical output every run -- which is what makes the pipeline tests
meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..config import Settings
from ..models import Transcript, TranscriptSegment
from .base import Transcriber, register_transcriber

SAMPLE_MEETING: List[tuple] = [
    (0.0, 12.0, "Priya", "Alright, let's start. Agenda today is the launch timeline, the pricing page, and the support backlog."),
    (12.0, 31.0, "Arjun", "On the timeline -- backend is done, but the payments integration is still failing on refunds. I need two more days."),
    (31.0, 47.0, "Priya", "Two days pushes us to Thursday. Can we ship without refunds and follow up next week?"),
    (47.0, 66.0, "Arjun", "We can, as long as support knows. Refund requests would have to be handled manually until then."),
    (66.0, 84.0, "Meera", "Support can absorb that. Volume is around ten a day. I'll write a manual refund runbook before Thursday."),
    (84.0, 103.0, "Priya", "Then it's decided -- we ship Thursday without automated refunds, and Arjun lands the fix the following Monday."),
    (103.0, 124.0, "Meera", "On pricing: the new page converts at four percent versus two point five on the old one, so I say we roll it out fully."),
    (124.0, 141.0, "Arjun", "Sample size? If that's a week of traffic I'd want another week before we commit."),
    (141.0, 158.0, "Meera", "It's nine days and about twelve hundred sessions. I'll pull the significance numbers and share by Friday."),
    (158.0, 176.0, "Priya", "Good. Last thing -- the support backlog is at eighty tickets. Meera, can you get that under thirty by month end?"),
    (176.0, 192.0, "Meera", "Yes, if I get one more part-time person. Otherwise realistically fifty."),
    (192.0, 208.0, "Priya", "I'll take the hiring ask to Kiran this week. Open question is whether the budget allows it."),
    (208.0, 219.0, "Priya", "That's everything. I'll send notes out after this."),
]


class MockTranscriber(Transcriber):
    """Returns a fixed transcript so the rest of the pipeline can be exercised."""

    name = "mock"

    def transcribe(self, audio_path: Path) -> Transcript:
        audio_path = Path(audio_path)

        sidecar_json = audio_path.with_suffix(".json")
        if sidecar_json.exists():
            return self._from_json(sidecar_json, audio_path)

        sidecar_txt = audio_path.with_suffix(".txt")
        if sidecar_txt.exists():
            return self._from_text(sidecar_txt, audio_path)

        segments = [
            TranscriptSegment(start=start, end=end, text=text, speaker=speaker)
            for start, end, speaker, text in SAMPLE_MEETING
        ]
        return Transcript(segments=segments, language="en-IN", source=str(audio_path))

    def _from_json(self, path: Path, source: Path) -> Transcript:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [TranscriptSegment(**entry) for entry in payload.get("segments", [])]
        return Transcript(
            segments=segments,
            language=payload.get("language", "unknown"),
            source=str(source),
        )

    def _from_text(self, path: Path, source: Path) -> Transcript:
        """Treat each non-empty line as one segment, ~4 words per second."""
        segments: List[TranscriptSegment] = []
        cursor = 0.0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            speaker = None
            if ":" in line[:40]:
                head, _, tail = line.partition(":")
                if head and len(head.split()) <= 4:
                    speaker, line = head.strip(), tail.strip()
            if not line:
                continue
            duration = max(len(line.split()) / 4.0, 1.0)
            segments.append(
                TranscriptSegment(
                    start=cursor, end=cursor + duration, text=line, speaker=speaker
                )
            )
            cursor += duration
        return Transcript(segments=segments, language="en", source=str(source))


def _factory(settings: Settings) -> Transcriber:
    return MockTranscriber(settings)


register_transcriber("mock", _factory)
