"""The orchestrator: audio in, structured summary out, delivered everywhere."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .audio import validate_audio_file
from .config import Settings, get_settings
from .delivery.base import DeliveryResult, get_channel
from .extraction.base import get_extractor
from .models import MeetingSummary, Transcript
from .transcription.base import get_transcriber

logger = logging.getLogger(__name__)

#: Called as ``progress(stage, message)`` so CLI and Streamlit can both report.
ProgressCallback = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:
    logger.info("[%s] %s", stage, message)


@dataclass
class PipelineResult:
    """Everything one run produced, including per-channel delivery outcomes."""

    transcript: Transcript
    summary: MeetingSummary
    deliveries: List[DeliveryResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def delivered_ok(self) -> bool:
        return all(result.success for result in self.deliveries)


class MeetingSummarizerPipeline:
    """Wires transcription -> extraction -> delivery.

    Providers are resolved from settings at construction time, so a misspelled
    provider name fails immediately rather than after a long transcription.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.progress = progress or _noop
        self.transcriber = get_transcriber(self.settings)
        self.extractor = get_extractor(self.settings)

    def run(
        self,
        audio_path: Path,
        title: str = "",
        channels: Optional[List[str]] = None,
    ) -> PipelineResult:
        started = time.monotonic()
        audio_path = validate_audio_file(Path(audio_path))

        self.progress("transcribe", f"Transcribing {audio_path.name} via {self.transcriber.name}")
        transcript = self.transcriber.transcribe(audio_path)
        self.progress(
            "transcribe",
            f"Got {len(transcript.segments)} segments "
            f"({transcript.duration / 60:.1f} min, language={transcript.language})",
        )

        self.progress("extract", f"Extracting summary via {self.extractor.name}")
        summary = self.extractor.extract(transcript, title=title)
        self.progress(
            "extract",
            f"{len(summary.decisions)} decision(s), "
            f"{len(summary.action_items)} action item(s)",
        )

        deliveries = self._deliver(summary, channels or ["file"])
        elapsed = time.monotonic() - started
        self.progress("done", f"Completed in {elapsed:.1f}s")

        return PipelineResult(
            transcript=transcript,
            summary=summary,
            deliveries=deliveries,
            elapsed_seconds=elapsed,
        )

    def _deliver(
        self, summary: MeetingSummary, channels: List[str]
    ) -> List[DeliveryResult]:
        results: List[DeliveryResult] = []
        for name in channels:
            try:
                channel = get_channel(name, self.settings)
            except ValueError as exc:
                results.append(DeliveryResult(name, False, str(exc)))
                continue

            self.progress("deliver", f"Sending via {name}")
            try:
                result = channel.send(summary)
            except Exception as exc:  # noqa: BLE001 - a channel must never abort the run
                logger.exception("Channel %s raised", name)
                result = DeliveryResult(name, False, f"unexpected error: {exc}")
            results.append(result)
            self.progress("deliver", str(result))
        return results


def summarize(
    audio_path: Path,
    title: str = "",
    channels: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
    progress: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """Convenience wrapper for one-shot use."""
    pipeline = MeetingSummarizerPipeline(settings=settings, progress=progress)
    return pipeline.run(audio_path, title=title, channels=channels)
