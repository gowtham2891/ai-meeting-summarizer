"""Sarvam AI Saaras speech-to-text-translate transcriber.

Saaras is used rather than plain STT because meeting audio in this workload is
routinely Tenglish (Telugu/English codemix); Saaras returns a normalised
transcript instead of two half-broken monolingual ones.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

from ..audio import probe_duration, split_audio
from ..config import Settings
from ..models import Transcript, TranscriptSegment
from .base import Transcriber, register_transcriber

logger = logging.getLogger(__name__)

#: Sarvam rejects uploads above this size; chunking keeps requests under it.
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


class SarvamTranscriptionError(RuntimeError):
    """Raised when Sarvam returns a non-recoverable error."""


class SarvamTranscriber(Transcriber):
    name = "sarvam"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.require(
            settings.sarvam_api_key, "SARVAM_API_KEY", "sarvam"
        )
        self.endpoint = f"{settings.sarvam_base_url.rstrip('/')}/speech-to-text-translate"

    # -- public API ---------------------------------------------------------

    def transcribe(self, audio_path: Path) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        chunks = split_audio(audio_path, self.settings.chunk_seconds)
        segments: List[TranscriptSegment] = []
        language = "unknown"
        offset = 0.0

        for chunk_path, chunk_start in chunks:
            payload = self._post_chunk(chunk_path)
            language = payload.get("language_code") or language
            chunk_segments = self._parse_segments(payload, offset=chunk_start or offset)
            segments.extend(chunk_segments)
            offset = max((s.end for s in segments), default=offset)

        if not segments:
            raise SarvamTranscriptionError(
                "Sarvam returned no transcript segments for this recording."
            )

        return Transcript(
            segments=segments, language=language, source=str(audio_path)
        )

    # -- internals ----------------------------------------------------------

    def _post_chunk(self, chunk_path: Path) -> Dict[str, Any]:
        """POST one audio chunk, retrying transient failures with backoff."""
        size = chunk_path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise SarvamTranscriptionError(
                f"Chunk {chunk_path.name} is {size / 1e6:.1f} MB, above Sarvam's "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit. Lower CHUNK_SECONDS."
            )

        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                with open(chunk_path, "rb") as handle:
                    response = httpx.post(
                        self.endpoint,
                        headers={"api-subscription-key": self.api_key},
                        files={"file": (chunk_path.name, handle, "audio/wav")},
                        data={"model": self.settings.sarvam_model},
                        timeout=self.settings.request_timeout,
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if attempt == self.settings.max_retries:
                    break
                backoff = 2 ** (attempt - 1)
                logger.warning(
                    "Sarvam request failed (attempt %d/%d): %s -- retrying in %ss",
                    attempt,
                    self.settings.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        raise SarvamTranscriptionError(
            f"Sarvam transcription failed after {self.settings.max_retries} "
            f"attempts: {last_error}"
        ) from last_error

    def _parse_segments(
        self, payload: Dict[str, Any], offset: float
    ) -> List[TranscriptSegment]:
        """Normalise Sarvam's response into segments.

        Saaras returns word/segment timings when diarization is on and a bare
        ``transcript`` string otherwise, so both shapes are handled.
        """
        raw_segments = payload.get("diarized_transcript", {}).get("entries") or []
        segments: List[TranscriptSegment] = []

        for entry in raw_segments:
            text = (entry.get("transcript") or "").strip()
            if not text:
                continue
            start = float(entry.get("start_time_seconds", 0.0)) + offset
            end = float(entry.get("end_time_seconds", start)) + offset
            speaker = entry.get("speaker_id")
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=max(end, start),
                    text=text,
                    speaker=f"Speaker {speaker}" if speaker is not None else None,
                )
            )

        if segments:
            return segments

        # Fall back to the flat transcript field, spread over the chunk length.
        text = (payload.get("transcript") or "").strip()
        if not text:
            return []
        duration = float(payload.get("duration_seconds") or 0.0)
        return [
            TranscriptSegment(
                start=offset, end=offset + max(duration, 1.0), text=text
            )
        ]


def _factory(settings: Settings) -> Transcriber:
    return SarvamTranscriber(settings)


register_transcriber("sarvam", _factory)

__all__ = ["SarvamTranscriber", "SarvamTranscriptionError", "probe_duration"]
