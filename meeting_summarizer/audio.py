"""Audio probing and chunking.

ffmpeg is used when present but never required: if it is missing, or the
recording is already short enough, the file is passed through as a single
chunk. That keeps `pip install` friction low for anyone trying the project.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class AudioError(RuntimeError):
    """Raised when an input file cannot be used as meeting audio."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def validate_audio_file(path: Path) -> Path:
    """Check the file exists, is non-empty and has a media extension."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"File not found: {path}")
    if not path.is_file():
        raise AudioError(f"Not a file: {path}")
    if path.stat().st_size == 0:
        raise AudioError(f"File is empty: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def probe_duration(path: Path) -> Optional[float]:
    """Return duration in seconds, or ``None`` when ffprobe is unavailable."""
    if not ffmpeg_available():
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        payload = json.loads(result.stdout or "{}")
        duration = payload.get("format", {}).get("duration")
        return float(duration) if duration is not None else None
    except (subprocess.SubprocessError, ValueError, KeyError) as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return None


def split_audio(path: Path, chunk_seconds: int) -> List[Tuple[Path, float]]:
    """Split ``path`` into ``chunk_seconds`` pieces.

    Returns ``(chunk_path, start_offset_seconds)`` pairs, always at least one
    entry. Falls back to the original file when ffmpeg is missing or the audio
    is shorter than one chunk.
    """
    path = Path(path)
    if chunk_seconds <= 0:
        return [(path, 0.0)]

    duration = probe_duration(path)
    if duration is None:
        logger.info("ffmpeg not available -- sending %s as a single chunk", path.name)
        return [(path, 0.0)]
    if duration <= chunk_seconds:
        return [(path, 0.0)]

    temp_dir = Path(tempfile.mkdtemp(prefix="meeting-chunks-"))
    chunks: List[Tuple[Path, float]] = []
    start = 0.0
    index = 0

    while start < duration:
        chunk_path = temp_dir / f"chunk_{index:03d}.wav"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(chunk_seconds),
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(chunk_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600)
        except subprocess.SubprocessError as exc:
            raise AudioError(f"ffmpeg failed while chunking {path.name}: {exc}") from exc

        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append((chunk_path, start))
        start += chunk_seconds
        index += 1

    if not chunks:
        return [(path, 0.0)]

    logger.info("Split %s into %d chunks", path.name, len(chunks))
    return chunks
