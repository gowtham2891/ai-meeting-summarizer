"""Transcriber interface plus the registry the pipeline resolves against."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict

from ..config import Settings
from ..models import Transcript


class Transcriber(ABC):
    """Turns an audio file into a timestamped :class:`Transcript`."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe ``audio_path``, raising on unrecoverable provider errors."""


_REGISTRY: Dict[str, Callable[[Settings], Transcriber]] = {}


def register_transcriber(name: str, factory: Callable[[Settings], Transcriber]) -> None:
    _REGISTRY[name] = factory


def available_transcribers() -> list[str]:
    return sorted(_REGISTRY)


def get_transcriber(settings: Settings) -> Transcriber:
    """Build the transcriber named by ``settings.transcriber``."""
    try:
        factory = _REGISTRY[settings.transcriber]
    except KeyError:
        raise ValueError(
            f"Unknown transcriber '{settings.transcriber}'. "
            f"Available: {', '.join(available_transcribers())}"
        ) from None
    return factory(settings)
