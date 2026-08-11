"""Extractor interface: transcript in, structured :class:`MeetingSummary` out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict

from ..config import Settings
from ..models import MeetingSummary, Transcript


class Extractor(ABC):
    """Turns a transcript into the structured meeting summary."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def extract(self, transcript: Transcript, title: str = "") -> MeetingSummary:
        """Extract summary, decisions and action items from ``transcript``."""


_REGISTRY: Dict[str, Callable[[Settings], Extractor]] = {}


def register_extractor(name: str, factory: Callable[[Settings], Extractor]) -> None:
    _REGISTRY[name] = factory


def available_extractors() -> list[str]:
    return sorted(_REGISTRY)


def get_extractor(settings: Settings) -> Extractor:
    try:
        factory = _REGISTRY[settings.extractor]
    except KeyError:
        raise ValueError(
            f"Unknown extractor '{settings.extractor}'. "
            f"Available: {', '.join(available_extractors())}"
        ) from None
    return factory(settings)
