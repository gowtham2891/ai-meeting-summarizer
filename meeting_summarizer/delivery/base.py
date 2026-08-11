"""Delivery channels: where a finished summary gets sent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List

from ..config import Settings
from ..models import MeetingSummary


@dataclass
class DeliveryResult:
    """Outcome of one channel attempt. Never raises past the pipeline."""

    channel: str
    success: bool
    detail: str = ""

    def __str__(self) -> str:
        status = "ok" if self.success else "failed"
        return f"{self.channel}: {status} -- {self.detail}" if self.detail else f"{self.channel}: {status}"


class DeliveryChannel(ABC):
    """Sends a :class:`MeetingSummary` somewhere useful."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def send(self, summary: MeetingSummary) -> DeliveryResult:
        """Deliver ``summary``, returning a result rather than raising."""


_REGISTRY: Dict[str, Callable[[Settings], DeliveryChannel]] = {}


def register_channel(name: str, factory: Callable[[Settings], DeliveryChannel]) -> None:
    _REGISTRY[name] = factory


def available_channels() -> List[str]:
    return sorted(_REGISTRY)


def get_channel(name: str, settings: Settings) -> DeliveryChannel:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown delivery channel '{name}'. "
            f"Available: {', '.join(available_channels())}"
        ) from None
    return factory(settings)
