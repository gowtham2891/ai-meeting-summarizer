"""AI Meeting Summarizer Agent.

Transcribes meeting recordings, extracts decisions and action items with an
LLM, and delivers the summary to WhatsApp, email or disk.

Importing this package registers every built-in provider, so
``get_transcriber`` / ``get_extractor`` / ``get_channel`` resolve by name.
"""

from __future__ import annotations

from .config import ConfigError, Settings, get_settings
from .models import (
    ActionItem,
    Decision,
    MeetingSummary,
    Priority,
    Transcript,
    TranscriptSegment,
)
from .pipeline import MeetingSummarizerPipeline, PipelineResult, summarize

# Importing the provider modules is what populates the registries.
from .transcription import mock as _mock_transcriber  # noqa: F401
from .transcription import sarvam as _sarvam_transcriber  # noqa: F401
from .extraction import mock as _mock_extractor  # noqa: F401
from .extraction import llm as _llm_extractor  # noqa: F401
from .delivery import channels as _channels  # noqa: F401

__version__ = "1.0.0"

__all__ = [
    "ActionItem",
    "ConfigError",
    "Decision",
    "MeetingSummary",
    "MeetingSummarizerPipeline",
    "PipelineResult",
    "Priority",
    "Settings",
    "Transcript",
    "TranscriptSegment",
    "get_settings",
    "summarize",
    "__version__",
]
