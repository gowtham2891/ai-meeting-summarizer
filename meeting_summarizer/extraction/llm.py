"""LangChain-backed extraction against OpenAI or Gemini chat models.

Provider SDKs are imported lazily so the core package installs (and the test
suite runs) without pulling in every LLM client.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..models import ActionItem, Decision, MeetingSummary, Priority, Transcript
from .base import Extractor, register_extractor
from .prompts import JSON_INSTRUCTION, SYSTEM_PROMPT, build_extraction_prompt

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class ExtractionError(RuntimeError):
    """Raised when the model output cannot be turned into a summary."""


# --- schema handed to the model ------------------------------------------


class _ActionItemSchema(BaseModel):
    task: str
    owner: str = "Unassigned"
    due_date: Optional[str] = None
    priority: str = "medium"
    context: str = ""


class _DecisionSchema(BaseModel):
    decision: str
    rationale: str = ""
    made_by: str = "Team"


class _SummarySchema(BaseModel):
    """Mirrors :class:`MeetingSummary` but keeps dates as strings.

    Models emit dates in assorted formats; parsing is done here rather than
    letting Pydantic reject an otherwise good extraction.
    """

    title: str = Field(default="Meeting Summary")
    overview: str = ""
    key_points: List[str] = Field(default_factory=list)
    decisions: List[_DecisionSchema] = Field(default_factory=list)
    action_items: List[_ActionItemSchema] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)


def parse_due_date(raw: Optional[str]) -> Optional[date]:
    """Best-effort ISO date parsing; returns ``None`` rather than raising."""
    if not raw:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"null", "none", "n/a", "-", "tbd"}:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            from datetime import datetime

            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    logger.debug("Could not parse due date %r", raw)
    return None


def parse_priority(raw: Optional[str]) -> Priority:
    text = (raw or "").strip().lower()
    for member in Priority:
        if member.value == text:
            return member
    if text in {"urgent", "critical", "p0", "p1"}:
        return Priority.HIGH
    if text in {"minor", "nice to have", "p3"}:
        return Priority.LOW
    return Priority.MEDIUM


def loads_lenient(raw: str) -> Dict[str, Any]:
    """Parse JSON from a model response that may carry fences or prose."""
    if not raw or not raw.strip():
        raise ExtractionError("Model returned an empty response.")

    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"Model returned malformed JSON: {exc}") from exc

    raise ExtractionError("Model response contained no JSON object.")


def to_meeting_summary(
    payload: Dict[str, Any], transcript: Transcript, fallback_title: str = ""
) -> MeetingSummary:
    """Validate a raw payload and widen it into the domain model."""
    try:
        parsed = _SummarySchema.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(f"Extraction did not match the schema: {exc}") from exc

    overview = parsed.overview.strip() or "No overview was produced for this meeting."

    return MeetingSummary(
        title=(fallback_title or parsed.title or "Meeting Summary").strip(),
        overview=overview,
        key_points=[point for point in parsed.key_points if point.strip()],
        decisions=[
            Decision(
                decision=item.decision.strip(),
                rationale=item.rationale.strip(),
                made_by=item.made_by.strip() or "Team",
            )
            for item in parsed.decisions
            if item.decision.strip()
        ],
        action_items=[
            ActionItem(
                task=item.task.strip(),
                owner=item.owner.strip() or "Unassigned",
                due_date=parse_due_date(item.due_date),
                priority=parse_priority(item.priority),
                context=item.context.strip(),
            )
            for item in parsed.action_items
            if item.task.strip()
        ],
        open_questions=[q for q in parsed.open_questions if q.strip()],
        participants=parsed.participants or transcript.speakers(),
        duration_seconds=transcript.duration,
        language=transcript.language,
    )


# --- the extractor --------------------------------------------------------


class LangChainExtractor(Extractor):
    """Runs the extraction prompt through a LangChain chat model."""

    def __init__(self, settings: Settings, provider: str) -> None:
        super().__init__(settings)
        self.name = provider
        self.provider = provider
        self._model = None

    def _build_model(self):
        if self._model is not None:
            return self._model

        if self.provider == "openai":
            api_key = self.settings.require(
                self.settings.openai_api_key, "OPENAI_API_KEY", "openai"
            )
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ExtractionError(
                    "langchain-openai is not installed. "
                    "Run: pip install 'ai-meeting-summarizer[openai]'"
                ) from exc
            self._model = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=api_key,
                temperature=self.settings.llm_temperature,
                timeout=self.settings.request_timeout,
            )
        elif self.provider == "gemini":
            api_key = self.settings.require(
                self.settings.gemini_api_key, "GEMINI_API_KEY", "gemini"
            )
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ExtractionError(
                    "langchain-google-genai is not installed. "
                    "Run: pip install 'ai-meeting-summarizer[gemini]'"
                ) from exc
            self._model = ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                google_api_key=api_key,
                temperature=self.settings.llm_temperature,
            )
        else:  # pragma: no cover - guarded by the registry
            raise ExtractionError(f"Unsupported LLM provider '{self.provider}'")

        return self._model

    def extract(self, transcript: Transcript, title: str = "") -> MeetingSummary:
        if not transcript.segments:
            raise ExtractionError("Cannot extract a summary from an empty transcript.")

        model = self._build_model()
        prompt = build_extraction_prompt(
            transcript_text=transcript.to_dialogue(),
            duration_minutes=int(transcript.duration // 60),
            speakers=transcript.speakers(),
        )

        # Prefer native structured output; fall back to JSON-in-text parsing.
        try:
            structured = model.with_structured_output(_SummarySchema)
            result = structured.invoke(
                [("system", SYSTEM_PROMPT), ("human", prompt)]
            )
            payload = (
                result.model_dump() if isinstance(result, BaseModel) else dict(result)
            )
        except Exception as exc:  # noqa: BLE001 - provider-specific failures vary
            logger.info("Structured output unavailable (%s); using JSON mode", exc)
            response = model.invoke(
                [("system", SYSTEM_PROMPT), ("human", prompt + JSON_INSTRUCTION)]
            )
            content = getattr(response, "content", response)
            payload = loads_lenient(
                content if isinstance(content, str) else json.dumps(content)
            )

        return to_meeting_summary(payload, transcript, fallback_title=title)


def _openai_factory(settings: Settings) -> Extractor:
    return LangChainExtractor(settings, "openai")


def _gemini_factory(settings: Settings) -> Extractor:
    return LangChainExtractor(settings, "gemini")


register_extractor("openai", _openai_factory)
register_extractor("gemini", _gemini_factory)
