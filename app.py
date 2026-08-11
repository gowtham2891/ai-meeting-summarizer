"""Streamlit UI for the AI Meeting Summarizer.

    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from meeting_summarizer import __version__, get_settings
from meeting_summarizer.audio import AudioError, ffmpeg_available, validate_audio_file
from meeting_summarizer.config import ConfigError
from meeting_summarizer.delivery.base import available_channels
from meeting_summarizer.extraction.base import available_extractors
from meeting_summarizer.pipeline import MeetingSummarizerPipeline
from meeting_summarizer.transcription.base import available_transcribers

st.set_page_config(
    page_title="AI Meeting Summarizer",
    page_icon="🎙️",
    layout="wide",
)


def _load_secrets_into_env() -> None:
    """Bridge Streamlit Cloud secrets into the environment.

    Configuration is read from environment variables, so secrets added in the
    Streamlit Cloud dashboard are copied across before settings are resolved.
    Existing environment variables win, so a local .env still takes precedence.
    """
    try:
        items = list(st.secrets.items())
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return
    for key, value in items:
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)


_load_secrets_into_env()

PRIORITY_COLORS = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}


def sidebar_config():
    """Render provider controls and return the resolved settings."""
    settings = get_settings(refresh=True)

    with st.sidebar:
        st.title("🎙️ Meeting Summarizer")
        st.caption(f"v{__version__}")
        st.divider()

        st.subheader("Providers")
        transcribers = available_transcribers()
        settings.transcriber = st.selectbox(
            "Transcriber",
            transcribers,
            index=transcribers.index(settings.transcriber)
            if settings.transcriber in transcribers
            else 0,
            help="`mock` needs no API key and returns a fixed sample transcript.",
        )
        extractors = available_extractors()
        settings.extractor = st.selectbox(
            "Extractor",
            extractors,
            index=extractors.index(settings.extractor)
            if settings.extractor in extractors
            else 0,
            help="`mock` uses offline heuristics instead of an LLM.",
        )

        st.subheader("Delivery")
        channels = st.multiselect(
            "Channels",
            available_channels(),
            default=["file"],
            help="Email and WhatsApp need credentials in your .env file.",
        )

        st.divider()
        if settings.is_mock:
            st.success("Mock mode — no credentials required.")
        else:
            st.info("Live mode — reading credentials from .env")
        if not ffmpeg_available():
            st.caption("⚠️ ffmpeg not found: long recordings won't be chunked.")

    return settings, channels


def render_summary(summary) -> None:
    st.subheader(summary.title)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{int(summary.duration_seconds // 60)} min")
    col2.metric("Decisions", len(summary.decisions))
    col3.metric("Action Items", len(summary.action_items))
    col4.metric("Participants", len(summary.participants))

    st.markdown("#### Overview")
    st.write(summary.overview)

    if summary.key_points:
        st.markdown("#### Key Points")
        for point in summary.key_points:
            st.markdown(f"- {point}")

    if summary.decisions:
        st.markdown("#### Decisions")
        for decision in summary.decisions:
            with st.container(border=True):
                st.markdown(f"**{decision.decision}**")
                if decision.rationale:
                    st.caption(f"Rationale: {decision.rationale}")
                st.caption(f"Owner: {decision.made_by}")

    if summary.action_items:
        st.markdown("#### Action Items")
        st.dataframe(
            [
                {
                    "Owner": item.owner,
                    "Task": item.task,
                    "Due": item.due_date.isoformat() if item.due_date else "—",
                    "Priority": item.priority.value,
                    "Context": item.context,
                }
                for item in summary.action_items
            ],
            width="stretch",
            hide_index=True,
        )

    if summary.open_questions:
        st.markdown("#### Open Questions")
        for question in summary.open_questions:
            st.markdown(f"- {question}")


def main() -> None:
    settings, channels = sidebar_config()

    st.title("Meeting Summarizer Agent")
    st.caption(
        "Upload a recording — it gets transcribed, mined for decisions and "
        "action items, and delivered wherever you point it."
    )

    uploaded = st.file_uploader(
        "Meeting recording",
        type=["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "webm"],
        help="In mock mode any file works; the sample transcript is used.",
    )
    title = st.text_input("Title (optional)", placeholder="Weekly product sync")

    use_sample = st.checkbox(
        "Use the built-in sample meeting instead", value=not uploaded
    )

    if st.button("Summarize", type="primary", width="stretch"):
        if use_sample:
            audio_path = (
                Path(__file__).parent
                / "meeting_summarizer"
                / "data"
                / "sample_meeting.wav"
            )
        elif uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(uploaded.getbuffer())
                audio_path = Path(handle.name)
        else:
            st.warning("Upload a recording or tick the sample checkbox.")
            return

        log_area = st.empty()
        messages: list[str] = []

        def progress(stage: str, message: str) -> None:
            messages.append(f"**{stage}** — {message}")
            log_area.info("\n\n".join(messages))

        try:
            with st.spinner("Running the pipeline…"):
                validate_audio_file(audio_path)
                pipeline = MeetingSummarizerPipeline(
                    settings=settings, progress=progress
                )
                result = pipeline.run(
                    audio_path, title=title, channels=channels or ["file"]
                )
        except (AudioError, ConfigError, ValueError) as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - show provider errors in the UI
            st.error(f"Pipeline failed: {exc}")
            return

        st.success(f"Done in {result.elapsed_seconds:.1f}s")

        summary_tab, transcript_tab, delivery_tab, export_tab = st.tabs(
            ["Summary", "Transcript", "Delivery", "Export"]
        )

        with summary_tab:
            render_summary(result.summary)

        with transcript_tab:
            st.caption(
                f"{len(result.transcript.segments)} segments · "
                f"language: {result.transcript.language}"
            )
            for segment in result.transcript.segments:
                speaker = segment.speaker or "Speaker"
                st.markdown(
                    f"`{segment.timestamp()}` **{speaker}** — {segment.text}"
                )

        with delivery_tab:
            for delivery in result.deliveries:
                if delivery.success:
                    st.success(f"{delivery.channel}: {delivery.detail}")
                else:
                    st.error(f"{delivery.channel}: {delivery.detail}")

        with export_tab:
            markdown = result.summary.to_markdown()
            st.download_button(
                "Download Markdown",
                markdown,
                file_name="meeting-summary.md",
                mime="text/markdown",
            )
            st.download_button(
                "Download JSON",
                result.summary.model_dump_json(indent=2),
                file_name="meeting-summary.json",
                mime="application/json",
            )
            st.code(markdown, language="markdown")


if __name__ == "__main__":
    main()
