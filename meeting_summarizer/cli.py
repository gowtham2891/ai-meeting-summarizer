"""Command line interface.

    meeting-summarizer run recording.mp3 --channel file --channel console
    meeting-summarizer providers
    meeting-summarizer demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .audio import AudioError, ffmpeg_available
from .config import ConfigError, get_settings
from .health import check_settings
from .delivery.base import available_channels
from .extraction.base import available_extractors
from .models import MeetingSummary
from .pipeline import MeetingSummarizerPipeline, PipelineResult
from .transcription.base import available_transcribers


def _init_stdio() -> bool:
    """Make stdout UTF-8 where possible; report whether glyphs are safe.

    Windows consoles still default to cp1252, which raises UnicodeEncodeError
    on box-drawing glyphs *and* on any non-Latin transcript text. Reconfiguring
    to UTF-8 with ``errors="replace"`` fixes both; when that is not possible we
    fall back to ASCII symbols instead of crashing mid-run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform specific
                pass

    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "▸✓✗•".encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError):  # pragma: no cover - platform specific
        return False


UNICODE_OK = _init_stdio()

SYMBOLS = {
    "arrow": "▸" if UNICODE_OK else ">",
    "bullet": "•" if UNICODE_OK else "-",
    "check": "✓" if UNICODE_OK else "+",
    "cross": "✗" if UNICODE_OK else "x",
    "query": "?",
}

console = Console()

STAGE_STYLE = {
    "transcribe": "cyan",
    "extract": "magenta",
    "deliver": "yellow",
    "done": "green",
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


def _progress(stage: str, message: str) -> None:
    style = STAGE_STYLE.get(stage, "white")
    console.print(f"[{style}]{SYMBOLS['arrow']} {stage:<10}[/{style}] {message}")


def _render_summary(summary: MeetingSummary) -> None:
    console.print()
    console.print(Panel(summary.overview, title=summary.title, border_style="blue"))

    if summary.key_points:
        console.print("\n[bold]Key Points[/bold]")
        for point in summary.key_points:
            console.print(f"  {SYMBOLS['bullet']} {point}")

    if summary.decisions:
        console.print("\n[bold]Decisions[/bold]")
        for decision in summary.decisions:
            console.print(f"  {SYMBOLS['check']} {decision.decision}  [dim]({decision.made_by})[/dim]")

    if summary.action_items:
        table = Table(title="Action Items", show_lines=False, header_style="bold")
        table.add_column("Owner", style="cyan")
        table.add_column("Task")
        table.add_column("Due", style="dim")
        table.add_column("Priority")
        priority_colors = {"high": "red", "medium": "yellow", "low": "green"}
        for item in summary.action_items:
            colour = priority_colors.get(item.priority.value, "white")
            table.add_row(
                item.owner,
                item.task,
                item.due_date.isoformat() if item.due_date else "-",
                f"[{colour}]{item.priority.value}[/{colour}]",
            )
        console.print()
        console.print(table)

    if summary.open_questions:
        console.print("\n[bold]Open Questions[/bold]")
        for question in summary.open_questions:
            console.print(f"  {SYMBOLS['query']} {question}")


def _render_deliveries(result: PipelineResult) -> None:
    console.print("\n[bold]Delivery[/bold]")
    for delivery in result.deliveries:
        icon = (f"[green]{SYMBOLS['check']}[/green]" if delivery.success
                else f"[red]{SYMBOLS['cross']}[/red]")
        console.print(f"  {icon} {delivery.channel}: {delivery.detail}")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="meeting-summarizer")
def cli() -> None:
    """Transcribe meetings, extract action items, deliver summaries."""


@cli.command()
@click.argument("audio", type=click.Path(path_type=Path))
@click.option("--title", default="", help="Override the generated meeting title.")
@click.option(
    "--channel",
    "channels",
    multiple=True,
    help=f"Delivery channel, repeatable. Options: {', '.join(available_channels())}",
)
@click.option("--transcriber", default=None, help="Override the TRANSCRIBER setting.")
@click.option("--extractor", default=None, help="Override the EXTRACTOR setting.")
@click.option("--output-dir", default=None, help="Override OUTPUT_DIR.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def run(
    audio: Path,
    title: str,
    channels: Tuple[str, ...],
    transcriber: str | None,
    extractor: str | None,
    output_dir: str | None,
    verbose: bool,
) -> None:
    """Summarize a meeting recording at AUDIO."""
    _configure_logging(verbose)
    settings = get_settings(refresh=True)
    if transcriber:
        settings.transcriber = transcriber
    if extractor:
        settings.extractor = extractor
    if output_dir:
        settings.output_dir = output_dir

    console.print(f"[dim]{settings.describe()}[/dim]")
    if not ffmpeg_available():
        console.print(
            "[dim]ffmpeg not found -- long recordings will be sent unchunked.[/dim]"
        )

    try:
        pipeline = MeetingSummarizerPipeline(settings=settings, progress=_progress)
        result = pipeline.run(audio, title=title, channels=list(channels) or ["file"])
    except (AudioError, ConfigError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - surface provider failures cleanly
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        if verbose:
            console.print_exception()
        sys.exit(2)

    _render_summary(result.summary)
    _render_deliveries(result)
    console.print(f"\n[dim]Finished in {result.elapsed_seconds:.1f}s[/dim]")
    sys.exit(0 if result.delivered_ok else 3)


@cli.command()
def providers() -> None:
    """List the available providers and the current configuration."""
    settings = get_settings(refresh=True)

    table = Table(title="Providers", header_style="bold")
    table.add_column("Stage", style="cyan")
    table.add_column("Available")
    table.add_column("Selected", style="green")
    table.add_row(
        "transcriber", ", ".join(available_transcribers()), settings.transcriber
    )
    table.add_row("extractor", ", ".join(available_extractors()), settings.extractor)
    table.add_row("delivery", ", ".join(available_channels()), "-")
    console.print(table)

    console.print(f"\n[dim]{settings.describe()}[/dim]")
    console.print(f"[dim]ffmpeg available: {ffmpeg_available()}[/dim]")


@cli.command()
def check() -> None:
    """Verify the configured API keys actually work."""
    settings = get_settings(refresh=True)
    console.print(f"[dim]{settings.describe()}[/dim]")
    console.print()

    results = check_settings(settings)
    if not results:
        console.print("[yellow]Nothing to check for this configuration.[/yellow]")
        return

    for result in results:
        icon = (
            f"[green]{SYMBOLS['check']}[/green]"
            if result.ok
            else f"[red]{SYMBOLS['cross']}[/red]"
        )
        console.print(f"  {icon} [bold]{result.provider}[/bold]: {result.message}")

    failed = [result for result in results if not result.ok]
    console.print()
    if failed:
        console.print(f"[red]{len(failed)} check(s) failed.[/red]")
        sys.exit(2)
    console.print("[green]All checks passed.[/green]")

@cli.command()
@click.option("--output-dir", default="output", help="Where to write the summary.")
def demo(output_dir: str) -> None:
    """Run the full pipeline on the built-in sample meeting -- no API keys needed."""
    _configure_logging(False)
    settings = get_settings(refresh=True)
    settings.transcriber = "mock"
    settings.extractor = "mock"
    settings.output_dir = output_dir

    sample = Path(__file__).parent / "data" / "sample_meeting.wav"
    console.print("[bold]Running demo with the built-in sample meeting[/bold]\n")

    pipeline = MeetingSummarizerPipeline(settings=settings, progress=_progress)
    result = pipeline.run(sample, channels=["file"])

    _render_summary(result.summary)
    _render_deliveries(result)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
