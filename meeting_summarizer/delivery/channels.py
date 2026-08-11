"""Concrete delivery channels: file, console, email (SMTP) and WhatsApp (Twilio).

Every channel returns a :class:`DeliveryResult` instead of raising, so one
misconfigured destination never loses the summary that was already produced.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List

import httpx

from ..config import Settings
from ..models import MeetingSummary
from .base import DeliveryChannel, DeliveryResult, register_channel

logger = logging.getLogger(__name__)


def _slugify(text: str, limit: int = 50) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:limit] or "meeting-summary"


class FileChannel(DeliveryChannel):
    """Writes the summary to ``OUTPUT_DIR`` as markdown plus JSON."""

    name = "file"

    def send(self, summary: MeetingSummary) -> DeliveryResult:
        try:
            out_dir = Path(self.settings.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = _slugify(summary.title)

            md_path = out_dir / f"{stem}.md"
            md_path.write_text(summary.to_markdown(), encoding="utf-8")

            json_path = out_dir / f"{stem}.json"
            json_path.write_text(
                summary.model_dump_json(indent=2), encoding="utf-8"
            )
            return DeliveryResult("file", True, f"wrote {md_path} and {json_path}")
        except OSError as exc:
            return DeliveryResult("file", False, f"write failed: {exc}")


class ConsoleChannel(DeliveryChannel):
    """Prints the summary to stdout. Always available, never fails."""

    name = "console"

    def send(self, summary: MeetingSummary) -> DeliveryResult:
        print(summary.to_markdown())
        return DeliveryResult("console", True, "printed to stdout")


class EmailChannel(DeliveryChannel):
    """Sends the summary over SMTP as a multipart text/HTML message."""

    name = "email"

    def send(self, summary: MeetingSummary) -> DeliveryResult:
        settings = self.settings
        missing = [
            label
            for label, value in (
                ("SMTP_HOST", settings.smtp_host),
                ("EMAIL_FROM", settings.email_from),
                ("EMAIL_TO", settings.email_to),
            )
            if not value
        ]
        if missing:
            return DeliveryResult(
                "email", False, f"missing config: {', '.join(missing)}"
            )

        message = EmailMessage()
        message["Subject"] = summary.title
        message["From"] = settings.email_from
        message["To"] = ", ".join(settings.email_to)
        message.set_content(summary.to_plain_text())
        message.add_alternative(self._to_html(summary), subtype="html")

        try:
            with smtplib.SMTP(
                settings.smtp_host, settings.smtp_port, timeout=settings.request_timeout
            ) as server:
                if settings.smtp_use_tls:
                    server.starttls()
                if settings.smtp_user:
                    server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(message)
            return DeliveryResult(
                "email", True, f"sent to {len(settings.email_to)} recipient(s)"
            )
        except (smtplib.SMTPException, OSError) as exc:
            logger.error("Email delivery failed: %s", exc)
            return DeliveryResult("email", False, str(exc))

    def _to_html(self, summary: MeetingSummary) -> str:
        def items(values: List[str]) -> str:
            return "".join(f"<li>{value}</li>" for value in values)

        rows = "".join(
            f"<tr><td>{item.owner}</td><td>{item.task}</td>"
            f"<td>{item.due_date or '-'}</td><td>{item.priority.value}</td></tr>"
            for item in summary.action_items
        )
        return f"""\
<html><body style="font-family:system-ui,sans-serif;line-height:1.5">
<h2>{summary.title}</h2>
<p>{summary.overview}</p>
{f'<h3>Key Points</h3><ul>{items(summary.key_points)}</ul>' if summary.key_points else ''}
{f'<h3>Decisions</h3><ul>{items([d.decision for d in summary.decisions])}</ul>' if summary.decisions else ''}
{f'<h3>Action Items</h3><table border="1" cellpadding="6" cellspacing="0"><tr><th>Owner</th><th>Task</th><th>Due</th><th>Priority</th></tr>{rows}</table>' if rows else ''}
{f'<h3>Open Questions</h3><ul>{items(summary.open_questions)}</ul>' if summary.open_questions else ''}
</body></html>"""


class WhatsAppChannel(DeliveryChannel):
    """Sends the summary via the Twilio WhatsApp API.

    Twilio caps a message body at 1600 characters, so long summaries are split
    into numbered parts rather than silently truncated.
    """

    name = "whatsapp"
    MAX_BODY = 1500

    def send(self, summary: MeetingSummary) -> DeliveryResult:
        settings = self.settings
        missing = [
            label
            for label, value in (
                ("TWILIO_ACCOUNT_SID", settings.twilio_account_sid),
                ("TWILIO_AUTH_TOKEN", settings.twilio_auth_token),
                ("WHATSAPP_TO", settings.whatsapp_to),
            )
            if not value
        ]
        if missing:
            return DeliveryResult(
                "whatsapp", False, f"missing config: {', '.join(missing)}"
            )

        parts = self._chunk(summary.to_plain_text())
        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{settings.twilio_account_sid}/Messages.json"
        )
        sent = 0
        for recipient in settings.whatsapp_to:
            to = recipient if recipient.startswith("whatsapp:") else f"whatsapp:{recipient}"
            for index, part in enumerate(parts, start=1):
                body = part if len(parts) == 1 else f"({index}/{len(parts)})\n{part}"
                try:
                    response = httpx.post(
                        url,
                        auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                        data={
                            "From": settings.twilio_whatsapp_from,
                            "To": to,
                            "Body": body,
                        },
                        timeout=settings.request_timeout,
                    )
                    response.raise_for_status()
                    sent += 1
                except (httpx.HTTPError, httpx.TransportError) as exc:
                    logger.error("WhatsApp delivery to %s failed: %s", to, exc)
                    return DeliveryResult("whatsapp", False, str(exc))

        return DeliveryResult("whatsapp", True, f"sent {sent} message(s)")

    def _chunk(self, text: str) -> List[str]:
        """Split on paragraph boundaries, keeping each part under the cap."""
        if len(text) <= self.MAX_BODY:
            return [text]

        parts: List[str] = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > self.MAX_BODY:
                if current:
                    parts.append(current.rstrip())
                # A single over-long line still has to be hard-split.
                while len(line) > self.MAX_BODY:
                    parts.append(line[: self.MAX_BODY])
                    line = line[self.MAX_BODY :]
                current = line + "\n"
            else:
                current += line + "\n"
        if current.strip():
            parts.append(current.rstrip())
        return parts


def _register_all() -> None:
    register_channel("file", FileChannel)
    register_channel("console", ConsoleChannel)
    register_channel("email", EmailChannel)
    register_channel("whatsapp", WhatsAppChannel)


_register_all()
