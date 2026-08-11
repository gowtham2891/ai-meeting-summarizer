"""Prompt templates for the extraction stage.

Kept in their own module so prompts can be reviewed, diffed and tuned without
touching provider plumbing.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert meeting analyst. You read a raw meeting transcript and produce \
a precise, structured summary that a participant could act on without listening \
to the recording.

Rules you must follow:
1. Only state things the transcript supports. Never invent names, dates or numbers.
2. An action item requires a concrete task. If no owner is named, use "Unassigned".
3. A decision is something the group settled on, not something merely discussed.
4. Resolve relative dates ("Thursday", "next week") against the meeting date when \
one is supplied; otherwise leave due_date null.
5. Open questions are things explicitly left unresolved.
6. Write in clear, plain English even when the transcript is codemixed.
"""

EXTRACTION_PROMPT = """\
Analyse the meeting transcript below and return the structured summary.

{date_line}Meeting duration: {duration_minutes} minutes
Speakers detected: {speakers}

TRANSCRIPT
----------
{transcript}
----------

Produce:
- title: a short descriptive title (max 8 words)
- overview: 2-4 sentences on what the meeting covered and where it landed
- key_points: the substantive points raised (3-7 items)
- decisions: each with the decision, its rationale, and who made it
- action_items: each with task, owner, due_date (YYYY-MM-DD or null), priority \
(high/medium/low) and one line of context
- open_questions: anything explicitly left unresolved
- participants: speaker names you can identify from the transcript
"""

#: Appended when the provider cannot enforce a schema natively.
JSON_INSTRUCTION = """\

Return ONLY a JSON object with exactly these keys: title, overview, key_points, \
decisions, action_items, open_questions, participants. No markdown fences, no \
commentary before or after the JSON.
"""


def build_extraction_prompt(
    transcript_text: str,
    duration_minutes: int,
    speakers: list[str],
    meeting_date: str = "",
) -> str:
    return EXTRACTION_PROMPT.format(
        date_line=f"Meeting date: {meeting_date}\n" if meeting_date else "",
        duration_minutes=duration_minutes,
        speakers=", ".join(speakers) if speakers else "not labelled",
        transcript=transcript_text,
    )
