# AI Meeting Summarizer Agent

[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-meeting-summarizerr.streamlit.app)
[![CI](https://github.com/gowtham2891/ai-meeting-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/gowtham2891/ai-meeting-summarizer/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**▶ Try it live — [ai-meeting-summarizerr.streamlit.app](https://ai-meeting-summarizerr.streamlit.app)** · runs in mock mode, no credentials needed — or paste your own API keys in the sidebar to drive the live providers.

An agentic pipeline that turns a meeting recording into a structured, actionable
summary — and delivers it to WhatsApp, email, or disk before anyone has to ask
"what did we decide?"

Built for Indian workplace audio, where meetings routinely switch between Telugu
and English mid-sentence. Transcription runs through **Sarvam Saaras**, which
handles codemixed speech rather than forcing a single-language model to guess.

---

## What it does

```
 recording.mp3
      │
      ▼
┌─────────────────┐   Sarvam Saaras STT (codemix-aware)
│  Transcription  │   → timestamped, speaker-labelled segments
└────────┬────────┘
         ▼
┌─────────────────┐   LangChain + GPT-4o / Gemini, schema-enforced
│   Extraction    │   → decisions, action items, owners, due dates
└────────┬────────┘
         ▼
┌─────────────────┐   WhatsApp (Twilio) · Email (SMTP) · Markdown + JSON
│    Delivery     │
└─────────────────┘
```

The output is a validated `MeetingSummary` object, not a wall of prose:

| Field | What it holds |
| --- | --- |
| `overview` | 2–4 sentence recap |
| `key_points` | The substantive points raised |
| `decisions` | What was settled, the rationale, and who called it |
| `action_items` | Task, owner, due date, priority, context |
| `open_questions` | What was explicitly left unresolved |

---

## Quickstart

```bash
git clone https://github.com/gowtham2891/ai-meeting-summarizer.git
cd ai-meeting-summarizer
pip install -e ".[dev,ui]"

# Run the whole pipeline offline — no API keys required
meeting-summarizer demo
```

`demo` runs the built-in sample meeting through transcription, extraction and
delivery, and writes markdown + JSON to `output/`.

### Web UI

```bash
streamlit run app.py
```

Upload a recording, pick your providers in the sidebar, and get a tabbed view of
the summary, the raw transcript, delivery results, and downloadable exports.

### CLI

```bash
# Summarize a real recording, deliver to disk and the terminal
meeting-summarizer run standup.mp3 --channel file --channel console

# Go live: Sarvam for speech, GPT-4o for extraction, straight to WhatsApp
meeting-summarizer run sync.m4a \
    --transcriber sarvam \
    --extractor openai \
    --channel whatsapp \
    --channel email

# Inspect what's registered and which providers are selected
meeting-summarizer providers
meeting-summarizer check        # verify your API keys work
```

Exit codes: `0` success · `1` bad input or config · `2` pipeline failure ·
`3` summary produced but a delivery channel failed.

---

## Mock mode

Every stage ships a credential-free implementation, selected by default:

- **`mock` transcriber** — returns a fixed 13-segment sample meeting, or reads a
  `.json` / `.txt` sidecar sitting next to your audio file.
- **`mock` extractor** — rule-based extraction (decision markers, action-phrase
  patterns, priority hints). Deterministic, offline, and the quality floor the
  LLM extractor is expected to beat.

This is what makes the project runnable on a fresh clone, keeps CI honest, and
lets the 113-test suite assert on real pipeline behaviour instead of mocks-all-
the-way-down.

---

## Configuration

Copy `.env.example` to `.env` and fill in only what you need.

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRANSCRIBER` | `mock` or `sarvam` | `mock` |
| `EXTRACTOR` | `mock`, `openai` or `gemini` | `mock` |
| `SARVAM_API_KEY` | Sarvam AI key | — |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI credentials | `gpt-4o-mini` |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini credentials | `gemini-2.0-flash` |
| `SMTP_HOST` … `EMAIL_TO` | Email delivery | — |
| `TWILIO_ACCOUNT_SID` … `WHATSAPP_TO` | WhatsApp delivery | — |
| `CHUNK_SECONDS` | Audio chunk length before upload | `600` |
| `MAX_RETRIES` | Retries on transient provider errors | `3` |
| `OUTPUT_DIR` | Where the file channel writes | `output` |

Install the provider extra you actually use:

```bash
pip install -e ".[openai]"   # or ".[gemini]"
```

---

## Design notes

**Provider registries.** Transcribers, extractors and delivery channels each
register into a name → factory map. Adding a provider means writing one class and
one `register_*` call; the CLI, the UI and the pipeline pick it up automatically.
Unknown provider names fail at construction, before a long transcription burns
API budget.

**Delivery never loses a summary.** Channels return a `DeliveryResult` instead of
raising, so a misconfigured SMTP server can't destroy work that already succeeded.
The pipeline reports per-channel outcomes and exits `3` — summary intact.

**Chunking is optional, not assumed.** Recordings longer than `CHUNK_SECONDS` are
split with ffmpeg and timestamps are re-offset on reassembly. If ffmpeg isn't
installed, the file is sent whole rather than the tool refusing to run.

**Structured output with a fallback.** The LLM extractor first tries the model's
native structured-output mode; if the provider doesn't support it, it retries in
JSON mode and parses leniently (fenced blocks, surrounding prose, and assorted
date formats are all handled).

**Console encoding.** Windows terminals default to cp1252, which raises
`UnicodeEncodeError` on both box glyphs and non-Latin transcript text. The CLI
reconfigures stdout to UTF-8 and falls back to ASCII symbols when it can't — a
regression test pins this.

---

## Project structure

```
meeting_summarizer/
├── models.py              # Pydantic domain models + markdown/plaintext rendering
├── config.py              # Env-driven settings, one source of truth
├── audio.py               # Validation, ffprobe duration, ffmpeg chunking
├── pipeline.py            # Orchestration + progress callbacks
├── transcription/         # base registry · sarvam · mock
├── extraction/            # base registry · llm (LangChain) · mock · prompts
└── delivery/              # base registry · file · console · email · whatsapp
app.py                     # Streamlit UI
tests/                     # 113 tests, all offline
```

---

## Testing

```bash
pytest -v
pytest --cov=meeting_summarizer --cov-report=term-missing
```

The suite covers model validation and rendering, heuristic extraction, lenient
JSON parsing, Sarvam response shapes (diarized and flat), retry-then-fail
behaviour, WhatsApp message chunking, delivery-failure isolation, and every CLI
exit code. No network, no credentials.

---

## Roadmap

- [ ] Speaker diarization with real name resolution from calendar invites
- [ ] Incremental summaries for recurring meetings (what changed since last time)
- [ ] Slack and Notion delivery channels
- [ ] Live streaming transcription instead of post-hoc file upload

---

## Bring your own API keys

The demo runs in mock mode, but you don't have to take its word for the live
path. Open **Use your own API keys** in the sidebar, paste a Sarvam, OpenAI or Gemini key,
and press **Test connections**. Each provider is probed with the smallest
authenticated request it supports, and you get a specific verdict rather than a
generic failure:

| Verdict | What it means |
| --- | --- |
| Key works — 68 models available | Authenticated and ready |
| Key rejected: API key not valid | Wrong, revoked, or mistyped key |
| Rate limited — valid but throttled | The key is fine; the quota isn't |
| Quota exhausted / API not enabled | Actionable: says exactly what to fix |

Keys entered this way stay in your browser session. They are **never** written
to `os.environ` — a deployed Streamlit app serves every visitor from a single
process, so a key placed in the environment would leak into other visitors'
sessions. That property is pinned by a test that drives the real app and fails
the build if a key ever reaches the process environment.

### If a key is missing

Pick a live provider without its key and the app says so before it runs
anything, naming the variable, the provider that needs it, and where to get
one — then opens the key panel for you. The run is blocked rather than
failing part way through:

```
Add an API key to continue.
- YouTube Data API key - needed for the 'youtube' data source
  get one at console.cloud.google.com

Open "Use your own API keys" in the sidebar and paste it there,
or switch the provider back to `mock` to use the offline demo.
```

The CLI does the same and exits `1`:

```bash
$ meeting-summarizer check
```

From the terminal:

```bash
meeting-summarizer check
```

Exit code `0` if every configured credential works, `2` if any fails.

---

## Deploy your own

Ready for [Streamlit Community Cloud](https://share.streamlit.io): free, and it
redeploys on every push to `main`.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python **3.11**.
4. Set the custom subdomain to `ai-meeting-summarizerr` (this app's URL).
5. Deploy — the first build takes a couple of minutes.

No secrets are needed for the demo. To switch it to the live providers, open
**Settings → Secrets** in the Streamlit dashboard and paste:

```toml
TRANSCRIBER = "sarvam"
EXTRACTOR = "openai"
SARVAM_API_KEY = "your-sarvam-key"
OPENAI_API_KEY = "your-openai-key"
```

`app.py` copies those secrets into the environment before settings are resolved,
so the exact same configuration works locally through `.env` and in the cloud
through the dashboard — no code changes either way.

> **ffmpeg note.** Recordings longer than `CHUNK_SECONDS` are split with ffmpeg.
> Streamlit Cloud images don't ship it, so add a `packages.txt` containing
> `ffmpeg` if you need chunking in the cloud. It is left out by default because
> it adds noticeably to cold-start time, and the code sends the file unchunked
> when ffmpeg is absent rather than failing.

---

## License

MIT — see [LICENSE](LICENSE).

**Ganesh Gowtham Dupati** · [GitHub](https://github.com/gowtham2891) · gowthamdupati28@gmail.com
