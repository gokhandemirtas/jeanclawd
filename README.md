# jeanclawd

A personal AI assistant framework built with Python, Pydantic, and local LLMs. Modular
skill-based architecture with persistent memory, multi-modal I/O (text + voice + vision),
a REPL CLI, and a REST/WebSocket API served to a mobile app over Tailscale.

Package name: `jeanclawd` (see `pyproject.toml`).

## Features

- **Local-first LLM** — runs entirely on-device via LM Studio (or any OpenAI-compatible
  endpoint), no cloud round-trips for chat, routing, or memory.
- **Two-tier intent routing** — primary classifier + per-skill secondary sub-intent
  classification with structured outputs. Not tool-use / function-calling.
- **Skill registry** — 12 skills: weather, finance, home control, information, reminders,
  newsfeed, transportation, notes, search, Spotify, nutrition, chat.
- **Voice in / out** — wake word (`hey jarvis`), Whisper STT (mlx), local Supertonic TTS
  (plus LM Studio / OpenAI-compatible providers) with six friendly voice IDs and
  persona-aware expression tags.
- **Vision** — VLM-backed image description (`POST /vision`, `/vision` command).
- **10 personas** — switchable identities with distinct tones and voices, including an alien observer,
  a systems-thinking mentor, an improvisational storyteller, and Mr Meowsky the cat.
- **Persistent memory** — completed turns are stored as `Interaction` documents and sent to
  LangMem, which maintains one compact structured user profile in MongoDB using the generalist model.
- **Research archive RAG** — completed deep-research reports are chunked into MongoDB-backed
  research documents and can be searched across reports for follow-up and comparison questions.
- **Explicit settings** — persona, voice, Spotify device, and news length live in a small
  `UserSettings` document and are never mixed with learned user facts.
- **Resumable sessions** — every turn belongs to a UUID-backed session that can be listed
  and resumed from the CLI or API.
- **API server** — 22 routes (aiohttp), OpenAPI 3.0 spec, Swagger UI at `/docs`.
- **Multi-interface** — CLI REPL, REST/WebSocket API, Telegram bot.

## Requirements

- **Python** `>=3.11.14`
- **[uv](https://github.com/astral-sh/uv)** for dependency management (never `pip` —
  keeps `pyproject.toml` / `uv.lock` in sync)
- **Docker Desktop** (or another Docker Engine with Compose) for the bundled MongoDB
- **MongoDB** (port 27017, db `langbox`)
- **LM Studio** (or another OpenAI-compatible LLM server on `:1234`) for inference

The supported and tested environment is macOS on Apple Silicon. Windows is not supported:
the dependency set includes MLX-based packages (`mlx-audio` and `mlx-whisper`), and MLX
does not publish compatible Windows wheels. `uv sync` therefore cannot install the complete
environment on Windows. Linux may work for non-MLX paths, but is not currently a tested
target.

## Installation

```bash
# Clone, start MongoDB, then sync dependencies via uv
docker compose -f db/docker-compose.yml up -d
uv sync

# Copy and edit env
cp .env.template .env
$EDITOR .env
```

The Compose file uses `admin` / `admin` and the `langbox` database by default. If you
change those values, use the same values in `.env` before starting the application.
Verify MongoDB is running with:

```bash
docker compose -f db/docker-compose.yml ps
```

## Configuration

All runtime config lives in `.env`. See [`.env.template`](.env.template) for the full key
list. Key variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `INFERENCE_PROVIDER` | LLM backend | `lmstudio` |
| `MODEL_GENERALIST` | Large model: chat, wrapping, routing, reasoning | `qwen3-30b-a3b-instruct-2507-mlx` |
| `MODEL_RESEARCH` | Dedicated model for deep research and synthesis | `Qwen3-14B-4bit` |
| `MODEL_INTENT` | Lean model for stateless classification | `meta-llama-3.1-8b-instruct` |
| `MODEL_VLM` | Vision-language model for `/vision` | `qwen/qwen3-vl-4b` |
| `MODEL_PLANNING` | Optional dedicated model for multi-step planning | unset (uses generalist) |
| `MEMORY_MAX_ITEMS` | Maximum relevant memories injected into a chat turn | `2` |
| `MEMORY_MAX_CHARS` | Maximum injected memory text size | `500` |
| `MEMORY_COOLDOWN_TURNS` | Turns before a recently used memory can be reused | `4` |
| `MAX_HISTORY_MESSAGES` | Recent in-process messages retained for immediate follow-ups | `12` |
| `CHAT_PROMPT_MAX_TOKENS` | Approximate hard budget for a complete CHAT prompt | `4000` |
| `CHAT_CONTEXT_MAX_TOKENS` | Maximum tokens allocated to summary and recent turns | `1400` |
| `SUMMARY_MAX_CHARS` | Maximum rolling-summary size | `1200` |
| `HISTORY_*_MAX_CHARS` | Maximum size of compact recent user/assistant entries | `400` / `600` |
| `MODEL_HOST` | LLM endpoint | `http://localhost:1234/v1` |
| `STT_PROVIDER` | Speech-to-text backend | `mlx` (Apple Silicon) or `whisper` |
| `TTS_PROVIDER` | Text-to-speech backend | `local` / `lmstudio` / `remote` |
| `MONGODB_*` | MongoDB connection | `localhost:27017`, db `langbox` |
| `*_API_KEY` | External service credentials | Required when using the skill that owns the service |

The external-service credentials are not optional for their corresponding skills. The
application can start without them, but a request that activates a service skill will need
that service's API key or account credentials configured in `.env`.

### Recommended models (LM Studio, 64 GB unified RAM)

| Role | Model | Why |
|------|-------|-----|
| Generalist | `qwen3-30b-a3b-instruct-2507-mlx` (MLX 4-bit) | 30B total / 3B active per token; ~80–100 tok/s on M1 Max/Ultra; strong rule-following for the 13-intent classifier; instruct-tuned (not thinking) for low latency on voice turns |
| Intent | `meta-llama-3.1-8b-instruct` | 8B is the sweet spot for structured JSON outputs on routing / topic extraction / query rewrite; pairs cleanly with the 30B generalist |
| VLM | `qwen/qwen3-vl-4b` | Small VL model that fits comfortably in RAM |

Download via LM Studio UI, or from the CLI:
```bash
lms pull lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit
lms pull lmstudio-community/meta-llama-3.1-8b-instruct
lms pull lmstudio-community/Qwen3-VL-4B-Instruct-MLX
```

## Running

```bash
# CLI REPL (most common)
uv run python main.py

# With voice I/O
uv run python main.py --speak --listen

# With API server on :8000
uv run python main.py --server

# API server standalone
uv run python -m api.server

# Auto-restart on crash
uv run python main.py --keepalive

# Resume a previous session
uv run python main.py --session SESSION_UUID
```

## Live monitor

`monitor.py` is a separate Textual dashboard for watching the running application. Start
the assistant first, then run the monitor in another terminal:

```bash
uv run python monitor.py
```

It tails `logs/app.log` as newline-delimited JSON and shows the application log levels
`DEBUG`, `INFO`, `WARN`, and `ERROR`. The **Events** tab displays structured log events
with their extra fields. The **LLM PERF** filter highlights
`llm_generation_completed` events, including prompt/completion token counts, server and
end-to-end throughput, and server/client time-to-first-token measurements.

The monitor also displays live process resources: CPU usage, system memory, swap when
available, NVIDIA VRAM when NVML is available, and the current LLM context-window usage.
The context meter is read from `logs/context.json`, which the application updates while
preparing LLM requests. It is separate from `app.log` and is not a conversational or
database log.

To inspect a different JSON-lines log file:

```bash
uv run python monitor.py --log path/to/another.log
```

The monitor starts at the end of an existing file and displays new entries as they arrive.
It does not inspect MongoDB, external-service logs, or historical entries before startup
unless they are appended after the monitor begins. Use `a` to toggle auto-scroll, `c` to
clear the dashboard, `y` to copy the selected or filtered results, and `q` to quit.

Last active application session:

```bash
uv run python main.py --session e4f97131-0c00-439b-b25b-41ff8e88774c
```

### LM Studio performance

Every native LM Studio generation writes a structured `llm_generation_completed`
event to `logs/app.log`. The event includes LM Studio's tokenizer-backed
`prompt_tokens`, `completion_tokens`, `server_tokens_per_second`, and
`server_ttft_ms`, plus client-observed `client_ttft_ms`, `wall_time_ms`, and
`measured_tokens_per_second`.

Watch the measurements while JeanClawd is running:

```bash
tail -f logs/app.log | jq 'select(.event == "llm_generation_completed")'
```

For comparing models on the same machine, use `server_tokens_per_second` and
`server_ttft_ms`. For perceived application performance, use
`client_ttft_ms` and `end_to_end_tokens_per_second`. Run several requests with
the same prompt and generation limit, discard the first request if it includes
model loading, and compare medians rather than a single result.

CLI flags: `--speak`, `--emote`, `--server`, `--listen`, `--track_camera`,
`--telegram`, `--keepalive`, `--session SESSION_UUID`.

Telegram requires `TELEGRAM_BOT_TOKEN` in `.env`; start it with:

```bash
uv run python main.py --telegram
```

## Slash commands

15 in-REPL commands — see [CODEBASE.md](CODEBASE.md) for the full table. Highlights:

- `/persona [id] [voice]` — switch identity
- `/change-model` — choose an available model by purpose
- `/memories [q]` — list or search long-term memory
- `/forget-memory <field>` — clear one LangMem profile field
- `/help` — list all commands

## Architecture

The application has three interfaces, but one canonical memory model:

```
CLI / REST / Telegram
  → query resolution and intent routing
  → registered skill or CHAT handler
  → response (optionally streamed or spoken)
  → Interaction + background LangMem profile update

LangMem uses the configured `MODEL_GENERALIST` through the local OpenAI-compatible endpoint.
It stores one structured profile per user in the MongoDB `LangMem` collection. The MongoDB
store uses the configured local embedding model and MongoDB vector index for semantic retrieval.

Completed deep-research reports are indexed separately in `ResearchDocuments` and
`ResearchChunks`. The archive currently uses BM25 retrieval so it works with the local setup
without requiring an embedding service. Questions mentioning prior research or comparing
research reports retrieve passages from the archive and preserve the reports' source markers.
```

The interfaces follow the same conceptual stages, while API and Telegram use the shared
intent-classifier entry point and the CLI keeps its own streaming path. The CLI
also resolves memory-backed phrases such as “my favourite band” before routing. In-process
conversation history is only for short-term follow-ups; cross-session knowledge comes from
the LangMem profile.

Sessions are separate from both settings and learned memory. A session stores its UUID,
title, rolling summary, and recent `Interaction` records. If no title is supplied, the
first completed turn is given a strict two- or three-word title by the small intent model.
Resuming a session restores that conversational context without making its facts
session-specific.

Two-tier routing is a settled design decision: the application routes requests through
intent classification and skill-specific handlers rather than relying on model tool use.

For the full architecture reference (every skill, every env var, every API route, the
interface pipelines, known issues, and the document inventory) see
[CODEBASE.md](CODEBASE.md).

## Project layout

```
jeanclawd/
├── main.py                  # CLI entry + REPL loop
├── commands.py              # slash command implementations
├── connectors/               # discoverable external-service connectors
├── pyproject.toml           # package "jeanclawd", uv-managed deps
├── agents/                  # routing, intent classification, personas
├── providers/               # lmstudio, llamacpp, openai_compatible, openai
├── skills/                  # 12 skills + planner + telegram
├── db/                      # Beanie ODM + MongoDB
├── api/                     # aiohttp server (22 routes) + OpenAPI
├── stt/  tts/               # speech I/O providers
├── utils/                   # langmem_service, inference_gate, ...
├── storage/  memory/        # runtime state directories, created automatically if absent
├── fixtures/                # tickers.json + test fixtures
└── logs/                    # runtime application logs
```

### Adding a connector

Create a module under `connectors/` with a concrete `Connector` subclass from
`connectors.base`. Set `service_id`, `display_name`, and optionally `description`, then
implement `is_connected()` and `connect()`. The registry discovers it automatically, so it
appears in `/connect` autocomplete without editing `commands.py`.

## Development

```bash
uv sync                                # install
uv add <package>                       # add a dep (never pip install)
uv run ruff check .                    # lint
uv run ruff format .                   # format (tab indent by default)
```

Code style: **ruff with tab indentation** (project default). Lint/format on edited files
before committing.

## MongoDB inspection

`utils/mongodb_access.py` provides bounded, read-only access using the `MONGODB_*`
variables in `.env`. `utils/mongodb_access.py` provides bounded, read-only diagnostics;
it cannot insert, update, or delete data.

## Known issues

- **Reasoning fallback is rarely used** — `skills/conversation/reasoning_engine.py` has a
  legacy import path that should be cleaned up if non-streaming reasoning is enabled.

See [CODEBASE.md § Known Issues](CODEBASE.md#known-issues--discrepancies) for the full
list and reproduction details.
