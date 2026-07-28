# pydantic-aida

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
- **6 personas** — switchable identities with distinct tones and voices.
- **Persistent memory** — completed turns are stored as `Interaction` documents and sent to
  LangMem, which maintains one compact structured user profile in MongoDB using the generalist model.
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
- **MongoDB** (port 27017, db `langbox`)
- **LM Studio** (or another OpenAI-compatible LLM server on `:1234`) for inference

## Installation

```bash
# Clone, then sync dependencies via uv
uv sync

# Copy and edit env
cp .env.template .env
$EDITOR .env
```

## Configuration

All runtime config lives in `.env`. See [`.env.template`](.env.template) for the full key
list. Key variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `INFERENCE_PROVIDER` | LLM backend | `lmstudio` |
| `MODEL_GENERALIST` | Large model: chat, wrapping, routing, reasoning | `qwen3-30b-a3b-instruct-2507-mlx` |
| `MODEL_RESEARCH` | Dedicated model for deep-search planning and synthesis | `Qwen3-14B-4bit` |
| `MODEL_INTENT` | Lean model for stateless classification | `meta-llama-3.1-8b-instruct` |
| `MODEL_VLM` | Vision-language model for `/vision` | `qwen/qwen3-vl-4b` |
| `MODEL_HOST` | LLM endpoint | `http://localhost:1234/v1` |
| `STT_PROVIDER` | Speech-to-text backend | `mlx` (Apple Silicon) or `whisper` |
| `TTS_PROVIDER` | Text-to-speech backend | `local` / `lmstudio` / `remote` |
| `MONGODB_*` | MongoDB connection | `localhost:27017`, db `langbox` |
| `*_API_KEY` | External services | Tavily, OpenRouteService, Spotify, Spoonacular, Hue |

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

# Supervised auto-restart
uv run python main.py --keepalive

# Resume a previous session
uv run python main.py --session SESSION_UUID
```

Last active application session:

```bash
uv run python main.py --session e4f97131-0c00-439b-b25b-41ff8e88774c
```

CLI flags: `--debug`, `--speak`, `--emote`, `--server`, `--listen`, `--track_camera`,
`--telegram`, `--keepalive`, `--session SESSION_UUID`.

Telegram requires `TELEGRAM_BOT_TOKEN` in `.env`; start it with:

```bash
uv run python main.py --telegram
```

## Slash commands

16 in-REPL commands — see [CODEBASE.md](CODEBASE.md) for the full table. Highlights:

- `/persona [id] [voice]` — switch identity
- `/model [name]` — fuzzy-pick a loaded LM Studio model
- `/memories [q]` — list or search long-term memory
- `/planner <task>` — multi-step tool use
- `/forget-memory <field>` — clear one LangMem profile field
- `/help` — list all commands

## Architecture

The application has three interfaces, but one canonical memory model:

```
CLI / REST / Telegram / MCP
  → query resolution and intent routing
  → registered skill or CHAT handler
  → response (optionally streamed or spoken)
  → Interaction + background LangMem profile update

LangMem uses the configured `MODEL_GENERALIST` through the local OpenAI-compatible endpoint.
It stores one structured profile per user in the MongoDB `LangMem` collection. The MongoDB
store uses the configured local embedding model and MongoDB vector index for semantic retrieval.
```

The interfaces follow the same conceptual stages, while API, Telegram, and MCP use the
shared intent-classifier entry point and the CLI keeps its own streaming path. The CLI
also resolves memory-backed phrases such as “my favourite band” before routing. In-process
conversation history is only for short-term follow-ups; cross-session knowledge comes from
the LangMem profile.

Sessions are separate from both settings and learned memory. A session stores its UUID,
title, rolling summary, and recent `Interaction` records. If no title is supplied, the
first completed turn is given a strict two- or three-word title by the small intent model.
Resuming a session restores that conversational context without making its facts
session-specific.

Two-tier routing is a settled design decision — tool-use was explicitly rejected because
models don't call tools consistently in practice. See
[`.claude/memory/project_architecture.md`](.claude/memory/project_architecture.md) for
rationale.

For the full architecture reference (every skill, every env var, every API route, the
interface pipelines, known issues, and the document inventory) see
[CODEBASE.md](CODEBASE.md).

## Project layout

```
pydantic-aida/
├── main.py                  # CLI entry + REPL loop
├── commands.py              # 16 slash commands
├── pyproject.toml           # package "jeanclawd", uv-managed deps
├── agents/                  # routing, intent classification, personas
├── providers/               # lmstudio, llamacpp, openai_compatible, openai
├── skills/                  # 12 skills + planner + telegram
├── db/                      # Beanie ODM + MongoDB
├── api/                     # aiohttp server (22 routes) + OpenAPI
├── stt/  tts/               # speech I/O providers
├── utils/                   # langmem_service, inference_gate, ...
├── storage/  memory/        # local state + feedback notes
├── fixtures/                # tickers.json + test fixtures
└── logs/                    # app.log + supervisord logs
```

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
variables in `.env`. The project MCP server exposes this through `mongo_inventory` and
`mongo_find`; neither tool can insert, update, or delete data.

For a general MongoDB connection available directly to Codex, MongoDB also publishes an
official Codex Plugin bundling its MCP server and MongoDB agent skills. Add the marketplace
with `codex plugin marketplace add mongodb/agent-skills`, start a new Codex session, then
install the MongoDB plugin from `/plugins`. Configure it in read-only mode before use.

## Known issues

- **Planner is explicit-only** — `/planner`, `@planner`, and the planner API/MCP entry
  points work, but natural-language planner routing is not a registered skill.
- **Reasoning fallback is rarely used** — `skills/conversation/reasoning_engine.py` has a
  legacy import path that should be cleaned up if non-streaming reasoning is enabled.

See [CODEBASE.md § Known Issues](CODEBASE.md#known-issues--discrepancies) for the full
list and reproduction details.
