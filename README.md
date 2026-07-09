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
- **Voice in / out** — wake word (`hey jarvis`), Whisper STT (mlx), streaming TTS
  (pocket-tts / LM Studio / OpenAI-compatible) with 8 voice IDs.
- **Vision** — VLM-backed image description (`POST /vision`, `/look` command).
- **7 personas** — switchable identities with distinct tones and voices; `adaptive` evolves
  from your journal.
- **Persistent memory** — mem0 + Qdrant for long-term facts; in-session history with
  automatic compression; topic tracking for follow-up resolution.
- **API server** — 22 routes (aiohttp), OpenAPI 3.0 spec, Swagger UI at `/docs`.
- **Multi-interface** — CLI REPL, REST/WebSocket API, Telegram bot.

## Requirements

- **Python** `>=3.11.14`
- **[uv](https://github.com/astral-sh/uv)** for dependency management (never `pip` —
  keeps `pyproject.toml` / `uv.lock` in sync)
- **MongoDB** (port 27017, db `langbox`)
- **Qdrant** (port 6333) for mem0
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
| `MODEL_INTENT` | Lean model for stateless classification | `meta-llama-3.1-8b-instruct` |
| `MODEL_EMBEDDING` | Embedding model for mem0 / RAG | `text-embedding-granite-embedding-30m-english` |
| `MODEL_VLM` | Vision-language model for `/look` | `qwen/qwen3-vl-4b` |
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
| Embedding | `text-embedding-granite-embedding-30m-english` | 30M params, 384-dim, fast |
| VLM | `qwen/qwen3-vl-4b` | Small VL model that fits comfortably in RAM |

Download via LM Studio UI, or from the CLI:
```bash
lms pull lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit
lms pull lmstudio-community/meta-llama-3.1-8b-instruct
lms pull ibm-granite/granite-embedding-30m-english-GGUF
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
```

CLI flags: `--debug`, `--speak`, `--emote`, `--server`, `--listen`, `--track_camera`,
`--keepalive`.

## Slash commands

17 in-REPL commands — see [CODEBASE.md](CODEBASE.md) for the full table. Highlights:

- `/persona [id] [voice]` — switch identity
- `/model [name]` — fuzzy-pick a loaded LM Studio model
- `/memories [q]` — list or search long-term memory
- `/planner <task>` — multi-step tool use
- `/analyze` — full persona build + journal compaction
- `/compact-memory` — LLM dedupe + merge mem0
- `/help` — list all commands

## Architecture

```
User input
  → resolve_query()            # rewrite follow-up using current topic
  → resolve_action_query()     # substitute "my favourite X" from mem0  [CLI only]
  → consume_pending_intent()   # locked routing from a prior clarifying "?"
  → generalist_route()         # primary: generalist model + 3-turn history + confidence
  → confidence < 0.75 && != CHAT  →  CHAT
  → route(intent, query)       # dispatch to skill.handle()
  → skill: secondary sub-intent classification → sub-handler
  → needs_wrapping? handle_conversation() : append_to_history()
  → streaming via on_token callback
  → journal write + mem0 batch (120s inactivity)
  → optional TTS
```

Two-tier routing is a settled design decision — tool-use was explicitly rejected because
models don't call tools consistently in practice. See
[`.claude/memory/project_architecture.md`](.claude/memory/project_architecture.md) for
rationale.

For the full architecture reference (every skill, every env var, every API route, the
divergent CLI/API pipelines, known issues, and the document inventory) see
[CODEBASE.md](CODEBASE.md).

## Project layout

```
pydantic-aida/
├── main.py                  # CLI entry + REPL loop
├── commands.py              # 17 slash commands
├── pyproject.toml           # package "jeanclawd", uv-managed deps
├── agents/                  # routing, intent classification, personas
├── providers/               # lmstudio, llamacpp, openai_compatible, openai
├── skills/                  # 12 skills + planner + personalizer + journal + telegram
├── db/                      # Beanie ODM (12 models) + MongoDB
├── api/                     # aiohttp server (22 routes) + OpenAPI
├── stt/  tts/               # speech I/O providers
├── utils/                   # memory_client, llm_structured_output, embedder, ...
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

## Known issues

- **Two divergent pipelines** — CLI runs `resolve_action_query()` (mem0 preference
  substitution); API does not. Mobile-app users get no "my favourite X" resolution.
- **`PLANNER` intent unreachable via routing** — declared in `IntentLiteral` but no skill
  registered; natural-language plan requests hit the fallback string.
- **Broken ReAct import** — `skills/conversation/reasoning_engine.py:118` imports a
  nonexistent `agents.router` module. Rarely surfaces because the streaming path bypasses
  reasoning.
- **`/save` imports `langchain_core`** which is not a declared dependency. May fail if not
  transitively installed.

See [CODEBASE.md § Known Issues](CODEBASE.md#known-issues--discrepancies) for the full
list and reproduction details.
