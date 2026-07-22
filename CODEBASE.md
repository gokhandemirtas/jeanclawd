# Codebase architecture

This document describes the current application. The project is pre-production and does
not preserve compatibility with the removed mem0, daily conversation, journal, adaptive
persona, personalizer, or reflection subsystems.

## Runtime entry points

- `main.py` runs the interactive CLI, optional voice input/output, Telegram, and optional
  HTTP server.
- `api/server.py` exposes the aiohttp REST and WebSocket API, including resumable sessions.
- `mcp_server.py` exposes selected capabilities through MCP.
- `commands.py` owns the CLI slash-command surface.

`db/init.py` must initialise Beanie before code accesses a document model. Runtime persona
and voice choices are loaded from `UserSettings` through `db/preferences.py`.

## Request flow

The CLI request path is:

```text
user input
  -> resolve_query()                 follow-up/topic rewrite
  -> resolve_action_query()          resolve "my favourite X" from MemoryFact
  -> consume_pending_intent()        honour a skill clarification lock
  -> generalist_route()              history-aware primary routing
  -> route()                         dispatch to a registered skill
  -> handle_conversation()           optional persona-aware response wrapping
  -> record_interaction()            persist evidence and deferred memory work
  -> optional TTS
```

`agents.intent_classifier.run_intent_classifier()` is the shared API and Telegram entry
path. Planner calls are handled explicitly because `PLANNER` is not a registered skill.

`skills/registry.py` registers the domain skills. CHAT is handled directly by
`skills/conversation/skill.py`, which owns the in-process history, rolling summary, topic
tracking, and memory-context injection.

## Memory

The canonical pipeline is:

```text
Interaction -> MemoryWorkItem -> MemoryFact -> Qdrant projection
```

### Interaction

Every completed turn is stored immediately as an immutable `Interaction`. The same write
path creates a durable `MemoryWorkItem`, so deferred extraction does not depend on an
in-memory timer or queue.

### Deferred extraction

`utils/memory_service.py` owns the worker. Once the application has been idle for ten
minutes, it coalesces pending work and uses the local generalist model with a Pydantic
schema to extract only durable facts explicitly stated by the user. A regex pre-filter
avoids an LLM call for turns with no personal signal.

Extraction excludes inferred sensitive traits, politics, religion, health, temporary
plans, world facts, and assistant-only statements. Each stored fact includes confidence,
source interaction IDs, first-seen and last-confirmed timestamps, and lifecycle status.
A new value for an existing key supersedes the prior active value.

`utils/inference_gate.py` serialises foreground local-model work and deferred extraction.
Background work yields whenever a foreground request is active or waiting.

### Recall

MongoDB `MemoryFact` documents are authoritative. `utils/memory_index.py` projects facts
into Qdrant collection `langbox_memory_facts` for semantic recall. Qdrant is disposable:
inactive facts are filtered, and the projection can be rebuilt from MongoDB with
`rebuild_memory_index()` after Qdrant loss or an embedding-dimension change.

CHAT receives two conservative views:

- a compact profile built from recent active facts;
- semantically relevant facts for sufficiently substantive queries.

Weather and greeting generation read canonical facts directly. The action resolver uses
semantic recall for commands such as “play my favourite band.”

CLI memory controls:

- `/memories [query]` lists or searches active facts;
- `/forget <id-prefix>` marks one fact forgotten and updates its index projection;
- `/apocalypse` removes interactions, work items, facts, and the Qdrant projection while
  preserving explicit settings.

## Conversation state

`skills/conversation/skill.py` maintains a `deque` of recent `HumanMessage` and
`AIMessage` objects. Before full history is evicted, older messages are compressed into a
rolling summary. This state exists for follow-ups inside the running process; persistent
cross-session knowledge comes only from active `MemoryFact` documents.

### Resumable sessions

`Session` documents provide durable conversation-thread identity. Each session has a UUID,
optional or automatically generated title, persisted rolling summary, and timestamps. `Interaction.session_id` links
every completed turn to its session. Resuming a UUID restores the saved summary and the
most recent interactions into the in-process conversation context. Learned `MemoryFact`
records remain user-wide rather than session-scoped.

The current named topic is extracted after responses and used to resolve short pronoun or
demonstrative follow-ups. A pending-intent lock routes answers to skill clarification
questions back to the requesting skill.

## Data model

Beanie registers these documents:

| Document | Purpose |
|---|---|
| `Interaction` | Immutable completed user/assistant turn |
| `Session` | Resumable conversation thread and persisted summary |
| `MemoryWorkItem` | Durable deferred-extraction outbox item |
| `MemoryFact` | Canonical learned fact with provenance and lifecycle |
| `UserSettings` | Explicit persona, voice, Spotify-device, and news-length settings |
| `Weather` | Cached forecasts |
| `Newsfeed` | Cached RSS content |
| `Plans` | Planner outputs |
| `Note` | Categorised notes |
| `Reminders` | Reminders and timers |
| `ServiceCredentials` | Per-service tokens and credentials |
| `Credentials` | Legacy Hue credential shape |
| `HueConfiguration` | Cached Hue lights and groups |

Learned user knowledge and explicit application settings are deliberately separate.

## Personas and settings

`agents/persona.py` defines six static personas: `jean`, `jerk`, `beth`, `kimi`, `borg`,
and `fred`. `UserSettings` persists the selected persona and voice plus news length and an
optional Spotify device. `db/preferences.py` is the only settings persistence layer.

There is no adaptive persona or separate user-persona profile. CHAT builds user context
directly from active memory facts.

## API

`api/server.py` registers 22 routes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/query` | Synchronous routed query |
| GET | `/query/ws` | Streaming query WebSocket |
| GET | `/tts/voices` | Available voices |
| GET | `/personas` | Available personas |
| GET/POST | `/persona` | Read or set active persona |
| POST | `/persona/voice` | Set active voice |
| GET | `/voice/ws` | Voice round-trip WebSocket |
| POST | `/voice` | Create an asynchronous voice job |
| GET/DELETE | `/voice/{job_id}` | Read or cancel a voice job |
| POST | `/vision` | Describe an image and inject context |
| GET | `/plans` | Saved plans |
| GET | `/notes` | Saved notes |
| GET | `/reminders` | Reminders |
| GET | `/sessions` | List resumable sessions |
| POST | `/sessions` | Create a session |
| POST | `/sessions/{session_id}/resume` | Resume a session |
| POST | `/sessions/{session_id}/close` | Close a session |
| GET | `/openapi.json` | OpenAPI specification |
| GET | `/docs` | Swagger UI |

## CLI commands

`commands.py` registers 15 commands:

```text
/clear  /history  /persona  /voice  /news-length
/model  /connect  /memories /forget /apocalypse
/ctx    /note     /planner  /look   /help
```

## Inference and providers

`providers/` supports LM Studio, llama.cpp, OpenAI-compatible endpoints, and OpenAI.
`utils/llm_structured_output.py` validates structured model responses against Pydantic
models. `agents/agent_factory.py` provides the message-based streaming interface used by
conversation flows.

Important model settings are `MODEL_GENERALIST`, `MODEL_INTENT`, `MODEL_EMBEDDING`,
`MODEL_VLM`, `MODEL_HOST`, and `MODEL_CTX`. Infrastructure uses `MONGODB_*` and
`QDRANT_HOST`/`QDRANT_PORT`.

## Validation

Use the project environment and keep the lockfile current:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q .
```

The repository currently has only a small ad-hoc classifier test; new memory behaviour
should be covered with isolated tests around extraction, supersession, forgetting, worker
retries, and Qdrant-unavailable fallbacks.
