# Codebase architecture

This document describes the current application. The project is a local-first personal
assistant with CLI, REST/WebSocket, and Telegram interfaces.

## Runtime flow

Requests are resolved, routed to a skill, answered, and recorded as immutable `Interaction`
documents. `skills/conversation/skill.py` owns short-term in-process history and injects the
LangMem profile when useful. Sessions are separate MongoDB documents and remain independent
from user profile memory.

## LangMem memory

`utils/langmem_service.py` is the only user-memory integration.

1. A completed turn is written to the `Interaction` collection.
2. A background LangMem manager receives the user and assistant messages.
3. LangMem updates one `UserMemory` Pydantic profile per user.
4. The profile is persisted through `MongoDBStore` in the `LangMem` collection.

The manager uses `MODEL_GENERALIST` through the configured OpenAI-compatible endpoint. Its
instructions explicitly separate User and Assistant perspectives, ignore conversational noise,
and prefer a compact profile over independent memory shards. The MongoDB store uses the local
`MODEL_EMBEDDING` model with `EMBEDDING_DIM` for semantic retrieval; Qdrant and a separate
vector database are not used.

The profile contains optional identity fields and compact lists for likes, dislikes, interests,
relationships, and durable context. `/memories` lists profile fields, `/forget-memory <field>`
clears one field, and `/apocalypse` clears interactions and the LangMem profile.

## MongoDB documents

Beanie registers these application documents:

| Document | Purpose |
|---|---|
| `Interaction` | Immutable completed user/assistant turn |
| `Session` | Resumable conversation thread and summary |
| `UserSettings` | Explicit persona, voice, Spotify-device, and news-length settings |
| `Weather` | Cached forecasts |
| `Newsfeed` | Cached RSS content |
| `Plans` | Planner outputs |
| `Note` | Categorised notes |
| `Reminders` | Reminders and timers |
| `ServiceCredentials` | Per-service tokens and credentials |
| `Credentials` | Hue credential shape |
| `HueConfiguration` | Cached Hue lights and groups |

LangMem owns its separate MongoDB collection and is not registered as a Beanie document.

## Connectors

`connectors/` contains the `Connector` base class, automatic module discovery, and the
built-in Spotify and Philips Hue connection flows. Adding a concrete connector module makes
it available to `/connect` and its autocomplete picker. `utils/auth/` retains compatibility
imports for skills that use the older `AuthProvider` names.

## Providers and configuration

`providers/` supports LM Studio, llama.cpp, OpenAI-compatible endpoints, and OpenAI. The
generalist model is configured with `MODEL_GENERALIST`; `MODEL_INTENT` is used for lean
classification and `MODEL_VLM` for vision. LangMem uses `MODEL_GENERALIST`, `MODEL_HOST`,
`INFERENCE_PROVIDER`, and `OPENAI_API_KEY` when needed by the local endpoint.

## Validation

Use the project environment and keep the lockfile current:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q .
```
