# pydantic-aida — Codebase Reference

> Verified against the actual source on 2026-06-24. This document describes the code as it
> really is, not as earlier docs (`CLAUDE.md`, `ANALYSIS.md`) describe it. Known divergences
> between docs and code are called out explicitly.

## Identity

- **Package name:** `jeanclawd` (`pyproject.toml`), though the directory is `pydantic-aida`.
- **What it is:** A personal AI assistant framework built with Python, Pydantic, and local
  LLMs. Modular skill-based architecture with persistent memory, multi-modal I/O (text +
  voice + vision), a REPL CLI, and a REST/WebSocket API served to a mobile app over Tailscale.
- **Python:** `>=3.11.14`. Dependency manager: **uv** (never pip — see
  `.claude/memory/feedback_use_uv.md`). Formatting: **ruff** with tab indentation.

## Routing Architecture — Primary + Secondary Intent Classification

> **Design decision (settled):** the assistant uses **primary + secondary intent
> classification with internal routing**, NOT tool-use / function-calling. Tool-use was
> evaluated and explicitly rejected because models do not call tools consistently in
> practice. Do not re-propose tool-use. See `.claude/memory/project_architecture.md`.

The pipeline is two-stage:

1. **Primary routing** — one LLM call classifies the query into a top-level intent, with
   recent conversation history so follow-ups route correctly.
2. **Secondary routing** — the chosen skill classifies its own sub-intent internally
   (another structured-output LLM call) and dispatches to a sub-handler.

This keeps routing deterministic and per-skill, avoids a second generic "wrapping" call for
skills that can produce final text directly, and lets each skill own its own sub-domain
logic.

### Primary routing

| Component | File | Model | Sees history? | Returns |
|-----------|------|-------|---------------|---------|
| `generalist_route()` | `agents/generalist/router.py` | `MODEL_GENERALIST` (via `get_provider()`) | Yes — last 3 exchanges | `(intent, confidence)` |
| `_keyword_classify()` | `agents/intent_classifier/intent_classifier.py` | none (regex) | No | `IntentLiteral \| None` (SEARCH shortcut) |
| `classify()` | `agents/intent_classifier/intent_classifier.py` | `MODEL_INTENT` (lean, via `get_intent_provider()`) | No (only `current_topic` hint) | `IntentLiteral` |

**The actual primary path** (in both `main.py` and the API) is: keyword shortcut →
`generalist_route()` → confidence gate. The stateless `classify()` still exists and is
importable, but `main.py` no longer calls it directly; `run_intent_classifier()` (the API
entry point) keeps it available as a fallback path but also prefers `generalist_route()`.

Confidence gate (`main.py:199`, mirrored in `run_intent_classifier`):
```python
if confidence < 0.75 and intent != "CHAT":
    intent = "CHAT"
```

**13 top-level intents** (`IntentLiteral` in `agents/intent_classifier/intent_classifier.py`):
`HOME_CONTROL`, `WEATHER`, `FINANCE_STOCKS`, `TRANSPORTATION`, `REMINDER`, `NEWSFEED`,
`INFORMATION_QUERY`, `NOTES`, `SEARCH`, `SPOTIFY`, `PLANNER`, `NUTRITION`, `CHAT`.
(`NUTRITION` was added after the earlier docs were written — old docs say 12.)

### Secondary routing (inside each skill)

Each skill runs its own structured-output classifier before dispatching. Reference
implementations:

| Skill | File | Sub-intents | Classifier |
|-------|------|-------------|------------|
| Notes | `skills/notes/skill.py` + `prompts.py` | `CREATE \| LIST \| READ \| DELETE` | `_classify_sub_intent()` → `_NotesIntentResponse` |
| Reminder | `skills/reminder/skill.py` + `prompts.py` | `LIST \| TIMER \| REMINDER` (+ datetime/description extraction) | `_classify_intent()` → `ReminderIntentResponse` |
| Home Control | `skills/home_control/skill.py` | `ALL \| GROUP \| LIGHT` × `on: true\|false\|null (toggle)`; also status/list query detection | `_classify_intent()` → `HomeControlIntentResponse` |
| Weather | `skills/weather/skill.py` + `prompts.py` | `location` + `period` (`CURRENT\|TODAY\|TOMORROW\|DAY_AFTER`) | `_classify_intent()` → `WeatherIntentResponse` |
| Finance | `skills/finance/skill.py` + `prompts.py` | `ticker` + `dataType` (`CURRENT\|HISTORICAL`) + `period`; candidate-limited via `fixtures/tickers.json` | `_classify_intent()` → `FinanceIntentResponse` |

### Dispatch and wrapping

`agents/intent_router/intent_router.py:route()`:
- Looks up the skill in `SKILL_MAP` (falls back to `skills/fallback.py`).
- Calls `skill.handle(user_input, **kwargs)` (introspects signature for `on_token` /
  `voice_mode`).
- If `skill.needs_wrapping is True` → passes raw output through
  `skills/conversation/skill.py:handle_conversation()` to synthesise natural language and
  stores the exchange in `_history`.
- If `needs_wrapping is False` → stores the exchange directly via `append_to_history()`.

`Skill` dataclass (`skills/base.py`):
```python
@dataclass
class Skill:
    id: str
    description: str
    handle: Callable
    system_prompt: Optional[str] = None
    needs_wrapping: bool = True
    auth_provider: Any = None
```

### Two parallel pipeline implementations (inconsistency)

There are **two implementations of the same pipeline** that differ in one important step:

1. **CLI** — `main.py` inlines the pipeline (lines ~187–209):
   `resolve_query` → **`resolve_action_query`** → `consume_pending_intent` →
   `generalist_route` → `route`.
   *Only the CLI calls `resolve_action_query()`* (`agents/action_resolver/agent.py`), which
   substitutes personal preferences from mem0 ("play my favourite band" → "play Death").

2. **API** — `api/server.py` calls `run_intent_classifier()`
   (`agents/intent_classifier/__init__.py`):
   `consume_pending_intent` → `resolve_query` → `_keyword_classify` → `generalist_route` →
   `route` → `append_to_journal`.
   *The API does NOT call `resolve_action_query()`*, so personal-preference substitution
   does not happen for mobile-app users.

### Follow-up handling

- **`_current_topic`** (`skills/conversation/skill.py`) — a named entity extracted after
  each response. `resolve_query()` rewrites short/pronoun-heavy follow-ups into
  self-contained queries before classification. `_is_followup()` gates rewriting to avoid
  topic bleed on long queries.
- **`_pending_intent`** — when a skill's response ends with `?`, the next turn is locked to
  the same skill (skips classification). Set via `set_pending_intent()`, consumed via
  `consume_pending_intent()`. Also used by the persona-question system to route answers
  back to CHAT.

## Message Flow (CLI)

```
User input
  → resolve_query()            # rewrite follow-up using _current_topic
  → resolve_action_query()     # substitute "my favourite X" from mem0  [CLI only]
  → consume_pending_intent()   # locked routing from a prior clarifying "?"
  → generalist_route()         # primary: generalist model + 3-turn history + confidence
       (fallback: _keyword_classify shortcut for SEARCH)
  → confidence < 0.75 && != CHAT  →  CHAT
  → route(intent, query)       # dispatch to skill.handle()
  → skill: secondary sub-intent classification → sub-handler
  → needs_wrapping? handle_conversation() : append_to_history()
  → streaming via on_token callback
  → enqueue("journal", append_to_journal())   # DB write + mem0 batch (120s inactivity)
  → optional TTS
```

## Conversation Memory

**In-session** (`skills/conversation/skill.py`):
- `_history` — `deque(maxlen=20)` of `HumanMessage`/`AIMessage` pairs (10 exchanges).
- When full, oldest half compressed into `_rolling_summary` via `_compress_oldest()`
  before eviction.
- `reset_session()` clears all in-memory state (called after compaction).
- `get_recent_history(n)` returns `(user, assistant)` string pairs — used by the router
  and ReAct engine.

**Persistent** (`utils/memory_client.py` + Qdrant):
- mem0 extracts facts with the local LLM; vectors stored in Qdrant collection
  `langbox_memories`.
- mem0's `EmbedderFactory` and `LlmFactory` are monkey-patched to use the project's local
  `_LangboxLlm` (wraps `get_provider().respond()`) and `_LangboxEmbedder` (shared
  `all-MiniLM-L6-v2` via `utils/embedder.py`). No HF download, no OpenAI key.
- `skills/journal/skill.py` batches exchanges and flushes to mem0 after 120s inactivity
  (`_INACTIVITY_SECONDS`) via the task queue.
- v1/v2 mem0 memory format normalisation handled in `_LangboxLlm.generate_response()`.
- `compact_memories()` — LLM deduplicates/merges all facts, drops the Qdrant collection,
  re-inserts compacted facts directly.

**Topic tracking** — `_current_topic` extracted after each non-CHAT response via
`_safe_extract_topic()` (structured output, keeps previous value on failure).

## Skill Registry

`skills/registry.py` registers **12 skills** in `SKILL_MAP`:

| Intent | Skill module | `needs_wrapping` | Notes |
|--------|--------------|------------------|-------|
| CHAT | `skills/conversation/skill.py:handle_chat` | False | Owns `_history`, ReAct, persona injection |
| WEATHER | `skills/weather/` | True | DB-cached forecasts (Open-Meteo) |
| FINANCE_STOCKS | `skills/finance/` | False | yfinance + `fixtures/tickers.json`, self-comments |
| HOME_CONTROL | `skills/home_control/` | True | Philips Hue; retries on stale cache |
| INFORMATION_QUERY | `skills/information/` | True | Wikipedia lookup |
| REMINDER | `skills/reminder/` | True | timers + reminders, persisted to `Reminders` |
| NEWSFEED | `skills/newsfeed/` | True | RSS via `feedparser` |
| TRANSPORTATION | `skills/transportation/` | True | OpenRouteService directions |
| NOTES | `skills/notes/` | True | CRUD notes with categories |
| SEARCH | `skills/search/` | True | Wikipedia → web search (Tavily/ddgs); deep research mode |
| SPOTIFY | `skills/spotify/` | True | playback control via OAuth |
| NUTRITION | `skills/nutrition/` | True | Spoonacular recipes/nutrition |

> **PLANNER is NOT in `SKILL_MAP`.** `PLANNER` is a valid `IntentLiteral` and both routers
> can return it, but no skill is registered for it, so `get_skill("PLANNER")` returns
> `fallback_skill` ("That capability isn't available yet"). The planner is intentionally
> invoked explicitly via the `/planner` command, the Telegram skill, and the API
> `/query`+`/planner` path — not via intent routing. If the primary router ever classifies
> a natural query as `PLANNER`, it will hit the fallback. (Possible follow-up: either
> register a planner skill or remove `PLANNER` from the router's intent list.)

> **`skills/chat.py` is dead code.** It defines a separate simple `chat_skill` (no history,
> uses `persona.DESCRIPTION` legacy constant) that is NOT imported by `registry.py`. The
> registered CHAT skill is `skills/conversation/skill.py:handle_chat`. `skills/chat.py`
> can be deleted.

## ReAct Reasoning Engine

`skills/conversation/reasoning_engine.py` — activated for complex CHAT queries
(comparisons, pronoun+lookup patterns) via `should_use_reasoning()`. Loop: Thought →
Action → Query → Observation (max 5 steps). Tools: `INFORMATION_QUERY`, `SEARCH`,
`WEATHER`, `FINANCE_STOCKS`, `RESPOND`.

> **Broken import (bug).** `reasoning_engine.py:118` does
> `from agents.router import route_intent`, but **`agents/router.py` does not exist**
> (only `agents/generalist/router.py` and `agents/intent_router/intent_router.py` exist).
> Any ReAct step that selects a non-RESPOND action will raise `ImportError` at line 161.
> This rarely surfaces because `should_use_reasoning()` is restrictive (requires history +
> comparison/pronoun-lookup keywords), and the streaming path in `handle_chat` bypasses
> reasoning entirely. Fix: import `route` from `agents.intent_router.intent_router` or
> `run_intent_classifier` from `agents.intent_classifier`.

## Planner

`skills/planner/skill.py:run_planner()` — an outlines-style constrained tool-selection
loop (max 10 steps). Selects one tool per step via structured output, calls the target
skill's `handle()`, accumulates observations, then synthesises a final plan. Tools
exposed: `WEATHER`, `SEARCH`, `INFORMATION_QUERY`, `FINANCE_STOCKS`, `NEWSFEED`,
`REMINDER`, `DONE`. **`HOME_CONTROL` is deliberately excluded** (real-world side effects).
Serialised by an `asyncio.Lock` (one plan at a time). Results saved to the `Plans`
collection. Invoked via `/planner <task>`, Telegram, and `POST /query` with a
`/planner`-prefixed body.

## Agents

| Agent | File | Role |
|-------|------|------|
| Intent Classifier | `agents/intent_classifier/` | `classify()` (stateless, lean model) + `run_intent_classifier()` (full pipeline, API entry) |
| Generalist Router | `agents/generalist/router.py` | `generalist_route()` — primary, history-aware, confidence |
| Intent Router | `agents/intent_router/intent_router.py` | `route()` — dispatch to skill + wrapping |
| Action Resolver | `agents/action_resolver/agent.py` | Substitutes "my favourite X" from mem0 (CLI only) |
| Greeting | `agents/greeting/` | Session-opening greeting in active persona |
| Persona | `agents/persona.py` | 7 personas + I/O capability notes + adaptive hot-swap |
| Adaptive Persona | `agents/adaptive_persona.py` | Evolves "adaptive" persona from journal history |
| Agent Factory | `agents/agent_factory.py` | `create_llm()` + LM Studio/OpenAI/OpenAI-compatible LLM wrappers |

## LLM Providers

`providers/__init__.py` — `get_provider()` (generalist) and `get_intent_provider()` (lean;
reuses main provider when `MODEL_INTENT == MODEL_GENERALIST` to avoid duplicate LM Studio
handles).

| Backend (`INFERENCE_PROVIDER`) | File | Notes |
|--------------------------------|------|-------|
| `lmstudio` (default) | `providers/lmstudio.py` | Uses `lmstudio` SDK; `respond_stream` |
| `llamacpp` | `providers/llamacpp.py` | Optional dep `llama-cpp-python` |
| `openai_compatible` | `providers/openai_compatible.py` | requests-based, streaming via SSE |
| `openai` | `providers/openai.py` | Async OpenAI client |

`providers/base.py:InferenceProvider` — ABC with `complete_stream()`, `complete()` (validates
JSON into a Pydantic model), `respond()` (raw text), `embed()`, `context_length()`.

`utils/llm_structured_output.py:generate_structured_output()` — thin wrapper over
`get_provider().complete(messages, pydantic_model)`. Used everywhere for sub-intent
classification and structured extraction.

`agents/agent_factory.py:create_llm()` returns a streaming-capable LLM
(`ainvoke`/`astream`/`bind`) for the conversation skill; `_LMStudioAgent` is a separate
invoke-style agent used by the finance skill's `_comment_on_data()`.

## Personas

`agents/persona.py` — **7 personas** (old docs say 6; `adaptive` was added):

| ID | Name | Default voice | Tone |
|----|------|---------------|------|
| `jean` (default) | Jean | — | Warm, witty, capable |
| `jerk` | Jerk | marius | Sarcastic, condescending |
| `beth` | Beth | eponine | Gloomy, melancholic |
| `kimi` | Kimi | cosette | Bubbly, enthusiastic |
| `borg` | Borg | alba | Logical, emotionless |
| `fred` | Fred | jean | Paranoid, doom-mongering |
| `adaptive` | Adaptive | — | Hot-swapped from `AgentPersona` DB doc, evolved from journal |

`get_active_identity()` / `get_active_preamble()` build the system prompt; the preamble
also lists all registered skill descriptions as "capabilities". I/O capability notes
(TTS/voice-input constraints) are appended when `--speak`/`--listen` are active.

> **Voice ID has two sources of truth.** `agents.persona.get_active_voice_id()` reads the
> static `PERSONAS[pid]["voice_id"]` (the persona's *default*), while the runtime-mutable
> voice lives in `tts.tts.active_voice_id` / `set_active_voice_id()`. `main.py` and
> `api/server.py` both do `get_active_voice_id() or active_voice_id` to reconcile. The
> `/voice` and `/persona` commands and the API `POST /persona` all call
> `tts.tts.set_active_voice_id()` and persist via `db.preferences.update_preferences()`.

## Personalizer

`skills/personalizer/skill.py` — extracts user facts into `UserPersona` and injects them
into the CHAT system prompt:
- **Tiered question system** (tiers 0–3) defined via `json_schema_extra` on `UserPersona`
  fields. Tier 1 asked freely, tier 2 after rapport, tier 3 only when relevant, tier 0
  inferred never asked. `get_next_persona_question()` is data-driven from field metadata.
- Regex pre-filter (`_PERSONAL_SIGNAL`) skips the LLM when no personal signal is present.
- `analyze_persona_from_log()` (from raw exchanges) + `analyze_persona_from_memories()`
  (from mem0 facts) merge into `UserPersona` (only fills empty fields).
- `get_persona_context()` returns a cached formatted string injected into CHAT.
- Persona questions are throttled: warmup 3 exchanges, then every 5 (`_PERSONA_WARMUP`,
  `_PERSONA_QUESTION_INTERVAL`). User invitations ("ask me anything") force a question.
- `/analyze` command runs the full pipeline: persona extraction → memory pass → compact
  uncompacted days → write journal entries → evolve adaptive persona.

## Database (MongoDB via Beanie)

`db/init.py` registers **12 document models** with Beanie. `db/schemas.py` defines **13**
`Document` subclasses — `VoiceSettings` is defined but **NOT registered**, so it cannot be
used through Beanie's document interface (likely dead/legacy).

| Document | Collection | Purpose |
|----------|-----------|---------|
| `Conversations` | — | Daily Q&A exchanges (one doc per date) + `compact_status` |
| `Journal` | `Journal` | Compacted daily summaries |
| `Weather` | — | Cached forecasts (by date + location) |
| `Reminders` | — | Scheduled reminders/timers with `is_completed` |
| `Plans` | — | Planner outputs (`ask` + `plan`) |
| `Note` | `Notes` | CRUD notes with `NoteCategory` (read/listen/watch/eat/visit) |
| `UserPersona` | — | Tiered user profile (name, location, preferences, …) |
| `AgentPersona` | `AgentPersona` | Evolved adaptive persona (one doc per `persona_id`) |
| `Preferences` | — | *embedded BaseModel*, not a Document: `agentPersona`, `agentVoiceId`, `favSpotifyDevice` |
| `ServiceCredentials` | — | Per-service OAuth tokens (Spotify, Hue) |
| `Credentials` | — | Legacy Hue username |
| `HueConfiguration` | — | Smart-home state snapshot (lights + groups) |
| `Newsfeed` | — | Cached RSS content |
| `VoiceSettings` | — | **Defined but NOT registered with Beanie** |

`ConversationExchange` is an embedded model (`timestamp`, `question`, `answer`).
`db/preferences.py` loads/saves `Preferences` (embedded in `UserPersona`) and applies them
on startup via `apply_preferences()`.

## API Server (aiohttp, port 8000)

`api/server.py` — **22 routes**, OpenAPI spec auto-generated from the skills registry
(`api/openapi.py`), Swagger UI at `/docs`. CORS middleware allows `*`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/query` | Sync text query (or `/planner <task>`) via `run_intent_classifier` |
| GET | `/query/ws` | Streaming text query (status/text_chunk/done/error events) |
| GET | `/tts/voices` | List TTS voice IDs + default |
| GET | `/personas` | List all personas |
| GET/POST | `/persona` | Get/set active persona |
| POST | `/persona/voice` | Set active voice |
| POST | `/voice` | Audio → async TTS job (returns `job_id`) |
| GET | `/voice/{job_id}` | Poll voice job result |
| DELETE | `/voice/{job_id}` | Cancel voice job |
| GET | `/voice/ws` | Full voice round-trip WebSocket |
| POST | `/vision` | Image → VLM description, injected into history |
| GET | `/compact` | WebSocket: compact a day's conversations → journal |
| GET | `/conversations` | List conversation dates |
| GET | `/conversation/{date}` | Exchanges for a date |
| GET | `/journal` | Journal entries (`?date=` or `?limit=`) |
| GET | `/plans` | Saved planner plans |
| GET | `/notes` | All notes |
| GET | `/reminders` | Reminders (`?include_completed=true`) |
| GET | `/openapi.json` | OpenAPI 3.0 spec |
| GET | `/docs` | Swagger UI |

Voice jobs run on a single-worker `ThreadPoolExecutor` (`_voice_executor`) via
`tts/pipeline.py:process_voice_job` / `process_voice_ws`.

## Voice I/O

**TTS** (`tts/tts.py`) — 3 providers via `TTS_PROVIDER`:
- `local` (default) — `pocket_tts` (Kokoro-based), streams via `sounddevice`.
- `lmstudio` — LM Studio TTS endpoint.
- `remote` — OpenAI-compatible HTTP endpoint.
- 8 voice IDs: `alba, marius, javert, jean, fantine, cosette, eponine, azelma` (default),
  mapped to Kokoro voices via `_KOKORO_VOICE_MAP`. Model adapters loaded dynamically from
  `tts/models/{TTS_MODEL}.py` (e.g. `orpheus-3b.py`, `vibevoice-realtime-0.5b.py`).
- `speak()` plays aloud (Escape key interrupts via `_escape_listener`); `synthesise()`
  writes to a file. Speech is sanitised (markdown/images/URLs stripped) first.

**STT** (`stt/`) — 2 providers via `STT_PROVIDER`:
- `whisper` — `openai-whisper`.
- `mlx` (default) — `mlx-whisper` (Apple Silicon).
- Wake word "hey jarvis" via `openwakeword` (`stt/wakeword.py`); VAD-based recording via
  `stt/recorder.py` (`webrtcvad`). Whisper hallucination guard in `main.py`
  (single-word-repeat detection).

**VLM** (`utils/vlm.py`) — image description via `MODEL_VLM` (e.g. moondrama/llava/gemma3);
used by `/look` command and `POST /vision`.

## Slash Commands

`commands.py` — **17 commands** (registered in `_COMMANDS`):

| Command | Handler | Notes |
|---------|---------|-------|
| `/save` | `cmd_save` | Summarise session (uses `_get_llm`; imports `langchain_core` — see note) |
| `/clear` | `cmd_clear` | Wipe `_history` |
| `/history` | `cmd_history` | Print session history |
| `/analyze` | `cmd_analyze` | Full persona build + compact + journal + evolve adaptive |
| `/persona [id] [voice]` | `cmd_persona` | Interactive fuzzy picker or direct; persists to `Preferences` |
| `/voice [id]` | `cmd_voice` | Switch voice only; persists to `Preferences` |
| `/model [name]` | `cmd_model` | Fuzzy picker from loaded LM Studio models; resets `providers._provider` |
| `/connect <service>` | `cmd_connect` | OAuth for `spotify` / `hue` (background task) |
| `/memories [q]` | `cmd_memories` | List all or semantic search |
| `/compact-memory` | `cmd_compact_memory` | LLM dedupe + merge mem0 |
| `/pluck-memory <n>` | `cmd_pluck_memory` | Delete memory #n |
| `/flush-memory` | `cmd_flush_memory` | Drop Qdrant `langbox_memories` collection |
| `/ctx` | `cmd_ctx` | Context-window usage bar (tiktoken) |
| `/note [title]` | `cmd_note` | Extract note from conversation context |
| `/planner <task>` | `cmd_planner` | Run `run_planner()` |
| `/look <path> [q]` | `cmd_look` | VLM image description → inject into history |
| `/help` | `cmd_help` | List commands |

> **`cmd_save` imports `langchain_core`** (`from langchain_core.messages import ...`), but
> `langchain_core` is not a declared dependency in `pyproject.toml` (it may arrive
> transitively via `mem0ai`/`lmstudio`). If `/save` fails with `ModuleNotFoundError`, this
> is why — it should use the project's own `utils.messages` instead.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERENCE_PROVIDER` | `lmstudio` | `lmstudio \| llamacpp \| openai_compatible \| openai` |
| `MODEL_GENERALIST` | — | Large model: chat, wrapping, reasoning, routing |
| `MODEL_INTENT` | `MODEL_GENERALIST` | Lean model for stateless `classify()` |
| `MODEL_EMBEDDING` | — | Embedding model name |
| `MODEL_VLM` | — | Vision model for image description |
| `MODEL_HOST` | `http://localhost:1234/v1` | LLM endpoint |
| `MODEL_CTX` | `8192` | Context window size fallback |
| `OPENAI_API_KEY` | `lm-studio` | Key for OpenAI-compatible endpoints |
| `EMBEDDING_DIM` | `384` | Embedding dimension (all-MiniLM-L6-v2) |
| `MONGODB_USER/PASSWORD/HOST/PORT/DB` | — | MongoDB (`db` is `langbox`) |
| `QDRANT_HOST/PORT` | `localhost:6333` | Qdrant for mem0 |
| `STT_PROVIDER` | `mlx` | `whisper \| mlx` |
| `TTS_PROVIDER` | `local` | `local \| lmstudio \| remote` |
| `TTS_MODEL` | — | Adapter filename under `tts/models/` |
| `TTS_HOST` | (derived) | TTS endpoint for lmstudio/remote |
| `TTS_API_KEY` | `OPENAI_API_KEY` | TTS API key for remote |
| `TTS_LANG` | `en-us` | TTS language |
| `ORS_API_KEY` | — | OpenRouteService (transportation) |
| `TAVILY_API_KEY` | — | Tavily web search |
| `HUE_BRIDGE_IP` | — | Philips Hue bridge |
| `SPOTIFY_CLIENT_ID/SECRET` | — | Spotify OAuth |
| `SPOONACULAR_API_KEY` | — | Nutrition/recipes |

## Project Structure

```
pydantic-aida/
├── main.py                      # CLI entry + REPL loop + keepalive restart loop
├── commands.py                  # 17 slash commands
├── monitor.py                   # (utility/monitoring)
├── supervisord.conf             # supervisord keepalive config
├── pyproject.toml               # package "jeanclawd", uv-managed deps
├── agents/
│   ├── intent_classifier/       # classify() (stateless) + run_intent_classifier() (API pipeline)
│   ├── intent_router/           # route() — dispatch + wrapping
│   ├── generalist/router.py     # generalist_route() — primary router (history + confidence)
│   ├── action_resolver/         # resolve_action_query() — mem0 preference substitution (CLI only)
│   ├── greeting/                # session greeting generator
│   ├── adaptive_persona.py      # evolve "adaptive" persona from journal
│   ├── agent_factory.py         # create_llm() + LLM wrappers
│   └── persona.py               # 7 personas + I/O capability notes
├── providers/                   # lmstudio, llamacpp, openai_compatible, openai
├── skills/
│   ├── base.py                  # Skill dataclass
│   ├── registry.py              # SKILL_MAP (12 skills; PLANNER NOT registered)
│   ├── fallback.py              # fallback_skill
│   ├── chat.py                  # DEAD CODE — not imported by registry
│   ├── conversation/            # CHAT: handle_chat, handle_conversation, ReAct, history, topic
│   ├── weather/ finance/ home_control/ information/ reminder/ newsfeed/
│   ├── transportation/ notes/ search/ spotify/ nutrition/
│   ├── planner/                 # run_planner() — invoked via command/API, not routing
│   ├── personalizer/            # UserPersona extraction + tiered questions
│   ├── journal/                 # append_to_journal + compact + journal_writer
│   ├── camera_tracking/         # (camera tracking skill)
│   └── telegram/                # Telegram bot interface
├── db/                          # Beanie init (12 models) + schemas (13) + preferences
├── api/                         # aiohttp server (22 routes) + models + openapi
├── utils/                       # memory_client, llm_structured_output, messages, embedder,
│   # log/logger, resource_monitor, boot_info, task_queue, vlm, search, journal_index, http_client
├── utils/auth/                  # spotify + hue OAuth providers
├── stt/                         # whisper + mlx providers, wakeword, recorder
├── tts/                         # tts.py + pipeline.py + models/ adapters
├── storage/                     # (storage)
├── memory/                      # feedback memory markdown
├── logs/                        # app.log + supervisord logs
└── fixtures/                    # tickers.json + test fixtures
```

## Running Locally

```bash
# Start infrastructure (MongoDB, Qdrant) — see db/ for docker-compose
# Configure .env from .env.template (MODEL_GENERALIST, MONGODB_*, QDRANT_*, API keys)

uv sync                                   # install deps (never pip)
python main.py [--debug] [--speak] [--emote] [--server] [--listen] [--track_camera]
python main.py --keepalive                 # supervisord-style auto-restart on crash
python -m api.server                       # API server standalone
```

CLI flags: `--debug` (verbose logs), `--speak` (TTS output), `--emote` (emotion tags),
`--server` (start API on :8000), `--listen` (mic input, wake word "hey jarvis"),
`--track_camera` (camera tracking), `--keepalive` (wrapper that restarts on crash;
shuts down supervisord first to avoid port conflicts).

## Known Issues / Discrepancies

1. **Broken ReAct import** — `skills/conversation/reasoning_engine.py:118` imports
   `from agents.router import route_intent` (nonexistent). Any non-RESPOND ReAct action
   crashes. Fix: use `agents.intent_router.intent_router.route` or
   `agents.intent_classifier.run_intent_classifier`.
2. **Two divergent pipelines** — CLI runs `resolve_action_query()` (mem0 preference
   substitution); API does not. Mobile-app users get no "my favourite X" resolution.
3. **`PLANNER` intent unreachable via routing** — declared in `IntentLiteral` but no
   skill in `SKILL_MAP`; a natural-language plan request classified as `PLANNER` hits the
   fallback string.
4. **`VoiceSettings` document not registered** — defined in `db/schemas.py` but absent
   from `db/init.py` collections list; unusable via Beanie.
5. **`skills/chat.py` is dead code** — a second, simpler CHAT skill not imported anywhere.
6. **`/save` imports `langchain_core`** which is not a declared dependency — may fail if
   not transitively installed.
7. **Stale docs** — `CLAUDE.md` and `ANALYSIS.md` describe the old stateless small-model
   classifier as primary (now `generalist_route()`), say 12 intents (now 13, `NUTRITION`
   added), 6 personas (now 7, `adaptive` added), "11 document models" (now 13 defined, 12
   registered), and omit the secondary sub-intent routing pattern entirely. `ANALYSIS.md`
   also documents a `/voice` command and a Rich boot-info system not in `CLAUDE.md`.
