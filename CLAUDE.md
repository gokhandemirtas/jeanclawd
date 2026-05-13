# pydantic-aida

A personal AI assistant framework built with Python, Pydantic, and local LLMs. Modular skill-based architecture with persistent memory, multi-modal I/O, and a REST/WebSocket API.

## Two-Agent Design

**Intent Agent** (`agents/intent_classifier/`) — uses `MODEL_INTENT` (small, lean model). Receives only the current query + a lightweight `current_topic` hint. No conversation history, no memory. Returns an `IntentLiteral`. Fast and isolated.

**Generalist Agent** (`skills/conversation/skill.py`) — uses `MODEL_GENERALIST`. Owns the full sliding-window history, rolling summary, memory injection, and reasoning engine. Handles CHAT and wraps skill outputs into natural language.

The Generalist tracks `_current_topic` after each response; `main.py` passes it to `classify()` as a one-line context hint so the Intent Agent can correctly route follow-ups ("what about Apple?" after a Tesla stock query → `FINANCE_STOCKS`, not `CHAT`).

## Stack

- **LLM backends:** LM Studio, llama.cpp, or any OpenAI-compatible endpoint
- **Database:** MongoDB (via Beanie ORM + motor async driver)
- **Vector DB:** Qdrant (for mem0 semantic memory)
- **API server:** aiohttp (port 8000)
- **Memory:** mem0ai with all-MiniLM-L6-v2 embeddings
- **Voice:** pocket-tts (TTS), openai-whisper (STT)
- **Structured output:** Pydantic models via `utils/llm_structured_output.py`

## Project Structure

```
pydantic-aida/
├── main.py                     # CLI entry point + REPL loop
├── commands.py                 # Slash command handlers
├── agents/
│   ├── intent_classifier/      # Stateless intent classification (12 intents)
│   ├── intent_router/          # Routes intents to skills
│   ├── action_resolver/        # Rewrites queries using mem0 preferences
│   ├── greeting/               # Startup greeting generator
│   ├── agent_factory.py        # LLM instance factory
│   └── persona.py              # 6 built-in personas
├── skills/
│   ├── base.py                 # Skill dataclass
│   ├── registry.py             # SKILL_MAP — all registered skills
│   ├── conversation/           # Chat + ReAct reasoning engine
│   │   ├── skill.py            # handle_chat, handle_conversation, query rewriting
│   │   └── reasoning_engine.py # ReAct loop (max 5 steps)
│   └── [weather, finance, home_control, information, reminder,
│        newsfeed, transportation, notes, search, spotify,
│        journal, personalizer, planner, camera_tracking, telegram]/
├── db/
│   ├── init.py                 # Beanie initialization
│   ├── schemas.py              # 11 document models
│   └── preferences.py          # Preferences + persona loading
├── api/
│   ├── server.py               # 19 aiohttp routes
│   └── models.py               # API request/response models
└── utils/
    ├── memory_client.py        # mem0 + Qdrant integration
    ├── messages.py             # SystemMessage, HumanMessage, AIMessage
    └── resource_monitor.py     # Context window tracking
```

## Architecture: Message Flow

```
User input
  → resolve_query()           # Rewrites follow-up questions via _current_topic
  → resolve_action_query()    # Substitutes personal prefs from mem0
  → classify()                # Stateless intent classifier → IntentLiteral
  → route(intent, query)      # Dispatches to skill.handle()
  → Skill handler             # Fetch data + optional LLM wrapping
  → Streaming via on_token callback
  → append_to_journal()       # DB + mem0 batch write
  → Optional TTS
```

## Intent System

**12 intents:** `HOME_CONTROL`, `WEATHER`, `FINANCE_STOCKS`, `TRANSPORTATION`, `REMINDER`, `NEWSFEED`, `INFORMATION_QUERY`, `NOTES`, `SEARCH`, `SPOTIFY`, `PLANNER`, `CHAT`

**Pending intent (locked routing):** When a skill response ends with `?`, the next turn skips classification and re-routes to the same skill. Implemented in `main.py` via `_pending_intent`.

**Follow-up rewriting:** `_current_topic` (set after each non-CHAT response) is used to expand ambiguous follow-ups like "what about tomorrow?" into standalone queries before classification. Happens in `skills/conversation/skill.py`.

**Known limitation:** The intent classifier is stateless — it has no access to conversation history. Follow-up questions that don't contain enough context may be misclassified. The `_current_topic` + `_pending_intent` mechanisms partially compensate, but are not robust for multi-turn flows that cross intent boundaries.

## Conversation Memory

**In-session (deque, maxlen=20):**
- `_history` in `skills/conversation/skill.py` — HumanMessage + AIMessage pairs
- When full, oldest half compressed into `_rolling_summary` before eviction
- Injected into CHAT system prompt

**Persistent (mem0 + Qdrant):**
- Exchanges batched in groups of 5 or after 60s inactivity
- mem0 extracts facts with local LLM, stores vectors in Qdrant
- Searched by action resolver (preferences) and CHAT skill (context injection)

**Topic tracking:**
- `_current_topic` — string extracted after each non-CHAT skill response
- Used to make follow-up questions self-contained before re-classification

## Skill Architecture

```python
@dataclass
class Skill:
    id: str                      # Maps to IntentLiteral
    description: str
    handle: Callable
    system_prompt: Optional[str] # For LLM wrapping pass
    needs_wrapping: bool         # True = raw data → natural language
    auth_provider: Any
```

Skills with `needs_wrapping=True` pass their raw output through `handle_conversation()` for natural language synthesis.

## Database Schema (MongoDB)

| Collection | Purpose |
|-----------|---------|
| `Conversations` | Daily Q&A exchanges (one doc per date) |
| `Journal` | Compacted conversation summaries |
| `Weather` | Cached forecasts (by date + location) |
| `Reminders` | Scheduled tasks with datetime |
| `Plans` | Planner agent outputs |
| `Note` | CRUD notes with category |
| `UserPersona` | Name, location, likes/dislikes, preferences |
| `Preferences` | agentPersona, agentVoiceId, favSpotifyDevice |
| `ServiceCredentials` | OAuth tokens (Spotify, Hue) |
| `HueConfiguration` | Smart home state snapshot |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERENCE_PROVIDER` | `lmstudio` | Backend: lmstudio/llamacpp/openai_compatible |
| `MODEL_GENERALIST` | — | Large model for chat, wrapping, reasoning |
| `MODEL_INTENT` | `MODEL_GENERALIST` | Small model for intent classification (e.g. `llama-3.2-1b-instruct`) |
| `MODEL_HOST` | `http://localhost:1234/v1` | LLM endpoint |
| `MODEL_CTX` | `8192` | Context window size |
| `MONGODB_HOST/PORT/USER/PASSWORD` | — | MongoDB connection |
| `QDRANT_HOST/PORT` | — | Qdrant connection |
| `OPENAI_API_KEY` | `lm-studio` | Key for OpenAI-compatible endpoints |

## API Server (port 8000)

Key endpoints:
- `POST /query` — Sync text query
- `WS /query/ws` — Streaming text query (JSON events: status, text_chunk, done, error)
- `POST /voice` — Audio → response (async job)
- `WS /voice/ws` — Full voice round-trip
- `GET|POST /persona` — Get/set active persona
- `WS /compact` — Compact daily conversations → journal

## Personas (agents/persona.py)

| ID | Name | Tone |
|----|------|------|
| `jean` | Jean (default) | Warm, witty, capable |
| `jerk` | Jerk | Sarcastic, condescending |
| `beth` | Beth | Gloomy, melancholic |
| `kimi` | Kimi | Bubbly, enthusiastic |
| `borg` | Borg | Logical, emotionless |
| `fred` | Fred | Paranoid, doom-mongering |

## Slash Commands

`/save`, `/clear`, `/history`, `/analyze`, `/model [name]`, `/connect <service>`, `/memories [q]`, `/compact-memory`, `/pluck-memory <n>`, `/flush-memory`, `/ctx`, `/note [title]`, `/planner <task>`, `/help`

## ReAct Reasoning Engine

`skills/conversation/reasoning_engine.py` — activated for complex CHAT queries (comparisons, multi-step, pronouns).

Loop: Thought → Action → Query → Observation (max 5 steps).
Available tools: `INFORMATION_QUERY`, `SEARCH`, `WEATHER`, `FINANCE_STOCKS`, `RESPOND`.

## Running Locally

```bash
# Start infrastructure
cd db && docker-compose up -d

# Install dependencies
pip install -e .

# Run CLI
python main.py [--debug] [--speak] [--emote] [--server]
```
