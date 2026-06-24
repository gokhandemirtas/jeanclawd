# Project Analysis: pydantic-aida

## Overview
This is a personal AI assistant framework built with Python, Pydantic, and local LLMs. It features a modular skill-based architecture with persistent memory, multi-modal I/O, and a REST/WebSocket API.

## Key Features
- **Two-Agent Design**:
  - Intent Agent (small, lean model for classification)
  - Generalist Agent (handles conversation and reasoning)
- **Modular skill-based architecture** with persistent memory
- **Multi-modal I/O** capabilities (text and voice)
- **REST/WebSocket API** for interaction

## Architecture
- **LLM Backends**: LM Studio, llama.cpp, or OpenAI-compatible endpoints
- **Database**: MongoDB (via Beanie ORM + motor async driver)
- **Vector DB**: Qdrant (for mem0 semantic memory)
- **API server**: aiohttp (port 8000)
- **Memory**: mem0ai with all-MiniLM-L6-v2 embeddings
- **Voice**: pocket-tts (TTS), openai-whisper (STT)

## Agents System
The project uses a multi-agent architecture with several specialized agents:

### Agent Types:
1. **Action Resolver** (`/agents/action_resolver/`) - Resolves abstract action queries using user memories
2. **Adaptive Persona Agent** (`/agents/adaptive_persona.py`) - Synthesizes adaptive persona from conversation history
3. **Agent Factory** (`/agents/agent_factory.py`) - Creates LLM instances for different backends (LM Studio, OpenAI-compatible)
4. **Generalist Router** (`/agents/generalist/router.py`) - Routes queries using generalist model with conversation history
5. **Greeting Agent** (`/agents/greeting/greeting_agent.py`) - Generates greetings using persona information
6. **Intent Classifier** (`/agents/intent_classifier/`) - Classifies user input into 12 intent categories
7. **Intent Router** (`/agents/intent_router/`) - Routes intents to appropriate skill handlers
8. **Persona Management** (`/agents/persona.py`) - Manages built-in and adaptive personas

### Key Agent Components:
- **Intent Classification**: 12 intent types including HOME_CONTROL, WEATHER, FINANCE_STOCKS, TRANSPORTATION, REMINDER, NEWSFEED, INFORMATION_QUERY, NOTES, SEARCH, SPOTIFY, PLANNER, CHAT
- **Conversation History**: Uses sliding window with rolling summary for long conversations
- **Persona System**: 6 built-in personas (Jean, Jerk, Beth, Kimi, Borg, Fred) plus adaptive persona
- **Memory Integration**: Uses mem0 with Qdrant for semantic memory and preference retrieval

## Boot Information System
The project includes a sophisticated boot information display system that uses the Rich library:

### Key Components:
1. **Resource Information Table** - Shows system resources like RAM/Unified Memory, GPU VRAM
2. **Models Table** - Displays configured models (Generalist, Intent, Embedding, VLM, TTS)
3. **Backend Information** - Shows system details (OS, CPU/GPU specs)
4. **Connection Status** - Monitors service connectivity (Inference, Database, API)
5. **Flag Information** - Displays runtime flags (Debug, Emote, TTS, Voice input, Camera tracking)
6. **Persona Information** - Shows active persona and voice settings

### Features:
- Uses Rich library for beautiful terminal formatting
- Animated gradient text (Jeanclawd logo)
- Live updating resource monitoring
- Color-coded status indicators (green for online, red for offline)
- Interactive display showing real-time system information

## Project Structure
```
pydantic-aida/
├── main.py                     # CLI entry point + REPL loop
├── commands.py                 # Slash command handlers
├── agents/                     # Intent classification and routing
│   ├── action_resolver/        # Resolves personal preferences in queries
│   ├── adaptive_persona.py     # Synthesizes adaptive persona from conversation history
│   ├── agent_factory.py        # LLM instance factory for different backends
│   ├── generalist/             # Generalist agent components
│   │   └── router.py           # Routes queries using generalist model with history
│   ├── greeting/               # Greeting generation
│   │   └── greeting_agent.py   # Greeting agent implementation
│   ├── intent_classifier/      # Stateless intent classification (12 intents)
│   │   ├── intent_classifier.py
│   │   └── prompt.py           # Intent classification prompt
│   ├── intent_router/          # Routes intents to skills
│   │   └── intent_router.py    # Intent routing logic
│   └── persona.py              # Persona management and built-in personas
├── skills/                     # Skill implementations
├── db/                         # Database integration
├── api/                        # API server
└── utils/                      # Utility functions
```

## Database Schema (MongoDB)
The project uses MongoDB with Beanie ORM for its database operations. The schema includes the following collections:

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

## Command System
The project includes a comprehensive slash command system for interacting with the AI assistant:

### Available Commands:
1. **/save** - Summarize and save the current session to the database
2. **/clear** - Wipe the in-memory conversation history  
3. **/history** - Print session history to the terminal
4. **/analyze** - Extract personal facts from conversation into persona profile
5. **/persona** - Switch persona and voice (with 'adaptive' option)
6. **/voice** - Set active voice ID for the agent (newly added command)
7. **/model** - Switch the active model
8. **/connect** - Authenticate services (spotify | hue)
9. **/memories** - List or search mem0 memories
10. **/compact-memory** - Deduplicate and merge memories using the LLM
11. **/pluck-memory** - Delete specific memory by index
12. **/flush-memory** - Delete all mem0 memories
13. **/ctx** - Show context window usage for current session
14. **/note** - Save a note from conversation context
15. **/planner** - Run autonomous multi-step planning agent
16. **/look** - Describe image via VLM and inject into context
17. **/help** - Show available commands

### New /voice Command Implementation:
The new `/voice` command allows users to set the active voice ID for the agent:

- **Usage**: `/voice` (interactive fuzzy picker) or `/voice <id>` (direct ID)
- **Functionality**: Updates both in-memory state and database preferences
- **Integration**: Works with existing voice system from `tts.tts` module
- **Persistence**: Saves changes to the Preferences database table via `update_preferences()`