# Development state

## Branch

`feature/langgraph-utilization`

## Original objective

Replace brittle keyword-first routing with context-aware intent routing so
that natural-language references to past research are not misrouted as new
research commands.

Reproduction case:

```text
I want to discuss about heavy metal music
we did some research earlier didn't we
tell me about the research we did in this topic
```

The final turn must not trigger deep research merely because it contains the
word `research`.

## Design direction

- The LLM router is the primary semantic decision-maker.
- The router receives bounded session context including the rolling summary
  retained after deep research clears detailed history.
- Deterministic classification is limited to explicit, syntax-anchored
  commands (`research heavy metal`, `deep research black metal`, `sear: …`).
- Natural-language mentions of `research`, `search`, `weather`, and similar
  words must not force a skill.
- No phrase/regex exceptions for individual conversational paraphrases.

## Work completed

### Semantic routing (original)
- Added `AGENTS.md` with semantic-routing rules.
- Added bounded session-summary context to router history.
- Replaced broad keyword-first classifier with `_explicit_command_classify()`.
- Kept `_keyword_classify` as a compatibility alias.
- Added tests distinguishing explicit research commands from conversational
  references.

### Code review — style and correctness (`c4c4f2e`)
Conducted a full codebase review with Codex (OpenAI, `wD:p2`) as verifier.

**Fixed:**
- `Optional[X]` → `X | None` throughout (`agent_factory`, `db/schemas`,
  `skills/base`); removed `from typing import Optional` imports
- `Skill.handle` typed as `Callable[..., Any]`
- `db/schemas`: renamed `Credentials.hueUsername` → `hue_username` and
  `HueConfiguration.lastUpdated` → `last_updated`; updated `hue_client.py`
- `intent_router`: replaced `try/except TypeError` control flow with
  `inspect.iscoroutinefunction`; sync handlers now run via
  `asyncio.to_thread` so the event loop is never blocked
- `task_queue`: re-raise `CancelledError` instead of swallowing; fixed
  double `task_done()` (our own regression — `finally` already calls it)
- `agent_factory`: narrow SSE parse `except` to `(JSONDecodeError, KeyError,
  IndexError)`; `_to_openai_messages` raises `TypeError` on unknown type
- `intent_classifier`: removed duplicate `import time` inside `classify()`;
  wrapped CLI test body in `if __name__ == "__main__":`
- `generalist/router`: moved deferred `import time` to module scope; renamed
  duplicate rule label `6.` to `6b.` in routing prompt
- `action_resolver`: moved deferred `import os` to module scope
- `api/server`: removed duplicate `import base64` inside `_decode_image()`
- `skills/registry`: reformatted to consistent tab indentation
- Deleted dead `skills/chat.py` (superseded by `skills/conversation/skill.py`)
- Fixed pre-existing `E741` ambiguous variable `l` in `hue_client.py`

### Memory and conversation fixes (`357479b`)

**Bugs fixed (verified with Codex):**

- **History lost-update** (`_compress_oldest`): replaced `_history.clear() +
  extend(pre-await snapshot)` with identity-based filter — messages appended
  during the LLM `await` now survive compression.
- **Topic stale after fallback** (`handle_chat`): plain-text retry path now
  fires `asyncio.ensure_future(_safe_extract_topic())` so follow-up queries
  resolve correctly after a structured-output failure.
- **Research context duplication**: archive RAG is skipped when the latest-
  report context is already loaded, preventing the same report from appearing
  twice in the prompt.

**Session topic persistence (M1):**
- `Session` schema gains `current_topic` and `last_research_report` fields.
- Both are written on every topic change (`_safe_extract_topic`,
  `handle_chat` structured path) and after deep-research compaction.
- Session resume restores them synchronously — follow-up resolution works
  immediately after a session switch without an extra LLM call.

### Remote STT provider (`357479b`)
- Added `stt/remote_provider.py` implementing `STTProvider` over any
  OpenAI-compatible `/v1/audio/transcriptions` endpoint.
- Wired into `get_stt_provider()` as `STT_PROVIDER=remote`.
- Configure via `STT_HOST`, `STT_MODEL`, `STT_LANGUAGE`, `STT_API_KEY`,
  `STT_TIMEOUT`.
- TTS remote offload was already supported via `TTS_PROVIDER=remote` +
  `TTS_HOST`.

## Known limitations / open items

- **M4 — global conversation state**: `_history`, `_rolling_summary`,
  `_current_topic`, `_pending_intent`, and related state are module globals.
  This is intentional for the current single-user, single-household design.
  If multi-user support is added later, these will need to be scoped per
  `user_id`.
- **M3 — LangMem store payload format**: `forget_memory()` handles two
  observed payload shapes (`{"content": {…}}` vs flat `{…}`). The
  `langmem` dependency is not pinned to an exact version; a format change
  could silently corrupt a forget operation.
- **Event-loop blocking in non-CHAT skills**: sync skill handlers (e.g.
  finance, home-control) are now run via `asyncio.to_thread` in
  `intent_router`. Any skill that bypasses this path (direct call) still
  blocks the loop.

## Validation

- Ruff passes for all modified files.
- `main.py` has 14 pre-existing E402 import-order warnings (not introduced
  by this branch).
