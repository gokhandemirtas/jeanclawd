"""Persistent conversation memory via mem0 + Qdrant.

Stores facts extracted from each conversation exchange so that cross-session
context ("remember I told you I live in London?") can be recalled semantically
rather than relying on full journal dumps.

mem0 handles fact extraction (via our local LLM) and vector storage (Qdrant).
The shared all-MiniLM-L6-v2 embedder (utils/embedder.py) is injected after init
so both this module and journal_index share a single loaded model.
"""

from __future__ import annotations

import os
from typing import Optional

from utils.log import logger

_memory: Optional["Memory"] = None


class _LangboxLlm:
    """Duck-typed mem0 LLM provider wrapping the project's ChatLlamaCpp.

    mem0 calls generate_response() to extract facts from conversation messages.
    Returning a plain string (the model's raw output) is sufficient — mem0 parses
    the JSON itself. json_repair is applied as a safety net.
    """

    def generate_response(self, messages, response_format=None, tools=None, tool_choice=None) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from agents.agent_factory import create_llm
        from json_repair import repair_json

        llm = create_llm(
            model_name=os.environ["MODEL_GENERALIST"],
            temperature=0.0,
            max_tokens=256,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.0,
        )
        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = llm.invoke(lc_messages)
        text = response.content.strip()

        # Ensure the output is valid JSON before mem0 parses it
        if response_format and response_format.get("type") == "json_object":
            text = repair_json(text)
            # mem0 expects {"memory": [{"text": "...", "event": "ADD|UPDATE|DELETE|NONE"}]}
            # Local LLMs sometimes return malformed items (nested lists, bare strings).
            # Normalise to dicts while preserving event values — forcing ADD on everything
            # breaks mem0's dedup phase which relies on NONE/UPDATE/DELETE events.
            _VALID_EVENTS = {"ADD", "UPDATE", "DELETE", "NONE"}
            try:
                import json as _json
                parsed = _json.loads(text)
                if isinstance(parsed, dict) and "memory" in parsed:
                    items = parsed["memory"]
                    if isinstance(items, list):
                        normalised = []
                        for item in items:
                            if isinstance(item, dict) and "text" in item:
                                # Already well-formed — preserve event as-is
                                normalised.append(item)
                            elif isinstance(item, list) and item:
                                # ["fact text", "EVENT"] or ["fact text"]
                                fact = str(item[0])
                                event = str(item[1]).upper() if len(item) > 1 and str(item[1]).upper() in _VALID_EVENTS else "ADD"
                                normalised.append({"text": fact, "event": event})
                            elif isinstance(item, str) and item:
                                # "EVENT: fact text" or bare fact string
                                upper = item.upper()
                                event = "ADD"
                                fact = item
                                for ev in _VALID_EVENTS:
                                    if upper.startswith(f"{ev}:"):
                                        event = ev
                                        fact = item[len(ev) + 1:].strip()
                                        break
                                normalised.append({"text": fact, "event": event})
                        parsed["memory"] = normalised
                        text = _json.dumps(parsed)
            except Exception:
                pass
        return text


class _LangboxEmbedder:
    """Duck-typed mem0 embedder that delegates to the shared all-MiniLM singleton."""

    def embed(self, text: str, memory_action: str = "add") -> list[float]:
        from utils.embedder import embed
        return embed(text)


def _get_memory() -> "Memory":
    global _memory
    if _memory is None:
        from mem0 import Memory

        # Patch both factories before from_config() so no external providers
        # are instantiated — no HF download, no OpenAI key validation.
        _embedder = _LangboxEmbedder()
        _llm = _LangboxLlm()
        try:
            from mem0.utils import factory as _factory
            _factory.EmbedderFactory.create = staticmethod(lambda provider, config, *a, **kw: _embedder)
            _factory.LlmFactory.create = staticmethod(lambda provider, config, *a, **kw: _llm)
        except (ImportError, AttributeError):
            logger.warning("[memory_client] Could not patch factories — falling back to post-init replace")

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": os.environ.get("QDRANT_HOST", "localhost"),
                    "port": int(os.environ.get("QDRANT_PORT", "6333")),
                    "collection_name": "langbox_memories",
                    "embedding_model_dims": 384,  # all-MiniLM-L6-v2-q8_0.gguf
                },
            },
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
        }

        _memory = Memory.from_config(config)
        # Ensure our instances are set even if the factory patch path was skipped
        _memory.llm = _llm
        _memory.embedding_model = _embedder

        logger.debug("[memory_client] Initialised mem0 with Qdrant + local LLM + all-MiniLM")

    return _memory


def add_exchange(user_msg: str, assistant_msg: str, user_id: str = "default") -> None:
    """Extract facts from a conversation turn and store them in Qdrant."""
    try:
        mem = _get_memory()
        mem.add(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            user_id=user_id,
        )
    except Exception:
        logger.exception("[memory_client] Failed to store exchange")


def compact_memories(user_id: str = "default") -> tuple[int, int]:
    """LLM-squash duplicate/redundant memories and replace the store.

    Returns (original_count, compacted_count).
    """
    import json
    import os
    import uuid
    from datetime import datetime, timezone

    from utils.embedder import embed

    mem = _get_memory()

    # 1. Fetch all stored memories
    raw = mem.get_all(filters={"user_id": user_id}, top_k=1000)
    results = raw.get("results", raw) if isinstance(raw, dict) else raw
    if not results:
        return 0, 0

    entries = []
    for r in results:
        text = r.get("memory") or r.get("text") if isinstance(r, dict) else str(r)
        if text:
            entries.append(text)

    original_count = len(entries)
    if original_count == 0:
        return 0, 0

    # 2. Ask the LLM to deduplicate and merge
    from pydantic import BaseModel
    from utils.llm_structured_output import generate_structured_output

    class _CompactedMemories(BaseModel):
        facts: list[str]

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(entries))
    compacted = generate_structured_output(
        model_name=os.environ["MODEL_GENERALIST"],
        user_prompt=f"Stored facts:\n{numbered}",
        system_prompt=(
            "You are given a numbered list of personal memory facts about the user. "
            "Your task:\n"
            "1. Remove exact or near-duplicate facts — keep the most specific version.\n"
            "2. Merge facts that say the same thing differently into one clear statement.\n"
            "3. Discard any fact that is world knowledge, a calculation, or not personal to the user.\n"
            "Return the cleaned list in the `facts` field. One fact per item. No explanations."
        ),
        pydantic_model=_CompactedMemories,
        max_tokens=1024,
    )

    compacted_facts = [f.strip() for f in compacted.facts if f.strip()]
    if not compacted_facts:
        logger.warning("[memory_client] compact_memories: LLM returned empty list — aborting")
        return original_count, original_count

    # 3. Flush all existing points from Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port)

    all_ids = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            "langbox_memories",
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        all_ids.extend(p.id for p in batch)
        if next_offset is None:
            break
        offset = next_offset

    if all_ids:
        client.delete("langbox_memories", points_selector=all_ids)

    # 4. Re-insert compacted facts directly (bypasses LLM extraction)
    now = datetime.now(timezone.utc).isoformat()
    points = []
    for fact in compacted_facts:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(fact),
                payload={
                    "memory": fact,
                    "user_id": user_id,
                    "hash": str(hash(fact)),
                    "created_at": now,
                    "updated_at": now,
                },
            )
        )
    client.upsert("langbox_memories", points=points)

    # Reset singleton so mem0's internal state reflects the new collection
    global _memory
    _memory = None

    logger.debug(f"[memory_client] compact: {original_count} → {len(compacted_facts)}")
    return original_count, len(compacted_facts)


def search_memories(query: str, user_id: str = "default", limit: int = 5) -> list[str]:
    """Return facts relevant to the query, or [] on failure."""
    try:
        mem = _get_memory()
        raw = mem.search(query, filters={"user_id": user_id}, limit=limit)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        memories = []
        for r in results:
            # mem0 returns dicts; key is 'memory' in current versions
            text = r.get("memory") or r.get("text") if isinstance(r, dict) else str(r)
            if text:
                memories.append(text)
        return memories
    except Exception:
        logger.exception("[memory_client] Search failed")
        return []
