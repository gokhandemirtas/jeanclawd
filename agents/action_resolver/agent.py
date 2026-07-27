"""Action resolver: resolves abstract action queries using user memories.

Sits between query rewriting and intent classification. When the user says
"play my favourite band", this agent:
  1. Detects: action verb + personal preference reference
  2. Searches memories for the referenced preference
  3. Asks the LLM to rewrite "play my favourite band" → "play Death"
  4. Returns the concrete query so the classifier sees "play Death" → SPOTIFY

Only fires when both conditions are met (action verb + "my favourite/preferred X"),
so the happy path (plain queries) costs nothing.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field

from utils.log import logger

_ACTION_PREFIX = re.compile(
    r"^(?:play|pause|skip|resume|stop|turn\s+(?:on|off)|set|dim|remind(?:\s+me)?|queue|add|shuffle)",
    re.IGNORECASE,
)
_PERSONAL_PREF = re.compile(
    r"\bmy\s+(?:favourite|favorite|preferred|best|top)\s+(\w+)",
    re.IGNORECASE,
)

_SYSTEM = (
    "From the memory facts, extract the specific value that answers the question. "
    "Return only the value itself — a short name or phrase, no explanation, no punctuation."
    "Example: question='What is the user\\'s favourite band?', memories='User loves the band Death' → 'Death'"
)


class _Resolved(BaseModel):
    value: str = Field(description="The extracted value (e.g. a band name, artist, song)")


async def resolve_action_query(user_input: str) -> str:
    """Return a concrete query with personal references substituted from memory.

    Returns user_input unchanged if no resolution is needed or possible.
    """
    import os

    from utils.langmem_service import search_relevant_memories
    from utils.llm_structured_output import generate_structured_output

    if not _ACTION_PREFIX.search(user_input.strip()):
        return user_input

    match = _PERSONAL_PREF.search(user_input)
    if not match:
        return user_input

    pref_type = match.group(1)  # e.g. "band", "genre", "artist"
    memory_query = f"favourite {pref_type}"

    memories = await search_relevant_memories(memory_query, limit=3)
    if not memories:
        logger.debug(f"[action_resolver] no memory for '{memory_query}' — passing through")
        return user_input

    logger.debug(f"[action_resolver] memories for '{memory_query}': {memories}")

    memory_text = "\n".join(f"- {m}" for m in memories)
    user_prompt = f'Question: What is the user\'s favourite {pref_type}?\n\nMemory facts:\n{memory_text}'

    try:
        result = await asyncio.to_thread(
            generate_structured_output,
            model_name=os.environ["MODEL_GENERALIST"],
            user_prompt=user_prompt,
            system_prompt=_SYSTEM,
            pydantic_model=_Resolved,
            max_tokens=20,
        )
        value = result.value.strip().strip("…. ?!")
        # Reject if the extracted value is mostly non-alphanumeric (LLM garbage)
        non_space = value.replace(" ", "")
        if non_space and sum(1 for c in non_space if c.isalnum()) / len(non_space) >= 0.6:
            resolved = _PERSONAL_PREF.sub(value, user_input, count=1)
            logger.debug(f"[action_resolver] '{user_input}' → '{resolved}'")
            return resolved
        logger.debug(f"[action_resolver] extracted value '{value}' rejected as garbage — passing through")
    except Exception:
        logger.exception("[action_resolver] resolution failed — passing through")

    return user_input
