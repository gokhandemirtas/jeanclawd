"""Persona builder — extracts user facts from a day's conversation log before compaction.

Runs as part of the compaction flow, on the raw exchanges, before they are summarised.
Uses a fast regex pre-filter to skip the LLM entirely when no personal signals are present.
Merges new findings into the existing UserPersona document in MongoDB.

The persona is injected into AIDA's CHAT system prompt so she can address the user naturally.
"""

import asyncio
import re
from datetime import datetime
from typing import Optional

from utils.log import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PersonaUpdate(BaseModel):
    name: str = "unknown"
    location: str = "unknown"
    communication_style: str = "unknown"  # "formal" | "casual" | "technical" | "mixed" | "unknown"
    likes: list[str] = []
    dislikes: list[str] = []


_EXTRACT_PROMPT = """You are a silent observer analysing a day's conversation log between a user and their AI assistant.

IMPORTANT: The assistant may be wrong or hallucinating. Only extract facts the USER themselves expressed.

Rules:
- name: only if the user refers to themselves by name or confirms a name they were addressed by.
- location: city or country only if explicitly mentioned. Do not infer from topics.
- communication_style: infer from how the user writes — "formal", "casual", "technical", or "mixed".
- likes / dislikes: concrete things the user expressed preference for or against. Empty list if none found.
- Use "unknown" for string fields where nothing was found.

Return a JSON object matching the schema exactly."""


# ---------------------------------------------------------------------------
# Shared persona context (read by conversation skill)
# ---------------------------------------------------------------------------

_cached_persona_context: Optional[str] = None


def get_persona_context() -> Optional[str]:
    """Return a formatted persona string for injection into AIDA's system prompt."""
    return _cached_persona_context


def _build_persona_context(persona) -> str:
    lines = ["## About the user you are talking to:"]

    if persona.name:
        lines.append(f"- Their name is {persona.name}. Address them by name naturally when it fits.")
    if persona.location:
        lines.append(f"- Location: {persona.location}")
    if persona.communication_style:
        lines.append(f"- Communication style: {persona.communication_style}")
    if persona.likes:
        lines.append(f"- Likes: {', '.join(persona.likes)}")
    if persona.dislikes:
        lines.append(f"- Dislikes: {', '.join(persona.dislikes)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _build_changes(existing, update: PersonaUpdate) -> dict:
    changes = {}

    for field in ("name", "location", "communication_style"):
        val = getattr(update, field)
        if val and val != "unknown" and not getattr(existing, field):
            changes[field] = val

    for field in ("likes", "dislikes"):
        current: list = getattr(existing, field) or []
        new_items = [x for x in getattr(update, field) if x not in current]
        if new_items:
            changes[field] = current + new_items

    return changes


# ---------------------------------------------------------------------------
# Pre-filter regex
# ---------------------------------------------------------------------------

_PERSONAL_SIGNAL = re.compile(
    r"\b(i('m| am| was| were| have| had| work| worked| like| love| hate| enjoy| prefer|"
    r" believe| think| feel| vote| practice| follow| play| read| watch| listen)|"
    r"my (name|birthday|birth|age|job|wife|husband|partner|kids?|children|family|religion|"
    r"faith|politics?|hobby|hobbies|interest|passion|fav(ou?rite)?)|"
    r"i('m| am) (a |an |into |from )|born (in|on)|i don'?t (like|enjoy|eat|drink|believe|support))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_sync(log: str) -> PersonaUpdate:
    import os
    from utils.llm_structured_output import generate_structured_output

    return generate_structured_output(
        model_name=os.environ["MODEL_GENERALIST"],
        user_prompt=log,
        system_prompt=_EXTRACT_PROMPT,
        pydantic_model=PersonaUpdate,
        max_tokens=256,
    )


# ---------------------------------------------------------------------------
# Public entry point — called before compaction so raw exchanges are not lost
# ---------------------------------------------------------------------------

async def analyze_persona_from_log(exchanges: list) -> None:
    """Extract user facts from a day's raw exchanges and merge into UserPersona."""
    global _cached_persona_context

    all_user_text = " ".join(ex.question for ex in exchanges)
    if not _PERSONAL_SIGNAL.search(all_user_text):
        return

    lines = []
    for ex in exchanges:
        lines.append(f"User: {ex.question}")
        lines.append(f"Assistant: {ex.answer}")
    log = "\n".join(lines)

    try:
        from db.schemas import UserPersona

        loop = asyncio.get_running_loop()
        update = await loop.run_in_executor(None, lambda: _extract_sync(log))

        existing = await UserPersona.find_one()
        if existing is None:
            existing = await UserPersona(last_updated=datetime.now()).insert()

        changes = _build_changes(existing, update)
        if not changes:
            return

        changes["last_updated"] = datetime.now()

        from beanie.operators import Set
        await existing.update(Set(changes))

        updated = await UserPersona.find_one()
        _cached_persona_context = _build_persona_context(updated)
        logger.debug(f"[personalizer] Persona updated: {list(changes.keys())}")

    except Exception as e:
        logger.error(f"[personalizer] Update failed: {e}")


async def start_personalizer() -> str | None:
    """Load existing persona from DB on startup."""
    global _cached_persona_context
    try:
        from db.schemas import UserPersona
        existing = await UserPersona.find_one()
        if existing:
            _cached_persona_context = _build_persona_context(existing)
            return "[personalizer] Loaded existing persona from DB."
    except Exception as e:
        logger.warning(f"[personalizer] Could not load existing persona: {e}")
