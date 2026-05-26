"""Persona builder — extracts user facts from a day's conversation log before compaction.

Runs as part of the compaction flow, on the raw exchanges, before they are summarised.
Uses a fast regex pre-filter to skip the LLM entirely when no personal signals are present.
Merges new findings into the existing UserPersona document in MongoDB.

The persona is injected into the CHAT system prompt so the assistant can address
the user naturally and fill in query gaps (e.g. missing location for weather).
"""

import asyncio
import re
from datetime import datetime
from typing import Optional

from utils.log import logger
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Extraction schema — mirrors the askable fields of UserPersona
# ---------------------------------------------------------------------------


class PersonaUpdate(BaseModel):
    name: str = "unknown"
    location: str = "unknown"
    country: str = "unknown"
    occupation: str = "unknown"
    diet: str = "unknown"
    interests: list[str] = []
    music: list[str] = []
    age: Optional[int] = None
    gender: str = "unknown"
    relationship_status: str = "unknown"
    has_children: Optional[bool] = None
    languages: list[str] = []
    nationality: str = "unknown"
    political_leaning: str = "unknown"
    religion: str = "unknown"
    education: str = "unknown"
    communication_style: str = "unknown"
    character_traits: list[str] = []
    likes: list[str] = []
    dislikes: list[str] = []

    @field_validator("age", mode="before")
    @classmethod
    def _coerce_age(cls, v):
        return None if v in (None, "", "unknown") else v

    @field_validator("has_children", mode="before")
    @classmethod
    def _coerce_has_children(cls, v):
        return None if v in (None, "", "unknown") else v


_EXTRACT_PROMPT = """You are a silent observer analysing a conversation log between a user and their AI assistant.

IMPORTANT: The assistant may be wrong or hallucinating. Only extract facts the USER themselves expressed.

Rules:
- name: only if the user refers to themselves by name or confirms a name they were addressed by.
- location: city or town only if explicitly mentioned by the user. Do not infer from topics.
- country: country only if explicitly mentioned.
- occupation: job or profession only if the user states it.
- diet: e.g. "vegan", "vegetarian", "pescatarian", "omnivore" — only if stated.
- interests: concrete hobbies or interests the user mentioned. Empty list if none.
- music: music genres or artists the user listens to or likes (e.g. "heavy metal", "jazz", "The Beatles"). Use this field — NOT likes — for anything music-related. Empty list if none.
- age: numeric age only if stated. null otherwise.
- gender: only if the user explicitly states how they identify.
- relationship_status: e.g. "single", "married", "partnered" — only if stated.
- has_children: true/false only if stated. null otherwise.
- languages: languages the user speaks, only if mentioned.
- nationality: only if explicitly stated.
- political_leaning: infer from explicit statements. "I don't like left wing" → "right-leaning". "I hate conservatives" → "left-leaning". Use plain descriptors like "right-leaning", "left-leaning", "centrist", "libertarian". Only "unknown" if truly no signal.
- religion: only if the user explicitly states a religious belief or lack thereof.
- education: e.g. "bachelor's", "PhD", "high school" — only if stated.
- communication_style: infer from how the user writes — "formal", "casual", "technical", or "mixed".
- character_traits: personality traits observable from the user's messages (e.g. "direct", "curious", "humorous"). Empty list if unclear.
- likes / dislikes: concrete things the user expressed preference for or against. Empty list if none found.
- Use "unknown" for string fields where nothing was found. Use null for optional numeric/bool fields.

Return a JSON object matching the schema exactly."""


# ---------------------------------------------------------------------------
# Shared persona context (read by conversation skill)
# ---------------------------------------------------------------------------

_cached_persona_context: Optional[str] = None


def get_persona_context() -> Optional[str]:
    """Return a formatted persona string for injection into the CHAT system prompt."""
    return _cached_persona_context


def _build_persona_context(persona) -> str:
    lines = ["## About the user you are talking to:"]

    if persona.name:
        lines.append(f"- Name: {persona.name}. Address them by name naturally when it fits.")
    if persona.location:
        lines.append(f"- Location: {persona.location}")
    if persona.country:
        lines.append(f"- Country: {persona.country}")
    if persona.occupation:
        lines.append(f"- Occupation: {persona.occupation}")
    if persona.diet:
        lines.append(f"- Diet: {persona.diet}")
    if persona.interests:
        lines.append(f"- Interests: {', '.join(persona.interests)}")
    if persona.music:
        lines.append(f"- Music: {', '.join(persona.music)}")
    if persona.age is not None:
        lines.append(f"- Age: {persona.age}")
    if persona.gender:
        lines.append(f"- Gender: {persona.gender}")
    if persona.relationship_status:
        lines.append(f"- Relationship status: {persona.relationship_status}")
    if persona.has_children is not None:
        lines.append(f"- Has children: {'yes' if persona.has_children else 'no'}")
    if persona.languages:
        lines.append(f"- Languages: {', '.join(persona.languages)}")
    if persona.nationality:
        lines.append(f"- Nationality: {persona.nationality}")
    if persona.political_leaning:
        lines.append(f"- Political leaning: {persona.political_leaning}")
    if persona.religion:
        lines.append(f"- Religion: {persona.religion}")
    if persona.education:
        lines.append(f"- Education: {persona.education}")
    if persona.communication_style:
        lines.append(f"- Communication style: {persona.communication_style}")
    if persona.character_traits:
        lines.append(f"- Character traits: {', '.join(persona.character_traits)}")
    if persona.likes:
        lines.append(f"- Likes: {', '.join(persona.likes)}")
    if persona.dislikes:
        lines.append(f"- Dislikes: {', '.join(persona.dislikes)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persona question picker — data-driven from UserPersona field metadata
# ---------------------------------------------------------------------------


def get_next_persona_question(persona) -> Optional[str]:
    """Return the question for the next empty askable field, ordered by tier.

    Reads tier/question metadata from UserPersona.model_fields so any new field
    added to the schema with json_schema_extra={"tier": N, "question": "..."} is
    automatically picked up here — no other changes needed.

    Scalar fields with any stored value are permanently skipped.
    List fields are skipped once they have at least one entry.
    """
    from db.schemas import UserPersona

    candidates = []
    for field_name, field_info in UserPersona.model_fields.items():
        extra = field_info.json_schema_extra
        if not isinstance(extra, dict):
            continue
        tier = extra.get("tier", 0)
        question = extra.get("question", "")
        if tier < 1 or not question:
            continue
        val = getattr(persona, field_name, None)
        if isinstance(val, list):
            if val:
                continue
        else:
            if val is not None and val != "":
                continue
        candidates.append((tier, field_name, question))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


def _build_changes(existing, update: PersonaUpdate) -> dict:
    changes = {}

    scalar_fields = (
        "name", "location", "country", "occupation", "diet",
        "gender", "relationship_status", "nationality",
        "political_leaning", "religion", "education", "communication_style",
    )
    for field in scalar_fields:
        val = getattr(update, field)
        if val and val != "unknown" and not getattr(existing, field):
            changes[field] = val

    if update.age is not None and existing.age is None:
        changes["age"] = update.age

    if update.has_children is not None and existing.has_children is None:
        changes["has_children"] = update.has_children

    list_fields = ("interests", "music", "languages", "character_traits", "likes", "dislikes")
    for field in list_fields:
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
    r" believe| think| feel| vote| practice| follow| play| read| watch| listen|"
    r" speak| eat| drink| identify)|"
    r"my (name|birthday|birth|age|job|wife|husband|partner|kids?|children|family|religion|"
    r"faith|politics?|hobby|hobbies|interest|passion|fav(ou?rite)?|diet|language|nationality|"
    r"education|background|degree|career|occupation)|"
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
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def analyze_persona_from_log(exchanges: list) -> None:
    """Extract user facts from a day's raw exchanges and merge into UserPersona."""
    global _cached_persona_context

    # Filter to exchanges where the user said something personally revealing,
    # then cap to the most recent 20 to keep the prompt within model limits.
    signal_exchanges = [ex for ex in exchanges if _PERSONAL_SIGNAL.search(ex.question)]
    if not signal_exchanges:
        return
    signal_exchanges = signal_exchanges[-20:]

    lines = []
    for ex in signal_exchanges:
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


async def analyze_persona_from_memories() -> int:
    """Extract user facts from all mem0 memories and merge into UserPersona.

    Returns the number of fields updated.
    """
    global _cached_persona_context

    import asyncio
    from utils.memory_client import get_all_memories

    memories = await asyncio.get_running_loop().run_in_executor(None, get_all_memories)
    if not memories:
        return 0

    # Memories are already compact facts — filter for personal signals then extract.
    personal = [m for m in memories if _PERSONAL_SIGNAL.search(m)]
    if not personal:
        return 0

    log = "\n".join(f"- {m}" for m in personal)

    try:
        from db.schemas import UserPersona
        from datetime import datetime

        loop = asyncio.get_running_loop()
        update = await loop.run_in_executor(None, lambda: _extract_sync(log))

        existing = await UserPersona.find_one()
        if existing is None:
            existing = await UserPersona(last_updated=datetime.now()).insert()

        changes = _build_changes(existing, update)
        if not changes:
            return 0

        changes["last_updated"] = datetime.now()
        from beanie.operators import Set
        await existing.update(Set(changes))

        updated = await UserPersona.find_one()
        _cached_persona_context = _build_persona_context(updated)
        logger.debug(f"[personalizer] Persona updated from memories: {list(changes.keys())}")
        return len(changes)

    except Exception as e:
        logger.error(f"[personalizer] Memory analysis failed: {e}")
        return 0


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
