"""Synthesise the Adaptive persona from recent journal entries and the user persona."""

import json
import os
import re
from datetime import datetime

from utils.log import logger

_EVOLVE_PROMPT = """\
You are crafting the character identity for a personal AI assistant that has grown \
through extended conversations with one specific user.

Based on the journal entries and user profile below, write a 2-3 sentence character \
description for this AI assistant — in second person ("You are...").

Describe:
- The tone and communication style the assistant has developed with this user
- What topics or interests the assistant naturally leans into
- The emotional register of the relationship (e.g. warm and playful, calm and precise)

Do not describe the user. Describe the assistant's evolved identity.
Also write a short greeting style hint (one sentence, lowercase) that fits this voice.

Respond as valid JSON only — no markdown, no extra text:
{"description": "...", "greeting_style": "..."}"""


async def evolve_adaptive_persona() -> bool:
    """Generate an updated Adaptive persona from recent journals and user context.

    Persists to the AgentPersona collection and hot-swaps the in-memory PERSONAS entry.
    Returns True if evolution succeeded, False otherwise.
    """
    try:
        from db.schemas import AgentPersona, Journal
        from skills.personalizer.skill import get_persona_context

        journals = await Journal.find_all().sort("-datestamp").limit(10).to_list()
        if not journals:
            logger.debug("[adaptive] No journal entries yet — skipping evolution")
            return False

        journal_text = "\n\n".join(f"[{j.datestamp}]\n{j.summary}" for j in journals)
        user_context = get_persona_context()

        from agents.agent_factory import create_llm
        from utils.messages import HumanMessage, SystemMessage

        llm = create_llm(
            model_name=os.environ.get("MODEL_GENERALIST"),
            temperature=0.6,
            max_tokens=256,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1,
        )

        content = ""
        if user_context:
            content += f"{user_context}\n\n"
        content += f"## Recent journal entries:\n{journal_text}"

        response = await llm.ainvoke([
            SystemMessage(content=_EVOLVE_PROMPT),
            HumanMessage(content=content),
        ])

        raw = response.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            logger.warning(f"[adaptive] Could not parse JSON from LLM response: {raw[:200]}")
            return False

        data = json.loads(match.group())
        description = data.get("description", "").strip()
        greeting_style = data.get("greeting_style", "").strip()

        if not description or not greeting_style:
            logger.warning("[adaptive] LLM returned empty fields — skipping")
            return False

        existing = await AgentPersona.find_one(AgentPersona.persona_id == "adaptive")
        if existing is None:
            await AgentPersona(
                persona_id="adaptive",
                description=description,
                greeting_style=greeting_style,
                last_evolved=datetime.now(),
                journal_count=len(journals),
            ).insert()
        else:
            existing.description = description
            existing.greeting_style = greeting_style
            existing.last_evolved = datetime.now()
            existing.journal_count = len(journals)
            await existing.save()

        from agents.persona import load_adaptive_persona
        load_adaptive_persona(description, greeting_style)

        logger.info(f"[adaptive] Persona evolved from {len(journals)} journal entry/entries")
        return True

    except Exception as e:
        logger.error(f"[adaptive] Evolution failed: {e}")
        return False


async def load_adaptive_persona_from_db() -> bool:
    """Load a previously synthesised adaptive persona from DB into memory at startup.

    Returns True if a stored description was found and loaded.
    """
    try:
        from db.schemas import AgentPersona

        stored = await AgentPersona.find_one(AgentPersona.persona_id == "adaptive")
        if stored is None:
            return False

        from agents.persona import load_adaptive_persona
        load_adaptive_persona(stored.description, stored.greeting_style, stored.voice_id)

        logger.debug(f"[adaptive] Loaded persona from DB (last evolved: {stored.last_evolved.date()})")
        return True

    except Exception as e:
        logger.warning(f"[adaptive] Could not load from DB: {e}")
        return False
