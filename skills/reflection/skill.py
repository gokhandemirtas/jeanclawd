"""Reflection skill — reviews past exchanges and stores self-critique notes.

Future override mechanism (not built yet) will retrieve high-signal reflections
(e.g. category="routing") and inject them into the router/CHAT prompts to nudge
future behavior. For now, this module only CAPTURES reflections.

Triggers:
- Manual: /reflect slash command (commands.py:cmd_reflect) — user-driven
- Automatic: piggybacked on the CHAT debounced personalizer
  (skills/conversation/skill.py:_personalizer_fire) — fires on chat idle
"""

import os
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from utils.log import logger

from db.schemas import ConversationExchange, Reflection
from skills.reflection.prompts import REFLECTION_SYSTEM_PROMPT, REFLECTION_USER_PROMPT_TEMPLATE
from utils.llm_structured_output import generate_structured_output


ReflectionCategory = Literal["routing", "factual", "tone", "completeness", "other"]


class _ReflectionItem(BaseModel):
    observation: str = Field(
        description="One sentence: what went wrong, including the user query and the actual (wrong) assistant response."
    )
    lesson: str = Field(
        description="One to two sentences: what to do next time. Be specific and actionable — name the trigger, the wrong intent/skill, and the right one."
    )
    category: ReflectionCategory


class _ReflectionResponse(BaseModel):
    reflections: list[_ReflectionItem] = Field(
        default_factory=list,
        description="0-3 self-critique notes. Empty list if the exchanges were fine.",
    )


def _format_exchanges(exchanges: list[ConversationExchange]) -> str:
    """Render exchanges as a numbered list for the LLM prompt."""
    lines = []
    for i, ex in enumerate(exchanges, 1):
        q = ex.question.strip().replace("\n", " ")
        a = ex.answer.strip().replace("\n", " ")
        if len(a) > 400:
            a = a[:400] + "…"
        lines.append(f"--- Exchange {i} ---")
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
        lines.append("")
    return "\n".join(lines)


async def reflect_on_exchanges(
    exchanges: list[ConversationExchange],
    *,
    source: str = "manual",
) -> int:
    """Review exchanges and persist any new reflections to MongoDB.

    Returns the number of reflections saved. Skips silently if exchanges is empty
    or the LLM call fails — reflection is observability, not correctness.
    """
    if not exchanges:
        return 0

    try:
        result = generate_structured_output(
            model_name=os.environ["MODEL_GENERALIST"],
            user_prompt=REFLECTION_USER_PROMPT_TEMPLATE.format(
                n=len(exchanges),
                format_exchanges=_format_exchanges(exchanges),
            ),
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            pydantic_model=_ReflectionResponse,
            temperature=0.2,
            max_tokens=600,
        )
    except Exception as e:
        logger.warning(f"[reflection] LLM call failed ({source}): {e}")
        return 0

    if not result.reflections:
        logger.debug(f"[reflection] no misfires detected ({source}, {len(exchanges)} exchange(s))")
        return 0

    saved = 0
    for item in result.reflections:
        # Use the first exchange's query/response as the source for the doc.
        # When reflecting on multiple exchanges, the LLM usually pinpoints one
        # of them in `observation`, so this gives a reasonable breadcrumb.
        first_q = exchanges[0].question[:200]
        first_a = exchanges[0].answer[:300]
        # Embed the `lesson` (the actionable rule) so the future override
        # retrieval can do MongoDB Atlas $vectorSearch directly. Embedding is
        # best-effort — a failure here logs a warning but still saves the
        # reflection (the doc is queryable by text fallback later).
        try:
            from providers import get_provider
            import asyncio as _aio
            embedding = await _aio.to_thread(get_provider().embed, item.lesson)
        except Exception as e:
            logger.debug(f"[reflection] embedding failed (saving without): {e}")
            embedding = None

        doc = Reflection(
            observation=item.observation,
            lesson=item.lesson,
            category=item.category,
            source_query=first_q,
            source_response=first_a,
            created_at=datetime.now(),
            embedding=embedding,
        )
        try:
            await doc.insert()
            saved += 1
            logger.debug(
                f"[reflection] saved [{item.category}]: {item.observation[:100]}"
            )
        except Exception as e:
            logger.warning(f"[reflection] insert failed: {e}")

    return saved


async def get_recent_exchanges(n: int = 5) -> list[ConversationExchange]:
    """Return the last N exchanges from today's Conversations doc.

    Falls back to empty list on any failure (db down, no doc yet, etc.).
    """
    from db.schemas import Conversations

    try:
        today = datetime.now().date()
        doc = await Conversations.find_one(Conversations.date == today)
        if not doc or not doc.exchanges:
            return []
        return doc.exchanges[-n:]
    except Exception as e:
        logger.debug(f"[reflection] get_recent_exchanges failed: {e}")
        return []
