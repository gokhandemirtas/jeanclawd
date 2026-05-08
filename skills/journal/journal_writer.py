"""Write a first-person Journal entry from the day's compacted summary, then update persona."""

import asyncio
import os
from datetime import date

from agents.persona import AGENT_NAME
from db.schemas import Conversations, Journal
from utils.log import logger

_JOURNAL_PROMPT = f"""You are {AGENT_NAME}, a personal AI assistant. Write a short journal entry (3-6 sentences)
from your own first-person perspective about today's conversations with your user.
Be warm, reflective, and curious. Reference the user by name if you know it.
Note anything interesting, surprising, or worth remembering. Write naturally — this is your private journal."""


async def write_journal_entry(for_date: date) -> str:
    """Write AIDA's first-person journal entry for the given date.

    Reads Conversations.compacted, calls the LLM, saves a Journal document,
    then fires persona analysis in the background.

    Returns the journal summary string.
    """
    doc = await Conversations.find_one(Conversations.date == for_date)
    if doc is None or not doc.compacted:
        raise ValueError(f"No compacted summary available for {for_date}")

    from utils.messages import HumanMessage, SystemMessage
    from agents.agent_factory import create_llm

    llm = create_llm(
        model_name=os.environ.get("MODEL_GENERALIST"),
        temperature=0.5,
        max_tokens=512,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1,
    )
    response = await llm.ainvoke([
        SystemMessage(content=_JOURNAL_PROMPT),
        HumanMessage(content=f"Summary of today's conversations:\n{doc.compacted}"),
    ])
    narrative = response.content.strip()

    existing = await Journal.find_one(Journal.datestamp == for_date)
    if existing is None:
        await Journal(datestamp=for_date, summary=narrative).insert()
    else:
        existing.summary = narrative
        await existing.save()

    logger.info(f"[journal] Written entry for {for_date}")

    # Index in Qdrant for semantic journal search (fire-and-forget)
    import asyncio
    asyncio.create_task(_index_journal(for_date, narrative))

    return narrative


async def _index_journal(for_date, narrative: str) -> None:
    import asyncio
    try:
        from utils.journal_index import index_journal_entry
        await asyncio.to_thread(index_journal_entry, for_date, narrative)
    except Exception:
        from utils.log import logger
        logger.exception("[journal] Failed to index journal entry")
