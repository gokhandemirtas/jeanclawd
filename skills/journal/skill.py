"""Journal skill — appends exchanges to the daily Conversations document."""

import asyncio
from datetime import date, datetime

from db.schemas import Conversations, ConversationExchange, Journal
from utils.log import logger


async def append_to_journal(question: str, answer: str) -> None:
    """Append a Q&A exchange to today's Conversations document and store facts in mem0."""
    today = date.today()
    exchange = ConversationExchange(timestamp=datetime.now(), question=question, answer=answer)

    doc = await Conversations.find_one(Conversations.date == today)
    if doc is None:
        await Conversations(date=today, exchanges=[exchange]).insert()
    else:
        doc.exchanges.append(exchange)
        await doc.save()

    # Store extracted facts in mem0 (runs in thread to avoid blocking the event loop)
    asyncio.create_task(_store_memory(question, answer))


async def _store_memory(question: str, answer: str) -> None:
    try:
        from utils.memory_client import add_exchange
        await asyncio.to_thread(add_exchange, question, answer)
    except Exception:
        logger.exception("[journal] Failed to store memory")


async def get_latest_journal_summary() -> str | None:
    """Return the summary from the most recent Journal entry."""
    journal = await (
        Journal.find()
        .sort(-Journal.datestamp)
        .first_or_none()
    )
    return journal.summary if journal else None
