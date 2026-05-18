"""Journal skill — appends exchanges to the daily Conversations document."""

import asyncio
from datetime import date, datetime

from db.schemas import Conversations, ConversationExchange, Journal
from utils.log import logger

_INACTIVITY_SECONDS = 120

_pending: list[tuple[str, str]] = []
_flush_handle: asyncio.TimerHandle | None = None


async def append_to_journal(question: str, answer: str) -> None:
    """Append a Q&A exchange to today's Conversations document and buffer for mem0."""
    today = date.today()
    exchange = ConversationExchange(timestamp=datetime.now(), question=question, answer=answer)

    doc = await Conversations.find_one(Conversations.date == today)
    if doc is None:
        await Conversations(date=today, exchanges=[exchange]).insert()
    else:
        doc.exchanges.append(exchange)
        await doc.save()

    _enqueue_memory(question, answer)


def _enqueue_memory(question: str, answer: str) -> None:
    global _flush_handle

    _pending.append((question, answer))

    if _flush_handle is not None:
        _flush_handle.cancel()

    loop = asyncio.get_event_loop()
    _flush_handle = loop.call_later(_INACTIVITY_SECONDS, _schedule_flush)


def _schedule_flush() -> None:
    from utils.task_queue import enqueue
    enqueue("mem0_flush", _flush_memory())


async def _flush_memory() -> None:
    global _pending, _flush_handle

    batch, _pending = _pending, []
    _flush_handle = None

    if not batch:
        return

    logger.debug(f"[journal] Flushing {len(batch)} exchange(s) to mem0")
    try:
        from utils.memory_client import add_exchanges
        await asyncio.to_thread(add_exchanges, batch)
    except Exception:
        logger.exception("[journal] Failed to flush memory batch")


async def get_latest_journal_summary() -> str | None:
    """Return the summary from the most recent Journal entry."""
    journal = await (
        Journal.find()
        .sort(-Journal.datestamp)
        .first_or_none()
    )
    return journal.summary if journal else None
