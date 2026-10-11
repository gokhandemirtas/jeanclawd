import asyncio

from utils.log import logger

_queue: asyncio.Queue | None = None


def get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


def enqueue(name: str, coro) -> None:
    get_queue().put_nowait((name, coro))
    from utils.observability import emit
    emit("queue_enqueued", task=name, depth=get_queue().qsize())


def pending_count() -> int:
    return get_queue().qsize()


def drain() -> int:
    q = get_queue()
    count = 0
    while True:
        try:
            q.get_nowait()
            q.task_done()
            count += 1
        except asyncio.QueueEmpty:
            break
    return count


async def worker() -> None:
    q = get_queue()
    while True:
        name, coro = await q.get()
        from utils.observability import span
        try:
            with span("queue_task", task=name, depth=q.qsize()):
                await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[queue] Task '{name}' failed")
        finally:
            q.task_done()
