"""One cooperative inference lease for foreground replies and background work.

Foreground requests (user-initiated) and background tasks (memory extraction)
both submit to the same model server. The server queues requests internally, so
there is no need to gate background work — allowing concurrent extraction avoids
silent drops when the event loop is busy with foreground turns.

background_inference() uses a semaphore (limit 1) to prevent multiple
extractions from queuing simultaneously against the model server.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

_lock: asyncio.Lock | None = None
_foreground_count = 0
_extraction_sem: asyncio.Semaphore | None = None


def _get_lock() -> asyncio.Lock:
	global _lock
	if _lock is None:
		_lock = asyncio.Lock()
	return _lock


def _get_extraction_sem() -> asyncio.Semaphore:
	global _extraction_sem
	if _extraction_sem is None:
		_extraction_sem = asyncio.Semaphore(1)
	return _extraction_sem


def foreground_active() -> bool:
	return _foreground_count > 0


@asynccontextmanager
async def foreground_inference():
	"""Reserve local inference for a user-initiated request."""
	global _foreground_count
	_foreground_count += 1
	try:
		async with _get_lock():
			yield
	finally:
		_foreground_count -= 1


@asynccontextmanager
async def background_inference():
	"""Run background work (memory extraction) without blocking foreground requests.

	Uses a dedicated semaphore so at most one extraction runs at a time, while
	never preventing a foreground request from starting immediately.
	"""
	async with _get_extraction_sem():
		yield True
