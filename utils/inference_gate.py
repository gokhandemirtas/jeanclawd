"""One cooperative inference lease for foreground replies and background work."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

_lock: asyncio.Lock | None = None
_foreground_count = 0


def _get_lock() -> asyncio.Lock:
	global _lock
	if _lock is None:
		_lock = asyncio.Lock()
	return _lock


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
	"""Run maintenance only when no foreground request is waiting or active."""
	if foreground_active():
		yield False
		return
	async with _get_lock():
		if foreground_active():
			yield False
			return
		yield True
