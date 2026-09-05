"""Centralized operational events, request correlation, and in-process metrics."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Iterator

from utils.log import logger

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_request_fields: ContextVar[dict] = ContextVar("request_fields", default={})
_metrics_lock = Lock()
_counters: dict[str, int] = defaultdict(int)
_durations: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "max_ms": 0.0})


def current_request_id() -> str | None:
	return _request_id.get()


def emit(name: str, *, level: str = "INFO", **fields) -> None:
	"""Record an event with the current request correlation fields."""
	base = dict(_request_fields.get())
	if _request_id.get():
		base["request_id"] = _request_id.get()
	base.update(fields)
	with _metrics_lock:
		_counters[name] += 1
	if name.startswith("skill_execution_") and base.get("skill"):
		base["_display_name"] = f"{name} [{base['skill']}]"
	logger.event(name, level=level, **base)


@contextmanager
def request_context(interface: str, operation: str, **fields) -> Iterator[str]:
	"""Correlate and time one top-level request."""
	rid = uuid.uuid4().hex
	token_id = _request_id.set(rid)
	token_fields = _request_fields.set({"interface": interface, "operation": operation, **fields})
	started = time.perf_counter()
	emit("request_started")
	try:
		yield rid
	except Exception as exc:
		emit("request_failed", level="ERROR", error_type=type(exc).__name__)
		raise
	finally:
		emit("request_completed", duration_ms=round((time.perf_counter() - started) * 1000, 2))
		_request_fields.reset(token_fields)
		_request_id.reset(token_id)


@contextmanager
def span(name: str, **fields):
	"""Time a nested operation and emit success/failure events."""
	started = time.perf_counter()
	emit(f"{name}_started", **fields)
	try:
		yield
	except Exception as exc:
		duration_ms = round((time.perf_counter() - started) * 1000, 2)
		emit(f"{name}_failed", level="ERROR", duration_ms=duration_ms, error_type=type(exc).__name__, **fields)
		raise
	else:
		duration_ms = round((time.perf_counter() - started) * 1000, 2)
		with _metrics_lock:
			stats = _durations[name]
			stats["count"] = int(stats["count"]) + 1
			stats["total_ms"] = float(stats["total_ms"]) + duration_ms
			stats["max_ms"] = max(float(stats["max_ms"]), duration_ms)
		emit(f"{name}_completed", duration_ms=duration_ms, **fields)


def metric(name: str, value: int = 1, **fields) -> None:
	"""Increment a named counter and emit the measurement event."""
	with _metrics_lock:
		_counters[name] += value
	emit("metric", metric=name, value=value, **fields)


def snapshot() -> dict:
	"""Return a JSON-safe snapshot of counters and duration aggregates."""
	with _metrics_lock:
		return {
			"counters": dict(_counters),
			"durations": {key: dict(value) for key, value in _durations.items()},
		}
