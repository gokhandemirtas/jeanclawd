"""Agent pipeline state — written by the pipeline, read by the TUI.

A single module-level dataclass updated synchronously by the pipeline worker
and polled by the TUI's set_interval timer.  No locks needed: both reader and
writer run in the same asyncio event-loop thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PipelineState:
    current: str = "idle"
    label: str = ""
    completed: set[str] = field(default_factory=set)
    error: bool = False
    node_times: dict[str, float] = field(default_factory=dict)  # node_id → seconds
    total_time: float = 0.0   # duration of the last completed request
    step_log: list[str] = field(default_factory=list)  # ReAct / research steps


_state = PipelineState()
_node_start: float = 0.0
_request_start: float = 0.0


def set_node(node: str, label: str = "") -> None:
    global _node_start, _request_start
    now = time.monotonic()
    prev = _state.current

    # Clear research steps at the start of each new request
    if prev == "idle" and node != "idle":
        _state.step_log.clear()

    # Record how long the previous node ran
    if prev not in ("idle", node) and _node_start > 0:
        _state.node_times[prev] = now - _node_start
        _state.completed.add(prev)

    # The TUI shows the total alongside per-node timings.  Derive it from those
    # same measurements so it always matches their sum rather than including
    # unassigned scheduling/rendering gaps between node transitions.
    if node == "idle" and _request_start > 0:
        _state.total_time = sum(_state.node_times.values())

    _state.current = node
    _state.label = label
    _state.error = False
    _node_start = now

    # Record request start on first non-idle node after a reset
    if node != "idle" and _request_start == 0:
        _request_start = now


def set_error() -> None:
    _state.error = True


def add_step(msg: str) -> None:
    """Append a ReAct / research progress line for the TUI to display."""
    _state.step_log.append(msg)


def reset() -> None:
    global _node_start, _request_start
    _state.current = "idle"
    _state.label = ""
    _state.completed.clear()
    _state.error = False
    _state.node_times.clear()
    _state.total_time = 0.0
    _state.step_log.clear()
    _node_start = 0.0
    _request_start = 0.0


def snapshot() -> PipelineState:
    return PipelineState(
        current=_state.current,
        label=_state.label,
        completed=set(_state.completed),
        error=_state.error,
        node_times=dict(_state.node_times),
        total_time=_state.total_time,
        step_log=list(_state.step_log),
    )
