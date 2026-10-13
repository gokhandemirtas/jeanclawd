"""Agent pipeline state — written by the pipeline, read by the TUI.

A single module-level dataclass updated synchronously by the pipeline worker
and polled by the TUI's set_interval timer.  No locks needed: both reader and
writer run in the same asyncio event-loop thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineState:
    current: str = "idle"
    label: str = ""            # e.g. "WEATHER" when current == "skill"
    completed: set[str] = field(default_factory=set)
    error: bool = False


_state = PipelineState()


def set_node(node: str, label: str = "") -> None:
    prev = _state.current
    if prev not in ("idle", node):
        _state.completed.add(prev)
    _state.current = node
    _state.label = label
    _state.error = False


def set_error() -> None:
    _state.error = True


def reset() -> None:
    _state.current = "idle"
    _state.label = ""
    _state.completed.clear()
    _state.error = False


def snapshot() -> PipelineState:
    return PipelineState(
        current=_state.current,
        label=_state.label,
        completed=set(_state.completed),
        error=_state.error,
    )
