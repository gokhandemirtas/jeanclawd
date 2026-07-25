"""Process-local confirmation state for compound requests."""

from agents.decomposer.agent import DecomposedTask

_pending: list[DecomposedTask] | None = None


def set_pending(tasks: list[DecomposedTask]) -> None:
	global _pending
	_pending = list(tasks)


def take_pending() -> list[DecomposedTask] | None:
	global _pending
	plan = _pending
	_pending = None
	return plan


def clear_pending() -> None:
	global _pending
	_pending = None


def has_pending() -> bool:
	return _pending is not None


def is_confirmation(text: str) -> bool:
	return text.strip().lower() in {"y", "yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "go ahead"}


def is_cancellation(text: str) -> bool:
	return text.strip().lower() in {"n", "no", "nope", "cancel", "stop", "never mind", "nevermind"}
