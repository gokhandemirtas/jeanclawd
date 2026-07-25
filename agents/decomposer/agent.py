from __future__ import annotations

import asyncio
from pydantic import BaseModel, field_validator

from agents.intent_classifier.intent_classifier import IntentLiteral
from agents.decomposer.prompt import DECOMPOSITION_PROMPT
from providers import get_intent_provider
from utils import logger

MAX_TASKS = 8


class DecomposedTask(BaseModel):
	description: str
	intent: IntentLiteral
	query: str


class DecompositionResult(BaseModel):
	is_compound: bool
	tasks: list[DecomposedTask]

	@field_validator("tasks")
	@classmethod
	def require_task(cls, value: list[DecomposedTask]) -> list[DecomposedTask]:
		if not value:
			raise ValueError("decomposition must contain at least one task")
		return value[:MAX_TASKS]


def _entity_hint(text: str) -> str:
	"""Best-effort GLiNER context; decomposition must work without GLiNER."""
	try:
		from utils.entity_extractor import extract_user_signals

		signals = extract_user_signals(text)
		entities = signals.get("entities", [])
		if entities:
			return f"Optional entity hints (do not segment from these): {entities}"
	except Exception as exc:  # GLiNER is an optional, heavyweight supporting model.
		logger.debug(f"[decomposer] GLiNER hints unavailable: {exc}")
	return ""


def _normalize(result: DecompositionResult, original: str) -> DecompositionResult:
	# Keep the model's result safe even when a permissive provider returns a stale
	# compound flag or more than the schema limit.
	result.tasks = result.tasks[:MAX_TASKS]
	result.is_compound = len(result.tasks) > 1
	if len(result.tasks) == 1 and not result.tasks[0].query.strip():
		result.tasks[0].query = original
	return result


def decompose_request_sync(user_input: str) -> DecompositionResult:
	hints = _entity_hint(user_input)
	content = user_input if not hints else f"{user_input}\n\n{hints}"
	try:
		result = get_intent_provider().complete(
			[
				{"role": "system", "content": DECOMPOSITION_PROMPT},
				{"role": "user", "content": content},
			],
			DecompositionResult,
		)
		return _normalize(result, user_input)
	except Exception as exc:
		logger.warning(f"[decomposer] Decomposition failed, using one task: {exc}")
		# Failure must not change existing single-request behavior: intent is still
		# selected by the existing router later in the pipeline.
		return DecompositionResult(
			is_compound=False,
			tasks=[DecomposedTask(description=user_input, intent="CHAT", query=user_input)],
		)


async def decompose_request(user_input: str) -> DecompositionResult:
	return await asyncio.to_thread(decompose_request_sync, user_input)
