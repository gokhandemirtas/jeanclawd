from __future__ import annotations

import asyncio
import re
from pydantic import BaseModel, field_validator

from agents.intent_classifier.intent_classifier import IntentLiteral
from agents.decomposer.prompt import DECOMPOSITION_PROMPT
from providers import get_intent_provider
from utils import logger

MAX_TASKS = 8
_BATCH_REQUEST_RE = re.compile(r"^\s*batch\s+request\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
_EXPLICIT_COMPOUND_RE = re.compile(
	r"(?:;|\n|\?|\b(?:and|also|then|plus|as well|followed by)\b)",
	re.IGNORECASE,
)


def extract_batch_request(text: str) -> str | None:
	"""Return the batch payload only when the user explicitly requests batching."""
	match = _BATCH_REQUEST_RE.match(text)
	if not match:
		return None
	return match.group(1).strip()

_INTENT_ALIASES: dict[str, IntentLiteral] = {
	"turn_light": "HOME_CONTROL",
	"turn_lights": "HOME_CONTROL",
	"home_control": "HOME_CONTROL",
	"set_reminder": "REMINDER",
	"create_reminder": "REMINDER",
	"research_topic": "SEARCH",
	"deep_research": "SEARCH",
	"search_topic": "SEARCH",
	"weather_lookup": "WEATHER",
	"stock_lookup": "FINANCE_STOCKS",
	"play_music": "SPOTIFY",
	"take_note": "NOTES",
	"recipe_lookup": "NUTRITION",
}


class DecomposedTask(BaseModel):
	description: str
	intent: IntentLiteral
	query: str

	@field_validator("intent", mode="before")
	@classmethod
	def normalize_intent_alias(cls, value: object) -> object:
		if isinstance(value, str):
			return _INTENT_ALIASES.get(value.strip().lower(), value)
		return value


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
	# Do not let the model invent multiple intents for a single sentence. A
	# A comma is ordinary conversational punctuation, not enough evidence that
	# the model found multiple requests. Semicolons, line breaks, question
	# boundaries, and explicit sequencing/conjunction words are stronger signals.
	if len(result.tasks) > 1 and not _EXPLICIT_COMPOUND_RE.search(original):
		result.tasks = [
			DecomposedTask(description=original, intent="CHAT", query=original)
		]
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
