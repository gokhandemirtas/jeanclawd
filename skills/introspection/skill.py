"""Answer questions from the application's persisted self-inspection records."""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from utils.log import logger
from utils.self_inspector import list_reflections, read_reflection

_BATCH_CHARS = 24_000


class _IntrospectionAnswer(BaseModel):
	answer: str = Field(description="A direct, evidence-based answer to the user's question")


def _record_batches(question: str) -> list[str]:
	blocks: list[str] = []
	for path in list_reflections(limit=100):
		try:
			record = read_reflection(path.rsplit("/", 1)[-1], max_bytes=64_000)
		except (OSError, ValueError, PermissionError) as exc:
			logger.info(f"[introspection] Skipping {path}: {exc}")
			continue
		blocks.append(f"\n--- {path} ---\n{record}")

	batches: list[str] = []
	current: list[str] = []
	current_chars = 0
	for block in blocks:
		if current and current_chars + len(block) > _BATCH_CHARS:
			batches.append("\n".join(current))
			current = []
			current_chars = 0
		current.append(block)
		current_chars += len(block)
	if current:
		batches.append("\n".join(current))
	return batches


async def _answer_batch(question: str, batch: str) -> str:
	from utils.llm_structured_output import generate_structured_output

	result = await asyncio.to_thread(
		generate_structured_output,
		model_name=os.environ["MODEL_GENERALIST"],
		user_prompt=(
			f"Question about this Assistant: {question}\n\n"
			"Answer using only the persisted introspection records below. "
			"Separate established facts from uncertainty. Do not invent capabilities, policies, "
			"retention periods, or implementation details.\n"
			+ batch
		),
		system_prompt="You answer questions about an application's implementation from evidence supplied by that application.",
		pydantic_model=_IntrospectionAnswer,
		max_tokens=1_024,
		temperature=0.1,
	)
	return result.answer


async def handle_introspection(question: str) -> str:
	"""Answer an implementation question without consulting external sources."""
	from utils.llm_structured_output import generate_structured_output

	batches = _record_batches(question)
	if not batches:
		return "I do not have any completed introspection records to answer that implementation question yet."

	try:
		batch_answers = [await _answer_batch(question, batch) for batch in batches]
		if len(batch_answers) == 1:
			return batch_answers[0]
		result = await asyncio.to_thread(
			generate_structured_output,
			model_name=os.environ["MODEL_GENERALIST"],
			user_prompt=(
				f"Question about this Assistant: {question}\n\n"
				"Synthesize a direct answer from these evidence-based analyses. "
				"Keep facts separate from uncertainty and do not add details absent from the analyses.\n\n"
				+ "\n\n".join(batch_answers)
			),
			system_prompt="You synthesize evidence-based answers about an application's implementation.",
			pydantic_model=_IntrospectionAnswer,
			max_tokens=1_024,
			temperature=0.1,
		)
		return result.answer
	except Exception:
		logger.exception("[introspection] Failed to answer implementation question")
		return "I could not read the introspection records reliably enough to answer that without guessing."
