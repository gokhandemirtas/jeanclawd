"""Canonical interaction recording and deferred user-memory extraction.

Interactions are evidence; MemoryFacts are curated knowledge.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from utils.inference_gate import background_inference, foreground_active
from utils.log import logger

_IDLE_SECONDS = 600
_POLL_SECONDS = 30
_MAX_BATCH = 20
_PROCESSING_TIMEOUT_SECONDS = 900
_worker_task: asyncio.Task | None = None

_PERSONAL_SIGNAL = re.compile(
	r"\b(i(?:'m| am| live| moved| work| like| love| hate| enjoy| prefer| have| speak| play| read| listen)|"
	r"my (?:name|home|city|job|work|wife|husband|partner|kids?|diet|favo(?:u)?rite|hobby|interest|language))\b",
	re.IGNORECASE,
)


class _ExtractedFact(BaseModel):
	namespace: str = Field(description="identity, preference, interest, relationship, or context")
	key: str = Field(description="stable snake_case attribute name")
	value: str = Field(description="short factual value stated by the user")
	confidence: float = Field(ge=0, le=1)
	source_interaction_id: str | None = Field(
		default=None, description="ID of the interaction that explicitly stated this fact"
	)


class _ExtractedFacts(BaseModel):
	facts: list[_ExtractedFact] = Field(default_factory=list)


_EXTRACTION_PROMPT = """Extract durable, user-specific memory facts from these conversation turns.
Only use facts explicitly stated by the user. Never infer sensitive traits, political views,
religion, health, or personality. Ignore requests, temporary plans, facts about the world,
and anything said only by the assistant. Return no facts when uncertain.

Use concise snake_case keys. Use `identity` for name/home location/work; `preference` for
stable likes, dislikes, diet and settings; `interest` for recurring hobbies; `relationship`
for explicitly stated family/relationship context; `context` only for durable context.
Each turn includes an interaction ID. Set source_interaction_id to the ID of the turn
that explicitly stated each fact. Never use an ID that is not present in the input.
"""


async def record_interaction(
	question: str,
	answer: str,
	*,
	user_id: str = "default",
	session_id: str | None = None,
	intent: str | None = None,
) -> None:
	"""Persist a completed turn immediately, then add durable deferred work."""
	from db.schemas import Interaction, MemoryWorkItem

	if session_id is None:
		from utils.session_service import get_active_session_id

		session_id = str(get_active_session_id() or "default")
	now = datetime.now()
	interaction = Interaction(
		user_id=user_id,
		session_id=session_id,
		created_at=now,
		user_message=question,
		assistant_message=answer,
		intent=intent,
	)
	await interaction.insert()
	await MemoryWorkItem(
		user_id=user_id, interaction_id=str(interaction.id), created_at=now
	).insert()
	from utils.session_service import touch_session

	await touch_session()
	from utils.session_service import maybe_generate_session_title

	await maybe_generate_session_title(question, answer)

async def _extract(interactions: list) -> list[_ExtractedFact]:
	candidates = [i for i in interactions if _PERSONAL_SIGNAL.search(i.user_message)]
	if not candidates:
		return []
	turns = "\n\n".join(
		f"Interaction ID: {i.id}\nUser: {i.user_message}\nAssistant: {i.assistant_message}"
		for i in candidates
	)
	from utils.llm_structured_output import generate_structured_output

	result = await asyncio.to_thread(
		generate_structured_output,
		model_name=os.environ["MODEL_GENERALIST"],
		user_prompt=turns,
		system_prompt=_EXTRACTION_PROMPT,
		pydantic_model=_ExtractedFacts,
		temperature=0.1,
		max_tokens=512,
	)
	valid = {"identity", "preference", "interest", "relationship", "context"}
	valid_ids = {str(i.id) for i in candidates}
	if len(candidates) == 1:
		for fact in result.facts:
			if fact.source_interaction_id is None:
				fact.source_interaction_id = str(candidates[0].id)
	return [
		fact
		for fact in result.facts
		if fact.namespace in valid
		and fact.key
		and fact.value
		and fact.source_interaction_id in valid_ids
	]


async def _upsert_fact(user_id: str, fact: _ExtractedFact, source_ids: list[str]) -> None:
	from db.schemas import MemoryFact

	now = datetime.now()
	existing = await MemoryFact.find_one(
		MemoryFact.user_id == user_id,
		MemoryFact.key == fact.key,
		MemoryFact.value == fact.value,
		MemoryFact.status == "active",
	)
	if existing:
		existing.last_confirmed_at = now
		existing.confidence = max(existing.confidence, fact.confidence)
		existing.source_interaction_ids = list(dict.fromkeys(existing.source_interaction_ids + source_ids))
		await existing.save()
		from utils.memory_index import index_fact
		await asyncio.to_thread(index_fact, existing)
		return

	prior = await MemoryFact.find(
		MemoryFact.user_id == user_id,
		MemoryFact.key == fact.key,
		MemoryFact.status == "active",
	).to_list()
	for old in prior:
		old.status = "superseded"
		await old.save()
		from utils.memory_index import index_fact
		await asyncio.to_thread(index_fact, old)

	stored = MemoryFact(
		user_id=user_id,
		namespace=fact.namespace,
		key=fact.key,
		value=fact.value,
		confidence=fact.confidence,
		source_interaction_ids=source_ids,
		first_observed_at=now,
		last_confirmed_at=now,
	)
	await stored.insert()
	from utils.memory_index import index_fact
	await asyncio.to_thread(index_fact, stored)


async def process_pending_memory() -> int:
	"""Coalesce sufficiently idle turns into one gated extraction pass."""
	from db.schemas import Interaction, MemoryWorkItem

	if foreground_active():
		return 0
	cutoff = datetime.now() - timedelta(seconds=_IDLE_SECONDS)
	processing_cutoff = datetime.now() - timedelta(seconds=_PROCESSING_TIMEOUT_SECONDS)
	stale_processing = await MemoryWorkItem.find(
		MemoryWorkItem.status == "processing",
		MemoryWorkItem.created_at <= processing_cutoff,
	).to_list()
	for item in stale_processing:
		item.status = "pending"
		await item.save()
	items = await MemoryWorkItem.find(
		MemoryWorkItem.status == "pending", MemoryWorkItem.created_at <= cutoff
	).sort("created_at").limit(_MAX_BATCH).to_list()
	if not items:
		return 0
	async with background_inference() as acquired:
		if not acquired:
			return 0
		try:
			# Claim the whole batch before doing inference. There is only one in-process
			# worker, and durable status keeps completed work from being reprocessed.
			for item in items:
				item.status = "processing"
				item.attempts += 1
				await item.save()

			interactions = []
			for item in items:
				interaction = await Interaction.get(item.interaction_id)
				if interaction:
					interactions.append(interaction)
			all_facts = 0
			for user_id in {item.user_id for item in items}:
				user_interactions = [i for i in interactions if i.user_id == user_id]
				facts = await _extract(user_interactions)
				for fact in facts:
					await _upsert_fact(user_id, fact, [fact.source_interaction_id])
				all_facts += len(facts)
			for item in items:
				item.status = "complete"
				item.error = None
				await item.save()
			logger.debug(f"[memory] processed {len(items)} interaction(s), stored {all_facts} fact(s)")
			return all_facts
		except Exception as exc:
			for item in items:
				item.status = "error" if item.attempts >= 3 else "pending"
				item.error = str(exc)[:500]
				await item.save()
			logger.exception("[memory] deferred extraction failed")
			return 0


async def _worker_loop() -> None:
	while True:
		try:
			await process_pending_memory()
		except asyncio.CancelledError:
			raise
		except Exception:
			logger.exception("[memory] worker poll failed")
		await asyncio.sleep(_POLL_SECONDS)


def start_memory_worker() -> None:
	global _worker_task
	if _worker_task is None or _worker_task.done():
		_worker_task = asyncio.create_task(_worker_loop(), name="memory-worker")


async def list_memories(query: str = "", *, user_id: str = "default", limit: int = 100) -> list:
	from db.schemas import MemoryFact
	facts = await MemoryFact.find(
		MemoryFact.user_id == user_id, MemoryFact.status == "active"
	).sort("-last_confirmed_at").to_list()
	if not query:
		return facts[:limit]
	needle = query.casefold()
	return [
		fact for fact in facts
		if needle in fact.key.casefold() or needle in fact.value.casefold()
	][:limit]


async def search_relevant_memories(query: str, user_id: str = "default", limit: int = 5) -> list[str]:
	from utils.memory_index import search_facts

	results = await asyncio.to_thread(search_facts, query, user_id, limit)
	if results:
		return results
	# Qdrant is only a projection; preserve basic recall from authoritative MongoDB.
	facts = await list_memories(query, user_id=user_id, limit=limit)
	return [f"{fact.key}: {fact.value}" for fact in facts]


async def rebuild_memory_index(user_id: str | None = None) -> int:
	"""Rebuild Qdrant from active MongoDB facts, which remains authoritative."""
	from db.schemas import MemoryFact
	from utils.memory_index import rebuild_index

	query = MemoryFact.find(MemoryFact.status == "active")
	if user_id is not None:
		query = MemoryFact.find(
			MemoryFact.user_id == user_id,
			MemoryFact.status == "active",
		)
	facts = await query.to_list()
	return await asyncio.to_thread(rebuild_index, facts)


async def get_fact_value(key: str, user_id: str = "default") -> str | None:
	from db.schemas import MemoryFact

	fact = await MemoryFact.find_one(
		MemoryFact.user_id == user_id,
		MemoryFact.key == key,
		MemoryFact.status == "active",
	)
	return fact.value if fact else None


async def get_profile_context(user_id: str = "default", limit: int = 20) -> str:
	"""Render active memory facts as conservative background context for CHAT."""
	from db.schemas import MemoryFact

	facts = await MemoryFact.find(
		MemoryFact.user_id == user_id, MemoryFact.status == "active"
	).sort("-last_confirmed_at").limit(limit).to_list()
	if not facts:
		return ""
	lines = [
		"## What you know about the user:",
		"Use these facts only when relevant. Do not volunteer them unprompted.",
	]
	lines.extend(f"- {fact.key.replace('_', ' ').title()}: {fact.value}" for fact in facts)
	return "\n".join(lines)
