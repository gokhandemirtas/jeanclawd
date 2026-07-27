"""Canonical interaction recording and deferred user-memory extraction.

Interactions are evidence; MemoryFacts are curated knowledge.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, model_validator

from utils.inference_gate import background_inference, foreground_active
from utils.log import logger

_IDLE_SECONDS = 600
_POLL_SECONDS = 30
_MAX_BATCH = 50
_PROCESSING_TIMEOUT_SECONDS = 900
_worker_task: asyncio.Task | None = None

_PERSONAL_SIGNAL = re.compile(
	r"\b(?:i(?:'m| am| live| moved| work| like| love| hate| enjoy| prefer| have| speak| play| read| listen)|"
	r"i\s+(?:also\s+)?(?:like|love|enjoy|prefer)|"
	r"my (?:name|home|city|job|work|wife|husband|partner|kids?|diet|favo(?:u)?rite|hobby|interest|language|birth(?:day| date)|gender)|"
	r"i was born)\b",
	re.IGNORECASE,
)
_NAME_WITH_QUESTION = re.compile(
	r"^\s*([A-Z][A-Za-z'-]{1,30})[,.!?]\s*(?:what(?:'s| is) your name|who are you)\b",
	re.IGNORECASE,
)
_EXPLICIT_PREFERENCE = re.compile(
	r"\bI\s+(?:(?P<negative>don't|do not|dislike|hate)\s+)?"
	r"(?P<verb>like|love|enjoy|prefer)\s+(?P<value>[^.!?]+)",
	re.IGNORECASE,
)
_EXPLICIT_BIRTH_DATE = re.compile(
	r"\b(?:i was born on|my birth date is|my birthday is)\s*:?[\s]*"
	r"(?P<value>(?:[A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4}))",
	re.IGNORECASE,
)
_INVALID_FACT_VALUES = frozenset({
	"me", "myself", "i", "you", "yourself", "the assistant", "assistant",
	"hello", "hi", "hey", "there", "something", "things", "it", "this", "that",
	"mine", "my own",
})
_SECOND_PERSON_VALUE = re.compile(
	r"^(?:you|your|yourself|the assistant|assistant)$", re.IGNORECASE
)
_FIRST_PERSON_SUBJECT = re.compile(
	r"^(?:i|i['’]m|i am|my|we|we['’]re|we are)\b", re.IGNORECASE
)


def _clean_fact_value(value: str) -> str:
	return " ".join(value.strip(" \t\r\n.,;:!?()[]{}\"'").split())


def _usable_fact_value(value: str) -> bool:
	compact = " ".join(value.casefold().split())
	return bool(compact) and compact not in _INVALID_FACT_VALUES and not _SECOND_PERSON_VALUE.fullmatch(compact)


def _durable_fact_value(value: str, *, namespace: str, key: str) -> bool:
	"""Reject model/entity output that is too small to be useful memory.

	Single words are valid for names, locations, and explicit preferences, but
	free-floating one-word interest/context values are usually entity-extractor
	noise (for example, a pronoun or a word copied from the assistant's turn).
	Those values should not reach MongoDB even if an upstream model emits them.
	"""
	value = _clean_fact_value(value)
	if not _usable_fact_value(value) or len(value) < 2:
		return False
	if namespace in {"interest", "context"} and key not in {"name", "location"}:
		return len(value.split()) >= 2
	return True


def _retrievable_fact(key: str, value: str) -> bool:
	"""Apply the same quality policy when reading older stored/indexed facts."""
	namespace = "interest" if key in {"interest", "work_on"} else "context"
	if key in {"name", "birth_date", "gender", "location", "home_location", "city"}:
		namespace = "identity"
	elif key in {"likes", "dislikes", "diet", "settings"}:
		namespace = "preference"
	return _durable_fact_value(value, namespace=namespace, key=key)


def _grounded_in_user_message(value: str, message: str) -> bool:
	"""Require extracted values to be present in the user's own words."""
	compact_value = _clean_fact_value(value).casefold()
	compact_message = " ".join(message.casefold().split())
	return (
		bool(compact_value)
		and _usable_fact_value(compact_value)
		and compact_value in compact_message
	)


def _explicitly_supported_fact(namespace: str, key: str, value: str, message: str) -> bool:
	"""Require a value to occur in the right first-person claim, not anywhere."""
	value = _clean_fact_value(value)
	message_casefolded = message.casefold()
	value_casefolded = value.casefold()
	value_start = message_casefolded.find(value_casefolded)
	if value_start < 0:
		return False
	prefix = message_casefolded[max(0, value_start - 100):value_start]

	if namespace == "preference":
		# Preferences about the Assistant are not user preferences. Also avoid
		# accepting a truncated relation tail from a larger object phrase.
		if re.search(r"\byour\b|\bthe assistant\b", prefix):
			return False
		return bool(re.search(r"\b(?:i|i['’]m|i am)\s+(?:also\s+)?(?:don't|do not|dislike|hate|like|love|enjoy|prefer)\s+(?:[^.!?]*\s+)?$", prefix))

	if namespace == "identity" and key in {"location", "home_location", "city"}:
		return bool(re.search(r"\b(?:i\s+(?:live|moved|am based)|my\s+(?:home|city|location))\s+(?:in|is|to)?\s*$", prefix))
	if namespace == "identity" and key == "name":
		return bool(re.search(r"\b(?:my name is|call me|i am|i['’]m)\s*$", prefix))
	if namespace == "interest" and key == "work_on":
		return bool(re.search(r"\b(?:i\s+(?:work|am working)\s+on|i['’]m\s+working\s+on|i\s+build)\s*$", prefix))
	if namespace == "interest":
		return bool(re.search(r"\b(?:my\s+(?:hobby|interest)\s+is|i am interested in|i enjoy)\s*$", prefix))
	return False


class _ExtractedFact(BaseModel):
	namespace: str = Field(
		default="", description="identity, preference, interest, relationship, or context"
	)
	key: str = Field(default="", description="stable snake_case attribute name")
	value: str = Field(default="", description="short factual value stated by the user")
	confidence: float = Field(default=0.8, ge=0, le=1)
	source_interaction_id: str | None = Field(
		default=None, description="ID of the interaction that explicitly stated this fact"
	)

	@model_validator(mode="before")
	@classmethod
	def normalize_compact_fact(cls, data: object) -> object:
		"""Normalize common small-model omissions/compressions before validation."""
		if not isinstance(data, dict):
			return data
		item = dict(data)
		namespaces = {"identity", "preference", "interest", "relationship", "context"}
		value = item.get("value")

		# Some local models compress {namespace, key, value} into:
		# {key: "identity", value: {name: "..."}}.
		if not item.get("namespace") and item.get("key") in namespaces:
			item["namespace"] = item["key"]
			item["key"] = ""
		if isinstance(value, dict):
			if not item.get("key") and len(value) == 1:
				item["key"], value = next(iter(value.items()))
			item["value"] = value
		if item.get("value") is not None and not isinstance(item["value"], str):
			item["value"] = str(item["value"])
		return item


class _ExtractedFacts(BaseModel):
	facts: list[_ExtractedFact] = Field(default_factory=list)


_EXTRACTION_PROMPT = """Extract durable, user-specific memory facts from these conversation turns.
Only use facts explicitly stated in the User line. The Assistant line is untrusted context and
must never be treated as evidence about the user. In particular, do not interpret the
assistant's first-person statements, guesses, jokes, questions, or claims as user facts.
Keep the perspectives separate: the User is the human; the Assistant is the AI. A User statement
such as "you are a master developer" describes the Assistant and is not a fact about the User.
A statement such as "I like you" records the User's feeling toward the Assistant, not an
Assistant trait. Never store facts about what the Assistant likes, is, knows, or can do as User facts.
Grammatical rule: first-person words ("I", "me", "my", "I'm") in the User line refer to the
human User; second-person words ("you", "your", "you're") refer to the Assistant. Therefore,
"I am the master developer" is a User identity/context fact, while "you are the master developer"
describes the Assistant. Preserve that direction exactly.
Every returned value must be copied from the corresponding User line and must be attributable
to that user's wording. Never infer sensitive traits, political views, religion, health, or
personality. Ignore requests, temporary plans, facts about the world, and anything said only
by the assistant. Do not extract facts from greetings, questions such as "remember me?", or
pronouns alone. Return no facts when uncertain.

Use concise snake_case keys. Use `identity` for name, birth_date, gender, home location/work; `preference` for
stable likes, dislikes, diet and settings; `interest` for recurring hobbies; `relationship`
for explicitly stated family/relationship context; `context` only for durable context.
Each turn includes an interaction ID. Set source_interaction_id to the ID of the turn
that explicitly stated each fact. Never use an ID that is not present in the input.
If a user begins a turn with a name followed by a question to the assistant (for
example, "Gokhan. what's your name?"), treat that leading name as the user's
identity name and return it as identity/name.
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
	candidates = []
	signals_by_id = {}
	from utils.entity_extractor import extract_user_signals
	for interaction in interactions:
		try:
			signals_by_id[str(interaction.id)] = await asyncio.to_thread(
				extract_user_signals, interaction.user_message
			)
		except Exception as exc:
			logger.warning(f"[memory] GLiNER2 unavailable ({exc}); using fallback signals")
			signals_by_id[str(interaction.id)] = {"entities": {}, "relations": {}}
	for index, interaction in enumerate(interactions):
		signals = signals_by_id[str(interaction.id)]
		entities = signals["entities"].get("entities", {})
		relations = signals["relations"].get("relation_extraction", {})
		if (
			_PERSONAL_SIGNAL.search(interaction.user_message)
			or _NAME_WITH_QUESTION.search(interaction.user_message)
			or any(entities.values())
			or any(relations.values())
		):
			candidates.append(interaction)
			continue
		# A user's short answer can carry the fact when the preceding turn asks
		# for it (e.g. "what's my name?" → "Gokhan"). Include both turns so
		# extraction has the context without treating arbitrary short messages as
		# durable facts.
		if index and len(interaction.user_message.split()) <= 4:
			previous = interactions[index - 1].user_message.lower()
			if "my name" in previous or "what do you call me" in previous:
				candidates.append(interaction)
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
	else:
		# Small local models frequently omit provenance despite the prompt. Keep
		# the fact, but attach it to the candidate turn containing its value (or
		# the first candidate as a conservative fallback).
		for fact in result.facts:
			if fact.source_interaction_id is None:
				value = fact.value.lower()
				match = next(
					(i for i in candidates if value and value in i.user_message.lower()),
					candidates[0],
				)
				fact.source_interaction_id = str(match.id)
	facts = [
		fact
		for fact in result.facts
		if fact.namespace in valid
		and fact.key
		and _durable_fact_value(fact.value, namespace=fact.namespace, key=fact.key)
		and fact.source_interaction_id in valid_ids
		and any(
			str(interaction.id) == fact.source_interaction_id
			and _grounded_in_user_message(fact.value, interaction.user_message)
			and _explicitly_supported_fact(
				fact.namespace, fact.key, fact.value, interaction.user_message
			)
			for interaction in candidates
		)
	]
	# Convert high-confidence GLiNER2 relations into facts without asking the
	# generalist model to reproduce obvious user statements.
	relation_keys = {
		"likes": ("preference", "likes"),
		"dislikes": ("preference", "dislikes"),
		"lives in": ("identity", "location"),
		"works on": ("interest", "work_on"),
	}
	for interaction in candidates:
		relations = signals_by_id[str(interaction.id)]["relations"].get(
		"relation_extraction", {}
	)
		for relation, output in relations.items():
			mapped = relation_keys.get(relation)
			if not mapped:
				continue
			for item in output:
				head = item.get("head", {}).get("text", "").strip()
				tail = item.get("tail", {}).get("text", "").strip()
				if (
					not _FIRST_PERSON_SUBJECT.match(head)
					or not _durable_fact_value(tail, namespace=mapped[0], key=mapped[1])
					or not _grounded_in_user_message(tail, interaction.user_message)
					or not _explicitly_supported_fact(
						mapped[0], mapped[1], tail, interaction.user_message
					)
				):
					continue
				if not any(f.namespace == mapped[0] and f.key == mapped[1] and f.value.lower() == tail.lower() for f in facts):
					facts.append(_ExtractedFact(namespace=mapped[0], key=mapped[1], value=tail, confidence=1.0, source_interaction_id=str(interaction.id)))
	# Preserve an unambiguous name supplied in the conversational form
	# "Gokhan. what's your name?" even if the local model returns no fact.
	for interaction in candidates:
		birth = _EXPLICIT_BIRTH_DATE.search(interaction.user_message)
		if birth and not any(f.namespace == "identity" and f.key == "birth_date" for f in facts):
			facts.append(
				_ExtractedFact(
					namespace="identity",
					key="birth_date",
					value=birth.group("value"),
					confidence=1.0,
					source_interaction_id=str(interaction.id),
				)
			)
		match = _NAME_WITH_QUESTION.match(interaction.user_message)
		if match and not any(f.namespace == "identity" and f.key == "name" for f in facts):
			facts.append(
				_ExtractedFact(
					namespace="identity",
					key="name",
					value=match.group(1),
					confidence=1.0,
					source_interaction_id=str(interaction.id),
				)
			)
		for match in _EXPLICIT_PREFERENCE.finditer(interaction.user_message):
			value = match.group("value").strip(" ,;")
			key = "dislikes" if match.group("negative") else "likes"
			if not _durable_fact_value(value, namespace="preference", key=key):
				continue
			if not any(
				f.namespace == "preference" and f.key == key and f.value.lower() == value.lower()
				for f in facts
			):
				facts.append(
					_ExtractedFact(
						namespace="preference",
						key=key,
						value=value,
						confidence=1.0,
						source_interaction_id=str(interaction.id),
					)
				)
	return facts


async def _upsert_fact(
	user_id: str, fact: _ExtractedFact, source_ids: list[str]
) -> None:
	from db.schemas import MemoryFact

	# Keep this guard at the persistence boundary as well: extraction providers
	# can change independently and callers may construct _ExtractedFact values.
	if not _durable_fact_value(fact.value, namespace=fact.namespace, key=fact.key):
		return
	fact.value = _clean_fact_value(fact.value)

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
		existing.source_interaction_ids = list(
			dict.fromkeys(existing.source_interaction_ids + source_ids)
		)
		await existing.save()
		from utils.memory_index import index_fact

		await asyncio.to_thread(index_fact, existing)
		await _update_profile(user_id, fact)
		return

	# Identity attributes are single-valued and should supersede an older value
	# (for example, a new home location). Preferences/interests are sets: liking
	# death metal must not remove an existing liking for Italian food.
	single_valued = {
		"name",
		"birth_date",
		"gender",
		"location",
		"home_location",
		"city",
	}
	prior = []
	if fact.key in single_valued:
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
	await _update_profile(user_id, fact)


async def _update_profile(user_id: str, fact: _ExtractedFact) -> None:
	"""Maintain the small identity projection without replacing MemoryFact."""
	from db.schemas import UserProfile

	if fact.namespace != "identity" or fact.key not in {
		"name",
		"birth_date",
		"gender",
		"location",
		"home_location",
		"city",
	}:
		return
	profile = await UserProfile.find_one(UserProfile.user_id == user_id)
	if profile is None:
		profile = UserProfile(user_id=user_id)
	if fact.key == "name":
		profile.display_name = fact.value
	elif fact.key == "birth_date":
		profile.birth_date = fact.value
	elif fact.key == "gender":
		profile.gender = fact.value
	elif fact.key in {"location", "home_location", "city"}:
		profile.location = fact.value
	profile.updated_at = datetime.now()
	await profile.save()


async def process_pending_memory(
	*, force: bool = False, include_complete: bool = False
) -> int:
	"""Coalesce pending turns into one gated extraction pass.

	Normally only turns older than the idle window are processed.  ``force`` is
	used by the CLI test command to run the same pipeline immediately.
	"""
	from db.schemas import Interaction, MemoryWorkItem

	if foreground_active():
		return 0
	cutoff = (
		datetime.now() if force else datetime.now() - timedelta(seconds=_IDLE_SECONDS)
	)
	processing_cutoff = datetime.now() - timedelta(seconds=_PROCESSING_TIMEOUT_SECONDS)
	stale_processing = await MemoryWorkItem.find(
		MemoryWorkItem.status == "processing",
		MemoryWorkItem.created_at <= processing_cutoff,
	).to_list()
	for item in stale_processing:
		item.status = "pending"
		await item.save()
	if include_complete:
		# Always process newly pending work first.  Sorting a combined pending /
		# complete query can fill the batch with old completed items and starve
		# the interaction the user is currently testing.
		items = (
			await MemoryWorkItem.find(
				MemoryWorkItem.status == "pending", MemoryWorkItem.created_at <= cutoff
			)
			.sort("created_at")
			.limit(_MAX_BATCH)
			.to_list()
		)
		if len(items) < _MAX_BATCH:
			historical = (
				await MemoryWorkItem.find(
					MemoryWorkItem.status == "complete", MemoryWorkItem.created_at <= cutoff
				)
				.sort("created_at")
				.limit(_MAX_BATCH - len(items))
				.to_list()
			)
			items.extend(historical)
	else:
		items = (
			await MemoryWorkItem.find(
				MemoryWorkItem.status == "pending", MemoryWorkItem.created_at <= cutoff
			)
			.sort("created_at")
			.limit(_MAX_BATCH)
			.to_list()
		)
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
			logger.debug(
				f"[memory] processed {len(items)} interaction(s), stored {all_facts} fact(s)"
			)
			return all_facts
		except Exception as exc:
			for item in items:
				item.status = "error" if item.attempts >= 3 else "pending"
				item.error = str(exc)[:500]
				await item.save()
			logger.exception("[memory] deferred extraction failed")
			if force:
				raise
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


async def list_memories(
	query: str = "", *, user_id: str = "default", limit: int = 100
) -> list:
	from db.schemas import MemoryFact

	facts = (
		await MemoryFact.find(MemoryFact.user_id == user_id, MemoryFact.status == "active")
		.sort("-last_confirmed_at")
		.to_list()
	)
	facts = [fact for fact in facts if _retrievable_fact(fact.key, fact.value)]
	if not query:
		return facts[:limit]
	needle = query.casefold()
	return [
		fact
		for fact in facts
		if needle in fact.key.casefold() or needle in fact.value.casefold()
	][:limit]


async def search_relevant_memories(
	query: str, user_id: str = "default", limit: int = 5
) -> list[str]:
	from utils.memory_index import search_facts

	results = await asyncio.to_thread(search_facts, query, user_id, limit)
	if results:
		return [
			item for item in results
			if (
				_retrievable_fact(item.split(":", 1)[0].strip(), item.split(":", 1)[1].strip())
				if ":" in item
				else _usable_fact_value(item)
			)
		]
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


async def get_profile_context(user_id: str = "default", limit: int = 50) -> str:
	"""Render active memory facts as conservative background context for CHAT."""
	from db.schemas import MemoryFact

	facts = (
		await MemoryFact.find(MemoryFact.user_id == user_id, MemoryFact.status == "active")
		.sort("-last_confirmed_at")
		.limit(limit)
		.to_list()
	)
	facts = [fact for fact in facts if _retrievable_fact(fact.key, fact.value)]
	if not facts:
		return ""
	lines = [
		"## USER PROFILE — facts about the human user only:",
		"Perspective rule: every fact below belongs to the User, never to the Assistant. For example, '- User likes X' means the human likes X. Do not reinterpret these as the Assistant's traits or preferences.",
		"Use these facts only when relevant. Do not volunteer them unprompted.",
	]
	lines.extend(
		f"- User's {fact.key.replace('_', ' ').lower()}: {fact.value}" for fact in facts
	)
	return "\n".join(lines)
