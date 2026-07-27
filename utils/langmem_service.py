"""LangMem integration for durable user profile memory.

Conversation turns remain in MongoDB as interaction history. LangMem maintains
one compact structured profile per user in a separate ``LangMem`` collection.
The MongoDB store is intentionally created without an index configuration, so
this integration does not enable embeddings or vector search.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from utils.inference_gate import background_inference
from utils.log import logger

_STORE_CONTEXT = None
_STORE = None
_MANAGER = None
_EXTRACTION_TASKS: set[asyncio.Task] = set()


class UserMemory(BaseModel):
	"""The single compact memory record LangMem maintains for each user."""

	name: str | None = Field(default=None, description="User's explicitly stated name")
	birth_date: str | None = Field(default=None, description="User's explicitly stated birth date")
	gender: str | None = Field(default=None, description="User's explicitly stated gender")
	location: str | None = Field(default=None, description="Where the user explicitly says they live")
	occupation: str | None = Field(default=None, description="User's explicitly stated occupation")
	likes: list[str] = Field(default_factory=list, description="Stable things the user explicitly likes")
	dislikes: list[str] = Field(default_factory=list, description="Stable things the user explicitly dislikes")
	interests: list[str] = Field(default_factory=list, description="Recurring interests explicitly stated by the user")
	relationships: list[str] = Field(default_factory=list, description="Explicitly stated family or relationship context")
	context: list[str] = Field(default_factory=list, description="Other durable user context explicitly stated by the user")


@dataclass(frozen=True)
class MemoryRecord:
	key: str
	value: str
	namespace: tuple[str, ...]


_INSTRUCTIONS = """Maintain one compact structured profile of the human User.

Only record information explicitly stated by the User message. The Assistant
message is context, never evidence. Keep the User and Assistant perspectives
separate: 'you', 'your', and claims about the Assistant are not facts about the
User. Ignore greetings, questions, jokes, temporary plans, incidental nouns,
and facts about the outside world. Do not infer personality, health, politics,
religion, or other sensitive traits. Prefer leaving a field unchanged over
guessing. Preserve meaningful phrases rather than splitting them into single
words. Merge duplicate list entries and remove entries that are not durable.
When the User explicitly corrects an existing fact, replace the old value.
"""


def _mongo_uri() -> str:
	uri = (
		f"mongodb://{os.environ['MONGODB_USER']}:{os.environ['MONGODB_PASSWORD']}@"
		f"{os.environ.get('MONGODB_HOST', 'localhost')}:{os.environ.get('MONGODB_PORT', '27017')}"
	)
	if "?" not in uri:
		uri += "?authSource=admin&serverSelectionTimeoutMS=5000&connectTimeoutMS=5000"
	return uri


def _store():
	global _STORE, _STORE_CONTEXT
	if _STORE is None:
		from langgraph.store.mongodb import MongoDBStore

		_STORE_CONTEXT = MongoDBStore.from_conn_string(
			conn_string=_mongo_uri(),
			db_name=os.environ.get("MONGODB_DB", "langbox"),
			collection_name="LangMem",
			# No index_config: this store uses ordinary MongoDB key/value access.
		)
		_STORE = _STORE_CONTEXT.__enter__()
	return _STORE


def _model():
	from langchain_openai import ChatOpenAI

	kwargs: dict[str, Any] = {
		"model": os.environ["MODEL_GENERALIST"],
		"api_key": os.environ.get("OPENAI_API_KEY", "lm-studio"),
		"temperature": 0,
	}
	if os.getenv("INFERENCE_PROVIDER", "lmstudio").lower() != "openai":
		kwargs["base_url"] = os.environ.get("MODEL_HOST", "http://localhost:1234/v1")
	return ChatOpenAI(**kwargs)


def _manager():
	global _MANAGER
	if _MANAGER is None:
		from langmem import create_memory_store_manager

		_MANAGER = create_memory_store_manager(
			_model(),
			schemas=[UserMemory],
			instructions=_INSTRUCTIONS,
			default_factory=lambda _config: UserMemory(),
			enable_inserts=False,
			enable_deletes=True,
			namespace=("memories", "{langgraph_user_id}"),
			store=_store(),
		)
	return _MANAGER


def _config(user_id: str) -> dict[str, dict[str, str]]:
	return {"configurable": {"langgraph_user_id": user_id}}


async def _extract_turn(
	question: str, answer: str, *, user_id: str, interaction_id: str
) -> None:
	try:
		async with background_inference() as acquired:
			if not acquired:
				return
			await _manager().ainvoke(
				{
					"messages": [
						{"role": "user", "content": question},
						{"role": "assistant", "content": answer},
					]
				},
				config=_config(user_id),
			)
		logger.debug(f"[langmem] processed interaction {interaction_id}")
	except asyncio.CancelledError:
		raise
	except Exception:
		logger.exception(f"[langmem] extraction failed for interaction {interaction_id}")


def _track_task(task: asyncio.Task) -> None:
	_EXTRACTION_TASKS.discard(task)
	if task.cancelled():
		return
	try:
		task.result()
	except Exception:
		logger.exception("[langmem] background extraction task failed")


async def record_interaction(
	question: str,
	answer: str,
	*,
	user_id: str = "default",
	session_id: str | None = None,
	intent: str | None = None,
) -> None:
	"""Persist the turn and submit it to LangMem in the background."""
	from db.schemas import Interaction

	if session_id is None:
		from utils.session_service import get_active_session_id

		session_id = str(get_active_session_id() or "default")
	interaction = Interaction(
		user_id=user_id,
		session_id=session_id,
		user_message=question,
		assistant_message=answer,
		intent=intent,
	)
	await interaction.insert()

	from utils.session_service import maybe_generate_session_title, touch_session

	await touch_session()
	await maybe_generate_session_title(question, answer)
	task = asyncio.create_task(
		_extract_turn(question, answer, user_id=user_id, interaction_id=str(interaction.id))
	)
	_EXTRACTION_TASKS.add(task)
	task.add_done_callback(_track_task)


async def _profile_item(user_id: str):
	items = await _store().asearch(("memories", user_id), limit=20)
	return items[0] if items else None


def _profile_from_value(value: dict[str, Any] | None) -> UserMemory:
	if not value:
		return UserMemory()
	content = value.get("content") if isinstance(value, dict) else None
	if isinstance(content, dict):
		value = content
	try:
		return UserMemory.model_validate(value)
	except Exception:
		return UserMemory()


def _profile_lines(profile: UserMemory) -> list[tuple[str, str]]:
	lines: list[tuple[str, str]] = []
	for key in ("name", "birth_date", "gender", "location", "occupation"):
		value = getattr(profile, key)
		if value:
			lines.append((key, value))
	for key in ("likes", "dislikes", "interests", "relationships", "context"):
		for value in getattr(profile, key):
			if isinstance(value, str) and value.strip():
				lines.append((key, value.strip()))
	return lines


async def list_memories(
	query: str = "", *, user_id: str = "default", limit: int = 100
) -> list[MemoryRecord]:
	item = await _profile_item(user_id)
	if item is None:
		return []
	needle = query.casefold().strip()
	records = [
		MemoryRecord(key=key, value=value, namespace=tuple(item.namespace))
		for key, value in _profile_lines(_profile_from_value(item.value))
	]
	if needle:
		records = [record for record in records if needle in f"{record.key}: {record.value}".casefold()]
	return records[: max(0, limit)]


async def search_relevant_memories(
	query: str, user_id: str = "default", limit: int = 5
) -> list[str]:
	"""Perform lightweight local key/value recall without vector search."""
	return [
		f"{record.key}: {record.value}"
		for record in await list_memories(query, user_id=user_id, limit=limit)
	]


async def get_fact_value(key: str, user_id: str = "default") -> str | None:
	for record in await list_memories(user_id=user_id, limit=100):
		if record.key == key:
			return record.value
	return None


async def get_profile_context(user_id: str = "default", limit: int = 50) -> str:
	records = await list_memories(user_id=user_id, limit=limit)
	if not records:
		return ""
	lines = [
		"## USER PROFILE — facts about the human user only:",
		"Perspective rule: every fact below belongs to the User, never to the Assistant.",
		"Use these facts only when relevant. Do not volunteer them unprompted.",
	]
	lines.extend(f"- User's {record.key.replace('_', ' ')}: {record.value}" for record in records)
	return "\n".join(lines)


async def forget_memory(field: str, *, user_id: str = "default") -> bool:
	item = await _profile_item(user_id)
	if item is None or field not in UserMemory.model_fields:
		return False
	profile = _profile_from_value(item.value)
	if isinstance(getattr(profile, field), list):
		setattr(profile, field, [])
	else:
		setattr(profile, field, None)
	payload = dict(item.value)
	if isinstance(payload.get("content"), dict):
		payload["content"] = profile.model_dump()
	else:
		payload = profile.model_dump()
	await _store().aput(tuple(item.namespace), item.key, payload)
	return True


async def clear_memories(*, user_id: str = "default") -> int:
	items = await _store().asearch(("memories", user_id), limit=1000)
	for item in items:
		await _store().adelete(tuple(item.namespace), item.key)
	return len(items)
