"""LangMem integration for durable user memory.

Conversation turns remain in MongoDB as interaction history. LangMem maintains
per-turn free-form memory records in a ``LangMem`` collection, with vector
search for semantic retrieval. The user's name is stored separately in a
``profile`` namespace as a single structured document so it can be fetched
instantly without a search.
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
_NAME_MANAGER = None
_EMBEDDINGS = None
_EXTRACTION_TASKS: set[asyncio.Task] = set()


class Memory(BaseModel):
	"""One durable fact about the user — stored as a separate document per fact."""

	content: str = Field(description="A single explicit, durable fact about the user")


class UserName(BaseModel):
	"""Single structured document for instant name access — stored separately."""

	name: str | None = Field(default=None, description="User's explicitly stated name")


@dataclass(frozen=True)
class MemoryRecord:
	key: str
	value: str
	namespace: tuple[str, ...]


_INSTRUCTIONS = """Record individual durable facts about the human User as separate memory entries.

Each memory must capture exactly ONE piece of explicitly stated information.
Keep each memory concise — a short phrase or sentence.

Record: location, occupation, relationships (family members, pets), interests,
likes, dislikes, birth date, gender, and any other stable personal context
the User explicitly provides.

Do NOT record:
- The user's name (stored separately)
- Information about the Assistant or AI system
- Greetings, questions, jokes, or temporary plans
- One-time events or session-specific context
- Sensitive inferences (health conditions, politics, religion, sexuality)

Prefer the User's own wording. When the User corrects a fact, remove the old
memory. Do not create a memory if the User has not explicitly shared durable
personal information.
"""

_NAME_INSTRUCTIONS = """Maintain the user's name only.

Record the name only when the User explicitly introduces themselves by name.
Do not record nicknames, usernames, or names of other people.
Do not infer the name from context. Never create a name from Assistant wording.
"""


def _mongo_uri() -> str:
	uri = (
		f"mongodb://{os.environ['MONGODB_USER']}:{os.environ['MONGODB_PASSWORD']}@"
		f"{os.environ.get('MONGODB_HOST', 'localhost')}:{os.environ.get('MONGODB_PORT', '27017')}"
	)
	if "?" not in uri:
		uri += "?authSource=admin&directConnection=true&serverSelectionTimeoutMS=5000&connectTimeoutMS=5000"
	return uri


def _store():
	global _STORE, _STORE_CONTEXT
	if _STORE is None:
		from langgraph.store.mongodb import MongoDBStore, create_vector_index_config

		index_config = create_vector_index_config(
			dims=int(os.environ["EMBEDDING_DIM"]),
			embed=_embeddings(),
			fields=["content"],
			name="langmem_vector_index",
		)
		_STORE_CONTEXT = MongoDBStore.from_conn_string(
			conn_string=_mongo_uri(),
			db_name=os.environ.get("MONGODB_DB", "langbox"),
			collection_name="LangMem",
			index_config=index_config,
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


def _embeddings():
	global _EMBEDDINGS
	if _EMBEDDINGS is None:
		from langchain_openai import OpenAIEmbeddings

		class _LMStudioEmbeddings(OpenAIEmbeddings):
			"""Send text compatible with Granite's local tokenizer."""

			@staticmethod
			def _clean(text: str) -> str:
				# Granite's tokenizer rejects emoji variation selectors. Removing
				# the selectors preserves the readable emoji/text and avoids making
				# an otherwise valid conversation turn fail memory retrieval.
				return text.replace("︎", "").replace("️", "")

			def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
				return super().embed_documents([self._clean(text) for text in texts], **kwargs)

			async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
				return await super().aembed_documents([self._clean(text) for text in texts], **kwargs)

		kwargs: dict[str, Any] = {
			"model": os.environ["MODEL_EMBEDDING"],
			"api_key": os.environ.get("OPENAI_API_KEY", "lm-studio"),
			# LM Studio's OpenAI-compatible endpoint expects raw strings, not
			# the integer token IDs used by LangChain's tiktoken path.
			"check_embedding_ctx_length": False,
		}
		if os.getenv("INFERENCE_PROVIDER", "lmstudio").lower() != "openai":
			kwargs["base_url"] = os.environ.get("MODEL_HOST", "http://localhost:1234/v1")
		_EMBEDDINGS = _LMStudioEmbeddings(**kwargs)
	return _EMBEDDINGS


def _manager():
	global _MANAGER
	if _MANAGER is None:
		from langmem import create_memory_store_manager

		_MANAGER = create_memory_store_manager(
			_model(),
			schemas=[Memory],
			instructions=_INSTRUCTIONS,
			enable_inserts=True,
			enable_deletes=True,
			namespace=("memories", "{langgraph_user_id}"),
			store=_store(),
		)
	return _MANAGER


def _name_manager():
	global _NAME_MANAGER
	if _NAME_MANAGER is None:
		from langmem import create_memory_store_manager

		_NAME_MANAGER = create_memory_store_manager(
			_model(),
			schemas=[UserName],
			instructions=_NAME_INSTRUCTIONS,
			default_factory=lambda _config: UserName(),
			enable_inserts=False,
			enable_deletes=False,
			namespace=("profile", "{langgraph_user_id}"),
			store=_store(),
		)
	return _NAME_MANAGER


def _config(user_id: str) -> dict[str, dict[str, str]]:
	return {"configurable": {"langgraph_user_id": user_id}}


def _item_content(value: Any) -> str:
	"""Extract the text string from a LangGraph store item value.

	LangMem wraps Pydantic model dumps under a "content" key, so
	Memory(content="fact") arrives as {"content": {"content": "fact"}}.
	"""
	if isinstance(value, dict):
		c = value.get("content")
		if isinstance(c, str):
			return c.strip()
		if isinstance(c, dict):
			# LangMem wrapping: {"content": {"content": "fact"}}
			nested = c.get("content")
			if isinstance(nested, str):
				return nested.strip()
			# Old UserMemory structured profile — no usable text
			return ""
		if isinstance(value.get("text"), str):
			return value["text"].strip()
	if isinstance(value, str):
		return value.strip()
	return ""


async def _extract_turn(
	question: str, answer: str, *, user_id: str, interaction_id: str
) -> None:
	try:
		async with background_inference() as acquired:
			if not acquired:
				logger.debug(f"[langmem] skipped extraction (foreground busy) for {interaction_id}")
				return
			messages = [
				{"role": "user", "content": question},
				{"role": "assistant", "content": answer},
			]
			cfg = _config(user_id)
			await _manager().ainvoke({"messages": messages}, config=cfg)
			await _name_manager().ainvoke({"messages": messages}, config=cfg)
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


async def close_memory_store() -> None:
	"""Cancel background extraction and close the MongoDB store cleanly."""
	global _STORE_CONTEXT, _STORE, _MANAGER, _NAME_MANAGER, _EMBEDDINGS

	tasks = list(_EXTRACTION_TASKS)
	for task in tasks:
		if not task.done():
			task.cancel()
	if tasks:
		await asyncio.gather(*tasks, return_exceptions=True)
	_EXTRACTION_TASKS.clear()

	if _STORE_CONTEXT is not None:
		try:
			_STORE_CONTEXT.__exit__(None, None, None)
		except Exception:
			logger.exception("[langmem] failed to close MongoDB store")
		finally:
			_STORE_CONTEXT = None
			_STORE = None
			_MANAGER = None
			_NAME_MANAGER = None
			_EMBEDDINGS = None


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
	from utils.observability import emit, span

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
	with span("interaction_write", intent=intent or "unknown"):
		await interaction.insert()

	from utils.session_service import maybe_generate_session_title, touch_session

	await touch_session()
	await maybe_generate_session_title(question, answer)
	task = asyncio.create_task(
		_extract_turn(question, answer, user_id=user_id, interaction_id=str(interaction.id))
	)
	emit("memory_extraction_queued", intent=intent or "unknown")
	_EXTRACTION_TASKS.add(task)
	task.add_done_callback(_track_task)


async def get_user_name(user_id: str = "default") -> str | None:
	"""Return the user's name directly from the profile namespace — no search needed."""
	items = await _store().asearch(("profile", user_id), limit=1)
	if not items:
		return None
	v = items[0].value
	content = v.get("content") if isinstance(v, dict) else v
	if isinstance(content, dict):
		return content.get("name")
	if isinstance(v, dict):
		return v.get("name")
	return None


async def list_memories(
	query: str = "", *, user_id: str = "default", limit: int = 100
) -> list[MemoryRecord]:
	items = await _store().asearch(("memories", user_id), limit=limit)
	needle = query.casefold().strip()
	records = []
	for item in items:
		content = _item_content(item.value)
		if not content:
			continue
		if needle and needle not in content.casefold():
			continue
		records.append(MemoryRecord(key=item.key, value=content, namespace=tuple(item.namespace)))
	return records[: max(0, limit)]


async def search_relevant_memories(
	query: str, user_id: str = "default", limit: int = 5
) -> list[str]:
	"""Return the most semantically relevant memory records for the query."""
	items = await _store().asearch(("memories", user_id), query=query, limit=limit)
	return [c for item in items if (c := _item_content(item.value))]


async def get_fact_value(key: str, user_id: str = "default") -> str | None:
	"""Retrieve a single fact by field concept. 'name' is fetched directly; others via search."""
	if key == "name":
		return await get_user_name(user_id)
	results = await search_relevant_memories(key, user_id=user_id, limit=1)
	return results[0] if results else None


async def get_profile_context(user_id: str = "default", limit: int = 50) -> str:
	name = await get_user_name(user_id)
	records = await list_memories(user_id=user_id, limit=limit)
	if not name and not records:
		return ""
	lines = [
		"## USER PROFILE — facts about the human user only:",
		"Perspective rule: every fact below belongs to the User, never to the Assistant.",
		"Use these facts only when relevant. Do not volunteer them unprompted.",
	]
	if name:
		lines.append(f"- User's name: {name}")
	lines.extend(f"- {record.value}" for record in records)
	return "\n".join(lines)


async def forget_memory(query_or_key: str, *, user_id: str = "default") -> bool:
	"""Delete memories by exact store key or by keyword match against content."""
	items = await _store().asearch(("memories", user_id), limit=1000)
	q = query_or_key.strip()
	for item in items:
		if item.key == q:
			await _store().adelete(tuple(item.namespace), item.key)
			return True
	q_lower = q.casefold()
	deleted = 0
	for item in items:
		if q_lower in _item_content(item.value).casefold():
			await _store().adelete(tuple(item.namespace), item.key)
			deleted += 1
	return deleted > 0


async def clear_memories(*, user_id: str = "default") -> int:
	mem_items = await _store().asearch(("memories", user_id), limit=1000)
	prof_items = await _store().asearch(("profile", user_id), limit=10)
	for item in mem_items:
		await _store().adelete(tuple(item.namespace), item.key)
	for item in prof_items:
		await _store().adelete(tuple(item.namespace), item.key)
	return len(mem_items) + len(prof_items)
