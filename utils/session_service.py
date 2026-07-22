"""Lifecycle and restoration helpers for resumable conversation sessions."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

_active_session_id: UUID | None = None


class _SessionTitle(BaseModel):
	title: str = Field(description="A concise two- or three-word conversation title")


def get_active_session_id() -> UUID | None:
	return _active_session_id


async def create_session(*, user_id: str = "default", title: str | None = None) -> UUID:
	from db.schemas import Session

	now = datetime.now()
	session_id = uuid4()
	await Session(
		session_id=session_id,
		user_id=user_id,
		title=title,
		created_at=now,
		last_active_at=now,
	).insert()
	await _activate(session_id, summary=None, interactions=[])
	return session_id


async def initialize_session(*, user_id: str = "default") -> UUID:
	"""Create the initial session once the database is ready."""
	if _active_session_id is not None:
		return _active_session_id
	return await create_session(user_id=user_id)


async def resume_session(session_id: str | UUID, *, user_id: str = "default") -> UUID:
	from db.schemas import Interaction, Session

	parsed = session_id if isinstance(session_id, UUID) else UUID(session_id)
	session = await Session.find_one(Session.session_id == parsed, Session.user_id == user_id)
	if session is None:
		raise ValueError(f"Session not found: {parsed}")
	interactions = await Interaction.find(
		Interaction.session_id == str(parsed), Interaction.user_id == user_id
	).sort("-created_at").limit(20).to_list()
	interactions.reverse()
	await _activate(parsed, summary=session.summary, interactions=interactions)
	session.status = "active"
	session.last_active_at = datetime.now()
	await session.save()
	return parsed


async def list_sessions(*, user_id: str = "default", limit: int = 20) -> list:
	from db.schemas import Session

	return await Session.find(Session.user_id == user_id).sort("-last_active_at").limit(limit).to_list()


async def close_session(session_id: str | UUID | None = None) -> None:
	from db.schemas import Session

	parsed = session_id or _active_session_id
	if parsed is None:
		return
	if not isinstance(parsed, UUID):
		parsed = UUID(parsed)
	session = await Session.find_one(Session.session_id == parsed)
	if session:
		session.status = "closed"
		session.last_active_at = datetime.now()
		await session.save()


async def update_session_summary(summary: str) -> None:
	if _active_session_id is None:
		return
	from db.schemas import Session

	session = await Session.find_one(Session.session_id == _active_session_id)
	if session:
		session.summary = summary
		session.last_active_at = datetime.now()
		await session.save()


async def touch_session() -> None:
	if _active_session_id is None:
		return
	from db.schemas import Session

	session = await Session.find_one(Session.session_id == _active_session_id)
	if session:
		session.last_active_at = datetime.now()
		await session.save()


async def maybe_generate_session_title(user_message: str, assistant_message: str) -> None:
	"""Give an untitled session a short title based on its first completed turn."""
	if _active_session_id is None:
		return
	from db.schemas import Session

	session = await Session.find_one(Session.session_id == _active_session_id)
	if session is None or session.title:
		return

	from utils.llm_structured_output import generate_structured_output

	prompt = (
		"Create a neutral title for this conversation. Return exactly two or three words. "
		"Use title case, no punctuation, and do not include the user's name.\n\n"
		f"User: {user_message}\nAssistant: {assistant_message[:1000]}"
	)
	try:
		result = await asyncio.to_thread(
			generate_structured_output,
			model_name=os.environ.get("MODEL_INTENT") or os.environ["MODEL_GENERALIST"],
			user_prompt=prompt,
			system_prompt="You generate short conversation titles.",
			pydantic_model=_SessionTitle,
			temperature=0.2,
			max_tokens=32,
		)
		words = [word.strip(".,!?;:") for word in result.title.split() if word.strip(".,!?;:")]
		if len(words) not in (2, 3):
			return
		session.title = " ".join(words)
		session.last_active_at = datetime.now()
		await session.save()
	except Exception:
		# Title generation is decorative; never make a successful interaction fail.
		return


async def _activate(session_id: UUID, *, summary: str | None, interactions: list) -> None:
	global _active_session_id
	_active_session_id = session_id
	from skills.conversation.skill import restore_session_context

	restore_session_context(summary, interactions)
