"""MCP stdio server exposing the JeanClawd assistant pipeline to local clients."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()


@asynccontextmanager
async def _lifespan(_server: MCPServer) -> AsyncIterator[None]:
	"""Initialise the same durable services used by the HTTP server."""
	from db.init import db_init
	from utils.session_service import initialize_session

	await db_init()
	await initialize_session()
	try:
		yield
	finally:
		from utils.langmem_service import close_memory_store

		await close_memory_store()


server = MCPServer(
	name="JeanClawd",
	description="Drive the local JeanClawd assistant through its normal skill, routing, and memory pipeline.",
	lifespan=_lifespan,
)


@server.tool()
async def chat(query: str) -> str:
	"""Send a message through JeanClawd's full intent, skills, and memory pipeline."""
	query = query.strip()
	if not query:
		raise ValueError("query must not be empty")

	from agents.intent_classifier import run_intent_classifier

	return await run_intent_classifier(query)


@server.tool()
async def list_memories() -> str:
	"""Return the durable memory facts JeanClawd has stored for the default user."""
	from utils.langmem_service import list_memories as get_memories

	memories = await get_memories()
	return json.dumps(
		[{"key": memory.key, "value": memory.value} for memory in memories],
		indent=2,
	)


@server.tool()
async def forget_memory(query: str) -> bool:
	"""Delete memory facts whose key or content matches the supplied query."""
	query = query.strip()
	if not query:
		raise ValueError("query must not be empty")

	from utils.langmem_service import forget_memory as remove_memory

	return await remove_memory(query)


@server.tool()
async def list_sessions() -> str:
	"""List JeanClawd's 20 most recently active conversation sessions."""
	from utils.session_service import list_sessions as get_sessions

	sessions = await get_sessions()
	return json.dumps(
		[
			{
				"session_id": str(session.session_id),
				"title": session.title,
				"summary": session.summary,
				"created_at": session.created_at.isoformat(),
				"last_active_at": session.last_active_at.isoformat(),
				"status": session.status,
			}
			for session in sessions
		],
		indent=2,
	)


@server.tool()
async def resume_session(session_id: str) -> str:
	"""Make an existing JeanClawd session active and restore its conversation context."""
	from utils.session_service import resume_session as resume

	try:
		resumed_session_id = await resume(session_id)
	except (TypeError, ValueError) as exc:
		raise ValueError("session not found") from exc
	return str(resumed_session_id)


def main() -> None:
	"""Run the MCP server over standard input/output for Claude Code."""
	server.run(transport="stdio")


if __name__ == "__main__":
	main()
