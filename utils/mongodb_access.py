"""Read-only MongoDB access for diagnostics and MCP tools.

This module deliberately exposes no insert, update, or delete operations.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from bson.json_util import dumps, loads
from pymongo import AsyncMongoClient


def _database_name() -> str:
	return os.environ.get("MONGODB_DB", "langbox")


@asynccontextmanager
async def database() -> AsyncIterator[Any]:
	"""Yield the configured database and always close the client afterward."""
	uri = (
		f"mongodb://{os.environ['MONGODB_USER']}:{os.environ['MONGODB_PASSWORD']}@"
		f"{os.environ.get('MONGODB_HOST', 'localhost')}:{os.environ.get('MONGODB_PORT', '27017')}"
	)
	client = AsyncMongoClient(uri, timeoutMS=5000)
	try:
		await client.admin.command("ping")
		yield client[_database_name()]
	finally:
		await client.close()


async def inventory() -> list[dict[str, int | str]]:
	"""Return collection names and document counts without reading documents."""
	async with database() as db:
		result = []
		for name in sorted(await db.list_collection_names()):
			result.append({"collection": name, "count": await db[name].count_documents({})})
		return result


async def find_documents(
	collection: str, query: dict[str, Any] | None = None, *, limit: int = 20
) -> list[dict[str, Any]]:
	"""Read a bounded set of documents, converting BSON values to JSON-safe values."""
	if not collection or collection.startswith("system."):
		raise ValueError("invalid collection")
	limit = max(1, min(limit, 100))
	async with database() as db:
		documents = await db[collection].find(query or {}).limit(limit).to_list()
		return [loads(dumps(document)) for document in documents]
