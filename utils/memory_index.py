"""Qdrant projection for canonical MemoryFacts.

MongoDB remains authoritative.  This collection only provides semantic recall.
"""

from __future__ import annotations

import hashlib
import os

from utils.log import logger

COLLECTION = "langbox_memory_facts"
_client = None


def _get_client():
	global _client
	if _client is None:
		from qdrant_client import QdrantClient
		from qdrant_client.models import Distance, VectorParams
		from utils.embedder import get_embedding_dim

		_client = QdrantClient(
			host=os.environ.get("QDRANT_HOST", "localhost"),
			port=int(os.environ.get("QDRANT_PORT", "6333")),
		)
		dim = get_embedding_dim()
		existing = {item.name for item in _client.get_collections().collections}
		if COLLECTION in existing:
			info = _client.get_collection(COLLECTION)
			stored_dim = info.config.params.vectors.size
			if stored_dim != dim:
				_client.delete_collection(COLLECTION)
				existing.remove(COLLECTION)
		if COLLECTION not in existing:
			_client.create_collection(
				collection_name=COLLECTION,
				vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
			)
	return _client


def _point_id(fact_id: str) -> int:
	return int(hashlib.md5(fact_id.encode()).hexdigest(), 16) % (2**62)


def index_fact(fact) -> None:
	try:
		from qdrant_client.models import PointStruct
		from utils.embedder import embed

		text = f"{fact.key}: {fact.value}"
		_get_client().upsert(
			collection_name=COLLECTION,
			points=[
				PointStruct(
					id=_point_id(str(fact.id)),
					vector=embed(text),
					payload={
						"fact_id": str(fact.id),
						"user_id": fact.user_id,
						"key": fact.key,
						"value": fact.value,
						"status": fact.status,
					},
				)
			],
		)
	except Exception:
		logger.exception("[memory_index] Failed to index memory fact")


def search_facts(query: str, user_id: str = "default", limit: int = 5) -> list[str]:
	try:
		from qdrant_client.models import FieldCondition, Filter, MatchValue
		from utils.embedder import embed

		result = _get_client().query_points(
			collection_name=COLLECTION,
			query=embed(query),
			query_filter=Filter(
				must=[
					FieldCondition(key="user_id", match=MatchValue(value=user_id)),
					FieldCondition(key="status", match=MatchValue(value="active")),
				]
			),
			limit=limit,
		)
		return [f"{hit.payload['key']}: {hit.payload['value']}" for hit in result.points]
	except Exception:
		logger.debug("[memory_index] Semantic recall unavailable", exc_info=True)
		return []


def reset_index() -> None:
	"""Drop the disposable Qdrant projection; MongoDB remains authoritative."""
	global _client
	try:
		client = _get_client()
		existing = {item.name for item in client.get_collections().collections}
		if COLLECTION in existing:
			client.delete_collection(COLLECTION)
	finally:
		_client = None


def rebuild_index(facts) -> int:
	"""Recreate the disposable projection from authoritative MemoryFact documents."""
	reset_index()
	count = 0
	for fact in facts:
		if getattr(fact, "status", None) != "active":
			continue
		index_fact(fact)
		count += 1
	return count
