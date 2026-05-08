"""Qdrant index for AIDA's journal entries.

Stores each journal narrative as a vector so the journal can be searched
semantically ("what did I write about Gokhan's interest in AI?") rather than
scanned chronologically.

The Journal MongoDB document remains the canonical store. This is a secondary
index — write here after writing to MongoDB, search here for retrieval.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date
from typing import Optional

from utils.log import logger

_client: Optional["QdrantClient"] = None

COLLECTION = "langbox_journal"


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        _client = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
        )
        existing = {c.name for c in _client.get_collections().collections}
        if COLLECTION not in existing:
            from utils.embedder import EMBEDDING_DIM
            _client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"[journal_index] Created Qdrant collection '{COLLECTION}'")

    return _client


def _date_to_id(d: date) -> int:
    """Stable integer point ID derived from the date."""
    return int(hashlib.md5(str(d).encode()).hexdigest(), 16) % (2**62)


def index_journal_entry(datestamp: date, narrative: str) -> None:
    """Upsert a journal entry into Qdrant. Safe to call multiple times for the same date."""
    try:
        from qdrant_client.models import PointStruct
        from utils.embedder import embed

        client = _get_client()
        vector = embed(narrative)
        client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=_date_to_id(datestamp),
                    vector=vector,
                    payload={"datestamp": str(datestamp), "narrative": narrative},
                )
            ],
        )
        logger.info(f"[journal_index] Indexed entry for {datestamp}")
    except Exception:
        logger.exception("[journal_index] Failed to index journal entry")


def search_journal(query: str, limit: int = 3) -> list[dict]:
    """Return semantically relevant journal entries.

    Each result is a dict with 'datestamp' (str) and 'narrative' (str) keys.
    Returns [] on failure or if Qdrant is unavailable.
    """
    try:
        from utils.embedder import embed

        client = _get_client()
        vector = embed(query)
        hits = client.search(collection_name=COLLECTION, query_vector=vector, limit=limit)
        return [h.payload for h in hits]
    except Exception:
        logger.exception("[journal_index] Search failed")
        return []
