"""Vector index for introspection reflection files.

Embeddings are stored in MongoDB keyed by filename + content hash.
On each query, new or changed files are embedded and cached; the question
is then embedded and ranked by cosine similarity against all cached vectors.
"""
from __future__ import annotations

import hashlib

import numpy as np

from utils.log import logger

_EMBED_CHARS = 4_000  # truncation for embedding input
_TOP_K = 6


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


async def _embed(texts: list[str]) -> list[list[float]]:
    from utils.langmem_service import _embeddings
    embedder = _embeddings()
    return await embedder.aembed_documents(texts)


async def index_reflection(path: str) -> None:
    """Embed and cache a single reflection file. No-op if already up to date."""
    from db.schemas import IntrospectionEmbedding
    from utils.self_inspector import read_reflection

    filename = path.rsplit("/", 1)[-1]
    try:
        text = read_reflection(filename, max_bytes=64_000)
    except Exception as exc:
        logger.warning(f"[introspection_index] cannot read {filename}: {exc}")
        return

    h = _hash(text)
    existing = await IntrospectionEmbedding.find_one(IntrospectionEmbedding.filename == filename)
    if existing and existing.content_hash == h:
        return  # already indexed

    try:
        vector = (await _embed([text[:_EMBED_CHARS]]))[0]
    except Exception as exc:
        logger.error(f"[introspection_index] embedding failed for {filename}: {exc}")
        return

    if existing:
        existing.content_hash = h
        existing.embedding = vector
        await existing.save()
    else:
        await IntrospectionEmbedding(filename=filename, content_hash=h, embedding=vector).insert()
    logger.debug(f"[introspection_index] indexed {filename}")


async def search_introspections(
    question: str,
    paths: list[str],
    top_k: int = _TOP_K,
) -> list[str]:
    """Return the top_k most relevant reflection paths for question.

    Builds/updates the embedding index lazily — only embeds files that are
    new or whose content has changed since the last run.
    """
    from db.schemas import IntrospectionEmbedding
    from utils.self_inspector import read_reflection

    # Load all cached embeddings keyed by filename
    cached: dict[str, IntrospectionEmbedding] = {
        doc.filename: doc
        for doc in await IntrospectionEmbedding.find_all().to_list()
    }

    # Read each reflection and identify which need (re-)embedding
    contents: dict[str, str] = {}
    to_embed: list[str] = []

    for path in paths:
        filename = path.rsplit("/", 1)[-1]
        try:
            text = read_reflection(filename, max_bytes=64_000)
        except Exception as exc:
            logger.warning(f"[introspection_index] skipping {filename}: {exc}")
            continue
        contents[filename] = text
        h = _hash(text)
        if filename not in cached or cached[filename].content_hash != h:
            to_embed.append(filename)

    # Embed new/changed files in one batch call
    if to_embed:
        logger.debug(f"[introspection_index] embedding {len(to_embed)} new/changed records")
        snippets = [contents[f][:_EMBED_CHARS] for f in to_embed]
        try:
            vectors = await _embed(snippets)
        except Exception as exc:
            logger.error(f"[introspection_index] embedding failed: {exc}")
            # Fall back to returning all paths in recency order
            return paths[:top_k]

        for filename, vector in zip(to_embed, vectors):
            h = _hash(contents[filename])
            existing = cached.get(filename)
            if existing:
                existing.content_hash = h
                existing.embedding = vector
                await existing.save()
            else:
                doc = IntrospectionEmbedding(filename=filename, content_hash=h, embedding=vector)
                await doc.insert()
                cached[filename] = doc

    # Embed the question
    try:
        q_vec = (await _embed([question[:_EMBED_CHARS]]))[0]
    except Exception as exc:
        logger.error(f"[introspection_index] question embedding failed: {exc}")
        return paths[:top_k]

    # Rank by cosine similarity
    scores: list[tuple[float, str]] = []
    for path in paths:
        filename = path.rsplit("/", 1)[-1]
        if filename not in cached:
            continue
        score = _cosine(q_vec, cached[filename].embedding)
        scores.append((score, path))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = [path for _, path in scores[:top_k]]
    logger.debug(f"[introspection_index] top {len(top)} results, best score={scores[0][0]:.3f}" if scores else "[introspection_index] no scores")
    return top
