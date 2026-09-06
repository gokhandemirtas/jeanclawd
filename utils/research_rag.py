"""Persistence and retrieval for completed deep-research reports.

The first implementation intentionally uses MongoDB for durable storage and
BM25 for local hybrid-friendly retrieval. It avoids making the normal chat
path depend on an embedding server being available.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rank_bm25 import BM25Okapi

from db.schemas import ResearchChunk, ResearchDocument
from utils.log import logger

_SECTION_RE = re.compile(r"(?m)^#{2,3}\s+(.+?)\s*$")
_SOURCE_RE = re.compile(r"\[(S\d+)\]")
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


def resolve_report_path(report_name: str) -> Path:
	"""Resolve a stored report filename relative to the application root."""
	return Path(__file__).resolve().parents[1] / "deep-research" / Path(report_name).name


def _tokens(text: str) -> list[str]:
	return [word.casefold() for word in _WORD_RE.findall(text)]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
	if len(left) != len(right) or not left or not right:
		return 0.0
	left_norm = math.sqrt(sum(value * value for value in left))
	right_norm = math.sqrt(sum(value * value for value in right))
	if not left_norm or not right_norm:
		return 0.0
	return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _semantic_enabled() -> bool:
	return os.getenv("RESEARCH_RAG_SEMANTIC", "true").strip().lower() not in {
		"0",
		"false",
		"no",
		"off",
	}


def _split_chunks(content: str, max_chars: int = 3500) -> list[tuple[str, str]]:
	"""Split a report by headings, then by bounded paragraphs."""
	matches = list(_SECTION_RE.finditer(content))
	sections: list[tuple[str, str]] = []
	if not matches:
		sections.append(("Report", content))
	else:
		for index, match in enumerate(matches):
			start = match.end()
			end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
			sections.append((match.group(1).strip(), content[start:end].strip()))

	chunks: list[tuple[str, str]] = []
	for section, text in sections:
		paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
		current = ""
		for paragraph in paragraphs:
			if len(paragraph) > max_chars:
				if current:
					chunks.append((section, current))
					current = ""
				while len(paragraph) > max_chars:
					cut = paragraph.rfind(" ", 0, max_chars)
					cut = cut if cut > max_chars // 2 else max_chars
					chunks.append((section, paragraph[:cut].strip()))
					paragraph = paragraph[cut:].strip()
				if paragraph:
					current = paragraph
				continue
			if current and len(current) + len(paragraph) + 2 > max_chars:
				chunks.append((section, current))
				current = ""
			current = f"{current}\n\n{paragraph}".strip()
		if current:
			chunks.append((section, current))
	return chunks


def _report_metadata(content: str) -> tuple[str, str, list[str]]:
	title = re.search(r"(?m)^#\s+Deep Research:\s*(.+?)\s*$", content)
	query = re.search(r"(?m)^- Original request:\s*(.+?)\s*$", content)
	sources = re.findall(r"(?m)^\s*\d+\.\s+(https?://\S+)\s*$", content)
	return (
		title.group(1).strip() if title else "Research",
		query.group(1).strip() if query else title.group(1).strip() if title else "",
		list(dict.fromkeys(url.rstrip(".,)") for url in sources)),
	)


async def index_report(report_path: str, *, user_id: str = "default") -> str | None:
	"""Index a report idempotently and return its research id."""
	path = Path(report_path)
	stored_report_path = path.name
	try:
		content = path.read_text(encoding="utf-8")
	except (OSError, UnicodeError) as exc:
		logger.warning(f"[research_rag] Could not read report: {exc}")
		return None

	content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
	existing_records = await ResearchDocument.find(
		ResearchDocument.user_id == user_id,
		ResearchDocument.content_hash == content_hash,
	).to_list()
	existing = next(
		(
			document
			for document in existing_records
			if Path(document.report_path).name == stored_report_path
		),
		None,
	)
	if existing:
		if existing.report_path != stored_report_path:
			existing.report_path = stored_report_path
			await existing.save()
		return existing.research_id

	topic, original_query, source_urls = _report_metadata(content)
	parsed_chunks = _split_chunks(content)
	embeddings: list[list[float] | None] = [None] * len(parsed_chunks)
	if _semantic_enabled() and parsed_chunks:
		try:
			from utils.langmem_service import _embeddings

			embedding_inputs = [
				f"Topic: {topic}\nSection: {section}\n{chunk_content}"
				for section, chunk_content in parsed_chunks
			]
			embeddings = await _embeddings().aembed_documents(embedding_inputs)
		except Exception as exc:
			logger.warning(f"[research_rag] Semantic indexing unavailable; using BM25: {exc}")
	research_id = str(uuid4())
	generated_at = datetime.now(timezone.utc)
	await ResearchDocument(
		research_id=research_id,
		user_id=user_id,
		topic=topic,
		original_query=original_query,
		report_path=stored_report_path,
		generated_at=generated_at,
		content_hash=content_hash,
		source_urls=source_urls,
	).insert()

	for index, (section, chunk_content) in enumerate(parsed_chunks):
		await ResearchChunk(
			research_id=research_id,
			user_id=user_id,
			chunk_id=f"{research_id}:{index}",
			section=section,
			content=chunk_content,
			source_markers=list(dict.fromkeys(_SOURCE_RE.findall(chunk_content))),
			embedding=embeddings[index] if index < len(embeddings) else None,
		).insert()
	logger.info(f"[research_rag] Indexed '{topic}' with {len(parsed_chunks)} chunk(s)")
	return research_id


async def search_research(
	query: str, *, user_id: str = "default", limit: int = 8
) -> list[dict[str, str | list[str]]]:
	"""Return relevant chunks, diversifying results across research reports."""
	documents = await ResearchDocument.find(ResearchDocument.user_id == user_id).to_list()
	if not documents:
		return []
	doc_by_id = {document.research_id: document for document in documents}
	chunks = await ResearchChunk.find(ResearchChunk.user_id == user_id).to_list()
	if not chunks:
		return []

	corpus = [_tokens(f"{chunk.section} {chunk.content}") for chunk in chunks]
	query_tokens = _tokens(query)
	if not query_tokens:
		return []
	scores = BM25Okapi(corpus).get_scores(query_tokens)
	semantic_scores: list[float] = [0.0] * len(chunks)
	if _semantic_enabled() and any(chunk.embedding for chunk in chunks):
		try:
			from utils.langmem_service import _embeddings

			query_embedding = (await _embeddings().aembed_documents([query]))[0]
			semantic_scores = [
				_cosine_similarity(query_embedding, chunk.embedding or []) for chunk in chunks
			]
		except Exception as exc:
			logger.warning(
				f"[research_rag] Semantic retrieval unavailable; using BM25: {exc}"
			)

	# Rank fusion is more stable than mixing raw BM25 and cosine scales.
	max_bm25 = max(scores) if scores.size else 0.0
	min_bm25 = min(scores) if scores.size else 0.0
	bm25_range = max_bm25 - min_bm25
	lexical_scores = [
		(float(score) - min_bm25) / bm25_range if bm25_range else 0.0 for score in scores
	]
	semantic_weight = float(os.getenv("RESEARCH_RAG_SEMANTIC_WEIGHT", "0.6"))
	semantic_weight = min(1.0, max(0.0, semantic_weight))
	if not any(semantic_scores):
		semantic_weight = 0.0
	fused_scores = [
		(semantic_weight * semantic) + ((1.0 - semantic_weight) * lexical)
		for semantic, lexical in zip(semantic_scores, lexical_scores, strict=True)
	]
	ordered = sorted(
		zip(fused_scores, chunks, strict=True), key=lambda item: item[0], reverse=True
	)

	results: list[dict[str, str | list[str]]] = []
	per_report: dict[str, int] = {}
	for score, chunk in ordered:
		if score <= 0 and results:
			break
		if per_report.get(chunk.research_id, 0) >= 3:
			continue
		document = doc_by_id.get(chunk.research_id)
		if document is None:
			continue
		results.append(
			{
				"research_id": chunk.research_id,
				"topic": document.topic,
				"section": chunk.section,
				"content": chunk.content,
				"source_markers": chunk.source_markers,
				"generated_at": document.generated_at.isoformat(),
			}
		)
		per_report[chunk.research_id] = per_report.get(chunk.research_id, 0) + 1
		if len(results) >= max(1, limit):
			break
	return results


async def research_context(
	query: str, *, user_id: str = "default", limit: int = 8
) -> str:
	"""Format retrieved chunks as bounded, provenance-preserving prompt context."""
	results = await search_research(query, user_id=user_id, limit=limit)
	if not results:
		return ""
	blocks = []
	for index, result in enumerate(results, 1):
		markers = ", ".join(f"[{marker}]" for marker in result["source_markers"]) or "none"
		blocks.append(
			f"Research passage {index} — {result['topic']} / {result['section']} "
			f"(sources: {markers})\n{result['content']}"
		)
	return "\n\n".join(blocks)
