"""Persistence and retrieval for completed deep-research reports.

The first implementation intentionally uses MongoDB for durable storage and
BM25 for local hybrid-friendly retrieval. It avoids making the normal chat
path depend on an embedding server being available.
"""

from __future__ import annotations

import hashlib
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


def _tokens(text: str) -> list[str]:
	return [word.casefold() for word in _WORD_RE.findall(text)]


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
	try:
		content = path.read_text(encoding="utf-8")
	except (OSError, UnicodeError) as exc:
		logger.warning(f"[research_rag] Could not read report: {exc}")
		return None

	content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
	existing = await ResearchDocument.find_one(
		ResearchDocument.user_id == user_id,
		ResearchDocument.report_path == str(path),
		ResearchDocument.content_hash == content_hash,
	)
	if existing:
		return existing.research_id

	topic, original_query, source_urls = _report_metadata(content)
	research_id = str(uuid4())
	generated_at = datetime.now(timezone.utc)
	await ResearchDocument(
		research_id=research_id,
		user_id=user_id,
		topic=topic,
		original_query=original_query,
		report_path=str(path),
		generated_at=generated_at,
		content_hash=content_hash,
		source_urls=source_urls,
	).insert()

	for index, (section, chunk_content) in enumerate(_split_chunks(content)):
		await ResearchChunk(
			research_id=research_id,
			user_id=user_id,
			chunk_id=f"{research_id}:{index}",
			section=section,
			content=chunk_content,
			source_markers=list(dict.fromkeys(_SOURCE_RE.findall(chunk_content))),
		).insert()
	logger.info(f"[research_rag] Indexed '{topic}' with {index + 1} chunk(s)")
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
	ordered = sorted(
		zip(scores, chunks, strict=True), key=lambda item: item[0], reverse=True
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
