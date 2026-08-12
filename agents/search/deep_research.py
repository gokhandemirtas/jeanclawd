"""Search-focused ReAct loop for deep research queries.

Runs up to MAX_STEPS search iterations, letting the model decide when it has
gathered enough information to answer the original question comprehensively.
Completed work is written to a Markdown report under ``deep-research/`` so the
interactive context only retains a compact pointer to the document.
"""

import asyncio
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import unicodedata

from pydantic import BaseModel, Field

from agents.persona import get_active_identity
from utils.llm_structured_output import generate_structured_output
from utils.log import logger
from utils.messages import HumanMessage, SystemMessage
from utils.search import web_search

MAX_STEPS = 10
MIN_SEARCHES = 3


def _research_model() -> str:
	"""Use the dedicated research model, while keeping existing setups usable."""
	return os.environ.get("MODEL_RESEARCH") or os.environ["MODEL_GENERALIST"]

_SYSTEM_PROMPT = """You are a research assistant conducting deep research on behalf of the user.

## Your process
1. Decide what to search for based on the original query and what you already know.
2. Read the search result and decide: do I have enough to fully answer the original question?
   - If yes → action: RESPOND
   - If no → action: SEARCH with a refined or follow-up query
3. Each search should target something *new* — do not repeat a query you already ran.

## Rules
- Be specific with search queries. "Jean-Claude Van Damme age 2024" beats "JCVD".
- For shopping or product-link requests, treat every item in a user-provided list as a separate
  research target and search each named marketplace separately.
- Preserve marketplace constraints; do not substitute generic articles or other sellers.
- Only report URLs returned by search results. Never construct or guess product URLs.
- After 3+ observations, strongly prefer RESPOND unless a critical fact is clearly missing.
- Do not RESPOND before at least 3 distinct searches have been completed. Before then,
  always choose SEARCH, preferably using a different angle or source type.
- RESPOND with a comprehensive, well-structured answer — this is deep research, not a quick reply.
- Never mention the search process in your final answer. Just answer the question.
"""

_URL_RE = re.compile(r'https?://\S+')
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
_SOURCE_URL_RE = re.compile(r'(?im)^SourceURL:\s*(https?://\S+)\s*$')
_SUMMARIZE_THRESHOLD = 1000
_SUMMARIZE_INPUT_CAP = 12000  # chars fed to the summarizer LLM (~3k tokens)
_DEFAULT_LENGTH = "at least 4–6 substantial paragraphs"

_LENGTH_PATTERNS: list[tuple[re.Pattern, str | None]] = [
    (re.compile(r'\bin\s+(\d+)\s+paragraphs?\b', re.I), None),   # "in 2 paragraphs" → dynamic
    (re.compile(r'\bin\s+(\d+)\s+sentences?\b', re.I), None),     # "in 3 sentences" → dynamic
    (re.compile(r'\b(briefly|brief|short(?:ly)?)\b', re.I), "1–2 short paragraphs"),
    (re.compile(r'\bconcise(?:ly)?\b', re.I), "2–3 concise paragraphs"),
]


def _extract_length_hint(query: str) -> str:
    for pattern, fixed in _LENGTH_PATTERNS:
        m = pattern.search(query)
        if not m:
            continue
        if fixed:
            return fixed
        n = int(m.group(1))
        unit = "paragraph" if "paragraph" in m.group(0).lower() else "sentence"
        return f"{n} {unit}{'s' if n != 1 else ''}"
    return _DEFAULT_LENGTH


def _report_slug(topic: str) -> str:
	"""Create a stable, filesystem-safe filename from the research topic."""
	normalized = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode().lower()
	slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
	return (slug[:100].rstrip("-") or "research")


def _write_report(
	original_query: str,
	topic: str,
	answer: str,
	observations: list[str],
	references: list[str],
) -> Path:
	"""Persist readable summaries, not raw page/navigation captures."""
	report_dir = Path(__file__).resolve().parents[2] / "deep-research"
	report_dir.mkdir(parents=True, exist_ok=True)
	report_path = report_dir / f"{_report_slug(topic)}.md"
	notes = "\n\n".join(_format_report_observation(observation) for observation in observations)
	sources = "\n".join(f"{i}. {url}" for i, url in enumerate(references, 1)) or "No external sources were collected."
	content = (
		f"# Deep Research: {topic}\n\n"
		f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
		f"- Original request: {original_query}\n\n"
		"## Final answer\n\n"
		f"{_strip_reasoning(answer).strip()}\n\n"
		"## Research notes\n\n"
		f"{notes or 'No research notes were collected.'}\n\n"
		"## Sources\n\n"
		f"{sources}\n"
	)
	report_path.write_text(content, encoding="utf-8")
	return report_path


def _split_long_paragraphs(text: str, *, max_sentences: int = 3, max_chars: int = 560) -> str:
	"""Add readable breaks without changing the model's wording."""
	paragraphs: list[str] = []
	for paragraph in re.split(r"\n\s*\n", text.strip()):
		paragraph = re.sub(r"\s+", " ", paragraph).strip()
		if not paragraph:
			continue
		sentences = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])", paragraph)
		if len(sentences) <= max_sentences and len(paragraph) <= max_chars:
			paragraphs.append(paragraph)
			continue
		chunks: list[str] = []
		chunk: list[str] = []
		chunk_chars = 0
		for sentence in sentences:
			if chunk and (len(chunk) >= max_sentences or chunk_chars + len(sentence) + 1 > max_chars):
				chunks.append(" ".join(chunk))
				chunk, chunk_chars = [], 0
			chunk.append(sentence)
			chunk_chars += len(sentence) + 1
		if chunk:
			chunks.append(" ".join(chunk))
		paragraphs.extend(chunks)
	return "\n\n".join(paragraphs)


def _format_report_observation(observation: str) -> str:
	"""Render one search summary as a labelled note, not a fake finding."""
	match = re.match(r"Query:\s*(.*?)\nResult:\s*(.*)", observation, flags=re.DOTALL)
	if not match:
		return _split_long_paragraphs(_strip_reasoning(observation))
	query, result = match.groups()
	return f"### Search: {query.strip()}\n\n{_split_long_paragraphs(_strip_reasoning(result))}"


async def _complete_research(
	original_query: str,
	topic: str,
	answer: str,
	report_observations: list[str],
	references: list[str],
) -> tuple[str, list[str]]:
	report_path = _write_report(original_query, topic, answer, report_observations, references)
	from skills.conversation.skill import compact_after_deep_research
	await compact_after_deep_research(topic, str(report_path))
	return f"Finished research, please check the final document at: {report_path}", references


class _ResearchStep(BaseModel):
    thought: str = Field(description="What you know so far and what is still missing.")
    action: Literal["SEARCH", "RESPOND"] = Field(
        description="SEARCH to run another query, RESPOND when you have enough."
    )
    query: str = Field(
        description="The search query (if SEARCH) or your comprehensive final answer (if RESPOND)."
    )


def _extract_urls(text: str) -> tuple[str, list[str]]:
	"""Strip explicit source URLs, ignoring incidental page-navigation links."""
	urls: list[str] = []

	def _capture_source(match: re.Match) -> str:
		url = match.group(1).rstrip('.,);\'"')
		if url not in urls:
			urls.append(url)
		return ""

	text = _SOURCE_URL_RE.sub(_capture_source, text)
	# Remove markdown and bare URLs from page text without adding them as sources.
	text = _MD_LINK_RE.sub(lambda match: match.group(1), text)
	text = _URL_RE.sub("", text)
	return text.strip(), urls


async def _run_step(prompt: str) -> _ResearchStep:
    return await asyncio.to_thread(
        generate_structured_output,
        model_name=_research_model(),
        user_prompt=prompt,
        system_prompt=_SYSTEM_PROMPT,
        pydantic_model=_ResearchStep,
        max_tokens=512,
    )


async def _summarize(text: str, query: str) -> str:
    from agents.agent_factory import create_llm
    llm = create_llm(model_name=_research_model(), temperature=0.0, max_tokens=400)
    response = await llm.ainvoke([
        SystemMessage(content=(
            "Summarize the following search result. Keep every fact that is relevant to the query. "
            "Be concise — aim for 200–350 words. Plain prose, no headers."
        )),
        HumanMessage(content=f"Query: {query}\n\nSearch result:\n{text}"),
    ])
    return _split_long_paragraphs(_strip_reasoning(response.content))


def _strip_reasoning(text: str) -> str:
	"""Remove private reasoning/channel markers before text reaches a report."""
	if '<|message|>' in text:
		text = text.rsplit('<|message|>', 1)[-1]
	text = re.sub(r'<\|[^|]+\|>', '', text)
	text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
	# Some local model templates emit only the closing marker. In that case,
	# everything before it is still private reasoning and must be discarded.
	if re.search(r'</think>', text, flags=re.IGNORECASE):
		text = re.split(r'</think>', text, maxsplit=1, flags=re.IGNORECASE)[-1]
	return text.strip()


async def _search(query: str, references: list[str]) -> tuple[str, str]:
    raw = await web_search(query) or f"No results found for: {query}"

    cleaned, urls = _extract_urls(raw)
    for u in urls:
        if u not in references:
            references.append(u)
    source_block = ""
    if urls:
        source_block = "\n\nSources: " + ", ".join(
            f"[S{references.index(url) + 1}] {url}" for url in urls
        )

    full_result = cleaned + source_block
    if len(cleaned) > _SUMMARIZE_THRESHOLD:
        t = time.time()
        _status(f"Summarising result ({len(cleaned):,} chars)…")
        cleaned = await _summarize(cleaned[:_SUMMARIZE_INPUT_CAP], query)
        _status(f"Summarised in {time.time() - t:.1f}s")

    return cleaned + source_block, full_result


async def _synthesize(original_query: str, observations: list[str], references: list[str], length_hint: str) -> str:
    from agents.agent_factory import create_llm

    system = (
        f"{get_active_identity()} You have conducted deep research on behalf of the user. "
        f"Using the research below, write a thorough, well-structured response in {length_hint}. "
        "Use short paragraphs of no more than 2–3 sentences, and insert a blank line "
        "whenever the topic or comparison point changes. No bullet points unless comparing multiple items. "
        "Cite factual claims with source markers such as [S1] when supported by the research. "
        "Do not invent sources, citations, companies, manufacturers, or URLs. "
        "If the research does not support a claim, say that it could not be verified. "
        "Do not mention the research process — just answer the question directly."
    )
    research = "\n\n".join(f"Research note {i + 1}:\n{obs}" for i, obs in enumerate(observations))
    source_list = "\n".join(f"[S{i}] {url}" for i, url in enumerate(references, 1)) or "(none)"
    llm = create_llm(model_name=_research_model(), temperature=0.7, max_tokens=2048)
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"User question: {original_query}\n\nResearch:\n{research}\n\nAvailable sources:\n{source_list}"),
    ])
    return _split_long_paragraphs(_strip_reasoning(response.content))


def _status(msg: str) -> None:
	print(f"  ○ {msg}", flush=True)


async def deep_research(original_query: str, topic: str) -> tuple[str, list[str]]:
	t_total = time.time()
	length_hint = _extract_length_hint(original_query)
	observations: list[str] = []
	report_observations: list[str] = []
	searched_queries: set[str] = set()
	references: list[str] = []
	_status(f"Deep research started — topic: {topic}")

	for step_num in range(1, MAX_STEPS + 1):
		obs_text = (
			"\n\n".join(f"Observation {i + 1}:\n{obs}" for i, obs in enumerate(observations))
			if observations else "No searches run yet."
		)
		already_searched = ", ".join(f'"{q}"' for q in searched_queries) or "none"
		prompt = (
			f"Original question: {original_query}\n"
			f"Search topic extracted: {topic}\n\n"
			f"Completed searches: {len(observations)} (minimum before RESPOND: {MIN_SEARCHES})\n"
			f"Queries already searched: {already_searched}\n\n"
			f"{obs_text}\n\n"
			f"What is your next step? (step {step_num}/{MAX_STEPS})"
		)

		_status(f"[{step_num}/{MAX_STEPS}] Thinking…")
		t_step = time.time()
		try:
			step = await _run_step(prompt)
		except Exception as e:
			logger.error(f"[deep_research] Step failed: {e}")
			break

		_status(f"[{step_num}/{MAX_STEPS}] Thought in {time.time() - t_step:.1f}s — {step.action}")
		logger.debug(f"[deep_research] Thought: {step.thought}")

		if step.action == "RESPOND" and len(observations) < MIN_SEARCHES:
			# Enforce meaningful depth even when the local model decides too early
			# that its first result is sufficient.
			angles = (
				"historical context and scholarly sources",
				"specific examples and influence on later shows",
				"critical reception and contrasting perspectives",
			)
			fallback_query = f"{original_query} {angles[len(observations) % len(angles)]}"
			_status(
				f"[{step_num}/{MAX_STEPS}] One response is not enough — "
				f"continuing to search ({len(observations)}/{MIN_SEARCHES})"
			)
			step = _ResearchStep(
				thought="The minimum independent-source depth has not been reached.",
				action="SEARCH",
				query=fallback_query,
			)

		if step.action == "RESPOND":
			if observations:
				_status(f"[{step_num}/{MAX_STEPS}] Satisfied — composing answer from {len(observations)} search(es)")
				t_synth = time.time()
				answer = await _synthesize(original_query, observations, references, length_hint)
				result = await _complete_research(
					original_query, topic, answer, report_observations, references
				)
				logger.debug(f"[deep_research] Report written in {time.time() - t_synth:.1f}s")
				return result
			return await _complete_research(
				original_query, topic, step.query, report_observations, references
			)

		if step.query in searched_queries:
			_status(f"[{step_num}/{MAX_STEPS}] Duplicate query — stopping early")
			logger.debug("[deep_research] Duplicate query — stopping early")
			break

		searched_queries.add(step.query)
		_status(f"[{step_num}/{MAX_STEPS}] Searching: {step.query}")
		t_search = time.time()
		result, full_result = await _search(step.query, references)
		_status(f"[{step_num}/{MAX_STEPS}] Search done in {time.time() - t_search:.1f}s")
		observations.append(f"Query: {step.query}\nResult: {result}")
		# Publish the cleaned/summarized result, not the raw page capture. The latter
		# often contains navigation, related-content widgets, and social links.
		report_observations.append(f"Query: {step.query}\nResult: {result}")
		logger.debug(f"[deep_research] Observation: {result[:120]}…")

	if observations and references:
		_status(f"Max steps reached — synthesising from {len(observations)} finding(s)")
		t_synth = time.time()
		answer = await _synthesize(original_query, observations, references, length_hint)
		result = await _complete_research(
			original_query, topic, answer, report_observations, references
		)
		logger.debug(f"[deep_research] Report written in {time.time() - t_synth:.1f}s")
		return result
	_status(f"No findings collected — finishing in {time.time() - t_total:.1f}s")
	return await _complete_research(
		original_query,
		topic,
		"I wasn't able to find enough information to answer your question.",
		report_observations,
		references,
	)
