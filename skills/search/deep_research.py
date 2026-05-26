"""Search-focused ReAct loop for deep research queries.

Runs up to MAX_STEPS search iterations, letting the model decide when it has
gathered enough information to answer the original question comprehensively.
"""

import asyncio
import os
import re
import time
from typing import Literal

from pydantic import BaseModel, Field

from agents.persona import get_active_identity
from utils.llm_structured_output import generate_structured_output
from utils.log import logger
from utils.messages import HumanMessage, SystemMessage
from utils.search import search_wikipedia, web_search

MAX_STEPS = 6

_SYSTEM_PROMPT = """You are a research assistant conducting deep research on behalf of the user.

## Your process
1. Decide what to search for based on the original query and what you already know.
2. Read the search result and decide: do I have enough to fully answer the original question?
   - If yes → action: RESPOND
   - If no → action: SEARCH with a refined or follow-up query
3. Each search should target something *new* — do not repeat a query you already ran.

## Rules
- Be specific with search queries. "Jean-Claude Van Damme age 2024" beats "JCVD".
- After 3+ observations, strongly prefer RESPOND unless a critical fact is clearly missing.
- RESPOND with a comprehensive, well-structured answer — this is deep research, not a quick reply.
- Never mention the search process in your final answer. Just answer the question.
"""

_URL_RE = re.compile(r'https?://\S+')
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
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


class _ResearchStep(BaseModel):
    thought: str = Field(description="What you know so far and what is still missing.")
    action: Literal["SEARCH", "RESPOND"] = Field(
        description="SEARCH to run another query, RESPOND when you have enough."
    )
    query: str = Field(
        description="The search query (if SEARCH) or your comprehensive final answer (if RESPOND)."
    )


def _extract_urls(text: str) -> tuple[str, list[str]]:
    """Strip URLs from text, return (cleaned_text, deduplicated_url_list)."""
    urls: list[str] = []

    def _capture_md(m: re.Match) -> str:
        url = m.group(2).rstrip('.,);\'\"')
        if url not in urls:
            urls.append(url)
        return m.group(1)

    def _capture_bare(m: re.Match) -> str:
        url = m.group(0).rstrip('.,);\'\"')
        if url not in urls:
            urls.append(url)
        return ''

    text = _MD_LINK_RE.sub(_capture_md, text)
    text = _URL_RE.sub(_capture_bare, text)
    return text.strip(), urls


async def _run_step(prompt: str) -> _ResearchStep:
    return await asyncio.to_thread(
        generate_structured_output,
        model_name=os.environ["MODEL_GENERALIST"],
        user_prompt=prompt,
        system_prompt=_SYSTEM_PROMPT,
        pydantic_model=_ResearchStep,
        max_tokens=512,
    )


async def _summarize(text: str, query: str) -> str:
    from agents.agent_factory import create_llm
    llm = create_llm(temperature=0.0, max_tokens=400)
    response = await llm.ainvoke([
        SystemMessage(content=(
            "Summarize the following search result. Keep every fact that is relevant to the query. "
            "Be concise — aim for 200–350 words. Plain prose, no headers."
        )),
        HumanMessage(content=f"Query: {query}\n\nSearch result:\n{text}"),
    ])
    return response.content.strip()


async def _search(query: str, references: list[str]) -> str:
    wiki = await asyncio.to_thread(search_wikipedia, query)
    if wiki:
        raw = f"[Wikipedia] {wiki}"
    else:
        raw = await web_search(query) or f"No results found for: {query}"

    cleaned, urls = _extract_urls(raw)
    for u in urls:
        if u not in references:
            references.append(u)

    if len(cleaned) > _SUMMARIZE_THRESHOLD:
        t = time.time()
        _status(f"Summarising result ({len(cleaned):,} chars)…")
        cleaned = await _summarize(cleaned[:_SUMMARIZE_INPUT_CAP], query)
        _status(f"Summarised in {time.time() - t:.1f}s")

    return cleaned


async def _synthesize(original_query: str, observations: list[str], length_hint: str) -> str:
    from agents.agent_factory import create_llm

    system = (
        f"{get_active_identity()} You have conducted deep research on behalf of the user. "
        f"Using the findings below, write a thorough, well-structured response in {length_hint}. "
        "Use short paragraphs. No bullet points unless comparing multiple items. "
        "Do not mention the research process — just answer the question directly."
    )
    research = "\n\n".join(f"Finding {i + 1}:\n{obs}" for i, obs in enumerate(observations))
    llm = create_llm(temperature=0.7, max_tokens=2048)
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"User question: {original_query}\n\nResearch:\n{research}"),
    ])
    return response.content.strip()


def _status(msg: str) -> None:
    print(f"  ○ {msg}", flush=True)


async def deep_research(original_query: str, topic: str) -> tuple[str, list[str]]:
    t_total = time.time()
    length_hint = _extract_length_hint(original_query)
    _status(f"Deep research started — topic: {topic}")
    observations: list[str] = []
    searched_queries: set[str] = set()
    references: list[str] = []

    for step_num in range(1, MAX_STEPS + 1):
        obs_text = (
            "\n\n".join(f"Observation {i + 1}:\n{obs}" for i, obs in enumerate(observations))
            if observations else "No searches run yet."
        )
        already_searched = ", ".join(f'"{q}"' for q in searched_queries) or "none"
        prompt = (
            f"Original question: {original_query}\n"
            f"Search topic extracted: {topic}\n\n"
            f"Queries already searched: {already_searched}\n\n"
            f"{obs_text}\n\n"
            f"What is your next step? (step {step_num}/{MAX_STEPS})"
        )

        t_step = time.time()
        _status(f"[{step_num}/{MAX_STEPS}] Thinking…")
        try:
            step = await _run_step(prompt)
        except Exception as e:
            logger.error(f"[deep_research] Step failed: {e}")
            break
        _status(f"[{step_num}/{MAX_STEPS}] Thought in {time.time() - t_step:.1f}s")

        logger.debug(f"[deep_research] Thought: {step.thought}")

        if step.action == "RESPOND":
            _status(f"[{step_num}/{MAX_STEPS}] Satisfied — composing answer from {len(observations)} finding(s)")
            if observations:
                t_synth = time.time()
                answer = await _synthesize(original_query, observations, length_hint)
                _status(f"Answer composed in {time.time() - t_synth:.1f}s  |  total: {time.time() - t_total:.1f}s")
                return answer, references
            return step.query, references

        if step.query in searched_queries:
            _status(f"[{step_num}/{MAX_STEPS}] Duplicate query — stopping early")
            break

        searched_queries.add(step.query)
        t_search = time.time()
        _status(f"[{step_num}/{MAX_STEPS}] Searching: {step.query}")
        result = await _search(step.query, references)
        _status(f"[{step_num}/{MAX_STEPS}] Search done in {time.time() - t_search:.1f}s")
        observations.append(f"Query: {step.query}\nResult: {result}")
        logger.debug(f"[deep_research] Observation: {result[:120]}…")

    _status(f"Max steps reached — synthesising from {len(observations)} finding(s)")
    if observations:
        t_synth = time.time()
        answer = await _synthesize(original_query, observations, length_hint)
        _status(f"Answer composed in {time.time() - t_synth:.1f}s  |  total: {time.time() - t_total:.1f}s")
        return answer, references
    return "I wasn't able to find enough information to answer your question.", references
