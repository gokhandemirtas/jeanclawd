"""Search agent — LLM-driven entry point for SEARCH intent.

Two modes:
  - Quick search: Wikipedia with web_search fallback, single-shot.
  - Deep research: ReAct loop (see deep_research.py) that lets the model decide
    when it has enough information to answer comprehensively.

The quick path is a deterministic two-step lookup; the deep path is a true
agent with its own state, tool calls, and step budget. Both live here because
the dispatch key is the same (SEARCH intent) and the model decides between
them based on the user's phrasing.
"""

import asyncio
import os
import re

from pydantic import BaseModel

from agents.search.deep_research import deep_research
from utils.llm_structured_output import generate_structured_output
from utils.log import logger
from utils.search import search_wikipedia, web_search

_DEEP_RESEARCH_TRIGGERS = {
	"research",
	"deep research",
	"deep dive",
	"deep-dive",
	"find everything",
	"comprehensive",
	"in depth",
	"in-depth",
	"full breakdown",
	"tell me everything",
}
_SHOPPING_TERMS = ("find links", "product links", "where can i buy", "where to buy", "buy")
_MARKETPLACES = ("amazon", "ebay", "ali express", "aliexpress")

_EXTRACT_PROMPT = (
	"Extract a concise web search query from the user's message. "
	"Return only the essential keywords — no conversational filler, no punctuation. "
	"IMPORTANT: Never produce incomplete queries with trailing words like 'similar to ', 'about ', 'for ' with nothing after them. "
	"If the message is too vague to produce a complete query, return the message as-is. "
	"Examples: 'bring her back 2025 movie cast', 'weather Paris tomorrow', 'Python asyncio tutorial'."
)


class _SearchQuery(BaseModel):
	query: str


_TRIGGER_STRIP = re.compile(
	r"^(deep\s+research|research|search(\s+for)?|look\s+up|find(\s+information)?(\s+about)?|deep\s+dive(\s+into)?)\s+",
	flags=re.IGNORECASE,
)


def _strip_trigger(text: str) -> str:
	return _TRIGGER_STRIP.sub("", text).strip()


def _extract_topic(user_message: str, topic_hint: str = "") -> str:
	match = re.match(r"search\s*:\s*(.+)", user_message, flags=re.IGNORECASE)
	if match:
		return match.group(1).strip()
	try:
		user_prompt = user_message
		system_prompt = _EXTRACT_PROMPT
		if topic_hint:
			user_prompt = f"Conversation topic: {topic_hint}\n\nUser message: {user_message}"
			system_prompt = system_prompt + (
				"\n\nIf the user's message references something from the conversation topic "
				"(e.g. 'the lyrics', 'this dish', 'the recipe', 'it', 'that movie'), substitute the "
				"actual entity name from the topic into the search query so the query stands on its own."
			)
		result = generate_structured_output(
			model_name=os.environ["MODEL_INTENT"],
			user_prompt=user_prompt,
			system_prompt=system_prompt,
			pydantic_model=_SearchQuery,
		)
		logger.debug(
			f"[search] Extracted query: '{result.query}' from: '{user_message}' (topic_hint: {topic_hint!r})"
		)
		return _strip_trigger(result.query)
	except Exception as e:
		logger.warning(f"[search] Query extraction failed ({e}), using raw input")
		# Fall back to topic hint if the user's query clearly references it (e.g. "the lyrics")
		if topic_hint and re.search(
			r"\b(the|this|that|these|those|it)\b", user_message, re.IGNORECASE
		):
			enriched = f"{topic_hint} {_strip_trigger(user_message)}"
			logger.debug(f"[search] Enriched query with topic hint: '{enriched}'")
			return enriched
		return _strip_trigger(user_message)


def _is_deep_research(query: str) -> bool:
	q = query.lower()
	return any(trigger in q for trigger in _DEEP_RESEARCH_TRIGGERS)


def _is_shopping_request(query: str) -> bool:
	q = query.lower()
	return any(term in q for term in _SHOPPING_TERMS) and (
		any(marketplace in q for marketplace in _MARKETPLACES) or "link" in q
	)


async def run_search(query: str) -> str:
	from skills.conversation.skill import get_current_topic

	topic_hint = get_current_topic() or ""
	topic = _extract_topic(query, topic_hint)

	if _is_deep_research(query) or _is_shopping_request(query):
		logger.debug(f"[search] Deep research mode for '{topic}'")
		answer, references = await deep_research(original_query=query, topic=topic)
		if references:
			logger.debug(f"[search] {len(references)} reference(s) collected")
		return answer

	wiki = await asyncio.to_thread(search_wikipedia, topic)
	if wiki:
		logger.debug(f"[search] Wikipedia hit for '{topic}'")
		return f"User query: {topic}\n\nWikipedia: {wiki}"

	logger.debug(f"[search] Wikipedia miss for '{topic}', falling back to web search")
	results = await web_search(topic)
	if not results:
		return f"No results found for: {topic}"
	top = results.split("\n")[0]
	return f"User query: {topic}\n\nResult: {top}"
