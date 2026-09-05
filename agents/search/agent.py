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
from difflib import SequenceMatcher

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
_SEARCH_REDIRECTS = re.compile(
	r"^(?:please\s+)?(?:use|try)\s+(?:the\s+)?search\s+skill(?:\s+on\s+it)?$|^look\s+it\s+up$",
	flags=re.IGNORECASE,
)
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
	r"^(deep\s+research|research|search(\s+for)?|look\s+up|find(\s+information)?(\s+about)?|deep\s+dive(\s+into)?)\s*[:\-]?\s+",
	flags=re.IGNORECASE,
)


def _canonicalize_deep_research_trigger(query: str) -> str:
	"""Correct a close typo in the explicit deep-research command prefix."""
	match = re.match(r"^\s*(deep\s+\w+)(?=\s*[:\-\s])", query, flags=re.IGNORECASE)
	if not match:
		return query
	candidate = re.sub(r"\s+", " ", match.group(1).strip().lower())
	if candidate == "deep research" or SequenceMatcher(None, candidate, "deep research").ratio() < 0.90:
		return query
	return "deep research" + query[match.end(1) :]


def _strip_trigger(text: str) -> str:
	return _TRIGGER_STRIP.sub("", text).strip()


def _search_subject(text: str) -> str:
	"""Return the part after an explicit search/research command prefix."""
	return re.sub(
		r"^(?:deep\s+research|research|search(?:\s+for)?|look\s+up|"
		r"find(?:\s+information)?(?:\s+about)?|deep\s+dive(?:\s+into)?)"
		r"\s*[:\-]?\s*",
		"",
		text.strip(),
		flags=re.IGNORECASE,
	).strip()


def _should_use_topic_hint(query: str, topic_hint: str) -> bool:
	"""Use the prior topic only for redirects or explicit contextual pronouns."""
	if not topic_hint:
		return False
	if _SEARCH_REDIRECTS.fullmatch(query.strip()):
		return True
	subject = _search_subject(query).lower()
	return bool(
		re.fullmatch(
			r"(?:about|on|for|regarding)?\s*(?:it|this|that|these|those|"
			r"the topic|the above|the same)",
			subject,
		)
	)


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
	q = _canonicalize_deep_research_trigger(query).lower()
	return any(trigger in q for trigger in _DEEP_RESEARCH_TRIGGERS)


def _topic_hint_for_search(query: str, topic_hint: str) -> str:
	"""Preserve normal search locking; isolate only new deep-research topics."""
	if not _is_deep_research(query):
		return topic_hint
	return topic_hint if _should_use_topic_hint(query, topic_hint) else ""


def _is_shopping_request(query: str) -> bool:
	q = query.lower()
	return any(term in q for term in _SHOPPING_TERMS) and (
		any(marketplace in q for marketplace in _MARKETPLACES) or "link" in q
	)


async def run_search(query: str) -> str:
	from skills.conversation.skill import get_current_topic

	query = _canonicalize_deep_research_trigger(query)
	topic_hint = get_current_topic() or ""
	# A new explicit subject must not inherit the previous research topic. Only
	# redirects and pronoun-only follow-ups are allowed to use that context.
	hint = _topic_hint_for_search(query, topic_hint)
	topic = topic_hint if hint and _SEARCH_REDIRECTS.fullmatch(query.strip()) else _extract_topic(query, hint)

	if _is_deep_research(query) or _is_shopping_request(query):
		logger.debug(f"[search] Deep research mode for '{topic}'")
		answer, references = await deep_research(original_query=query, topic=topic)
		if references:
			logger.debug(f"[search] {len(references)} reference(s) collected")
		return answer

	# Explicit SEARCH requests should use live web results first. Wikipedia remains
	# a useful fallback when the web providers return nothing.
	results = await web_search(topic)
	if results:
		top = results.split("\n")[0]
		return f"User query: {topic}\n\nResult: {top}"

	wiki = await asyncio.to_thread(search_wikipedia, topic)
	if wiki:
		logger.debug(f"[search] Web search returned nothing; Wikipedia hit for '{topic}'")
		return f"User query: {topic}\n\nWikipedia: {wiki}"
	logger.debug(f"[search] No web or Wikipedia result for '{topic}'")
	return f"No results found for: {topic}"
