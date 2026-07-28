import time
from typing import Literal

from pydantic import BaseModel

from agents.intent_classifier.prompt import INTENT_PROMPT
from providers import get_intent_provider
from utils import logger

IntentLiteral = Literal[
	"HOME_CONTROL",
	"WEATHER",
	"FINANCE_STOCKS",
	"TRANSPORTATION",
	"REMINDER",
	"NEWSFEED",
	"INFORMATION_QUERY",
	"NOTES",
	"SEARCH",
	"SPOTIFY",
	"PLANNER",
	"NUTRITION",
	"CHAT",
]


class IntentResponse(BaseModel):
	intent: IntentLiteral


_SEARCH_PREFIXES = (
	"research ",
	"deep research",
	"search ",
	"search for ",
	"look up ",
	"find information",
	"find info",
	"deep dive",
	"tell me everything about",
	"find links",
	"find a link",
	"product link",
	"product links",
	"where can i buy",
	"where to buy",
)

# These are session/conversation-management requests, not web searches. Keep
# them deterministic because a small routing model may interpret "where were
# we" literally as a grammar/search query.
_CONVERSATIONAL_PATTERNS = (
	"where were we",
	"where did we leave off",
	"what were we discussing",
	"what did we discuss",
	"what were we talking about",
	"do you remember what we",
	"i was just testing",
	"just testing your ability",
)

_DATETIME_QUERIES = frozenset(
	{
		"todays date",
		"what is todays date",
		"whats todays date",
		"what is the date",
		"whats the date",
		"what is the date today",
		"whats the date today",
		"current date",
		"date today",
		"what time is it",
		"whats the time",
		"what is the time",
		"what is the time now",
		"current time",
		"time now",
		"what day is it",
		"whats today",
		"what is today",
		"current day",
		"what day is today",
		"what day",
		"what day of the week",
		"what day of the week is it",
		"what is the day today",
		"whats the day today",
	}
)

def _keyword_classify(text: str) -> IntentLiteral | None:
	"""Fast pre-check for high-confidence intents that small models routinely miss."""
	t = text.lower().strip()
	if any(pattern in t for pattern in _CONVERSATIONAL_PATTERNS):
		return "CHAT"
	# Reminder listing is unambiguous and must not fall through to NOTES just
	# because it contains the generic word "list".
	if "reminder" in t and any(
		word in t for word in ("list", "show", "view", "see", "upcoming", "schedule")
	):
		return "REMINDER"
	if any(t.startswith(p) or p in t for p in _SEARCH_PREFIXES):
		return "SEARCH"
	# Shopping requests must never be answered from the chat model's memory.
	if any(vendor in t for vendor in ("amazon", "ebay", "ali express", "aliexpress")) and any(
		word in t for word in ("link", "buy", "price", "product", "seller", "find", "search")
	):
		return "SEARCH"
	t_norm = t.replace("'", "").replace("\u2019", "")
	if t_norm in _DATETIME_QUERIES:
		return "INFORMATION_QUERY"
	return None


def classify(user_input: str, current_topic: str | None = None) -> IntentLiteral:
	"""Classify user_input into an intent.

	current_topic is a lightweight hint (e.g. "Tesla stock price") so the
	classifier can correctly route follow-up questions without needing full
	conversation history.

	Falls back to the generalist provider if the intent model returns bad output.
	"""
	from providers import get_provider

	shortcut = _keyword_classify(user_input)
	if shortcut:
		logger.debug(f"[intent] {shortcut} (keyword match)")
		return shortcut
	if current_topic:
		content = f"Recent topic: {current_topic}\nQuery: {user_input}"
	else:
		content = user_input
	messages = [
		{"role": "system", "content": INTENT_PROMPT},
		{"role": "user", "content": content},
	]
	import time

	try:
		t = time.time()
		response = get_intent_provider().complete(messages, IntentResponse)
		logger.debug(f"[intent] {response.intent} ({time.time() - t:.2f}s)")
		return response.intent
	except Exception as e:
		logger.warning(f"[intent] Small model failed ({e}), falling back to generalist")
		t = time.time()
		response = get_provider().complete(messages, IntentResponse)
		logger.debug(f"[intent] {response.intent} ({time.time() - t:.2f}s) [fallback]")
		return response.intent


def test():
	logger.info("Start")
	start = time.time()
	user_input = input("Ask me anything: ")
	intent = classify(user_input)
	logger.info(intent)
	logger.newline()
	logger.timing(time.time() - start)
