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
	"SEAR",
	"SPOTIFY",
	"PLANNER",
	"NUTRITION",
	"INTROSPECTION",
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

_SEAR_PREFIX = "sear:"

_FINANCE_PHRASES = (
	"stock",
	"share price",
	"share prices",
	"market cap",
	"market capitalization",
	"valuation",
	"ticker",
	"stock price",
	"stock quote",
	"share price",
	"trading at",
	"p/e",
	"pe ratio",
	"dividend",
)

_NOTE_PHRASES = (
	"create a note",
	"save a note",
	"make a note",
	"write a note",
	"list my notes",
	"show my notes",
	"read my notes",
	"delete my note",
)

# Explicit daily-news requests should always execute the NEWSFEED skill.  If
# these are left to the conversational router, it may answer from model
# knowledge and incorrectly present that answer as today's news.
_NEWSFEED_PHRASES = (
	"news today",
	"today's news",
	"todays news",
	"latest news",
	"current news",
	"news headlines",
	"today's headlines",
	"todays headlines",
	"headlines today",
	"world news",
	"global news",
	"what's happening in the world",
	"whats happening in the world",
	"what is happening in the world",
	"what's happening around the world",
	"whats happening around the world",
	"what is happening around the world",
	"global developments",
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

# References to something the Assistant just said are ordinary conversation,
# including when the referenced answer came from a live-data skill. Keep these
# deterministic so the intent model cannot mistake them for implementation
# questions merely because they mention a claim, source, or answer.
_PREVIOUS_ANSWER_PATTERNS = (
	"you previously mentioned",
	"you previously said",
	"you mentioned earlier",
	"you said earlier",
	"you just mentioned",
	"you just said",
	"as you mentioned",
	"as you said",
	"what you mentioned",
	"what you said",
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
	t = text.lower().strip().replace("\u2018", "'").replace("\u2019", "'")
	# Explicit SearXNG requests take priority over conversational shortcuts.
	if t.startswith(_SEAR_PREFIX) and t[len(_SEAR_PREFIX):].strip():
		return "SEAR"
	if any(pattern in t for pattern in _PREVIOUS_ANSWER_PATTERNS):
		return "CHAT"
	if any(pattern in t for pattern in _CONVERSATIONAL_PATTERNS):
		return "CHAT"
	if any(phrase in t for phrase in _NEWSFEED_PHRASES):
		return "NEWSFEED"
	if any(phrase in t for phrase in _NOTE_PHRASES):
		return "NOTES"
	# Stock and valuation requests need fresh provider data. Keep this ahead of
	# the model so a short query such as "Microsoft stock" cannot become CHAT.
	if any(phrase in t for phrase in _FINANCE_PHRASES) and not any(
		phrase in t for phrase in ("stock photo", "stock image", "stock footage")
	):
		return "FINANCE_STOCKS"
	# Reminder listing is unambiguous and must not fall through to NOTES just
	# because it contains the generic word "list".
	if "reminder" in t and any(
		word in t for word in ("list", "show", "view", "see", "upcoming", "schedule")
	):
		return "REMINDER"
	if "reminder" in t and any(
		phrase in t for phrase in ("what reminders", "my reminders", "reminders do i have")
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
