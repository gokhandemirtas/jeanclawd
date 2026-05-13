import os
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
	"CHAT",
]


class IntentResponse(BaseModel):
	intent: IntentLiteral


def classify(user_input: str, current_topic: str | None = None) -> IntentLiteral:
	"""Classify user_input into an intent.

	current_topic is a lightweight hint (e.g. "Tesla stock price") so the
	classifier can correctly route follow-up questions without needing full
	conversation history.

	Falls back to the generalist provider if the intent model returns bad output.
	"""
	from providers import get_provider
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
