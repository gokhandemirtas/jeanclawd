import os
import time
from typing import Literal

from pydantic import BaseModel

from agents.intent_classifier.prompt import INTENT_PROMPT
from providers import get_provider
from utils import logger

model_key: str = os.getenv("MODEL_GENERALIST", "")

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


def classify(user_input: str) -> IntentLiteral:
	provider = get_provider()
	messages = [
		{"role": "system", "content": INTENT_PROMPT},
		{"role": "user", "content": user_input},
	]
	response = provider.complete(messages, IntentResponse)
	return response.intent


def test():
	logger.info("Start")
	start = time.time()
	user_input = input("Ask me anything: ")
	intent = classify(user_input)
	logger.info(intent)
	logger.newline()
	logger.timing(time.time() - start)
