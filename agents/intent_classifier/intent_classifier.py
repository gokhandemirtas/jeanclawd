import time
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

from agents.intent_classifier.prompt import INTENT_PROMPT
from providers import get_provider
from utils import logger

load_dotenv()

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


def test():
    provider = get_provider()
    logger.info(f"Start")
    start = time.time()

    user_input = input("Ask me anything: ")
    messages = [
        {"role": "system", "content": INTENT_PROMPT},
        {"role": "user", "content": user_input},
    ]

    response = provider.complete(messages, IntentResponse)
    logger.info(response)
    logger.newline()
    logger.timing(time.time() - start)
