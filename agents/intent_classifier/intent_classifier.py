import time
from dotenv import load_dotenv
import lmstudio as lms
from pydantic import BaseModel
from typing import Literal
from agents.intent_classifier.prompt import INTENT_PROMPT
from utils import logger

import os

load_dotenv()
model_key = os.environ["MODEL_GENERALIST"]


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
    logger.info(f"Start using {model_key}")
    start = time.time()

    model = lms.llm(
        model_key,
        config={
            "gpu_offload": "max",
            "temperature": 0,
            "max_tokens": -1,
        },
    )
    chat = lms.Chat(INTENT_PROMPT)
    chat.add_user_message(input("Ask me anything"))

    for fragment in model.respond_stream(chat, response_format=IntentResponse):
        logger.stream(fragment.content)
    logger.newline()
    elapsed = time.time() - start
    logger.timing(elapsed)
