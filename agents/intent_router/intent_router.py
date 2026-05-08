import asyncio
import inspect

from agents.intent_classifier.intent_classifier import IntentLiteral
from skills import get_skill


def route(intent: IntentLiteral, user_input: str) -> str:
	skill = get_skill(intent)
	result = skill.handle(user_input)
	if inspect.isawaitable(result):
		return asyncio.run(result)
	return result
