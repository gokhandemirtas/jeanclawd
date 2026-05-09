import inspect
from collections.abc import Callable

from agents.intent_classifier.intent_classifier import IntentLiteral
from skills import get_skill


async def route(
    intent: IntentLiteral,
    user_input: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    skill = get_skill(intent)
    try:
        result = skill.handle(user_input, on_token=on_token)
    except TypeError:
        result = skill.handle(user_input)
    if inspect.isawaitable(result):
        return await result
    return result
