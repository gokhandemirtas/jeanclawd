import inspect
from collections.abc import Callable

from agents.intent_classifier.intent_classifier import IntentLiteral
from skills import get_skill


async def route(
    intent: IntentLiteral,
    user_input: str,
    on_token: Callable[[str], None] | None = None,
    voice_mode: bool = False,
) -> str:
    skill = get_skill(intent)
    sig = inspect.signature(skill.handle)
    kwargs: dict = {}
    if "on_token" in sig.parameters:
        kwargs["on_token"] = on_token
    if "voice_mode" in sig.parameters:
        kwargs["voice_mode"] = voice_mode
    try:
        result = skill.handle(user_input, **kwargs)
    except TypeError:
        result = skill.handle(user_input)
    if inspect.isawaitable(result):
        return await result
    return result
