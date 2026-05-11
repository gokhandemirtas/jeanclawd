import asyncio
from collections.abc import Callable

from agents.intent_classifier.intent_classifier import classify
from agents.intent_router.intent_router import route


async def run_intent_classifier(
    user_input: str,
    on_token: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    voice_mode: bool = False,
) -> str:
    intent = await asyncio.to_thread(classify, user_input)
    if on_status:
        on_status(intent.replace("_", " ").title())
    response = await route(intent, user_input, on_token=on_token, voice_mode=voice_mode)

    from skills.journal import append_to_journal
    asyncio.create_task(append_to_journal(user_input, response))

    return response
