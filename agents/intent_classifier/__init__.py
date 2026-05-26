import asyncio
from collections.abc import Callable

from agents.intent_classifier.intent_classifier import classify, _keyword_classify
from agents.intent_router.intent_router import route
from utils.log import logger


async def run_intent_classifier(
    user_input: str,
    on_token: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    voice_mode: bool = False,
) -> str:
    from skills.conversation.skill import consume_pending_intent, get_recent_history, resolve_query
    pending = consume_pending_intent()
    if pending:
        intent = pending
    else:
        user_input = await resolve_query(user_input)
        shortcut = _keyword_classify(user_input)
        if shortcut:
            logger.debug(f"[router] {shortcut} (keyword match)")
            intent = shortcut
        else:
            from agents.generalist.router import generalist_route
            intent, confidence = await generalist_route(user_input, get_recent_history(n=3))
            if confidence < 0.75 and intent != "CHAT":
                intent = "CHAT"
    if on_status:
        on_status(intent.replace("_", " ").title())
    response = await route(intent, user_input, on_token=on_token, voice_mode=voice_mode)

    from skills.journal import append_to_journal
    asyncio.create_task(append_to_journal(user_input, response))

    return response
