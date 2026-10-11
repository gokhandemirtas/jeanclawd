from collections.abc import Callable

from agents.intent_classifier.intent_classifier import (
	_explicit_command_classify,
	classify as classify,
)
from agents.intent_router.intent_router import route
from utils.log import logger


async def run_intent_classifier(
	user_input: str,
	on_token: Callable[[str], None] | None = None,
	on_status: Callable[[str], None] | None = None,
	voice_mode: bool = False,
) -> str:
	from utils.inference_gate import foreground_inference
	from utils.observability import emit

	async with foreground_inference():
		from skills.conversation.skill import consume_pending_intent, get_recent_history, resolve_query
		user_input = await resolve_query(user_input)
		shortcut = _explicit_command_classify(user_input)
		pending = consume_pending_intent()
		if shortcut:
			if pending:
				logger.debug(f"[router] explicit {shortcut} overrides pending intent lock: {pending}")
			logger.debug(f"[router] {shortcut} (keyword match)")
			intent = shortcut
			confidence = 1.0
			classification_method = "keyword"
		elif pending:
			intent = pending
			confidence = 1.0
			classification_method = "pending"
		else:
			from agents.generalist.router import generalist_route
			intent, confidence = await generalist_route(
				user_input, get_recent_history(n=3, include_summary=True)
			)
			if confidence < 0.75 and intent != "CHAT":
				intent = "CHAT"
			classification_method = "generalist"
		emit("intent_resolved", intent=intent, confidence=confidence, method=classification_method)
		if on_status:
			on_status(intent.replace("_", " ").title())
		response = await route(intent, user_input, on_token=on_token, voice_mode=voice_mode)

	from utils.langmem_service import record_interaction
	await record_interaction(user_input, response, intent=intent)

	return response
