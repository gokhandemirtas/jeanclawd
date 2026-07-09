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
	from skills.conversation.skill import (
		_safe_extract_topic,
		append_to_history,
		handle_conversation,
		set_pending_intent,
	)

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
		result = await result
	if skill.needs_wrapping:
		final = await handle_conversation(
			user_input, result, on_token=on_token, voice_mode=voice_mode
		)
	else:
		append_to_history(user_input, result)
		final = result
		# Non-wrapping skills (SPOTIFY, NOTES, NEWSFEED, FALLBACK) bypass
		# handle_conversation and therefore skip the topic extraction that lives
		# inside it. Without a topic, follow-up queries that reference context
		# ("the lyrics", "what is this song called") can't be resolved. CHAT
		# handles its own topic extraction internally — skip it here to avoid a
		# redundant second LLM call that would see the freshly-set topic as the
		# "previous" topic.
		if intent != "CHAT":
			await _safe_extract_topic(user_input, result)

	# Auto-lock the next turn to this intent when the response ends with a question
	# (e.g. "Which playlist?", "What time?"). The lock is single-use: consumed and
	# cleared by consume_pending_intent() on the next turn. If the user's reply
	# also ends with "?", a fresh lock is set; otherwise the chain breaks.
	if final.rstrip().endswith("?"):
		set_pending_intent(intent)

	return final
