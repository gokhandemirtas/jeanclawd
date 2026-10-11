import asyncio
import inspect
import re
from collections.abc import Callable

from agents.intent_classifier.intent_classifier import IntentLiteral
from skills import get_skill


_OFFER_MARKERS = re.compile(
	r"\b(?:i can|i could|i'll|i will|let me|would you like me to|want me to)\b",
	re.IGNORECASE,
)


def _lock_offered_intent(intent: IntentLiteral, response: str) -> None:
	"""Lock a clearly offered follow-up action for a short confirmation.

	CHAT often offers to perform a live-data action without ending in a
	question, e.g. "I can give you a quick overview of what's happening in the
	world."  Preserve that intent so "yeah, go ahead" can execute the skill.
	Do not lock arbitrary mentions of news: the response must also contain an
	explicit offer marker.
	"""
	if intent != "CHAT" or not _OFFER_MARKERS.search(response):
		return

	from agents.intent_classifier.intent_classifier import _explicit_command_classify
	from skills.conversation.skill import set_pending_intent

	offered_intent = _explicit_command_classify(response)
	if offered_intent in {"NEWSFEED", "WEATHER", "FINANCE_STOCKS", "SEARCH"}:
		set_pending_intent(offered_intent)


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
		maybe_compress_history,
		set_pending_intent,
	)
	await maybe_compress_history()

	skill = get_skill(intent)
	from utils.observability import span
	sig = inspect.signature(skill.handle)
	kwargs: dict = {}
	if "on_token" in sig.parameters:
		kwargs["on_token"] = on_token
	if "voice_mode" in sig.parameters:
		kwargs["voice_mode"] = voice_mode
	with span("skill_execution", skill=intent):
		if inspect.iscoroutinefunction(skill.handle):
			result = await skill.handle(user_input, **kwargs)
		else:
			result = await asyncio.to_thread(skill.handle, user_input, **kwargs)
	# Deep research already produces a user-facing document path. Wrapping it in
	# another LLM call can alter or omit that path, so preserve it verbatim.
	document_response = isinstance(result, str) and result.startswith(
		"Finished research, please check the final document at:"
	)
	if skill.needs_wrapping and not document_response:
		# Sear provenance is checked after synthesis. Streaming would expose an
		# unvalidated draft before the fail-closed check can replace it.
		conversation_on_token = None if intent == "SEAR" else on_token
		final = await handle_conversation(
			user_input, result, on_token=conversation_on_token, voice_mode=voice_mode
		)
		if intent == "SEAR":
			valid_ids: set[str] = set()
			current_id: str | None = None
			for line in result.splitlines():
				match = re.match(r"\[(S\d+)\]", line.strip())
				if match:
					current_id = match.group(1)
				elif line.strip() == "EvidenceStatus: page content fetched" and current_id:
					valid_ids.add(current_id)
					current_id = None
			cited_ids = set(re.findall(r"\[(S\d+)\]", final))
			if not valid_ids or not cited_ids or not cited_ids.issubset(valid_ids):
				final = "I couldn't produce a validated answer from readable SearXNG sources."
				if valid_ids:
					final += " The available source pages did not support a properly cited response."
				else:
					final += " No readable source pages were available."
				cited_ids = set()
			if not voice_mode and cited_ids:
				# The synthesis model may summarize away provenance. Keep the exact
				# URLs visible so the user can see which pages supplied the evidence.
				sources: list[str] = []
				current_url: str | None = None
				current_source_id: str | None = None
				for line in result.splitlines():
					stripped = line.strip()
					match = re.match(r"\[(S\d+)\]", stripped)
					if match:
						current_source_id = match.group(1)
					if stripped.startswith("SourceURL:"):
						current_url = stripped.removeprefix("SourceURL:").strip()
					elif stripped == "EvidenceStatus: page content fetched" and current_url and current_source_id in cited_ids:
						sources.append(current_url)
						current_url = None
				if sources:
					final = final.rstrip() + "\n\nSources consulted:\n" + "\n".join(
						f"- {url}" for url in sources
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

	# Keep a non-CHAT skill for one likely answer turn when it asks a question
	# (e.g. "Which playlist?", "What time?"). set_pending_intent rejects CHAT,
	# and the state also expires after its short TTL. Offered actions below use
	# the same handoff for confirmations such as "yeah, go ahead".
	if final.rstrip().endswith("?"):
		set_pending_intent(intent)
	_lock_offered_intent(intent, final)

	return final
