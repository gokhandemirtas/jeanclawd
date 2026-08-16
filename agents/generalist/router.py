import asyncio

from pydantic import BaseModel

from agents.intent_classifier.intent_classifier import IntentLiteral
from providers import get_intent_provider
from utils.log import logger

_ROUTER_PROMPT = """You are a query router. Given a conversation history and the user's latest message, return the single best intent.

Intents:
HOME_CONTROL — explicit commands to control smart home devices: lights, lamp, turn on/off, dim. Must be a direct imperative action, NOT a complaint or observation about temperature/comfort ("it's hot", "too cold", "I'm warm").
WEATHER — fetch LIVE weather data for a location not yet discussed
FINANCE_STOCKS — fetch LIVE stock/crypto prices not yet discussed
TRANSPORTATION — directions or route between two specific places
REMINDER — set a timer, alarm, or reminder
NEWSFEED — fetch today's news headlines or a current-events briefing. Use this
for explicit requests about today's/latest/current news, headlines, world news,
or what is happening in the world. Do not answer these from memory.
INFORMATION_QUERY — factual lookup requiring live or externally verified information: anything qualified as current, latest, today, now, present, serving, or incumbent; current people or offices; very recent events; obscure niche facts; specific URLs; local business info
NOTES — create or retrieve personal notes
SEARCH — user wants to search or research a topic. Triggers: "search for", "look up", "google", "research", "deep dive", "deep research", "tell me everything about", "find everything about", "comprehensive overview", "in depth", "in-depth", "full breakdown", "use the search skill", "use it" when referring to search
SPOTIFY — music playback control: play, pause, stop, resume, skip, next, previous, volume, now playing, queue, device management. Bare imperative words like "pause", "stop", "skip", "next", "resume" with no other context refer to media playback → SPOTIFY.
PLANNER — build a multi-step plan or itinerary
NUTRITION — any mention of recipe, calories, nutrition, ingredients, how to cook/make a dish, what to make with certain ingredients. Use this even if the model could answer from knowledge — the skill fetches structured recipe/nutrition data.
INTROSPECTION — questions about this Assistant's implementation, architecture, routing, skills, memory/storage, persistence, capabilities, or how it works internally. Use the application's persisted introspection evidence; do not use web search or generic AI knowledge.
CHAT — everything else: general knowledge, history, explanations, science, comparisons, opinions, follow-ups on anything already in the conversation

Critical rules — apply in order, stop at first match:
1. Short agreements, continuations, or confirmations ("yes", "no", "ok", "sure", "yes all", "sounds good", "go ahead", "tell me more", "please", "exactly") → always CHAT.
2. If the answer can be derived from ACTUAL SKILL DATA in the conversation history (e.g. a previous weather response, stock price, song title) → CHAT. Never re-fetch. CHAT RESPONSES DO NOT COUNT as skill data — they are conversational only and CANNOT represent executed actions. If the previous turn was CHAT, the new query is a fresh request, not a follow-up to executed data.
3. Comparisons or "which one" questions where both subjects are already in the history → CHAT.
4. recipe/calories/nutrition/ingredients/how to cook/how to make [dish] → NUTRITION. Do NOT route these to CHAT even though the model knows recipes.
5. Explicit research triggers ("research", "deep dive", "deep research", "tell me everything about", "find everything about", "in depth") → SEARCH. This overrides rule 6 even for topics the model knows.
6. General knowledge: history, background, explanations, how something works, founding stories, science → CHAT. The model already knows these things. NOTE: current date/time/day is NOT general knowledge — the model cannot know it, route to INFORMATION_QUERY.
6. Explicit corrections or redirects ("no I mean", "I meant", "not that", "I want X not Y") → CHAT.
7. Reactions, opinions, follow-ups on something just said → CHAT. EXCEPTION: a follow-up that contains an explicit imperative action command (verb + object, e.g. "turn them off", "skip it", "stop that") is NOT a reaction — it's a new action and must be routed to the relevant domain intent.
7a. Declarative feedback, agreement, explanation, or meta-commentary about the assistant's previous answer → CHAT. This includes statements about reliability, sources, limitations, accuracy, or what the assistant can/cannot do. Do not route such commentary to INFORMATION_QUERY or SEARCH merely because it mentions online sources, facts, or retrieval.
7b. Questions about the user's own identity, name, remembered profile, or relationship with the Assistant (for example, "remember me?", "what's my name?", or "what do you know about me?") → CHAT. These are conversational memory questions, not NOTES, INFORMATION_QUERY, or SEARCH, unless the user explicitly asks to save or retrieve a note.
7c. Use the meaning of the whole utterance and its conversational context to resolve perspective. Personal-memory and rapport questions about the relationship (for example, "what did I tell you yesterday?" or "do you remember me?") → CHAT. Pronouns such as "you", "your", "me", "us", and "our" are relationship cues, not keyword triggers.
7d. Questions about this application's internals — how requests are routed, how skills work, where conversations are stored, how sessions or memory are persisted, or what the codebase can do — → INTROSPECTION. This is for implementation questions about the Assistant, not generic questions about AI agents. Use INTROSPECTION even when the question is phrased conversationally (for example, "how do you decide whether to search?" or "where do our conversations live?"). Ordinary questions about what the user previously said remain CHAT.
8. Only use a domain intent when the user clearly needs NEW live data or a NEW action that isn't in the history.
9. A temporal qualifier such as "current", "latest", "today", "now", "present", "serving", or "incumbent" means the answer may have changed since training. Route it to the appropriate live-data intent, usually INFORMATION_QUERY or SEARCH. Never route such a question to CHAT merely because the underlying subject is general knowledge.
	For broad current-events requests ("what's happening in the world", "today's news", "latest headlines"), the appropriate intent is NEWSFEED.
10. Bare media-control imperatives EXACTLY matching one of these single words — "pause", "stop", "skip", "next", "resume", "play" — with no other words in the query and no light/device keyword → SPOTIFY. Multi-word phrases like "turn off", "switch off", "shut off" are NOT covered by this rule. Do not extend this rule to synonyms of "off" — the LLM tendency to do that is the exact bug this rule exists to prevent.
11. Explicit imperative action commands containing a device, object, or location keyword anywhere in the query — "lights on", "lights off", "turn the lights on", "turn the lights off", "turn the bedroom lights on", "lamp on", "kitchen lights off", "play X", "pause", "stop", "set a timer", "set an alarm", "volume to 50", "next", "skip" — ALWAYS go to the relevant domain intent (HOME_CONTROL, SPOTIFY, REMINDER, etc.), even after a CHAT response in history. Resolve pronouns ("them", "it") using recent history. This overrides rules 2, 7, and 12. The rule fires on presence of ANY device / object / location keyword, including but not limited to: lights, lamp, fan, tv, heater, humidifier, plug, switch, radio, speaker, kitchen, bedroom, living room, office, bathroom, hallway, garage, music, song, playlist, timer, alarm, reminder, note, volume. When uncertain whether a word is a device / object / location, route to the domain intent — those skills can always ask a clarifying question if truly ambiguous. This rule fires BEFORE rule 12.
12. BARE "turn on" / "turn off" / "switch on" / "switch off" / "shut on" / "shut off", or just "on" / "off" as the entire query — with NO device, object, or location keyword and no clear media context → CHAT (ambiguous; ask what to turn on/off). The entire query must consist of just the phrase plus stopwords like "the" / "a" / "please" / "it" — any content word beyond the phrase disqualifies this rule and pushes the query to rule 11. Examples that match: "turn on" → CHAT, "turn off" → CHAT, "switch off" → CHAT, "on" → CHAT, "off" → CHAT, "turn it off" → CHAT. Examples that do NOT match (rule 11 wins): "turn the lights on" → HOME_CONTROL, "switch the lamp off" → HOME_CONTROL, "turn off the fan" → HOME_CONTROL, "turn on the bedroom" → HOME_CONTROL. This rule is strict: any noun / device / location / room name disqualifies the query from this rule.
13. When in doubt → CHAT.
14. Queries with a demonstrative that points at the most recent assistant turn ("what is this song called", "what is that movie about", "who made this", "how long is this one", "is this any good") → CHAT. The answer is almost always in the conversation history; searching would be wasteful and confusing. Treat the demonstrative as a strong signal that the user is asking about something the assistant just mentioned.
15. Explicit note-taking triggers at or near the start of the query — "take a note", "take notes", "note:", "note -", "save a note", "save a note about", "add a note", "make a note", "make a note of", "jot down", "jot this down", "write down", "write this down" — → NOTES, even when the note content mentions cooking / recipes, stocks, calendar items, or any other domain-intent keyword. The note-taking prefix is a save command and takes priority over the content of the note itself. This overrides rule 4 (NUTRITION) and any other content-based domain intent. The NOTES skill's sub-classifier decides CREATE / LIST / READ / DELETE.

Examples:
  history: [microsoft stock → $406, tesla stock → $450]
  "which one is performing better?" → CHAT  (both prices in history, no new fetch needed)
  "yes all" → CHAT  (short confirmation)
  "tell me about Microsoft's company history" → CHAT  (general knowledge)
  "what is Microsoft's current market cap?" → FINANCE_STOCKS  (live metric not in history)
  "how about Apple?" → FINANCE_STOCKS  (new ticker, not in history)

  history: [lights turned on]
  "today was very hot, I can't take this much heat" → CHAT  (complaint/opinion, not a device command)
  "it's too warm in here" → CHAT  (observation, not a command)

  history: [london weather → 12°C, paris weather → 13°C]
  "which one is warmer?" → CHAT
  "nice. where should I go?" → CHAT
  "what about rome?" → WEATHER  (new location)

  history: [discussion about horror movies after 2024]
  "you have a search skill, use it" → SEARCH  (explicit redirect to search)
  "look it up" → SEARCH  (explicit redirect to search)
  "use it" → SEARCH  (when context is about finding/fetching something not in history)

  history: [any stock discussion]
  "no company history" → CHAT  (correction + general knowledge)
  "I want company history not stock" → CHAT  (explicit redirect)
  "yes please" → CHAT  (confirmation)

  "give me a recipe for pasta amatriciana" → NUTRITION
  "how many calories in an avocado" → NUTRITION
  "what can I make with chicken and rice" → NUTRITION
  "how do I make sushi" → NUTRITION

  "what is today's date?" → INFORMATION_QUERY  (the model cannot know the current date)
  "what time is it?" → INFORMATION_QUERY  (the model cannot know the current time)
  "what day is it?" → INFORMATION_QUERY  (the model cannot know the current day)
  "who is the current prime minister of the UK?" → INFORMATION_QUERY  (current office-holder; live lookup required)
  "what is the latest population estimate for Tokyo?" → INFORMATION_QUERY  (the value can change; verify it)

  history: [spotify → now playing Down for Life by Testament]
  "what is this song called?" → CHAT  (demonstrative points at the just-mentioned song — answer is in history)

  "turn the lights on" → HOME_CONTROL  (device keyword "lights" present, rule 11)
  "switch the lamp off" → HOME_CONTROL  (device keyword "lamp" present, rule 11)
  "turn on" → CHAT  (bare phrase, no device keyword, rule 12)
  "turn it off" → CHAT  (bare phrase, pronoun with no resolvable device from history, rule 12)

  "take a note to cook italian for my friends" → NOTES  (note trigger "take a note" prefix wins over "cook italian" content, rule 15)
  "note: I will cook tiramisu for my friends" → NOTES  ("note:" prefix wins over "cook tiramisu" content, rule 15)
  "save a note about my flight on Friday" → NOTES  (explicit "save a note about" trigger, rule 15)
  "jot down: buy milk" → NOTES  (note trigger, rule 15)

Respond with JSON only: {"intent": "<INTENT>", "confidence": <0.0-1.0>}

confidence: how certain you are this is the right domain intent (1.0 = unambiguous action, 0.5 = could be CHAT). Use < 0.7 when the query is conversational, emotional, or ambiguous."""


class _RouteDecision(BaseModel):
	intent: IntentLiteral
	confidence: float = 1.0


def _route_sync(
	query: str, history: list[tuple[str, str]]
) -> tuple[IntentLiteral, float]:
	lines = []
	for user_msg, assistant_msg in history[-3:]:
		lines.append(f"User: {user_msg}")
		lines.append(f"Assistant: {assistant_msg[:200]}")
	context = "\n".join(lines)
	content = f"{context}\n\nUser: {query}" if context else query

	messages = [
		{"role": "system", "content": _ROUTER_PROMPT},
		{"role": "user", "content": content},
	]
	# Routing is pure classification over a 14-rule prompt — uses the lean
	# MODEL_INTENT model (e.g. 8B) rather than the chat-tuned generalist.
	# 4-8x faster, better at rule-following structured output.
	result = get_intent_provider().complete(messages, _RouteDecision)
	return result.intent, result.confidence


async def generalist_route(
	query: str, history: list[tuple[str, str]]
) -> tuple[IntentLiteral, float]:
	"""Route a query using the generalist model with conversation history.

	Unlike the intent classifier, this sees recent exchanges so it can
	correctly handle conversational shifts, negations, and follow-ups.
	Falls back to CHAT on any routing failure so the conversation never hard-crashes.
	"""
	import time
	from agents.intent_classifier.intent_classifier import _keyword_classify

	shortcut = _keyword_classify(query)
	if shortcut:
		logger.debug(f"[router] {shortcut} (keyword match)")
		return shortcut, 1.0

	t = time.time()
	try:
		intent, confidence = await asyncio.to_thread(_route_sync, query, history)
		logger.debug(
			f"[router] {intent} (confidence={confidence:.2f}, {time.time() - t:.2f}s)"
		)
		return intent, confidence
	except Exception as e:
		logger.warning(f"[router] Routing failed ({e}), falling back to CHAT")
		return "CHAT", 1.0
