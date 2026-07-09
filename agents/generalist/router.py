import asyncio

from pydantic import BaseModel

from agents.intent_classifier.intent_classifier import IntentLiteral
from providers import get_provider
from utils.log import logger

_ROUTER_PROMPT = """You are a query router. Given a conversation history and the user's latest message, return the single best intent.

Intents:
HOME_CONTROL — explicit commands to control smart home devices: lights, lamp, turn on/off, dim. Must be a direct imperative action, NOT a complaint or observation about temperature/comfort ("it's hot", "too cold", "I'm warm").
WEATHER — fetch LIVE weather data for a location not yet discussed
FINANCE_STOCKS — fetch LIVE stock/crypto prices not yet discussed
TRANSPORTATION — directions or route between two specific places
REMINDER — set a timer, alarm, or reminder
NEWSFEED — fetch today's news headlines
INFORMATION_QUERY — real-time lookup of something a language model could NOT know: current date, time, or day of week, obscure niche facts, very recent events, specific URLs, local business info
NOTES — create or retrieve personal notes
SEARCH — user wants to search or research a topic. Triggers: "search for", "look up", "google", "research", "deep dive", "deep research", "tell me everything about", "find everything about", "comprehensive overview", "in depth", "in-depth", "full breakdown", "use the search skill", "use it" when referring to search
SPOTIFY — music playback control: play, pause, stop, resume, skip, next, previous, volume, now playing, queue, device management. Bare imperative words like "pause", "stop", "skip", "next", "resume" with no other context refer to media playback → SPOTIFY.
PLANNER — build a multi-step plan or itinerary
NUTRITION — any mention of recipe, calories, nutrition, ingredients, how to cook/make a dish, what to make with certain ingredients. Use this even if the model could answer from knowledge — the skill fetches structured recipe/nutrition data.
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
8. Only use a domain intent when the user clearly needs NEW live data or a NEW action that isn't in the history.
9. INFORMATION_QUERY / SEARCH only for things the model cannot know: breaking news, real-time events, local business info.
10. Bare media-control imperatives EXACTLY matching one of these single words — "pause", "stop", "skip", "next", "resume", "play" — with no other words in the query and no light/device keyword → SPOTIFY. Multi-word phrases like "turn off", "switch off", "shut off" are NOT covered by this rule. Do not extend this rule to synonyms of "off" — the LLM tendency to do that is the exact bug this rule exists to prevent.
11. "turn on", "turn off", "switch on", "switch off", "shut on", "shut off", or just "on"/"off" with no light/device keyword and no clear media context → CHAT (ambiguous — could be media, lights, or anything else; ask what to turn on/off).
12. Explicit imperative action commands with a device or object keyword ("lights off", "lights on", "lamp on", "kitchen lights off", "play X", "pause", "stop", "set a timer", "set an alarm", "volume to 50", "next", "skip") ALWAYS go to the relevant domain intent (HOME_CONTROL, SPOTIFY, REMINDER, etc.), even after a CHAT response in history. Resolve pronouns ("them", "it") using recent history. This overrides rules 2 and 7.
13. When in doubt → CHAT.

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
	result = get_provider().complete(messages, _RouteDecision)
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
