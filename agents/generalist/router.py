import asyncio
import os

from pydantic import BaseModel

from agents.intent_classifier.intent_classifier import IntentLiteral
from providers import get_provider
from utils.log import logger

_ROUTER_PROMPT = """You are a query router. Given a conversation history and the user's latest message, return the single best intent.

Intents:
HOME_CONTROL — control smart home devices: lights, lamp, thermostat, turn on/off, dim
WEATHER — fetch LIVE weather data for a location not yet discussed
FINANCE_STOCKS — fetch LIVE stock/crypto prices not yet discussed
TRANSPORTATION — directions or route between two specific places
REMINDER — set a timer, alarm, or reminder
NEWSFEED — fetch today's news headlines
INFORMATION_QUERY — real-time lookup of something a language model could NOT know (obscure niche facts, very recent events, specific URLs, local business info)
NOTES — create or retrieve personal notes
SEARCH — explicit web search ("search for", "look up", "google")
SPOTIFY — music playback or device management
PLANNER — build a multi-step plan or itinerary
NUTRITION — any mention of recipe, calories, nutrition, ingredients, how to cook/make a dish, what to make with certain ingredients. Use this even if the model could answer from knowledge — the skill fetches structured recipe/nutrition data.
CHAT — everything else: general knowledge, history, explanations, science, comparisons, opinions, follow-ups on anything already in the conversation

Critical rules — apply in order, stop at first match:
1. Short agreements, continuations, or confirmations ("yes", "no", "ok", "sure", "yes all", "sounds good", "go ahead", "tell me more", "please", "exactly") → always CHAT.
2. If the answer can be derived from data already in the conversation history → CHAT. Never re-fetch.
3. Comparisons or "which one" questions where both subjects are already in the history → CHAT.
4. recipe/calories/nutrition/ingredients/how to cook/how to make [dish] → NUTRITION. Do NOT route these to CHAT even though the model knows recipes.
5. General knowledge: history, background, explanations, how something works, founding stories, science → CHAT. The model already knows these things.
6. Explicit corrections or redirects ("no I mean", "I meant", "not that", "I want X not Y") → CHAT.
7. Reactions, opinions, follow-ups on something just said → CHAT.
8. Only use a domain intent when the user clearly needs NEW live data or a NEW action that isn't in the history.
9. INFORMATION_QUERY / SEARCH only for things the model cannot know: breaking news, real-time events, local business info.
10. When in doubt → CHAT.

Examples:
  history: [microsoft stock → $406, tesla stock → $450]
  "which one is performing better?" → CHAT  (both prices in history, no new fetch needed)
  "yes all" → CHAT  (short confirmation)
  "tell me about Microsoft's company history" → CHAT  (general knowledge)
  "what is Microsoft's current market cap?" → FINANCE_STOCKS  (live metric not in history)
  "how about Apple?" → FINANCE_STOCKS  (new ticker, not in history)

  history: [london weather → 12°C, paris weather → 13°C]
  "which one is warmer?" → CHAT
  "nice. where should I go?" → CHAT
  "what about rome?" → WEATHER  (new location)

  history: [any stock discussion]
  "no company history" → CHAT  (correction + general knowledge)
  "I want company history not stock" → CHAT  (explicit redirect)
  "yes please" → CHAT  (confirmation)

  "give me a recipe for pasta amatriciana" → NUTRITION
  "how many calories in an avocado" → NUTRITION
  "what can I make with chicken and rice" → NUTRITION
  "how do I make sushi" → NUTRITION

Respond with JSON only: {"intent": "<INTENT>"}"""


class _RouteDecision(BaseModel):
    intent: IntentLiteral


def _route_sync(query: str, history: list[tuple[str, str]]) -> IntentLiteral:
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
    return result.intent


async def generalist_route(query: str, history: list[tuple[str, str]]) -> IntentLiteral:
    """Route a query using the generalist model with conversation history.

    Unlike the intent classifier, this sees recent exchanges so it can
    correctly handle conversational shifts, negations, and follow-ups.
    Falls back to CHAT on any routing failure so the conversation never hard-crashes.
    """
    import time
    t = time.time()
    try:
        intent = await asyncio.to_thread(_route_sync, query, history)
        logger.debug(f"[router] {intent} ({time.time() - t:.2f}s)")
        return intent
    except Exception as e:
        logger.warning(f"[router] Routing failed ({e}), falling back to CHAT")
        return "CHAT"
