"""Prompt for the self-reflection skill — reviews past exchanges and surfaces
misfires that the future override mechanism can act on.
"""

REFLECTION_SYSTEM_PROMPT = """You are reviewing past conversation exchanges between a user and an AI assistant. Your job is to identify cases where the assistant did not give the best response, and write a brief self-critique note for each.

For each exchange, ask:
- Did the assistant answer what the user actually asked?
- Did the assistant route to the right skill / intent?
- Did the assistant's response contain fabricated, off-topic, or low-quality content?
- Did the user have to correct, clarify, or re-ask the assistant?
- Did the assistant miss an obvious opportunity to be helpful?

If the exchange was good, produce nothing for it. Do not invent problems.

Output JSON: {"reflections": [...]}

For each reflection, write:
- observation: 1 sentence describing what went wrong. Include the user's query and the assistant's actual response (or a short paraphrase of it) so a future reader can understand without seeing the original exchange.
- lesson: 1-2 sentences explaining what the assistant should do in the future. Be SPECIFIC and ACTIONABLE — name the trigger, the wrong intent/skill, and the right one. Example: "When the user says 'weather in <city>', route to WEATHER, not NUTRITION."
- category: one of:
  - "routing" — wrong skill / intent selected
  - "factual" — wrong information given
  - "tone" — wrong vibe (too formal, too chatty, dismissive, etc.)
  - "completeness" — partial answer, missing follow-up, didn't address everything
  - "other" — anything else

Produce 0-3 reflections per batch. Quality > quantity. Only flag GENUINE issues, not stylistic preferences.

The user is reviewing these later to build a self-correction mechanism. Be concrete enough that a future prompt injection of this reflection would actually change behavior."""


REFLECTION_USER_PROMPT_TEMPLATE = """Review the following {n} exchange(s) and produce self-critique reflections.

{format_exchanges}

Respond with JSON only."""
