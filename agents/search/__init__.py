"""Search agent — registered as the SEARCH intent handler.

The actual ReAct loop lives in `deep_research.py` (deep mode) and `agent.py`
(quick mode). This module just wires the entry point into the Skill registry
so the intent router can dispatch to it.
"""

from skills.base import Skill

from agents.search.agent import run_search

_SEARCH_PROMPT = (
	"You are a voice assistant. The user asked a question and the top web search result is provided below. "
	"Answer the user's question directly and conversationally using the information in the result. "
	"Do not list options, do not say 'here are some results', do not mention URLs or sources. "
	"If the result contains the answer, state it clearly and concisely. "
	"If the result is not sufficient to answer, say so briefly."
)

search_skill = Skill(
	id="SEARCH",
	description="Web search and deep research. Quick path uses Wikipedia + web search; deep mode runs a multi-step research agent.",
	system_prompt=_SEARCH_PROMPT,
	handle=run_search,
	needs_wrapping=True,
)

__all__ = ["search_skill", "run_search"]
