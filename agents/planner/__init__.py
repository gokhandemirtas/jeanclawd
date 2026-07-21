"""Planner agent — registered as the PLANNER intent handler.

The ReAct loop lives in `agent.py`; this module wires the entry point into
the Skill registry so the intent router can dispatch to it.

Note: PLANNER was previously a slash-command-only feature (it was never
registered in SKILL_MAP, so the main intent router would fall back to the
"capability not available" message). This registration makes it routable
from normal chat — e.g. "plan a 2-week Japan itinerary" will now actually
trigger the planner instead of the fallback.
"""

from skills.base import Skill

from agents.planner.agent import run_planner

_PLANNER_PROMPT = (
	"You are a voice assistant. The user asked for a plan or itinerary, "
	"and a planning agent has produced one below. Present it naturally — "
	"acknowledge the task briefly, then deliver the plan. Do not mention "
	"the planning process or any tools used."
)

planner_skill = Skill(
	id="PLANNER",
	description="Multi-step planning agent — calls other skills (weather, search, news, etc.) as needed, then synthesises a plan or itinerary.",
	system_prompt=_PLANNER_PROMPT,
	handle=run_planner,
	needs_wrapping=True,
)

__all__ = ["planner_skill", "run_planner"]
