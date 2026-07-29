"""Planner agent — ReAct loop that picks tools, gathers information, then synthesises a plan.

Unlike the search agent, the planner has a tool registry (WEATHER, SEARCH,
INFORMATION_QUERY, FINANCE_STOCKS, NEWSFEED, REMINDER, DONE) and delegates
each tool call back through the Skill registry. HOME_CONTROL is intentionally
excluded — it has real-world side effects that should not fire autonomously.

The agent owns: its own system prompt, the tool descriptions, the step
selection LLM call, the synthesis LLM call, the per-task asyncio.Lock, and
the budget cap. The plan result is persisted to Plans collection.

This module is the implementation. The Skill registration that lets the
intent router dispatch to it lives in `__init__.py`.
"""

import asyncio
import inspect
import os
from datetime import datetime
from typing import Literal

from utils.messages import HumanMessage, SystemMessage
from utils.log import logger
from pydantic import BaseModel

from agents.agent_factory import create_llm
from utils.llm_structured_output import generate_structured_output

MAX_STEPS = 10

# Skills exposed to the planner. HOME_CONTROL is excluded — it has real-world
# side effects (turns lights on/off) that should not happen autonomously.
_PLANNER_TOOLS = Literal[
	"WEATHER",
	"SEARCH",
	"INFORMATION_QUERY",
	"FINANCE_STOCKS",
	"NEWSFEED",
	"REMINDER",
	"DONE",
]

_TOOL_DESCRIPTIONS = {
	"WEATHER": "Get weather forecasts and current conditions for a location",
	"SEARCH": "Web search for any topic, person, place, product, or event",
	"INFORMATION_QUERY": "Look up factual knowledge via Wikipedia",
	"FINANCE_STOCKS": "Get stock prices and financial market data",
	"NEWSFEED": "Get latest news headlines and current events",
	"REMINDER": "Set a reminder or timer",
	"DONE": "Stop gathering information — enough data has been collected to write the plan",
}

_SELECT_TOOL_PROMPT = """You are a planning agent working step by step to complete a task.

Your task: {task}

Available tools:
{tool_list}

Steps completed so far:
{steps_so_far}

Rules:
- Call one tool at a time with a focused, specific query
- Do not repeat the same tool+query combination
- Use DONE when you have enough information to write a complete plan
- You have a maximum of {remaining} steps remaining

Decide the next tool to call."""

_SYNTHESIZE_PROMPT = """You are a planning assistant. Using the research data below, produce the right output for the user's task.

If the task requires gathering information FROM the user (e.g. an interview, a questionnaire, discovering their life story, getting to know them) — do NOT write a written plan. Instead, start doing it: one sentence to set expectations, then ask the first question. Let the conversation do the rest.

For all other tasks, write a clear and practical plan structured with markdown: headers, bullet points, and bold text where appropriate. Be specific and concise."""


class PlannerAction(BaseModel):
    tool: _PLANNER_TOOLS
    query: str


_lock = asyncio.Lock()


def _format_tool_list() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in _TOOL_DESCRIPTIONS.items())


def _format_steps(steps: list[tuple[str, str, str]]) -> str:
    if not steps:
        return "None yet."
    lines = []
    for i, (tool, query, result) in enumerate(steps, 1):
        lines.append(f"Step {i} — {tool}: {query}")
        lines.append(f"  Result: {result[:400]}")
    return "\n".join(lines)


async def _select_next_action(task: str, steps: list[tuple[str, str, str]]) -> PlannerAction:
    prompt = _SELECT_TOOL_PROMPT.format(
        task=task,
        tool_list=_format_tool_list(),
        steps_so_far=_format_steps(steps),
        remaining=MAX_STEPS - len(steps),
    )
    return await asyncio.to_thread(
        generate_structured_output,
		model_name=os.environ.get("MODEL_PLANNING") or os.environ["MODEL_GENERALIST"],
        user_prompt=prompt,
        system_prompt="You are a planning agent. Choose the next tool to call.",
        pydantic_model=PlannerAction,
        max_tokens=100,
    )


async def _call_skill(tool: str, query: str) -> str:
    from skills.registry import SKILL_MAP

    skill = SKILL_MAP.get(tool)
    if not skill:
        return f"Tool {tool} not found."
    if inspect.iscoroutinefunction(skill.handle):
        return await skill.handle(query=query)
    return skill.handle(query=query)


# Cap to keep the synthesis prompt within the model's context window.
# Search/information results can include Tavily raw_content which is
# 5-30K chars per result; without truncation, 3+ steps can blow past
# 50K tokens and llama.cpp rejects with "tokens to keep from the initial
# prompt is greater than the context length".
_MAX_STEP_CHARS = 2_000      # ~500 tokens per step
_MAX_RESEARCH_CHARS = 18_000  # ~4.5K tokens of research total


async def _synthesize(task: str, steps: list[tuple[str, str, str]]) -> str:
    parts: list[str] = []
    total = 0
    for tool, query, result in steps:
        truncated = (
            result if len(result) <= _MAX_STEP_CHARS
            else result[:_MAX_STEP_CHARS] + "\n[...truncated to fit context window...]"
        )
        part = f"[{tool} — {query}]\n{truncated}"
        if total + len(part) > _MAX_RESEARCH_CHARS:
            remaining = len(steps) - len(parts)
            parts.append(f"[...{remaining} more step(s) omitted to fit context window...]")
            break
        parts.append(part)
        total += len(part)
    data = "\n\n".join(parts)
    logger.debug(f"[planner] synthesis input: {total:,} chars from {len(parts)}/{len(steps)} step(s)")

    llm = create_llm(temperature=0.5).bind(max_tokens=3072)
    response = await llm.ainvoke([
        SystemMessage(content=_SYNTHESIZE_PROMPT),
        HumanMessage(content=f"Task: {task}\n\nResearch:\n{data}"),
    ])
    return response.content.strip()


async def run_planner(task: str) -> str:
    from db.schemas import Plans

    if _lock.locked():
        return "A planning task is already in progress. Please wait until it finishes."

    async with _lock:
        logger.debug(f"[planner] starting: {task}")
        steps: list[tuple[str, str, str]] = []

        for step_num in range(1, MAX_STEPS + 1):
            action = await _select_next_action(task, steps)
            logger.debug(f"[planner] step {step_num} → {action.tool}({action.query!r})")

            if action.tool == "DONE":
                logger.debug("[planner] agent decided DONE")
                break

            result = await _call_skill(action.tool, action.query)
            logger.debug(f"[planner] step {step_num} ← {result[:300]}")
            steps.append((action.tool, action.query, result))

        if not steps:
            return "The planner could not gather any information for this task."

        logger.debug("[planner] synthesising final plan")
        plan = await _synthesize(task, steps)

        await Plans(created_at=datetime.now(), ask=task, plan=plan).insert()
        logger.debug("[planner] plan saved to database")

        return plan
