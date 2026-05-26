"""Search-focused ReAct loop for deep research queries.

Runs up to MAX_STEPS search iterations, letting the model decide when it has
gathered enough information to answer the original question comprehensively.
"""

import asyncio
import os
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from agents.persona import get_active_identity
from utils.llm_structured_output import generate_structured_output
from utils.log import logger
from utils.messages import HumanMessage, SystemMessage
from utils.search import search_wikipedia, web_search

MAX_STEPS = 6

_console = Console(stderr=True, force_terminal=True)

_SYSTEM_PROMPT = """You are a research assistant conducting deep research on behalf of the user.

## Your process
1. Decide what to search for based on the original query and what you already know.
2. Read the search result and decide: do I have enough to fully answer the original question?
   - If yes → action: RESPOND
   - If no → action: SEARCH with a refined or follow-up query
3. Each search should target something *new* — do not repeat a query you already ran.

## Rules
- Be specific with search queries. "Jean-Claude Van Damme age 2024" beats "JCVD".
- After 3+ observations, strongly prefer RESPOND unless a critical fact is clearly missing.
- RESPOND with a comprehensive, well-structured answer — this is deep research, not a quick reply.
- Never mention the search process in your final answer. Just answer the question.
"""


class _ResearchStep(BaseModel):
    thought: str = Field(description="What you know so far and what is still missing.")
    action: Literal["SEARCH", "RESPOND"] = Field(
        description="SEARCH to run another query, RESPOND when you have enough."
    )
    query: str = Field(
        description="The search query (if SEARCH) or your comprehensive final answer (if RESPOND)."
    )


async def _run_step(prompt: str) -> _ResearchStep:
    return await asyncio.to_thread(
        generate_structured_output,
        model_name=os.environ["MODEL_GENERALIST"],
        user_prompt=prompt,
        system_prompt=_SYSTEM_PROMPT,
        pydantic_model=_ResearchStep,
        max_tokens=512,
    )


async def _search(query: str) -> str:
    wiki = await asyncio.to_thread(search_wikipedia, query)
    if wiki:
        return f"[Wikipedia] {wiki}"
    results = await web_search(query)
    return results or f"No results found for: {query}"


async def _synthesize(original_query: str, observations: list[str]) -> str:
    from agents.agent_factory import create_llm

    system = (
        f"{get_active_identity()} You have conducted deep research on behalf of the user. "
        "Using the findings below, write a thorough, well-structured response. "
        "Use short paragraphs. No bullet points unless comparing multiple items. "
        "Do not mention the research process — just answer the question directly."
    )
    research = "\n\n".join(f"Finding {i + 1}:\n{obs}" for i, obs in enumerate(observations))
    llm = create_llm(temperature=0.7, max_tokens=1024)
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"User question: {original_query}\n\nResearch:\n{research}"),
    ])
    return response.content.strip()


async def deep_research(original_query: str, topic: str) -> str:
    observations: list[str] = []
    searched_queries: set[str] = set()

    for step_num in range(1, MAX_STEPS + 1):
        obs_text = (
            "\n\n".join(f"Observation {i + 1}:\n{obs}" for i, obs in enumerate(observations))
            if observations else "No searches run yet."
        )
        already_searched = ", ".join(f'"{q}"' for q in searched_queries) or "none"
        prompt = (
            f"Original question: {original_query}\n"
            f"Search topic extracted: {topic}\n\n"
            f"Queries already searched: {already_searched}\n\n"
            f"{obs_text}\n\n"
            f"What is your next step? (step {step_num}/{MAX_STEPS})"
        )

        with Live(
            Spinner("dots", text=Text(f"Researching… (step {step_num}/{MAX_STEPS})", style="dim")),
            console=_console,
            transient=True,
        ):
            try:
                step = await _run_step(prompt)
            except Exception as e:
                logger.error(f"[deep_research] Step failed: {e}")
                break

        logger.debug(f"[deep_research] Thought: {step.thought}")
        logger.debug(f"[deep_research] Action: {step.action} → {step.query[:80]}")

        if step.action == "RESPOND":
            if observations:
                with Live(
                    Spinner("dots", text=Text("Composing research summary…", style="dim")),
                    console=_console,
                    transient=True,
                ):
                    return await _synthesize(original_query, observations)
            return step.query

        if step.query in searched_queries:
            logger.debug("[deep_research] Duplicate query detected, stopping early")
            break

        searched_queries.add(step.query)
        with Live(
            Spinner("dots", text=Text(f"Searching: {step.query}", style="dim")),
            console=_console,
            transient=True,
        ):
            result = await _search(step.query)
        observations.append(f"Query: {step.query}\nResult: {result}")
        logger.debug(f"[deep_research] Observation: {result[:120]}…")

    logger.warning(f"[deep_research] Loop ended after {step_num} steps, synthesising from {len(observations)} observations")
    if observations:
        return await _synthesize(original_query, observations)
    return "I wasn't able to find enough information to answer your question."
