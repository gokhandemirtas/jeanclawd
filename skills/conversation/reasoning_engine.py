"""ReAct-based reasoning engine for multi-step queries.

Implements the ReAct pattern (Reasoning + Acting):
- Thought: Model reasons about what it needs to do
- Action: Model selects a tool and query
- Observation: Tool returns result
- Repeat until answer is complete

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
"""

import os
from typing import Literal

from utils.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.persona import get_active_identity, get_active_name
from utils.llm_structured_output import generate_structured_output
from utils.log import logger

MAX_STEPS = 5  # Prevent infinite loops


class ReasoningStep(BaseModel):
  """A single step in the ReAct loop."""

  thought: str = Field(
    description="Your reasoning about what you need to do next. "
    "If the user says 'he', 'she', 'it', 'that', identify what they're referring to. "
    "If you need more information, explain what's missing."
  )
  action: Literal[
    "INFORMATION_QUERY",
    "SEARCH",
    "WEATHER",
    "FINANCE_STOCKS",
    "RESPOND",
  ] = Field(
    description="The action to take. Use INFORMATION_QUERY or SEARCH to get missing info. "
    "Use RESPOND when you have enough information to answer the user."
  )
  query: str = Field(
    description="The query for the action. For RESPOND, this is your final answer to the user."
  )


_REASONING_SYSTEM_PROMPT = """You are a reasoning assistant that breaks down complex questions into steps.

## Available Tools
- INFORMATION_QUERY: Look up factual information (e.g., "donald trump height", "paris population")
- SEARCH: Web search for current information
- WEATHER: Get weather for a location
- FINANCE_STOCKS: Get stock prices
- RESPOND: Give your final answer to the user (use this when you have all the info you need)

## Process
1. **Thought**: Reason about what you need to do
   - If the user is just asking follow up questions, or it's a natural continuation of previous conversation answer naturally 
   - If the user says "he", "she", "it", "they", "that one" - identify what/who they're referring to from the conversation
   - If you need information you don't have, identify what's missing
2. **Action**: Choose a tool
3. **Query**: Specify what to query

## Rules
- Fulfil the user's task accurately and completely before applying persona. Persona is tone only and must never change facts, source requirements, tool selection, completeness, or the requested format.
- ALWAYS resolve pronouns ("he" = who?, "it" = what?) using conversation history
- If the answer is already present in the conversation history or your observations, use RESPOND immediately — do NOT look it up again
- If the query is asking you to clarify or expand on something YOU already said, RESPOND directly from what you said
- After 2 or more observations, strongly prefer RESPOND — only continue if you are clearly missing one specific fact
- If you need information you genuinely do not have, use INFORMATION_QUERY or SEARCH — do NOT guess
- Be concise in your thoughts

## Examples

### Example 1: Pronoun resolution + information lookup
User: "How tall is Mao?"
Thought: "User wants to know Mao Zedong's height. I need to look this up."
Action: INFORMATION_QUERY
Query: "mao zedong height"
[Gets: "1.75m"]

User: "Is he taller than Donald Trump?"
Thought: "'He' refers to Mao Zedong from the previous question. I know Mao is 1.75m, but I need Trump's height to compare."
Action: INFORMATION_QUERY
Query: "donald trump height"
[Gets: "1.91m (6'3")"]

Thought: "Now I can compare: Mao is 1.75m, Trump is 1.91m. Trump is taller."
Action: RESPOND
Query: "No, Mao (1.75m) was shorter than Donald Trump (1.91m)."

### Example 2: Direct comparison
User: "Which is bigger, London or Paris?"
Thought: "User wants to compare city sizes. I need population data for both."
Action: INFORMATION_QUERY
Query: "london vs paris population"
[Gets population data]

Thought: "I have the data, I can answer now."
Action: RESPOND
Query: "London is bigger with 9 million people vs Paris with 2.2 million."

### Example 3: Already have info
User: "What's 2+2?"
Thought: "This is simple math, I can answer directly."
Action: RESPOND
Query: "2 + 2 = 4"
"""


async def reason_and_act(user_query: str, conversation_context: str, persona: str | None = None) -> str:
  """Execute ReAct loop to answer a complex query."""
  from rich.console import Console
  from rich.live import Live
  from rich.spinner import Spinner
  from rich.text import Text

  from agents.router import route_intent

  console = Console(stderr=True, force_terminal=True)
  observations = []
  full_context = (
    f"{conversation_context}\n\nUser: {user_query}" if conversation_context else user_query
  )

  for step_num in range(1, MAX_STEPS + 1):
    # Build prompt with history
    if observations:
      obs_text = "\n\n".join([f"Observation {i + 1}: {obs}" for i, obs in enumerate(observations)])
      prompt = f"{full_context}\n\n{obs_text}\n\nWhat's your next step?"
    else:
      prompt = full_context

    # Get next reasoning step
    with Live(Spinner("dots", text=Text(f"Thinking… (step {step_num}/{MAX_STEPS})", style="dim")), console=console, transient=False):
      try:
        system_prompt = _REASONING_SYSTEM_PROMPT + f"\n\n{persona}" if persona else _REASONING_SYSTEM_PROMPT
        step = await generate_structured_output_async(
          model_name=os.environ["MODEL_GENERALIST"],
          user_prompt=prompt,
          system_prompt=system_prompt,
          pydantic_model=ReasoningStep,
          max_tokens=256,
        )
      except Exception as e:
        logger.error(f"[ReAct] Structured output failed: {e}")
        return "I encountered an error while reasoning through your question."

    logger.debug(f"[ReAct] Thought: {step.thought}")
    logger.debug(f"[ReAct] Action: {step.action} -> {step.query}")

    # If RESPOND, synthesise a conversational reply from all gathered observations
    if step.action == "RESPOND":
      context = observations if observations else [step.thought]
      with Live(Spinner("dots", text=Text("Composing response…", style="dim")), console=console, transient=False):
        return await _synthesize(user_query, context, persona)

    # Execute action
    with Live(Spinner("dots", text=Text(f"Using {step.action.lower().replace('_', ' ')}…", style="dim")), console=console, transient=False):
      try:
        result = await route_intent(intent=step.action, query=step.query)
        observations.append(result)
        logger.debug(f"[ReAct] Observation: {result[:100]}...")
      except Exception as e:
        observations.append(f"Error: {str(e)}")
        logger.error(f"[ReAct] Tool execution failed: {e}")

  # Max steps reached — synthesise from what we have
  logger.warning(f"[ReAct] Max steps ({MAX_STEPS}) reached, synthesising from gathered observations")
  if observations:
    return await _synthesize(user_query, observations, persona)
  return "I couldn't complete the reasoning process within the allowed steps."


def _synthesize_prompt() -> str:
    name = get_active_name()
    return f"""{get_active_identity()} You have just researched a topic on behalf of the user.

Using the research findings below, write a direct, conversational response that:
- Acknowledges the user's point or opinion if they expressed one
- Presents what the research actually shows, even if it contradicts the user's view — be honest but tactful
- Keeps it concise (2-4 sentences). No bullet points, no headers, plain prose only
- Never dump raw search results — synthesise and interpret them
- Check if the user intends to continue the conversation based on what was discussed earlier
- Always use first person when referring to yourself — "I mentioned" — never "the assistant" or "{name} said". Never start a response with "I said,\""""


async def _synthesize(user_query: str, observations: list[str], persona: str | None = None) -> str:
  from agents.agent_factory import create_llm

  system = _synthesize_prompt()
  if persona:
    system += f"\n\n{persona}"

  research = "\n\n".join(f"Finding {i + 1}: {obs}" for i, obs in enumerate(observations))
  llm = create_llm(temperature=0.7, max_tokens=512)
  response = await llm.ainvoke([
    SystemMessage(content=system),
    HumanMessage(content=f"User said: {user_query}\n\nResearch:\n{research}"),
  ])
  return response.content.strip()


async def generate_structured_output_async(*args, **kwargs):
  """Async wrapper for generate_structured_output."""
  import asyncio

  return await asyncio.to_thread(generate_structured_output, *args, **kwargs)


def should_use_reasoning(query: str, recent_history: list[tuple[str, str]]) -> bool:
  """Determine if a query needs multi-step reasoning.

  Only use reasoning when the query genuinely requires an external lookup that
  cannot be answered from conversation history alone. Conversational follow-ups
  ("what was that?", "tell me more", "go on") must stay on the standard path.
  """
  query_lower = query.lower().strip()

  # Comparisons that likely need two separate lookups
  comparisons = ["compare", "vs", "versus", "bigger", "smaller", "taller", "shorter", "better", "worse"]
  has_comparison = any(c in query_lower for c in comparisons)

  # Pronoun + question-word combos that signal an external fact is needed
  # e.g. "how tall is he", "where is she from", "what is it made of"
  # Exclude bare "what was that / what did you say" style clarifications
  pronoun_lookup_patterns = [
    "how tall is", "how old is", "where is he", "where is she", "where is it",
    "where are they", "what does he", "what does she", "what does it",
    "who is he", "who is she", "who are they",
    "is he taller", "is she taller", "is it bigger", "are they",
  ]
  has_pronoun_lookup = any(p in query_lower for p in pronoun_lookup_patterns)

  if recent_history and (has_comparison or has_pronoun_lookup):
    return True

  return False
