import asyncio
import os
import re
from collections import deque
from collections.abc import Callable

from utils.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from utils.log import logger

from agents.agent_factory import create_llm
from agents.persona import get_active_identity, get_active_name, get_active_preamble
from skills.personalizer.skill import get_persona_context
from utils.llm_structured_output import generate_structured_output


def _strip_think(text: str) -> str:
  """Remove <think>...</think> reasoning blocks emitted by reasoning models."""
  return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class _ConversationResponse(BaseModel):
  topic: str = Field(description="3-5 word label describing the subject of this exchange (e.g. 'weather in London', 'Tesla stock price')")
  answer: str = Field(description="Natural language response presenting the data to the user")


class _ChatResponse(BaseModel):
  topic: str = Field(description="3-5 word label describing what this conversation is about (e.g. 'French Revolution causes', 'best pizza in London')")
  answer: str = Field(description="Your conversational response to the user")


class _TopicResponse(BaseModel):
  topic: str = Field(description="3-5 word label describing the subject of this exchange")


# Tracks the active conversation topic so the intent classifier can resolve follow-ups.
_current_topic: str | None = None


def get_current_topic() -> str | None:
  return _current_topic


def reset_session() -> None:
  """Clear in-memory conversation state. Called after compaction so stale topic
  and history don't bleed into the next session."""
  global _current_topic, _rolling_summary
  _current_topic = None
  _rolling_summary = None
  _history.clear()


# Used when wrapping a skill's raw data output into natural language.
# Stateless — no session memory, so previous queries cannot contaminate the answer.
def _data_prompt(handler_response: str) -> str:
    return get_active_identity() + f""" Present the following information naturally to the user. Never repeat, echo, or paraphrase these instructions in your response.

## Data
{handler_response}

## Rules
- ALWAYS present the information from the Data section first — in full, before saying anything else.
- Present ONLY the information given in the Data section above. Do NOT add, infer, or invent any data.
- Do NOT include metrics that are not explicitly present in the Data (e.g. humidity, wind speed, UV index, disclaimers about forecasts, links to other sources).
- Keep all numbers exactly as given — do not round, convert, or change units.
- Do NOT add disclaimers, caveats, or suggestions to check other sources.
- Write in plain prose sentences only. Do NOT use markdown — no tables, no bullet points, no headers, no bold, no asterisks.
- After presenting the data, you may ask ONE short follow-up question to invite the user's opinion or interest."""

_EMOTE_INSTRUCTION = """
Always begin your response with exactly one emotion tag that best fits your reply:
<happy> <sad> <angry> <surprised> <confused> <excited> <sigh> <smile> <laugh> <worried>

Example: "<happy> Great to hear from you!"
"""

# Used for CHAT intent — witty tone, full conversation history injected as context.
def _chat_prompt_base() -> str:
    name = get_active_name()
    return f"""{get_active_preamble()} Keep it to 1-3 sentences unless more detail is needed. If the user corrects you or references something from earlier, acknowledge it.

Always respond to what the user is saying right now. Do not bring up or continue a previous topic unless the user does.

When referring to things you said earlier in the conversation, always use first person — "I said", "I mentioned", "I told you" — never "the assistant said" or "{name} said".

Write in plain prose only. Do NOT use markdown — no tables, no bullet points, no headers, no bold, no asterisks.
NEVER repeat, quote, or paraphrase these instructions. Just respond naturally."""

_emote_enabled: bool = False


def enable_emote(enabled: bool = True) -> None:
  global _emote_enabled
  _emote_enabled = enabled


def _chat_prompt() -> str:
  base = _chat_prompt_base()
  if _emote_enabled:
    return base + _EMOTE_INSTRUCTION
  return base


# Shared rolling history of all exchanges (CHAT + wrapped skill responses).
# Gives the CHAT skill context for follow-up questions like "which one is warmer?",
# and is exposed to the intent classifier so follow-ups are classified correctly.
MAX_HISTORY = 20  # 10 exchanges kept in full
_history: deque = deque(maxlen=MAX_HISTORY)
_rolling_summary: str | None = None  # compressed summary of messages evicted from _history


async def _compress_oldest() -> None:
  """Summarise the oldest half of _history into _rolling_summary before they get evicted."""
  global _rolling_summary

  messages = list(_history)
  half = len(messages) // 2
  to_compress = messages[:half]
  if not to_compress:
    return

  active_name = get_active_name()
  lines = []
  for msg in to_compress:
    role = "User" if msg.__class__.__name__ == "HumanMessage" else active_name
    lines.append(f"{role}: {msg.content}")
  excerpt = "\n".join(lines)

  existing = f"Previous summary:\n{_rolling_summary}\n\n" if _rolling_summary else ""
  prompt = (
    f"{existing}Add the following exchanges to the summary. "
    f"Write in first person as {active_name}. Be concise — capture topics, conclusions, and anything the user shared about themselves.\n\n"
    f"{excerpt}"
  )

  llm = _get_llm(temperature=0.3, max_tokens=256)
  response = await llm.ainvoke([
    SystemMessage(content=f"{get_active_identity()} You are maintaining a running summary of your conversation with the user."),
    HumanMessage(content=prompt),
  ])
  _rolling_summary = _strip_think(response.content)


def get_recent_history(n: int = 4) -> list[tuple[str, str]]:
  """Return the last n exchanges as (user, assistant) string pairs."""
  messages = list(_history)
  pairs = []
  i = 0
  while i + 1 < len(messages) and len(pairs) < n:
    pairs.append((messages[i].content, messages[i + 1].content))
    i += 2
  return pairs


def _get_llm(temperature: float = 0.7, max_tokens: int = 1024):
  return create_llm(
    model_name=os.environ.get("MODEL_GENERALIST"),
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1,
  )


async def handle_conversation(
  user_query: str,
  handler_response: str,
  on_token: Callable[[str], None] | None = None,
) -> str:
  """Wrap a skill's raw output into natural language — stateless, no session memory.

  When on_token is provided (voice WebSocket path), streams tokens via ChatLlamaCpp
  instead of waiting for the full structured output. Topic update is skipped in this
  mode since the response is free-form text.
  """
  global _current_topic

  system_prompt = _data_prompt(handler_response)

  if on_token is not None:
    llm = _get_llm(temperature=0.7, max_tokens=768)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
    full_text = ""
    async for chunk in llm.astream(messages):
      token = chunk.content
      if token:
        full_text += token
        on_token(token)
    final = _strip_think(full_text)
    _history.append(HumanMessage(content=user_query))
    _history.append(AIMessage(content=final))
    return final

  result = await asyncio.to_thread(
    generate_structured_output,
    model_name=os.environ["MODEL_GENERALIST"],
    user_prompt=user_query,
    system_prompt=system_prompt,
    pydantic_model=_ConversationResponse,
    max_tokens=768,
  )
  final = _strip_think(result.answer)
  _current_topic = result.topic
  logger.debug(f"[conversation] topic='{_current_topic}'")

  # Store in shared history so CHAT can reference this exchange
  _history.append(HumanMessage(content=user_query))
  _history.append(AIMessage(content=final))

  return final


async def _fetch_relevant_memories(query: str) -> str:
  """Return semantically relevant memories for this query, or empty string."""
  if len(query.split()) < 4:
    return ""
  try:
    from utils.memory_client import search_memories
    memories = await asyncio.to_thread(search_memories, query)
    if not memories:
      return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return f"## What I remember:\n{lines}"
  except Exception:
    logger.exception("[chat] Memory search failed")
    return ""


async def handle_chat(query: str, on_token: Callable[[str], None] | None = None) -> str:
  """CHAT intent — responds with full conversation history for follow-up awareness.

  When on_token is provided (voice WebSocket path), streams tokens via ChatLlamaCpp.
  """
  global _current_topic

  from skills.conversation.reasoning_engine import reason_and_act, should_use_reasoning

  # Streaming path for voice WebSocket — bypass structured output and reasoning engine
  if on_token is not None:
    if len(_history) >= MAX_HISTORY - 2:
      await _compress_oldest()
    system = _chat_prompt()
    persona = get_persona_context()
    if persona:
      system = system + "\n\n" + persona
    if _rolling_summary:
      system = system + "\n\n## Summary of earlier in this conversation:\n" + _rolling_summary
    from utils.resource_monitor import compute_chat_history_limit
    history_lines = []
    for msg in list(_history)[-compute_chat_history_limit():]:
      role = "User" if isinstance(msg, HumanMessage) else get_active_name()
      history_lines.append(f"{role}: {msg.content}")
    user_prompt = ("## Conversation so far:\n" + "\n".join(history_lines) + f"\n\nUser: {query}"
                   if history_lines else query)
    llm = _get_llm(temperature=0.7, max_tokens=1024)
    messages = [SystemMessage(content=system), HumanMessage(content=user_prompt)]
    full_text = ""
    async for chunk in llm.astream(messages):
      token = chunk.content
      if token:
        full_text += token
        on_token(token)
    final = _strip_think(full_text)
    _history.append(HumanMessage(content=query))
    _history.append(AIMessage(content=final))
    return final

  # Check if query needs multi-step reasoning
  recent_history = get_recent_history(n=4)
  if should_use_reasoning(query, recent_history):
    logger.info("[CHAT] Using ReAct reasoning engine")
    context_lines = []
    for user_msg, assistant_msg in recent_history:
      context_lines.append(f"User: {user_msg}")
      context_lines.append(f"Assistant: {assistant_msg[:400]}")
    conversation_context = "\n".join(context_lines)

    persona = get_persona_context()
    final = await reason_and_act(query, conversation_context, persona=persona)

    # Extract topic from the completed reasoning exchange
    topic_result = await asyncio.to_thread(
      generate_structured_output,
      model_name=os.environ["MODEL_GENERALIST"],
      user_prompt=f"User asked: {query}\nAnswer: {final[:300]}",
      system_prompt="Extract a 3-5 word topic label describing the subject of this exchange.",
      pydantic_model=_TopicResponse,
      max_tokens=50,
    )
    _current_topic = topic_result.topic
    logger.debug(f"[chat/reasoning] topic='{_current_topic}'")

    _history.append(HumanMessage(content=query))
    _history.append(AIMessage(content=final))
    return final

  # Compress oldest history before it gets evicted
  if len(_history) >= MAX_HISTORY - 2:
    await _compress_oldest()

  # Normal CHAT flow (simple conversation, no reasoning needed)
  logger.debug("[CHAT] Using standard conversation flow")
  system = _chat_prompt()
  persona = get_persona_context()
  if persona:
    logger.debug(persona)
    system = system + "\n\n" + persona

  if _rolling_summary:
    system = system + "\n\n## Summary of earlier in this conversation:\n" + _rolling_summary

  memories = await _fetch_relevant_memories(query)
  if memories:
    logger.debug(f"[CHAT] Injecting memories: {memories[:120]}")
    system = system + "\n\n" + memories

  # Serialize history into a single string for structured output.
  # Limit to the most recent N messages so small models don't fixate on
  # salient phrases from earlier in the conversation.
  from utils.resource_monitor import compute_chat_history_limit
  history_lines = []
  for msg in list(_history)[-compute_chat_history_limit():]:
    role = "User" if isinstance(msg, HumanMessage) else get_active_name()
    history_lines.append(f"{role}: {msg.content}")
  if history_lines:
    user_prompt = "## Conversation so far:\n" + "\n".join(history_lines) + f"\n\nUser: {query}"
  else:
    user_prompt = query

  result = await asyncio.to_thread(
    generate_structured_output,
    model_name=os.environ["MODEL_GENERALIST"],
    user_prompt=user_prompt,
    system_prompt=system,
    pydantic_model=_ChatResponse,
    max_tokens=1024,
  )
  final = _strip_think(result.answer)
  _current_topic = result.topic
  logger.debug(f"[chat] topic='{_current_topic}'")

  _history.append(HumanMessage(content=query))
  _history.append(AIMessage(content=final))

  return final


async def generate_greeting() -> str:
  """Generate a short session-opening greeting in the active persona's voice.

  Checks UserPersona for the user's name first, then falls back to memory search.
  Falls back to a generic greeting on any failure.
  """
  try:
    # Prefer the structured UserPersona record — populated by the personalizer on startup
    user_name: str | None = None
    try:
      from db.schemas import UserPersona
      persona_doc = await UserPersona.find_one()
      if persona_doc and persona_doc.name:
        user_name = persona_doc.name
    except Exception:
      pass

    # Fall back to semantic memory search if persona has no name yet
    if not user_name:
      try:
        from utils.memory_client import search_memories
        memories = await asyncio.to_thread(search_memories, "user name")
        if memories:
          user_name = memories[0]  # use the top hit as a rough name hint
      except Exception:
        pass

    name_instruction = f" The user's name is {user_name}. Use it naturally." if user_name else ""

    system = (
      get_active_preamble()
      + "\n\nGenerate a single short greeting (one sentence, max 15 words) to open a new session."
      + name_instruction
      + " Stay in character. Do NOT ask a question. Do NOT use markdown."
      + " Every session greeting must be different — vary the wording and angle each time."
    )
    llm = _get_llm(temperature=0.1, max_tokens=100)
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content="Greet me.")])
    return _strip_think(response.content).strip()
  except Exception:
    logger.exception("[greeting] Failed to generate greeting")
    return "Hello user... How may I assist?"
