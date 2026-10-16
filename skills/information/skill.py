import os
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from skills.information.prompts import informationIntentPrompt
from utils.llm_structured_output import generate_structured_output
from utils.log import logger
from utils.search import web_search


class QueryType(str, Enum):
  contextual = "contextual"
  general_knowledge = "general_knowledge"


class InformationIntentResponse(BaseModel):
  query_type: QueryType
  keyword: str


def _classify_intent(query: str) -> InformationIntentResponse:
  return generate_structured_output(
    model_name=os.environ["MODEL_GENERALIST"],
    user_prompt=query,
    system_prompt=informationIntentPrompt,
    pydantic_model=InformationIntentResponse,
  )


def _get_contextual_answer(keyword: str) -> str:
  now = datetime.now()
  if keyword == "current_time":
    return f"The current time is {now.strftime('%H:%M:%S')}."
  elif keyword == "current_date":
    return f"Today's date is {now.strftime('%A, %B %d, %Y')}."
  elif keyword == "current_day":
    return f"Today is {now.strftime('%A')}."
  else:
    return f"Current date and time: {now.strftime('%A, %B %d, %Y %H:%M:%S')}."


_DEFINITIONAL = __import__("re").compile(
  r"^(what(?:'s| is| are| was| were)?|who(?:'s| is| are)?|define|meaning of)\b",
  __import__("re").IGNORECASE,
)


def _grounded_query(query: str) -> str:
  """Append the active conversation topic to short definitional queries so
  the search is scoped to the relevant context (e.g. 'allspice jerk chicken'
  instead of 'what is allspice')."""
  try:
    from skills.conversation.skill import get_current_topic
    topic = get_current_topic()
  except Exception:
    topic = None
  if not topic:
    return query
  words = query.split()
  if len(words) > 6 or topic.lower() in query.lower():
    return query
  if not _DEFINITIONAL.match(query):
    return query
  return f"{query} {topic}"


async def handle_information_query(query: str) -> str:
  """Answer information queries only when live search provides evidence."""
  try:
    intent = _classify_intent(query)
    logger.debug(f"Information intent: query_type={intent.query_type}, keyword={intent.keyword}")
    if intent.query_type.value == "contextual":
      return _get_contextual_answer(intent.keyword)
  except Exception as e:
    logger.warning(f"[information] Intent classification failed ({e}), falling back to web search")

  search_query = _grounded_query(query)
  if search_query != query:
    logger.debug(f"[information] grounded query: '{query}' → '{search_query}'")
  web_results = await web_search(search_query)
  if web_results:
    return web_results

  logger.warning("[information] Search engines returned no verifiable results")
  return "I couldn't find reliable sources to verify that, so I won't guess."
