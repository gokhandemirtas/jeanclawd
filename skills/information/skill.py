import os
from datetime import datetime
from enum import Enum

from utils.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.agent_factory import create_llm
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


async def _llm_fallback(query: str) -> str:
  llm = create_llm(model_name=os.environ.get("MODEL_GENERALIST"), temperature=0.3)
  response = await llm.ainvoke([
    SystemMessage(content="Answer the user's question as accurately as possible using your training data."),
    HumanMessage(content=query),
  ])
  return f"I am answering this using my training data: {response.content}"


async def handle_information_query(query: str) -> str:
  """Handle general knowledge queries — DDG first, LLM fallback with disclaimer."""
  intent = _classify_intent(query)
  logger.debug(f"Information intent: query_type={intent.query_type}, keyword={intent.keyword}")

  if intent.query_type.value == "contextual":
    return _get_contextual_answer(intent.keyword)

  web_results = await web_search(query)
  if web_results:
    return web_results

  logger.debug("Search engines returned no results, falling back to LLM")
  return await _llm_fallback(query)
