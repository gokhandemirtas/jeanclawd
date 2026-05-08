import os
from typing import Optional

from utils.log import logger
from pydantic import BaseModel

from skills.reminder.create import handle_create_reminder
from skills.reminder.list import handle_list_reminders
from skills.reminder.prompts import reminderIntentPrompt
from skills.reminder.timer import handle_timer
from utils.llm_structured_output import generate_structured_output


class ReminderIntentResponse(BaseModel):
  type: str
  datetime: Optional[str] = ""
  description: Optional[str] = ""


def _classify_intent(query: str) -> dict:
  try:
    result = generate_structured_output(
      model_name=os.environ["MODEL_GENERALIST"],
      user_prompt=query,
      system_prompt=reminderIntentPrompt,
      pydantic_model=ReminderIntentResponse,
      max_tokens=256,
      temperature=0.0,
      repeat_penalty=1.15,
      top_p=0.95,
      top_k=40,
    )
    return result.model_dump()
  except Exception as e:
    logger.error(f"Failed to classify reminder intent. Error: {e}")
    return e


async def handle_reminder(query: str) -> str:
  """Handle timers and reminders — routes to create, list, or timer sub-handlers."""
  intent = _classify_intent(query)
  reminder_type = intent.get("type")
  datetime_str = intent.get("datetime", "")
  description = intent.get("description", "")

  if reminder_type:
    logger.debug(
      f"Detected secondary intent, Type: {reminder_type}, "
      f"DateTime: '{datetime_str}', Description: '{description}'"
    )

    match reminder_type:
      case "LIST":
        logger.debug("Listing today's reminders")
        return await handle_list_reminders()

      case "TIMER":
        if datetime_str and description:
          logger.debug(f"Timer request: {description}")
          return await handle_timer(datetime_str, description)
        else:
          logger.error("Date or description missing from TIMER request")
          return "Date or description missing"

      case "REMINDER":
        if datetime_str and description:
          logger.debug(f"Creating reminder: {datetime_str} - {description}")
          return await handle_create_reminder(datetime_str, description)
        else:
          logger.error("Date or description missing from CREATE REMINDER request")
          return "Date or description missing"

      case _:
        return "Could not determine if this is a timer or reminder request."
