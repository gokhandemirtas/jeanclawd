"""Handler for creating reminders."""

from datetime import datetime

from utils.log import logger

from db.schemas import Reminders
from skills.reminder.parser import format_reminder_display, parse_reminder_date


async def handle_create_reminder(datetime_str: str, description: str) -> str:
  """Create a new reminder."""
  if not description or description.strip() == "":
    return "Reminder cancelled - no description provided."

  if not datetime_str or datetime_str.strip() == "":
    return "Reminder cancelled - no date provided."

  parsed_datetime = parse_reminder_date(datetime_str)
  if not parsed_datetime:
    return f"Could not understand the date/time: '{datetime_str}'. Please try again."

  try:
    new_reminder = Reminders(
      reminder_datetime=parsed_datetime,
      description=description,
      created_at=datetime.now().date(),
      is_completed=False,
    )
    logger.debug(new_reminder)
    await new_reminder.insert()

    display_time = format_reminder_display(parsed_datetime)
    logger.debug(f"Reminder saved: {description} at {display_time}")
    return f"Reminder set for {display_time}: {description}"

  except Exception as e:
    logger.error(f"Failed to save reminder: {e}")
    return "Failed to save reminder. Please try again."
