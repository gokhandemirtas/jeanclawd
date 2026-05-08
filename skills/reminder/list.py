"""Handler for listing upcoming reminders."""

from datetime import datetime, timedelta

from utils.log import logger

from db.schemas import Reminders


async def handle_list_reminders() -> str:
  """List all reminders from today through the next 7 days."""
  try:
    now = datetime.now()
    end = datetime.combine((now + timedelta(weeks=1)).date(), datetime.max.time())

    reminders = await Reminders.find(
      Reminders.reminder_datetime >= now,
      Reminders.reminder_datetime <= end,
      Reminders.is_completed == False,
    ).sort("+reminder_datetime").to_list()

    if not reminders:
      return "You have no upcoming reminders for the next 7 days."

    reminder_lines = []
    for idx, reminder in enumerate(reminders, 1):
      date_str = reminder.reminder_datetime.strftime("%a %d %b, %I:%M %p")
      reminder_lines.append(f"{idx}. {date_str} - {reminder.description}")

    result = "Your upcoming reminders:\n" + "\n".join(reminder_lines)
    logger.debug(f"Found {len(reminders)} upcoming reminders")
    return result

  except Exception as e:
    logger.error(f"Failed to list reminders: {e}")
    return "Failed to retrieve reminders. Please try again."
