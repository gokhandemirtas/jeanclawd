"""Utility functions for parsing reminder dates and times from natural language."""

from datetime import datetime

import dateparser
from utils.log import logger

_DATEPARSER_SETTINGS = {
  "RELATIVE_BASE": datetime.now(),
  "PREFER_DATES_FROM": "future",
  "PREFER_DAY_OF_MONTH": "current",
  "RETURN_AS_TIMEZONE_AWARE": False,
}


def parse_time(date_input: str) -> int | None:
  """Parse a natural-language duration/date into seconds from now (for timers)."""
  try:
    parsed_dt = dateparser.parse(date_input, settings=_DATEPARSER_SETTINGS)
    if parsed_dt:
      delta = (parsed_dt - datetime.now()).total_seconds()
      logger.debug(f"Parsed '{date_input}' to '{delta:.0f}s from now")
      return int(delta)
    else:
      logger.warning(f"Could not parse time: {date_input}")
      return None
  except Exception as e:
    logger.error(f"Time parsing error for '{date_input}': {e}")
    return None


def parse_reminder_date(date_input: str) -> datetime | None:
  """Parse natural language date/time string into a datetime object."""
  try:
    parsed_dt = dateparser.parse(date_input, settings=_DATEPARSER_SETTINGS)
    if parsed_dt:
      logger.debug(f"Parsed '{date_input}' to '{parsed_dt.isoformat()}'")
      return parsed_dt
    else:
      logger.warning(f"Could not parse date: {date_input}")
      return None
  except Exception as e:
    logger.error(f"Date parsing error for '{date_input}': {e}")
    return None


def format_reminder_display(dt: datetime) -> str:
  """Format datetime for user-friendly display."""
  return dt.strftime("%Y-%m-%d %H:%M")
