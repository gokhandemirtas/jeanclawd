"""Utility functions for parsing reminder dates and times from natural language."""

from datetime import datetime

import fuzzydate as fd
from utils.log import logger


def parse_time(date_input: str) -> int:
  try:
    parsed_time = fd.to_seconds(date_input)
    if parsed_time:
      logger.debug(f"Parsed '{date_input}' to '{parsed_time}'")
      return parsed_time
    else:
      logger.warning(f"Could not parse time: {date_input}")
      return None
  except Exception as e:
    logger.error(f"Time parsing error for '{date_input}': {e}")
    return None


def parse_reminder_date(date_input: str) -> datetime | None:
  """Parse natural language date/time string into a datetime object."""
  try:
    parsed_datetime = fd.to_datetime(date_input)
    if parsed_datetime:
      logger.debug(f"Parsed '{date_input}' to '{parsed_datetime.isoformat()}'")
      return parsed_datetime
    else:
      logger.warning(f"Could not parse date: {date_input}")
      return None
  except Exception as e:
    logger.error(f"Date parsing error for '{date_input}': {e}")
    return None


def format_reminder_display(dt: datetime) -> str:
  """Format datetime for user-friendly display."""
  return dt.strftime("%Y-%m-%d %H:%M")
