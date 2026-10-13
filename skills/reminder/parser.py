"""Utility functions for parsing reminder dates and times from natural language."""

from datetime import datetime
import re

import dateparser
from utils.log import logger

_DATEPARSER_SETTINGS = {
  "PREFER_DATES_FROM": "future",
  "PREFER_DAY_OF_MONTH": "current",
  "RETURN_AS_TIMEZONE_AWARE": False,
}

_DAY_PARTS = {
  "morning": 9,
  "afternoon": 15,
  "evening": 18,
  "night": 21,
}

_DURATION = re.compile(
  r"^\s*(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s*$",
  re.IGNORECASE,
)
_DURATION_SECONDS = {
  "second": 1,
  "sec": 1,
  "minute": 60,
  "min": 60,
  "hour": 3600,
  "hr": 3600,
}


def _parse(date_input: str) -> datetime | None:
  """Parse dates, including dateparser-unrecognised day-part phrases."""
  settings = {**_DATEPARSER_SETTINGS, "RELATIVE_BASE": datetime.now()}
  parsed_dt = dateparser.parse(date_input, settings=settings)
  if parsed_dt:
    return parsed_dt

  # dateparser understands "tomorrow" and "next Monday", but not their
  # common day-part suffixes ("tomorrow afternoon"). Parse the calendar
  # expression separately and assign a useful conventional time.
  match = re.search(r"\b(morning|afternoon|evening|night)\b", date_input, re.IGNORECASE)
  if not match:
    return None
  calendar_text = re.sub(r"\b(morning|afternoon|evening|night)\b", "", date_input, flags=re.IGNORECASE)
  parsed_dt = dateparser.parse(calendar_text.strip(), settings=settings)
  if not parsed_dt:
    return None
  return parsed_dt.replace(hour=_DAY_PARTS[match.group(1).lower()], minute=0, second=0, microsecond=0)


def parse_time(date_input: str) -> int | None:
  """Parse a natural-language duration/date into seconds from now (for timers)."""
  try:
    # Date parsers resolve "5 minutes" relative to the current instant. That
    # can lose a second before the response is formatted, so parse explicit
    # countdown durations exactly.
    match = _DURATION.match(date_input)
    if match:
      unit = match.group(2).lower().rstrip("s")
      return int(float(match.group(1)) * _DURATION_SECONDS[unit])
    parsed_dt = _parse(date_input)
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
    parsed_dt = _parse(date_input)
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
