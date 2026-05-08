"""Timer functionality — in-memory asyncio-based timers."""

import asyncio
import time

from utils.log import logger

from skills.reminder.parser import parse_time

_active_timers: dict[str, asyncio.Task] = {}


async def _timer_callback(timer_id: str, description: str, delay_seconds: float):
  try:
    await asyncio.sleep(delay_seconds)
    print(f"\n⏰ Timer: {description}\n")
    logger.info(f"Timer fired: {description}")
  except asyncio.CancelledError:
    logger.debug(f"Timer cancelled: {description}")
  finally:
    _active_timers.pop(timer_id, None)


async def handle_timer(datetime_str: str, description: str) -> str:
  """Create an in-memory timer that prints the description when it expires."""
  logger.debug(f"Timer request: '{description}' at '{datetime_str}'")

  delay_seconds = parse_time(datetime_str)
  if not delay_seconds:
    return f"Could not understand the timer duration: '{datetime_str}'"

  delay_seconds = int(delay_seconds)
  if delay_seconds <= 0:
    return "The specified time is in the past. Please set a future time for the timer."

  timer_id = f"timer_{int(time.time())}_{description[:20]}"
  task = asyncio.create_task(_timer_callback(timer_id, description, delay_seconds))
  _active_timers[timer_id] = task

  minutes, seconds = divmod(int(delay_seconds), 60)
  hours, minutes = divmod(minutes, 60)
  parts = []
  if hours:
    parts.append(f"{hours}h")
  if minutes:
    parts.append(f"{minutes}m")
  if seconds or not parts:
    parts.append(f"{seconds}s")

  return f"Timer set for {' '.join(parts)}: {description}"
