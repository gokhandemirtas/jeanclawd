import os
import re
from collections import Counter
from datetime import datetime

import aiohttp
from utils.log import logger
from pydantic import BaseModel

from db.schemas import Weather
from skills.weather.prompts import weatherIntentPrompt
from skills.weather.weather_client import WeatherForecast, fetch_weather_forecast
from utils.llm_structured_output import generate_structured_output


class WeatherIntentResponse(BaseModel):
  location: str
  period: str


def _get_condition(hourly_str: str) -> str:
  parts = [p.strip() for p in hourly_str.split(",")]
  descriptions = [parts[i] for i in range(2, len(parts), 3) if i < len(parts)]
  if not descriptions:
    return "conditions unknown"
  return Counter(descriptions).most_common(1)[0][0].lower()


def _parse_day(day_str: str) -> tuple[int, int, int, str]:
  avg = int(re.search(r'avg temp: (\d+)', day_str).group(1))
  low = int(re.search(r'low: (\d+)', day_str).group(1))
  high = int(re.search(r'high: (\d+)', day_str).group(1))
  hourly_start = day_str.find('hourly: ')
  hourly = day_str[hourly_start + 8:] if hourly_start != -1 else day_str
  return avg, low, high, hourly


def _format_weather_data(data: dict, period: str = "FORECAST") -> str:
  today = data.get("today", {})
  location = today.get("location", "Unknown")
  current_temp = today.get("current_temperature", "?")
  forecast = today.get("forecast", [])

  day_labels = ["Today", "Tomorrow", "The day after tomorrow"]
  period_index = {"CURRENT": [0], "TODAY": [0], "TOMORROW": [1], "DAY_AFTER": [2]}
  indices = period_index.get(period, range(len(forecast)))

  lines = [f"Weather for {location}:", f"Current temperature: {current_temp}°C"]
  for i in indices:
    if i >= len(forecast):
      continue
    label = day_labels[i] if i < len(day_labels) else f"Day {i + 1}"
    avg, low, high, hourly = _parse_day(forecast[i])
    condition = _get_condition(hourly)
    lines.append(f"{label}: average {avg}°C, lows {low}°C and highs {high}°C with {condition}")

  return "\n".join(lines)


def _classify_intent(query: str) -> dict:
  try:
    result = generate_structured_output(
      model_name=os.environ["MODEL_GENERALIST"],
      user_prompt=query,
      system_prompt=weatherIntentPrompt,
      pydantic_model=WeatherIntentResponse,
    )

    location = result.location
    if location and location.lower() != "unknown_location" and location.lower() not in query.lower():
      logger.debug(f"Rejecting hallucinated location '{location}' not found in query")
      location = "UNKNOWN_LOCATION"

    return {"location": location, "period": result.period}

  except Exception as e:
    logger.error(f"Failed to generate structured output. Error: {e}")
    return {"location": "UNKNOWN_LOCATION", "period": "CURRENT"}


async def _query_weather(location: str, period: str) -> dict:
  datestamp = datetime.now().date()

  found = await Weather.find(Weather.datestamp == datestamp, Weather.location == location).to_list()
  if found:
    logger.debug(f"Today's forecast found for {location} in DB")
    today = found[0].model_dump(exclude={"id"})
  else:
    logger.debug(f"Fetching new forecast for {location}")
    weather_forecast = await fetch_weather_forecast(location)
    new_record = Weather(
      datestamp=datestamp,
      location=weather_forecast.location.lower(),
      current_temperature=weather_forecast.current_temperature,
      forecast=weather_forecast.forecast,
    )
    await new_record.insert()
    today = new_record.model_dump(exclude={"id"})

  return {"today": today}


async def handle_weather(query: str) -> str:
  """Handle weather information queries."""
  intent = _classify_intent(query)
  location = intent.get("location", "").lower()
  time_period = intent.get("period") or "CURRENT"

  logger.debug(f"Detected secondary intent, Location: {location}, period: {time_period}")

  if location == "unknown_location" or not location:
    return "I'd be happy to check the weather for you! Which location would you like the forecast for?"

  try:
    weather_data = await _query_weather(location, time_period)
    return _format_weather_data(weather_data, time_period)
  except ValueError as e:
    logger.warning(f"Failed to fetch weather for '{location}': {e}")
    return f"I couldn't find weather data for '{location}'. Could you specify a valid city or location?"
  except aiohttp.ClientResponseError as e:
    logger.warning(f"Weather API error for '{location}': {e.status} {e.message}")
    return "The weather service is temporarily unavailable. Please try again in a moment."
  except aiohttp.ClientError as e:
    logger.warning(f"Weather network error for '{location}': {e}")
    return "I couldn't reach the weather service right now. Please check your connection and try again."
