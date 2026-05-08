from datetime import datetime

from utils.log import logger
from pydantic import BaseModel, Field

from utils.http_client import HTTPClient

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_DESCRIPTIONS = {
  0: "Clear sky",
  1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
  45: "Fog", 48: "Depositing rime fog",
  51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
  61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
  71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
  77: "Snow grains",
  80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
  85: "Slight snow showers", 86: "Heavy snow showers",
  95: "Thunderstorm",
  96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class WeatherForecast(BaseModel):
  """Complete weather forecast for a location."""

  location: str = Field(..., description="Location/city name")
  current_temperature: int = Field(..., description="Current temperature in Celsius")
  forecast: list[str] = Field(
    default_factory=list,
    description="List of daily forecasts as formatted strings"
  )


def _wmo_desc(code: int) -> str:
  return WMO_DESCRIPTIONS.get(code, f"code {code}")


async def fetch_weather_forecast(location: str) -> WeatherForecast:
  """Fetch weather forecast for a location using Open-Meteo.

  Args:
      location: City or location name

  Returns:
      WeatherForecast: Structured weather forecast data

  Raises:
      ValueError: If location cannot be geocoded
  """
  async with HTTPClient() as client:
    geo = await client.get(GEOCODING_URL, params={"name": location, "count": 1, "language": "en", "format": "json"})
    results = geo.get("results")
    if not results:
      raise ValueError(f"Could not geocode location: {location}")

    result = results[0]
    lat, lon = result["latitude"], result["longitude"]
    resolved_name = result.get("name", location)
    logger.debug(f"Geocoded '{location}' → {resolved_name} ({lat}, {lon})")

    data = await client.get(FORECAST_URL, params={
      "latitude": lat,
      "longitude": lon,
      "current": "temperature_2m,weather_code",
      "hourly": "temperature_2m,weather_code",
      "forecast_days": 3,
      "timezone": "auto",
    })

  current_temp = int(round(data["current"]["temperature_2m"]))

  hourly_times = data["hourly"]["time"]
  hourly_temps = data["hourly"]["temperature_2m"]
  hourly_codes = data["hourly"]["weather_code"]

  days: dict[str, list] = {}
  for i, ts in enumerate(hourly_times):
    date_str = ts[:10]
    days.setdefault(date_str, []).append((ts, hourly_temps[i], hourly_codes[i]))

  forecast = []
  for date_str, entries in days.items():
    temps = [t for _, t, _ in entries]
    avg_temp = int(round(sum(temps) / len(temps)))
    min_temp = int(round(min(temps)))
    max_temp = int(round(max(temps)))

    hourly_parts = []
    for ts, temp, code in entries:
      hour = datetime.fromisoformat(ts).hour
      hour_end = (hour + 1) % 24
      desc = _wmo_desc(code)
      hourly_parts.append(f"{hour:02d}:00-{hour_end:02d}:00, {int(round(temp))} °C, {desc}")

    daily_str = f"{date_str}, avg temp: {avg_temp} °C, low: {min_temp} °C, high: {max_temp} °C, hourly: {', '.join(hourly_parts)}"
    forecast.append(daily_str)

  return WeatherForecast(
    location=resolved_name,
    current_temperature=current_temp,
    forecast=forecast,
  )
