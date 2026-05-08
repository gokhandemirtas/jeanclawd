from skills.base import Skill
from skills.weather.prompts import weatherIntentPrompt
from skills.weather.skill import handle_weather

weather_skill = Skill(
    id="WEATHER",
    description="Weather forecasts and current conditions for any location",
    system_prompt=weatherIntentPrompt,
    handle=handle_weather,
)
