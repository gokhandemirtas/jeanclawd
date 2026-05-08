import asyncio
import os
from typing import Literal

from utils.log import logger
from pydantic import BaseModel

from skills.transportation.ors_client import format_directions, geocode, get_directions
from skills.transportation.prompts import TRANSPORTATION_INTENT_PROMPT
from utils.llm_structured_output import generate_structured_output


class TransportationIntent(BaseModel):
    origin: str = "London"
    destination: str
    mode: Literal["driving-car", "foot-walking", "cycling-regular", "public-transport"] = "public-transport"


async def handle_transportation(query: str) -> str:
    logger.debug(f"TRANSPORTATION: {query}")

    try:
        intent: TransportationIntent = await asyncio.to_thread(
            generate_structured_output,
            model_name=os.environ["MODEL_GENERALIST"],
            user_prompt=query,
            system_prompt=TRANSPORTATION_INTENT_PROMPT,
            pydantic_model=TransportationIntent,
        )
    except Exception as e:
        logger.error(f"Failed to parse transportation intent: {e}")
        return "I couldn't understand the origin and destination from your query. Please try rephrasing."

    logger.debug(f"Route: {intent.origin!r} → {intent.destination!r} via {intent.mode}")

    try:
        origin_coords, dest_coords = await asyncio.gather(
            geocode(intent.origin),
            geocode(intent.destination),
        )
    except ValueError as e:
        logger.error(e)
        return f"I couldn't find that location: {e}"

    try:
        data = await get_directions(origin_coords, dest_coords, intent.mode)
    except Exception as e:
        logger.error(f"Directions request failed: {e!r}")
        return f"I couldn't retrieve directions ({e}). Please check your ORS API key or try again."

    return format_directions(intent.origin, intent.destination, data)
