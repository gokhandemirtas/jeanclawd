"""OpenRouteService client — geocoding and directions."""

import os

from utils.log import logger

from utils.http_client import HTTPClient

_BASE = "https://api.openrouteservice.org"


def _api_key() -> str:
    key = os.environ.get("ORS_API_KEY", "")
    if not key:
        raise ValueError("ORS_API_KEY is not set")
    return key


async def geocode(location: str) -> tuple[float, float]:
    """Return (longitude, latitude) for a place name."""
    async with HTTPClient(base_url=_BASE) as client:
        data = await client.get(
            "/geocode/search",
            params={"api_key": _api_key(), "text": location, "size": 1, "boundary.country": "GBR"},
        )
    features = data.get("features", [])
    if not features:
        raise ValueError(f"Could not geocode: {location!r}")
    lon, lat = features[0]["geometry"]["coordinates"]
    label = features[0]["properties"].get("label", location)
    logger.debug(f"Geocoded {location!r} → {label} ({lon}, {lat})")
    return lon, lat


async def get_directions(
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
) -> dict:
    """Return raw ORS directions response for the given mode."""
    payload = {"coordinates": [list(origin), list(destination)]}
    logger.debug(f"ORS request: mode={mode}, payload={payload}")
    async with HTTPClient(base_url=_BASE) as client:
        return await client.post(
            f"/v2/directions/{mode}/json",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key()}"},
        )


def format_directions(origin_name: str, dest_name: str, data: dict) -> str:
    """Format ORS directions response into readable text."""
    routes = data.get("routes", [])
    if not routes:
        return "No route found."

    route = routes[0]
    summary = route.get("summary", {})
    distance_km = round(summary.get("distance", 0) / 1000, 1)
    duration_min = round(summary.get("duration", 0) / 60)

    lines = [
        f"From {origin_name} to {dest_name}",
        f"Distance: {distance_km} km | Estimated time: {duration_min} min",
        "",
        "Steps:",
    ]

    segments = route.get("segments", [])
    for segment in segments:
        for step in segment.get("steps", []):
            instruction = step.get("instruction", "")
            dist = round(step.get("distance", 0))
            if instruction:
                lines.append(f"  • {instruction} ({dist} m)")

    return "\n".join(lines)
