from datetime import datetime
from typing import Optional

from db.schemas import UserPreferences


async def get_preferences() -> UserPreferences:
    prefs = await UserPreferences.find_one()
    if prefs is None:
        prefs = UserPreferences()
        await prefs.insert()
    return prefs


async def update_preferences(
    active_persona_id: Optional[str] = None,
    active_voice_id: Optional[str] = None,
    spotify_device_id: Optional[str] = None,
    spotify_device_name: Optional[str] = None,
) -> UserPreferences:
    prefs = await get_preferences()
    if active_persona_id is not None:
        prefs.active_persona_id = active_persona_id
    if active_voice_id is not None:
        prefs.active_voice_id = active_voice_id
    if spotify_device_id is not None:
        prefs.spotify_device_id = spotify_device_id
    if spotify_device_name is not None:
        prefs.spotify_device_name = spotify_device_name
    prefs.updated_at = datetime.now()
    await prefs.save()
    return prefs
