from datetime import datetime
from typing import Optional

from db.schemas import Preferences, UserPersona
from utils.log import logger


async def _get_persona() -> UserPersona:
    persona = await UserPersona.find_one()
    if persona is None:
        persona = UserPersona(last_updated=datetime.now())
        await persona.insert()
    return persona


async def get_preferences() -> Preferences:
    persona = await _get_persona()
    return persona.preferences


async def apply_preferences() -> None:
    """Load stored preferences and apply active persona and voice at startup."""
    try:
        prefs = await get_preferences()

        from agents.persona import set_active_persona
        set_active_persona(prefs.agentPersona)

        from tts.tts import set_active_voice_id
        set_active_voice_id(prefs.agentVoiceId)

        logger.debug(f"[preferences] Applied persona={prefs.agentPersona} voice={prefs.agentVoiceId}")
    except Exception as e:
        logger.warning(f"[preferences] Could not apply preferences: {e}")


async def update_preferences(
    agentPersona: Optional[str] = None,
    agentVoiceId: Optional[str] = None,
    favSpotifyDevice: Optional[str] = None,
) -> Preferences:
    persona = await _get_persona()
    if agentPersona is not None:
        persona.preferences.agentPersona = agentPersona
    if agentVoiceId is not None:
        persona.preferences.agentVoiceId = agentVoiceId
    if favSpotifyDevice is not None:
        persona.preferences.favSpotifyDevice = favSpotifyDevice
    persona.last_updated = datetime.now()
    await persona.save()
    return persona.preferences
