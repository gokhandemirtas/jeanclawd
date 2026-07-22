from typing import Literal, Optional

from db.schemas import UserSettings
from utils.log import logger


async def _get_settings() -> UserSettings:
	settings = await UserSettings.find_one(UserSettings.user_id == "default")
	if settings is None:
		settings = UserSettings()
		await settings.insert()
	return settings


async def get_preferences() -> UserSettings:
	return await _get_settings()


async def apply_preferences() -> None:
	"""Apply explicit runtime settings at startup."""
	try:
		settings = await _get_settings()
		from agents.persona import set_active_persona
		from tts.tts import set_active_voice_id

		set_active_persona(settings.agent_persona)
		set_active_voice_id(settings.agent_voice_id)
		logger.debug(
			f"[preferences] Applied persona={settings.agent_persona} voice={settings.agent_voice_id}"
		)
	except Exception as exc:
		logger.warning(f"[preferences] Could not apply preferences: {exc}")


async def update_preferences(
	agent_persona: Optional[str] = None,
	agent_voice_id: Optional[str] = None,
	fav_spotify_device: Optional[str] = None,
	news_length: Optional[Literal["short", "medium", "long"]] = None,
) -> UserSettings:
	settings = await _get_settings()
	if agent_persona is not None:
		settings.agent_persona = agent_persona
	if agent_voice_id is not None:
		settings.agent_voice_id = agent_voice_id
	if fav_spotify_device is not None:
		settings.fav_spotify_device = fav_spotify_device
	if news_length is not None:
		settings.news_length = news_length
	await settings.save()
	return settings
