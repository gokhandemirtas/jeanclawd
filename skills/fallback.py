from skills.base import Skill


def _handle_fallback(user_input: str) -> str:
	return "That capability isn't available yet. More skills are coming soon."


fallback_skill = Skill(
	id="FALLBACK",
	description="Fallback for unrecognized intents",
	system_prompt=None,
	handle=_handle_fallback,
	needs_wrapping=False,
)
