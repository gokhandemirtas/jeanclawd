from skills.base import Skill
from skills.fallback import fallback_skill
from skills.registry import SKILL_MAP


def get_skill(intent: str) -> Skill:
	return SKILL_MAP.get(intent, fallback_skill)
