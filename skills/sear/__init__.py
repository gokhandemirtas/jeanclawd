from skills.base import Skill
from skills.sear.skill import handle_sear


sear_skill = Skill(
	id="SEAR",
	description="Search the locally hosted SearXNG metasearch instance and curate ranked sources",
	handle=handle_sear,
)

__all__ = ["sear_skill", "handle_sear"]
