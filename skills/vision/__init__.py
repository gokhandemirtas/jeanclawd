from skills.vision.skill import handle_vision
from skills.base import Skill

vision_skill = Skill(
	id="VISION",
	description=(
		"Vision and image understanding — analyses uploaded photos and image documents. "
		"Payload-based capability: available when an image is supplied by an image-aware client."
	),
	handle=handle_vision,
	needs_wrapping=False,
	routable=False,
)

__all__ = ["handle_vision", "vision_skill"]
