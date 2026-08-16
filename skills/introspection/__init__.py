from skills.base import Skill
from skills.introspection.skill import handle_introspection

introspection_skill = Skill(
	id="INTROSPECTION",
	description="Evidence-based answers about this Assistant's implementation, architecture, routing, memory, and capabilities",
	system_prompt=None,
	handle=handle_introspection,
	needs_wrapping=False,
)
