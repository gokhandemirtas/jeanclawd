from skills.base import Skill
from skills.transportation.skill import handle_transportation

transportation_skill = Skill(
    id="TRANSPORTATION",
    description="Navigation and directions between locations (not geography questions)",
    system_prompt=None,
    handle=handle_transportation,
)
