from skills.base import Skill
from skills.newsfeed.skill import handle_newsfeed

newsfeed_skill = Skill(
    id="NEWSFEED",
    description="Latest news headlines and current events from BBC RSS",
    system_prompt=None,
    handle=handle_newsfeed,
    needs_wrapping=False,
)
