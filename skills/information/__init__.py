from skills.base import Skill
from skills.information.prompts import informationIntentPrompt
from skills.information.skill import handle_information_query

information_skill = Skill(
    id="INFORMATION_QUERY",
    description="General knowledge, factual questions, how-to queries, and current time/date",
    system_prompt=informationIntentPrompt,
    handle=handle_information_query,
)
