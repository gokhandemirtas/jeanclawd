from skills.base import Skill
from skills.nutrition.prompts import NUTRITION_SYSTEM_PROMPT
from skills.nutrition.skill import handle_nutrition

nutrition_skill = Skill(
    id="NUTRITION",
    description="Nutrition facts, calorie lookup, and recipe search",
    system_prompt=NUTRITION_SYSTEM_PROMPT,
    handle=handle_nutrition,
)
