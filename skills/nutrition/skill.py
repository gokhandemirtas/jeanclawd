import os
from typing import Literal

import aiohttp
from pydantic import BaseModel
from utils.log import logger

from skills.information.skill import handle_information_query
from skills.nutrition.nutrition_client import (
    get_recipe_by_id,
    lookup_nutrition,
    search_by_ingredient,
    search_recipes,
)
from skills.nutrition.prompts import NUTRITION_INTENT_PROMPT
from utils.llm_structured_output import generate_structured_output

QueryType = Literal["RECIPE_SEARCH", "NUTRITION_LOOKUP", "RECIPE_BY_INGREDIENTS"]


class NutritionIntent(BaseModel):
    query_type: QueryType
    query: str


def _classify_intent(user_query: str) -> NutritionIntent:
    try:
        return generate_structured_output(
            model_name=os.environ["MODEL_GENERALIST"],
            user_prompt=user_query,
            system_prompt=NUTRITION_INTENT_PROMPT,
            pydantic_model=NutritionIntent,
        )
    except Exception as e:
        logger.error(f"[nutrition] Intent classification failed: {e}")
        return NutritionIntent(query_type="RECIPE_SEARCH", query=user_query)


def _format_nutrition(data: dict, food: str) -> str:
    name = data.get("product_name") or food
    n = data.get("nutriments", {})

    def val(key: str, unit: str = "g") -> str:
        v = n.get(f"{key}_100g")
        return f"{v:.1f}{unit}" if v is not None else "n/a"

    return "\n".join([
        f"Nutrition for {name} (per 100g):",
        f"  Calories: {val('energy-kcal', 'kcal')}",
        f"  Protein: {val('proteins')}",
        f"  Carbs: {val('carbohydrates')}",
        f"  Fat: {val('fat')}",
        f"  Fiber: {val('fiber')}",
        f"  Sugar: {val('sugars')}",
        f"  Salt: {val('salt')}",
    ])


def _format_recipe(meal: dict) -> str:
    title = meal.get("strMeal", "Unknown")
    category = meal.get("strCategory", "")
    area = meal.get("strArea", "")
    source = meal.get("strSource", "") or meal.get("strYoutube", "")

    ingredients = []
    for i in range(1, 21):
        ing = meal.get(f"strIngredient{i}", "").strip()
        measure = meal.get(f"strMeasure{i}", "").strip()
        if ing:
            ingredients.append(f"{measure} {ing}".strip())

    instructions = meal.get("strInstructions", "")
    steps = [s.strip() for s in instructions.split("\n") if s.strip()][:6]

    lines = [f"Recipe: {title}"]
    if category or area:
        lines.append(f"  {area} | {category}".strip(" |"))
    if ingredients:
        lines.append(f"  Ingredients: {', '.join(ingredients[:8])}" + (" ..." if len(ingredients) > 8 else ""))
    if steps:
        lines.append("Steps:")
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step[:200]}")
        if len([s for s in instructions.split("\n") if s.strip()]) > 6:
            lines.append("  ...")
    if source:
        lines.append(f"  Source: {source}")
    return "\n".join(lines)


def _format_by_ingredient(meals: list[dict]) -> str:
    if not meals:
        return "No recipes found for those ingredients."
    lines = ["Recipes you can make:"]
    for m in meals[:5]:
        lines.append(f"  - {m['strMeal']}")
    return "\n".join(lines)


async def _fallback_to_information(query: str, reason: str) -> str:
    """Hand off to INFORMATION_QUERY when the nutrition/recipe APIs can't answer.

    INFORMATION_QUERY has its own web-search → LLM-with-disclaimer fallback chain
    (see skills/information/skill.py:handle_information_query), so this gives the
    user a real answer instead of a hard 'not found' dead end.
    """
    logger.debug(f"[nutrition] {reason} for {query!r}, falling back to INFORMATION_QUERY")
    return await handle_information_query(query)


async def handle_nutrition(query: str) -> str:
    intent = _classify_intent(query)
    logger.debug(f"[nutrition] query_type={intent.query_type}, query={intent.query!r}")

    try:
        if intent.query_type == "NUTRITION_LOOKUP":
            try:
                data = await lookup_nutrition(intent.query)
            except aiohttp.ClientResponseError:
                data = None
            if not data:
                meals = await search_recipes(intent.query)
                if meals:
                    full = await get_recipe_by_id(meals[0]["idMeal"])
                    if full:
                        return _format_recipe(full)
                return await _fallback_to_information(intent.query, "no nutrition or recipe data found")
            return _format_nutrition(data, intent.query)

        elif intent.query_type == "RECIPE_BY_INGREDIENTS":
            first_ingredient = intent.query.split(",")[0].strip()
            meals = await search_by_ingredient(first_ingredient)
            if not meals:
                return await _fallback_to_information(intent.query, "no recipes matched those ingredients")
            return _format_by_ingredient(meals)

        else:  # RECIPE_SEARCH
            meals = await search_recipes(intent.query)
            if not meals:
                return await _fallback_to_information(intent.query, "no recipe match")
            full = await get_recipe_by_id(meals[0]["idMeal"])
            if not full:
                return await _fallback_to_information(intent.query, "matched recipe but lookup failed")
            return _format_recipe(full)

    except aiohttp.ClientResponseError as e:
        logger.warning(f"[nutrition] API error: {e.status} {e.message}")
        return await _fallback_to_information(intent.query, f"API error {e.status}")
    except aiohttp.ClientError as e:
        logger.warning(f"[nutrition] Network error: {e}")
        return await _fallback_to_information(intent.query, "network error")
