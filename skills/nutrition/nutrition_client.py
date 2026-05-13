import aiohttp
from utils.log import logger

THEMEALDB_BASE = "https://www.themealdb.com/api/json/v1/1"
OPENFOODFACTS_BASE = "https://world.openfoodfacts.org/cgi/search.pl"

HEADERS = {"User-Agent": "pydantic-aida/1.0 (personal assistant)"}


async def search_recipes(query: str) -> list[dict]:
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.get(f"{THEMEALDB_BASE}/search.php", params={"s": query}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("meals") or []


async def search_by_ingredient(ingredient: str) -> list[dict]:
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.get(f"{THEMEALDB_BASE}/filter.php", params={"i": ingredient}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("meals") or []


async def get_recipe_by_id(meal_id: str) -> dict | None:
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.get(f"{THEMEALDB_BASE}/lookup.php", params={"i": meal_id}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            meals = data.get("meals")
            return meals[0] if meals else None


async def lookup_nutrition(food: str) -> dict | None:
    params = {
        "search_terms": food,
        "json": 1,
        "page_size": 1,
        "fields": "product_name,nutriments",
    }
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.get(OPENFOODFACTS_BASE, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            products = data.get("products", [])
            return products[0] if products else None
