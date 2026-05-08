import re

from utils.search import web_search


def _extract_topic(query: str) -> str:
    match = re.match(r"search\s*:\s*(.+)", query, flags=re.IGNORECASE)
    return match.group(1).strip() if match else query.strip()


async def handle_search(query: str) -> str:
    topic = _extract_topic(query)
    results = await web_search(topic)
    return results if results else f"No results found for: {topic}"
