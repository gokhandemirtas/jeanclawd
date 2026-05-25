import os
import re

from pydantic import BaseModel

from utils.llm_structured_output import generate_structured_output
from utils.log import logger
from utils.search import web_search

_EXTRACT_PROMPT = (
    "Extract a concise web search query from the user's message. "
    "Return only the essential keywords — no conversational filler, no punctuation. "
    "IMPORTANT: Never produce incomplete queries with trailing words like 'similar to ', 'about ', 'for ' with nothing after them. "
    "If the message is too vague to produce a complete query, return the message as-is. "
    "Examples: 'bring her back 2025 movie cast', 'weather Paris tomorrow', 'Python asyncio tutorial'."
)


class _SearchQuery(BaseModel):
    query: str


def _extract_topic(user_message: str) -> str:
    match = re.match(r"search\s*:\s*(.+)", user_message, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    try:
        result = generate_structured_output(
            model_name=os.environ["MODEL_INTENT"],
            user_prompt=user_message,
            system_prompt=_EXTRACT_PROMPT,
            pydantic_model=_SearchQuery,
        )
        logger.debug(f"[search] Extracted query: '{result.query}' from: '{user_message}'")
        return result.query.strip()
    except Exception as e:
        logger.warning(f"[search] Query extraction failed ({e}), using raw input")
        return user_message.strip()


async def handle_search(query: str) -> str:
    topic = _extract_topic(query)
    results = await web_search(topic)
    if not results:
        return f"No results found for: {topic}"
    top_result = results.split("\n")[0]
    return f"User query: {topic}\n\nTop result: {top_result}"
