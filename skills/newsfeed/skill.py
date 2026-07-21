import os
from datetime import date
from typing import Literal

import feedparser
from utils.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from utils.log import logger

from agents.agent_factory import create_llm
from db.preferences import get_preferences
from db.schemas import Newsfeed

_console = Console(stderr=True, force_terminal=True)

BBC_RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"

_NEWS_LENGTH_MAX_TOKENS: dict[str, int] = {
    "short": 512,
    "medium": 1024,
    "long": 2048,
}

_NEWS_LENGTH_PROMPTS: dict[str, str] = {
    "short": """\
You are a news presenter. Summarise the following headlines as a quick briefing for someone with limited time.

News Content:
{news_content}

Rules:
- 1-2 sentences per story, headline-style.
- Skip minor stories if necessary to keep the briefing tight.
- Do NOT merge or compress within a single story.
- Do NOT add any facts not present in the news content.
""",
    "medium": """\
You are a news presenter. Present the following headlines to the user, keeping each story's detail intact.

News Content:
{news_content}

Rules:
- Present each headline with its summary — do not cut detail.
- 2-3 sentences per story is typical; longer where the story needs it.
- Do NOT merge, compress, or omit stories.
- Do NOT add any facts not present in the news content.
""",
    "long": """\
You are a news presenter. Present the following headlines with full, thorough detail.

News Content:
{news_content}

Rules:
- Preserve every fact, name, number, quote, and detail from the source content.
- Render each story at full length — do not compress, paraphrase tightly, or omit.
- If a story in the source has 5 sentences, render all 5.
- Do NOT merge, compress, or omit stories.
- Do NOT add any facts not present in the news content.
""",
}


async def _fetch_news() -> str:
  today = date.today()

  cached = await Newsfeed.find_one(Newsfeed.datestamp == today)
  if cached:
    logger.debug("Using cached newsfeed from database")
    return cached.content

  try:
    feed = feedparser.parse(BBC_RSS_URL)
    if feed.bozo and not feed.entries:
      logger.warning(f"RSS feed parse error: {feed.bozo_exception}")
      return "Unable to fetch current news at this time."

    entries = feed.entries[:7]
    results = [f"- {e.get('title', 'No title')}: {e.get('summary', 'No summary available')}" for e in entries]
    full_content = "\n".join(results)

    newsfeed = Newsfeed(datestamp=today, content=full_content)
    await newsfeed.insert()
    logger.debug("Saved newsfeed to database")
    return full_content
  except Exception as e:
    logger.error(f"Failed to fetch RSS feed: {e}")
    return "Unable to fetch current news at this time."


async def handle_newsfeed(query: str) -> str:
  """Handle newsfeed queries by fetching and presenting BBC daily headlines."""
  prefs = await get_preferences()
  length: Literal["short", "medium", "long"] = prefs.newsLength
  max_tokens = _NEWS_LENGTH_MAX_TOKENS[length]

  with Live(Spinner("dots", text=Text("fetching news...", style="dim")), console=_console, transient=True):
    news_content = await _fetch_news()
  logger.debug(f"Fetched news: {news_content[:200]}...")

  llm = create_llm(
    model_name=os.environ.get("MODEL_GENERALIST"),
    temperature=0.3,
    max_tokens=max_tokens,
  )

  system_prompt = _NEWS_LENGTH_PROMPTS[length].format(news_content=news_content)

  with Live(Spinner("dots", text=Text("reading the news...", style="dim")), console=_console, transient=True):
    response = await llm.ainvoke([
      SystemMessage(content=system_prompt),
      HumanMessage(content=query),
    ])

  return response.content
