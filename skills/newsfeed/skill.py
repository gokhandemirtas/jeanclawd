import os
from datetime import date

import feedparser
from utils.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from utils.log import logger

from agents.agent_factory import create_llm
from db.schemas import Newsfeed

_console = Console(stderr=True, force_terminal=True)

BBC_RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"


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
  with Live(Spinner("dots", text=Text("fetching news...", style="dim")), console=_console, transient=True):
    news_content = await _fetch_news()
  logger.debug(f"Fetched news: {news_content[:200]}...")

  llm = create_llm(
    model_name=os.environ.get("MODEL_GENERALIST"),
    temperature=0.3,
    max_tokens=1024,
  )

  system_prompt = f"""You are a news presenter. Present the following headlines to the user, keeping each story's detail intact.

News Content:
{news_content}

Rules:
- Present each headline with its summary — do not cut detail.
- Do NOT merge, compress, or omit stories.
- Do NOT add any facts not present in the news content.
"""

  with Live(Spinner("dots", text=Text("reading the news...", style="dim")), console=_console, transient=True):
    response = await llm.ainvoke([
      SystemMessage(content=system_prompt),
      HumanMessage(content=query),
    ])

  return response.content
