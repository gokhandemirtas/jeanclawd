"""Shared web search utility used by both the search and information skills."""

import asyncio
import os
import re

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from utils.log import logger

_console = Console(stderr=True, force_terminal=True)
_TOP_K = 3


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _rank_by_relevance(query: str, results: list[str], top_k: int = _TOP_K) -> list[str]:
    if len(results) <= top_k:
        return results
    from rank_bm25 import BM25Okapi
    corpus = [_tokenize(r) for r in results]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
    return [results[i] for i in ranked[:top_k]]


def _search_ddg(query: str) -> list[str]:
    from ddgs import DDGS
    try:
        results = DDGS().text(query, max_results=5)
        lines = [
            f"- {r.get('title', '')}: {r.get('body', '').replace(chr(10), ' ')} ({r.get('href', '')})"
            for r in (results or [])
        ]
        logger.debug(f"DuckDuckGo returned {len(lines)} results for '{query}'")
        return lines
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def _search_tavily(query: str) -> list[str]:
    from tavily import TavilyClient
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        results = TavilyClient(api_key=api_key).search(query, max_results=5).get("results", [])
        lines = [
            f"- {r.get('title', '')}: {r.get('content', '').replace(chr(10), ' ')} ({r.get('url', '')})"
            for r in results
        ]
        logger.debug(f"Tavily returned {len(lines)} results for '{query}'")
        return lines
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


async def web_search(query: str) -> str | None:
    """Search DuckDuckGo and Tavily in parallel, rank by relevance, return top results."""
    with Live(Spinner("dots", text=Text(f"Searching: {query}", style="dim")), console=_console, transient=True):
        ddg_results, tavily_results = await asyncio.gather(
            asyncio.to_thread(_search_ddg, query),
            asyncio.to_thread(_search_tavily, query),
        )
    combined = ddg_results + tavily_results
    if not combined:
        return None
    ranked = _rank_by_relevance(query, combined)
    logger.debug(f"Ranked {len(combined)} results → top {len(ranked)}")
    for i, r in enumerate(ranked):
        logger.debug(f"  [{i+1}] {r[:120]}")
    return "\n".join(ranked)
