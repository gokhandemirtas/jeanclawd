"""Shared web search utility used by both the search and information skills."""

import asyncio
import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from utils.log import logger

_console = Console(stderr=True, force_terminal=True)
_TOP_K = 3
_MAX_SNIPPET_CHARS = 900
_BLOCKED_DOMAINS = frozenset(
	{
		"twitter.com",
		"x.com",
		"facebook.com",
		"instagram.com",
		"pinterest.com",
		"tiktok.com",
	}
)
_IMAGE_EXTENSIONS = frozenset({".gif", ".jpg", ".jpeg", ".png", ".webp", ".svg", ".ico"})


def _clean_url(url: str) -> str:
	"""Remove tracking parameters while preserving the useful destination URL."""
	try:
		parts = urlsplit(url.strip())
		if parts.scheme not in {"http", "https"} or not parts.netloc:
			return ""
		query = [
			(k, v)
			for k, v in parse_qsl(parts.query, keep_blank_values=True)
			if not k.lower().startswith(("utm_", "ved", "usg", "fbclid", "gclid"))
		]
		return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
	except ValueError:
		return ""


def _is_useful_url(url: str) -> bool:
	try:
		parts = urlsplit(url)
		domain = parts.netloc.lower().split(":", 1)[0].removeprefix("www.")
		return bool(domain) and domain not in _BLOCKED_DOMAINS and not any(
			parts.path.lower().endswith(ext) for ext in _IMAGE_EXTENSIONS
		)
	except ValueError:
		return False


def _clean_text(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
	"""Collapse page boilerplate and cap snippets before sending them to an LLM."""
	text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text or "")
	text = re.sub(r"<[^>]+>", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text[:limit].rstrip() + ("…" if len(text) > limit else "")


def _format_result(title: str, content: str, url: str) -> str | None:
	clean_url = _clean_url(url)
	if not clean_url or not _is_useful_url(clean_url):
		return None
	title = _clean_text(title, 180)
	content = _clean_text(content)
	if not title and not content:
		return None
	return f"- {title}: {content} ({clean_url})"


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
            line
            for r in (results or [])
            if (line := _format_result(r.get("title", ""), r.get("body", ""), r.get("href", "")))
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
        results = TavilyClient(api_key=api_key).search(
            query,
            max_results=5,
            search_depth="advanced",
            include_raw_content=False,
            include_images=False,
        ).get("results", [])
        lines = [
            line
            for r in results
            if (line := _format_result(r.get("title", ""), r.get("content", ""), r.get("url", "")))
        ]
        logger.debug(f"Tavily returned {len(lines)} results for '{query}'")
        return lines
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


def search_wikipedia(query: str) -> str | None:
    import wikipedia
    try:
        return wikipedia.summary(query, sentences=5, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=5)
        except Exception:
            return None
    except Exception:
        return None


async def web_search(query: str) -> str | None:
    """Search DuckDuckGo and Tavily in parallel, rank by relevance, return top results."""
    with Live(Spinner("dots", text=Text(f"Searching: {query}", style="dim")), console=_console, transient=True):
        ddg_results, tavily_results = await asyncio.gather(
            asyncio.to_thread(_search_ddg, query),
            asyncio.to_thread(_search_tavily, query),
        )
    # Deduplicate by normalized URL; the two providers commonly return the same page.
    combined: list[str] = []
    seen: set[str] = set()
    for result in ddg_results + tavily_results:
        match = re.search(r"\((https?://[^)]+)\)$", result)
        key = match.group(1) if match else result
        if key in seen:
            continue
        seen.add(key)
        combined.append(result)
    if not combined:
        return None
    ranked = _rank_by_relevance(query, combined)
    logger.debug(f"Ranked {len(combined)} results → top {len(ranked)}")
    for i, r in enumerate(ranked):
        logger.debug(f"  [{i+1}] {r}")
    return "\n".join(ranked)
