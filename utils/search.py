"""Shared web-search and page-extraction utilities.

Search providers are deliberately kept behind a small project-owned result model.  That
lets the research agent rank, clean, fetch, and cite evidence without depending on a
provider's presentation format.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from utils.log import logger

_console = Console(stderr=True, force_terminal=True)
_TOP_K = 3
_MAX_SNIPPET_CHARS = 900
_MAX_PAGE_CHARS = 12000
_BLOCKED_DOMAINS = frozenset(
	{"twitter.com", "x.com", "facebook.com", "instagram.com", "pinterest.com", "tiktok.com"}
)
_IMAGE_EXTENSIONS = frozenset({".gif", ".jpg", ".jpeg", ".png", ".webp", ".svg", ".ico"})


@dataclass(slots=True)
class SearchResult:
	title: str
	url: str
	snippet: str
	provider: str
	content: str = ""
	published_date: str | None = None

	@property
	def domain(self) -> str:
		return urlsplit(self.url).netloc.lower().removeprefix("www.").split(":", 1)[0]

	@property
	def text(self) -> str:
		return " ".join(part for part in (self.title, self.snippet, self.content) if part)


def _clean_url(url: str) -> str:
	try:
		parts = urlsplit(url.strip())
		if parts.scheme not in {"http", "https"} or not parts.netloc:
			return ""
		query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
			if not k.lower().startswith(("utm_", "ved", "usg", "fbclid", "gclid"))]
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
	text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text or "")
	text = re.sub(r"<[^>]+>", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text[:limit].rstrip() + ("…" if len(text) > limit else "")


def _result(title: str, snippet: str, url: str, provider: str, *, content: str = "", published_date: str | None = None) -> SearchResult | None:
	clean_url = _clean_url(url)
	if not clean_url or not _is_useful_url(clean_url):
		return None
	if not title and not snippet and not content:
		return None
	return SearchResult(
		title=_clean_text(title, 180),
		url=clean_url,
		snippet=_clean_text(snippet),
		provider=provider,
		content=_clean_text(content, _MAX_PAGE_CHARS),
		published_date=published_date,
	)


def _tokenize(text: str) -> list[str]:
	return re.findall(r"\w+", text.lower())


def _quality_score(result: SearchResult) -> float:
	"""Small source-quality prior; relevance still dominates the ranking."""
	domain = result.domain
	if domain in {"gov.uk", "ac.uk", "edu"} or domain.endswith((".gov", ".gov.uk", ".ac.uk", ".edu")):
		return 1.0
	if domain in {"nature.com", "science.org", "sciencedirect.com", "ncbi.nlm.nih.gov", "arxiv.org", "who.int"}:
		return 1.0
	if domain in {"medium.com", "substack.com", "quora.com", "reddit.com"}:
		return -0.65
	if domain.endswith((".org", ".edu")):
		return 0.35
	return 0.0


def _rank_results(query: str, results: list[SearchResult], top_k: int = _TOP_K) -> list[SearchResult]:
	if len(results) <= top_k:
		return results
	from rank_bm25 import BM25Okapi

	corpus = [_tokenize(r.text) for r in results]
	scores = BM25Okapi(corpus).get_scores(_tokenize(query))
	max_score = max(scores) if len(scores) else 0.0
	min_score = min(scores) if len(scores) else 0.0
	span = max(max_score - min_score, 1e-9)
	scored = sorted(
		((0.6 * ((scores[i] - min_score) / span) + 0.4 * _quality_score(r), i, r)
		 for i, r in enumerate(results)),
		key=lambda item: item[0], reverse=True,
	)
	selected: list[SearchResult] = []
	domains: set[str] = set()
	for _, _, result in scored:
		if result.domain in domains and len(scored) - len(selected) > top_k:
			continue
		selected.append(result)
		domains.add(result.domain)
		if len(selected) == top_k:
			break
	return selected


def _search_ddg(query: str) -> list[SearchResult]:
	from ddgs import DDGS
	try:
		results = DDGS().text(query, max_results=5)
		lines = [r for item in (results or []) if (r := _result(item.get("title", ""), item.get("body", ""), item.get("href", ""), "ddg"))]
		logger.debug(f"DuckDuckGo returned {len(lines)} results for '{query}'")
		return lines
	except Exception as e:
		logger.warning(f"DuckDuckGo search failed: {e}")
		return []


def _search_tavily(query: str) -> list[SearchResult]:
	from tavily import TavilyClient
	api_key = os.environ.get("TAVILY_API_KEY")
	if not api_key:
		return []
	try:
		results = TavilyClient(api_key=api_key).search(
			query, max_results=5, search_depth="advanced", include_raw_content=True, include_images=False,
		).get("results", [])
		lines = [r for item in results if (r := _result(item.get("title", ""), item.get("content", ""), item.get("url", ""), "tavily", content=item.get("raw_content", ""), published_date=item.get("published_date")))]
		logger.debug(f"Tavily returned {len(lines)} results for '{query}'")
		return lines
	except Exception as e:
		logger.warning(f"Tavily search failed: {e}")
		return []


def _extract_page(url: str) -> str:
	"""Extract readable text for providers (such as DDG) that return snippets only."""
	try:
		import httpx
		from bs4 import BeautifulSoup
		response = httpx.get(url, timeout=8.0, follow_redirects=True, headers={"User-Agent": "pydantic-aida/1.0"})
		response.raise_for_status()
		soup = BeautifulSoup(response.text, "html.parser")
		for tag in soup(("script", "style", "nav", "footer", "header", "aside", "form")):
			tag.decompose()
		return _clean_text(soup.get_text(" "), _MAX_PAGE_CHARS)
	except Exception as e:
		logger.debug(f"Page extraction failed for {url}: {e}")
		return ""


async def web_search_results(query: str) -> list[SearchResult]:
	with Live(Spinner("dots", text=Text(f"Searching: {query}", style="dim")), console=_console, transient=True):
		ddg_results, tavily_results = await asyncio.gather(asyncio.to_thread(_search_ddg, query), asyncio.to_thread(_search_tavily, query))
	combined: list[SearchResult] = []
	seen: set[str] = set()
	for result in ddg_results + tavily_results:
		if result.url in seen:
			continue
		seen.add(result.url)
		combined.append(result)
	if not combined:
		return []
	ranked = _rank_results(query, combined)
	missing_content = await asyncio.gather(*(asyncio.to_thread(_extract_page, r.url) if not r.content else asyncio.sleep(0, result="") for r in ranked))
	for result, extracted in zip(ranked, missing_content):
		if extracted:
			result.content = extracted
	logger.debug(f"Ranked {len(combined)} results → top {len(ranked)}")
	for i, result in enumerate(ranked):
		logger.debug(f"  [{i + 1}] {result.domain} ({result.provider}) {result.title}")
	return ranked


def _format_results(results: list[SearchResult]) -> str:
	return "\n".join(
		f"- {r.title}: {_clean_text(r.content or r.snippet, _MAX_PAGE_CHARS)}\nSourceURL: {r.url}"
		for r in results
	)


async def web_search(query: str) -> str | None:
	"""Search both providers, extract selected pages, rank, and return readable results."""
	results = await web_search_results(query)
	return _format_results(results) if results else None


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
