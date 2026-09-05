"""SearXNG-backed search skill for explicit ``sear:`` requests."""

import asyncio
import json
import os
import time
from urllib.parse import quote_plus, urlsplit

import aiohttp

from utils.log import logger

_PREFIX = "sear:"
_DEFAULT_URL = "http://127.0.0.1:8080"
_RESULT_LIMIT = 8
_PAGE_FETCH_LIMIT = 5
_PAGE_CONTENT_LIMIT = 3500
_UNREADABLE_DOMAINS = frozenset({
	"facebook.com", "instagram.com", "x.com", "twitter.com", "tiktok.com",
	"youtube.com", "youtu.be", "vimeo.com", "reddit.com",
})
_RAW_RESPONSE_PATH = "logs/searxng-latest.json"


def _query_from_input(user_input: str) -> str:
	query = user_input.strip()
	if query.lower().startswith(_PREFIX):
		query = query[len(_PREFIX):].strip()
	return query


async def _fetch_page_content(url: str) -> str:
	"""Extract readable page text so the answer can be grounded in page evidence."""
	domain = urlsplit(url).netloc.lower().removeprefix("www.")
	if domain in _UNREADABLE_DOMAINS:
		logger.debug(f"[sear] Skipping gated/social/video URL: {url}")
		return ""
	try:
		import httpx
		from bs4 import BeautifulSoup

		def extract() -> str:
			response = httpx.get(
				url,
				timeout=8.0,
				follow_redirects=True,
				headers={"User-Agent": "pydantic-aida-sear/1.0"},
			)
			response.raise_for_status()
			final_domain = urlsplit(str(response.url)).netloc.lower().removeprefix("www.")
			if final_domain in _UNREADABLE_DOMAINS:
				return ""
			content_type = response.headers.get("content-type", "").lower()
			if "text/html" not in content_type:
				return ""
			soup = BeautifulSoup(response.text, "html.parser")
			for tag in soup(("script", "style", "nav", "footer", "header", "aside", "form")):
				tag.decompose()
			text = " ".join(soup.get_text(" ").split())
			# Login walls and access-denied shells are not usable page evidence.
			lowered = text.lower()
			if len(text) < 200 or any(marker in lowered for marker in (
				"log in to facebook", "you must log in", "join facebook to connect",
				"access denied", "sign in to continue",
			)):
				return ""
			return text

		return await asyncio.to_thread(extract)
	except Exception as exc:
		logger.debug(f"[sear] Page extraction failed for {url}: {exc}")
		return ""


async def _format_results(query: str, payload: dict) -> str:
	seen: set[str] = set()
	results: list[dict] = []
	for item in payload.get("results", []):
		url = str(item.get("url", "")).strip()
		domain = urlsplit(url).netloc.lower().removeprefix("www.")
		if not url or domain in _UNREADABLE_DOMAINS or url in seen or not item.get("title"):
			continue
		seen.add(url)
		results.append(item)
	results.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
	results = results[:_RESULT_LIMIT]
	if not results:
		return f"No SearXNG results found for: {query}"

	from utils.observability import span
	page_results = results[:_PAGE_FETCH_LIMIT]
	with span("sear_page_fetch", query=query, result_count=len(page_results)):
		page_content = await asyncio.gather(*(_fetch_page_content(item["url"]) for item in page_results))
	for item, content in zip(page_results, page_content):
		item["page_content"] = content[:_PAGE_CONTENT_LIMIT].strip()
	fetched_count = sum(bool(item.get("page_content")) for item in page_results)
	from utils.observability import emit
	emit("sear_page_fetch_result", query=query, fetched_count=fetched_count, attempted_count=len(page_results))

	lines = [f"SearXNG results for: {query}", "Evidence was fetched from the selected source pages; use only this evidence.", ""]
	for index, item in enumerate(results, 1):
		item["source_id"] = f"S{index}"
		title = " ".join(str(item.get("title", "")).split())
		snippet = " ".join(str(item.get("content", "")).split())
		line = f"{index}. {title}"
		if snippet:
			line += f" — {snippet[:700]}"
		fetch_status = "page content fetched" if item.get("page_content") else "snippet only"
		lines.extend((
			f"[{item['source_id']}] {line}",
			f"   SourceURL: {item['url']}",
			f"   EvidenceStatus: {fetch_status}",
		))
		if item.get("page_content"):
			lines.append(f"   PageContent: {item['page_content']}")
	return "\n".join(lines)


async def handle_sear(user_input: str) -> str:
	"""Call the local SearXNG JSON API and return curated source material."""
	query = _query_from_input(user_input)
	if not query:
		return "Please provide a search term after `sear:`."

	base_url = os.environ.get("SEARXNG_URL", _DEFAULT_URL).rstrip("/")
	url = f"{base_url}/search?q={quote_plus(query)}&format=json"
	try:
		started = time.perf_counter()
		from utils.observability import span
		with span("sear_api_search", query=query, endpoint=base_url):
			timeout = aiohttp.ClientTimeout(total=float(os.environ.get("SEARXNG_TIMEOUT", "15")))
			async with aiohttp.ClientSession(timeout=timeout) as session:
				async with session.get(url, headers={"Accept": "application/json"}) as response:
					response.raise_for_status()
					raw_response = await response.text()
					payload = json.loads(raw_response)
		try:
			path = os.environ.get("SEARXNG_RAW_RESPONSE_PATH", _RAW_RESPONSE_PATH)
			parent = os.path.dirname(path)
			if parent:
				os.makedirs(parent, exist_ok=True)
			with open(path, "w", encoding="utf-8") as raw_file:
				raw_file.write(raw_response)
		except OSError as exc:
			logger.warning(f"[sear] Could not save raw SearXNG response: {exc}")
		if not isinstance(payload, dict):
			return "SearXNG returned an invalid response."
		logger.info(f"[sear] SearXNG returned {len(payload.get('results', []))} results in {time.perf_counter() - started:.2f}s")
		return await _format_results(query, payload)
	except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
		logger.warning(f"[sear] SearXNG request failed: {exc}")
		return "SearXNG is unavailable. Start it with `docker compose -f searxng/docker-compose.yml up -d`."
