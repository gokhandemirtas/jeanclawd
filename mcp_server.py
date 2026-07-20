"""jeanclawd MCP server — exposes the assistant's skills, persona, and state
as MCP tools, resources, and prompts so an LLM client (Claude Desktop,
Cursor, etc.) can talk to the agent directly.

Transport: stdio (default) — the LLM client launches this file as a subprocess
and communicates over JSON-RPC on stdin/stdout.

Run it:
    uv run mcp_server.py
    # or
    uv run mcp dev mcp_server.py    # opens the MCP inspector in a browser

Add to claude_desktop_config.json (~Library/Application Support/Claude/...):
    {
      "mcpServers": {
        "jeanclawd": {
          "command": "uv",
          "args": ["--directory", "/ABS/PATH/TO/pydantic-aida", "run", "mcp_server.py"]
        }
      }
    }

Destructive tools (apocalypse, flush-memory, pluck-memory, compact-memory)
are intentionally NOT exposed — LLM clients can accidentally invoke them.

Heavy skill modules are lazy-imported inside each tool function so the
MCP handshake stays fast (the client lists tools before any are called).
"""

import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

# MCP server runs as a standalone subprocess (Claude Desktop, mcp dev, etc.)
# so it doesn't inherit main.py's load_dotenv(). Pull .env from the project
# root before any tool is invoked — without MODEL_GENERALIST / MODEL_INTENT /
# DB_URL etc., skills fall back to placeholder error messages.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

mcp = FastMCP("jeanclawd")


def _redirect_rich_console_to_stderr() -> None:
	"""Re-point utils.logger's Rich console at stderr.

	utils.logger uses rich.Console() which writes to stdout by default.
	Any logger.info() call from a tool (notes, weather, etc.) would then
	corrupt the JSON-RPC channel. Swap to a stderr-backed console so
	logs land in the same stream as the startup banner.
	"""
	import utils.logger as _jean_logger
	from rich.console import Console

	_jean_logger._console = Console(file=sys.stderr, force_terminal=False, soft_wrap=False)

logger = logging.getLogger("jeanclawd.mcp")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)


_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _print_banner() -> None:
	tools = mcp._tool_manager.list_tools()
	resources = mcp._resource_manager.list_resources()
	prompts = mcp._prompt_manager.list_prompts()

	groups: dict[str, list[str]] = {
		"meta": [],
		"memory": [],
		"notes": [],
		"reminders": [],
		"skills": [],
		"reflection": [],
		"journal": [],
		"general": [],
	}
	for t in tools:
		n = t.name
		if n in {"list_personas", "get_active_persona", "set_persona", "list_voices", "get_active_voice", "set_voice", "show_context"}:
			groups["meta"].append(n)
		elif n in {"list_memories", "search_memories"}:
			groups["memory"].append(n)
		elif n in {"save_note", "list_notes", "read_note", "delete_note"}:
			groups["notes"].append(n)
		elif n in {"set_reminder", "set_timer", "list_reminders"}:
			groups["reminders"].append(n)
		elif n in {"reflect", "list_reflections"}:
			groups["reflection"].append(n)
		elif n in {"read_journal", "compact_conversations"}:
			groups["journal"].append(n)
		elif n in {"send_query"}:
			groups["general"].append(n)
		else:
			groups["skills"].append(n)

	w = sys.stderr.write
	w(f"\n{_CYAN}{_BOLD}┌─ jeanclawd MCP server{_RESET}\n")
	w(f"{_CYAN}│{_RESET}  transport: stdio (JSON-RPC on stdin/stdout)\n")
	w(f"{_CYAN}│{_RESET}  pid:       {os.getpid()}\n")
	w(f"{_CYAN}│{_RESET}  tools:     {len(tools)}\n")
	w(f"{_CYAN}│{_RESET}  resources: {len(resources)}\n")
	w(f"{_CYAN}│{_RESET}  prompts:   {len(prompts)}\n")
	w(f"{_CYAN}│{_RESET}\n")
	for label, names in groups.items():
		if not names:
			continue
		w(f"{_CYAN}│{_RESET}  {_BOLD}{label}{_RESET}\n")
		for n in names:
			w(f"{_CYAN}│{_RESET}    {_DIM}·{_RESET} {n}\n")
	w(f"{_CYAN}│{_RESET}\n")
	w(f"{_CYAN}│{_RESET}  {_BOLD}resources:{_RESET} ")
	w(", ".join(str(r.uri) for r in resources))
	w("\n")
	w(f"{_CYAN}│{_RESET}  {_BOLD}prompts:{_RESET}   ")
	w(", ".join(p.name for p in prompts))
	w("\n")
	w(f"{_CYAN}│{_RESET}\n")
	w(f"{_CYAN}│{_RESET}  {_DIM}stdout is reserved for the JSON-RPC protocol — all logs go to stderr.{_RESET}\n")
	w(f"{_CYAN}│{_RESET}  {_DIM}stop with Ctrl-C or kill -TERM {os.getpid()}{_RESET}\n")
	w(f"{_CYAN}└─ ready, waiting for client handshake...{_RESET}\n\n")
	sys.stderr.flush()
	logger.info("MCP server ready (pid=%d) — %d tools, %d resources, %d prompts", os.getpid(), len(tools), len(resources), len(prompts))


# ===========================================================================
# Meta / persona / voice / state
# ===========================================================================


@mcp.tool()
async def list_personas() -> str:
    """List all available persona IDs and their names/descriptions."""
    from agents.persona import PERSONAS
    lines = [f"- **{pid}**: {p['name']} — {p['description']}" for pid, p in PERSONAS.items()]
    return "Available personas:\n" + "\n".join(lines)


@mcp.tool()
async def get_active_persona() -> str:
    """Return the currently active persona's name, description, and greeting style."""
    from agents.persona import (
        get_active_name,
        get_active_persona_id,
        get_active_preamble,
    )
    return f"Active persona: **{get_active_persona_id()}** ({get_active_name()})\n\n{get_active_preamble()}"


@mcp.tool()
async def set_persona(persona_id: str) -> str:
    """Switch the active persona.

    Args:
        persona_id: The persona ID to activate (e.g. "jean", "borg", "fred").
                    Use list_personas to see available IDs.
    """
    from agents.persona import PERSONAS, set_active_persona
    if persona_id not in PERSONAS:
        return f"Unknown persona '{persona_id}'. Use list_personas to see valid IDs."
    set_active_persona(persona_id)
    return f"Switched active persona to **{persona_id}** ({PERSONAS[persona_id]['name']})."


@mcp.tool()
async def list_voices() -> str:
    """List all available TTS voice IDs."""
    from tts.tts import voice_ids
    return "Available TTS voices:\n" + "\n".join(f"- {v}" for v in voice_ids)


@mcp.tool()
async def get_active_voice() -> str:
    """Return the currently active TTS voice ID."""
    from agents.persona import get_active_voice_id
    return f"Active voice: **{get_active_voice_id()}**"


@mcp.tool()
async def set_voice(voice_id: str) -> str:
    """Switch the active TTS voice.

    Args:
        voice_id: The voice ID to activate (e.g. "azelma", "puck"). Use list_voices to see options.
    """
    from tts.tts import voice_ids
    from tts.tts import set_active_voice_id
    if voice_id not in voice_ids:
        return f"Unknown voice '{voice_id}'. Use list_voices to see valid IDs."
    set_active_voice_id(voice_id)
    return f"Switched active voice to **{voice_id}**."


@mcp.tool()
async def show_context() -> str:
    """Show the current LLM context window usage (used / max tokens + percent)."""
    return await asyncio.to_thread(_show_context_sync)


def _show_context_sync() -> str:
    from utils.resource_monitor import _context_used_tokens, get_context_max

    n_ctx = get_context_max() or 8192
    used = _context_used_tokens
    pct = used / n_ctx * 100
    bar_width = 40
    filled = int(bar_width * used / n_ctx)
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"Context usage: {used:,} / {n_ctx:,} tokens ({pct:.1f}%)\n[{bar}]"


# ===========================================================================
# Memory
# ===========================================================================


@mcp.tool()
async def list_memories(query: Optional[str] = None, limit: int = 20) -> str:
    """List all stored memories, or search for memories relevant to a query.

    Args:
        query: Optional search query. If provided, returns semantically similar memories.
               If omitted, returns the most recent memories.
        limit: Maximum number of memories to return (default 20).
    """
    from utils.memory_client import _get_memory

    memory = _get_memory()
    if query:
        results = memory.search(query=query, user_id="default", limit=limit)
        items = results.get("results", []) if isinstance(results, dict) else results
        if not items:
            return f"No memories matching '{query}'."
        lines = [f"- {m.get('memory', m)}" for m in items]
        return f"Memories matching '{query}':\n" + "\n".join(lines)
    else:
        items = memory.get_all(user_id="default", limit=limit).get("results", [])
        if not items:
            return "No memories stored yet."
        lines = [f"- {m.get('memory', m)}" for m in items]
        return "Recent memories:\n" + "\n".join(lines)


@mcp.tool()
async def search_memories(query: str, limit: int = 10) -> str:
    """Semantic search over stored memories.

    Args:
        query: Natural language search query.
        limit: Maximum number of memories to return (default 10).
    """
    return await list_memories(query=query, limit=limit)


# ===========================================================================
# Notes
# ===========================================================================


@mcp.tool()
async def save_note(title: str, content: str, category: Optional[str] = None) -> str:
    """Create a new note.

    Args:
        title: Plain text title for the note.
        content: Markdown body of the note.
        category: Optional category — one of: read, listen, watch, eat, visit.
    """
    from db.schemas import NoteCategory
    from skills.notes.create import handle_create_note_from_context
    cat: NoteCategory | None = None
    if category:
        if category not in ("read", "listen", "watch", "eat", "visit"):
            return f"Invalid category '{category}'. Must be one of: read, listen, watch, eat, visit."
        cat = category  # type: ignore[assignment]
    return await handle_create_note_from_context(title=title, content=content, category=cat)


@mcp.tool()
async def list_notes(category: Optional[str] = None) -> str:
    """List stored notes, optionally filtered by category.

    Args:
        category: Optional filter — one of: read, listen, watch, eat, visit.
    """
    from skills.notes.list import handle_list_notes
    query = f"list {category} notes" if category else "list all notes"
    return await handle_list_notes(query)


@mcp.tool()
async def read_note(title: str) -> str:
    """Read the full content of a specific note by its title.

    Args:
        title: Title (or partial title) of the note to read.
    """
    from skills.notes.read import handle_read_note
    return await handle_read_note(f"read note titled {title}")


@mcp.tool()
async def delete_note(title: str) -> str:
    """Delete a note by its title.

    Args:
        title: Title of the note to delete.
    """
    from skills.notes.delete import handle_delete_note
    return await handle_delete_note(f"delete note titled {title}")


# ===========================================================================
# Reminders
# ===========================================================================


@mcp.tool()
async def set_reminder(when: str, what: str) -> str:
    """Set a reminder for a future date/time.

    Args:
        when: Natural language date/time (e.g. "tomorrow at 3pm", "in 2 days", "next monday").
              Uses dateparser — handles relative and absolute expressions.
        what: What to be reminded about (e.g. "call mom", "submit report").
    """
    from skills.reminder.create import handle_create_reminder
    return await handle_create_reminder(when, what)


@mcp.tool()
async def set_timer(duration: str, what: str = "Timer") -> str:
    """Start a countdown timer.

    Args:
        duration: Natural language duration (e.g. "5 minutes", "30 seconds", "1 hour").
        what: Description shown when the timer fires (default: "Timer").
    """
    from skills.reminder.timer import handle_timer
    return await handle_timer(duration, what)


@mcp.tool()
async def list_reminders() -> str:
    """List all pending reminders."""
    from skills.reminder.list import handle_list_reminders
    return await handle_list_reminders()


# ===========================================================================
# Skills (pass-through tools)
# ===========================================================================


@mcp.tool()
async def weather(query: str) -> str:
    """Get current weather or forecast for a location.

    Args:
        query: Natural language weather request (e.g. "weather in London", "will it rain tomorrow in Paris").
    """
    from skills.weather.skill import handle_weather
    return await handle_weather(query)


@mcp.tool()
async def stocks(query: str) -> str:
    """Look up current stock or crypto prices.

    Args:
        query: Ticker or company name (e.g. "Tesla", "AAPL", "Bitcoin price").
    """
    from skills.finance.skill import handle_finance_stocks
    return await asyncio.to_thread(handle_finance_stocks, query)


@mcp.tool()
async def directions(query: str) -> str:
    """Get directions between two places.

    Args:
        query: Origin and destination (e.g. "directions from Putney to Chelsea", "how do I get from London to Paris").
    """
    from skills.transportation.skill import handle_transportation
    return await handle_transportation(query)


@mcp.tool()
async def news(query: str = "today's headlines") -> str:
    """Fetch current news headlines.

    Args:
        query: Optional filter (e.g. "AI news", "tech news today"). Defaults to today's headlines.
    """
    from skills.newsfeed.skill import handle_newsfeed
    return await handle_newsfeed(query)


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web for information. Use for things the model doesn't know.

    Args:
        query: Natural language search query.
    """
    from skills.search.skill import handle_search
    return await handle_search(query)


@mcp.tool()
async def deep_research(query: str) -> str:
    """Run an in-depth research task. Triggers the multi-step research agent.

    Use for: research, deep dive, find everything about, tell me everything about.
    More thorough than web_search but slower.

    Args:
        query: The topic to research in depth.
    """
    from skills.search.skill import handle_search
    return await handle_search(f"research {query}")


@mcp.tool()
async def home_control(query: str) -> str:
    """Control smart home devices (Hue lights, etc.).

    Args:
        query: Imperative command (e.g. "turn the lights on", "kitchen lights off", "dim the bedroom to 30%").
    """
    from skills.home_control.skill import handle_home_control
    return await handle_home_control(query)


@mcp.tool()
async def spotify(query: str) -> str:
    """Control Spotify playback or query the library.

    Args:
        query: Music command (e.g. "play Bohemian Rhapsody", "pause", "what's playing", "next").
    """
    from skills.spotify.skill import handle_spotify
    return await handle_spotify(query)


@mcp.tool()
async def recipe(query: str) -> str:
    """Search for a recipe by dish name, or get a recipe from ingredients.

    Args:
        query: Dish name (e.g. "iskender kebab") OR ingredients list (e.g. "what can I make with chicken and rice").
    """
    from skills.nutrition.skill import handle_nutrition
    return await handle_nutrition(query)


@mcp.tool()
async def nutrition(query: str) -> str:
    """Look up nutrition facts for a food.

    Args:
        query: Food name (e.g. "banana", "avocado", "oats"). NOT for cooked dishes.
    """
    from skills.nutrition.skill import handle_nutrition
    return await handle_nutrition(f"nutrition facts for {query}")


@mcp.tool()
async def info(query: str) -> str:
    """Look up general knowledge / facts. Falls back to web search and then to LLM.

    Args:
        query: A factual question (e.g. "what year was the Eiffel Tower built", "who is Marie Curie").
    """
    from skills.information.skill import handle_information_query
    return await handle_information_query(query)


@mcp.tool()
async def plan(task: str) -> str:
    """Build a multi-step plan for a complex task. The planner will call other skills as needed.

    Args:
        task: Description of the task to plan (e.g. "plan a weekend trip to Paris", "organize my work week").
    """
    from skills.planner.skill import run_planner
    return await run_planner(task)


@mcp.tool()
async def send_query(query: str) -> str:
    """Route a free-form query through jeanclawd's full intent classifier and skill handlers.

    Use this when you're not sure which specific tool to call — the agent will
    figure it out (weather, notes, reminders, chat, etc.) and return the result.

    Args:
        query: Any natural language request.
    """
    from agents.intent_router.intent_router import route
    from agents.intent_classifier.intent_classifier import classify
    intent = classify(query)
    return await route(intent, query)


# ===========================================================================
# Self-reflection
# ===========================================================================


@mcp.tool()
async def reflect(n: int = 3) -> str:
    """Run self-reflection on the most recent conversation exchanges.

    Stores any self-critique notes (routing mistakes, incomplete answers, etc.)
    to the Reflections collection. Use to feed the future self-correction
    mechanism with structured misfires.

    Args:
        n: Number of recent exchanges to review (default 3, max 20).
    """
    from skills.reflection.skill import get_recent_exchanges, reflect_on_exchanges
    n = max(1, min(n, 20))
    exchanges = await get_recent_exchanges(n)
    if not exchanges:
        return "No recent exchanges to reflect on."
    saved = await reflect_on_exchanges(exchanges, source="mcp /reflect")
    if saved:
        return f"Saved {saved} reflection(s) to the Reflections collection."
    return "No misfires detected — no reflections saved."


@mcp.tool()
async def list_reflections(category: Optional[str] = None, limit: int = 20) -> str:
    """List stored self-reflections, optionally filtered by category.

    Args:
        category: Optional filter — one of: routing, factual, tone, completeness, other.
        limit: Maximum number to return (default 20).
    """
    from db.schemas import Reflection
    query = Reflection.find()
    if category:
        if category not in ("routing", "factual", "tone", "completeness", "other"):
            return f"Invalid category '{category}'."
        query = Reflection.find(Reflection.category == category)
    items = await query.sort(-Reflection.created_at).limit(limit).to_list()
    if not items:
        return f"No reflections{' ['+category+']' if category else ''} found."
    lines = []
    for r in items:
        ts = r.created_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"- [{ts}] ({r.category}) {r.observation}\n    → Lesson: {r.lesson}")
    return f"Reflections{' ['+category+']' if category else ''} ({len(items)}):\n" + "\n".join(lines)


# ===========================================================================
# Journal
# ===========================================================================


@mcp.tool()
async def read_journal(for_date: Optional[str] = None) -> str:
    """Read the journal entry for a given date. Defaults to the most recent.

    Args:
        for_date: ISO date string (YYYY-MM-DD). Omit for the most recent entry.
    """
    from db.schemas import Journal
    if for_date:
        try:
            d = date.fromisoformat(for_date)
        except ValueError:
            return f"Invalid date '{for_date}'. Use YYYY-MM-DD."
        doc = await Journal.find_one(Journal.datestamp == d)
        if not doc:
            return f"No journal entry for {for_date}."
        return f"Journal entry for {for_date}:\n\n{doc.summary}"
    else:
        doc = await Journal.find().sort(-Journal.datestamp).limit(1).first_or_none()
        if not doc:
            return "No journal entries yet."
        return f"Most recent journal entry ({doc.datestamp.isoformat()}):\n\n{doc.summary}"


@mcp.tool()
async def compact_conversations(for_date: Optional[str] = None) -> str:
    """Compact and journal a day's conversations.

    Use to force a daily summary instead of waiting for the end-of-day cron.
    Skips destructive apocalypse/flush operations.

    Args:
        for_date: ISO date string (YYYY-MM-DD). Omit for today.
    """
    if for_date:
        try:
            d = date.fromisoformat(for_date)
        except ValueError:
            return f"Invalid date '{for_date}'. Use YYYY-MM-DD."
    else:
        d = date.today()
    from skills.journal.compact import compact_conversations as _compact
    return await _compact(d)


# ===========================================================================
# Resources
# ===========================================================================


@mcp.resource("persona://current")
def resource_active_persona() -> str:
    """The active persona's name, description, and greeting style."""
    from agents.persona import (
        get_active_name,
        get_active_persona_id,
        get_active_preamble,
    )
    return f"persona_id: {get_active_persona_id()}\nname: {get_active_name()}\n\n{get_active_preamble()}"


@mcp.resource("context://current")
def resource_context() -> str:
    """Current LLM context window usage."""
    from utils.resource_monitor import _context_used_tokens, get_context_max
    n_ctx = get_context_max() or 0
    used = _context_used_tokens
    pct = (used / n_ctx * 100) if n_ctx else 0
    return f"used: {used}\nmax: {n_ctx}\npercent: {pct:.1f}"


@mcp.resource("notes://recent")
async def resource_recent_notes() -> str:
    """The 10 most recently created notes."""
    from db.schemas import Note
    notes = await Note.find().sort(-Note.created_at).limit(10).to_list()
    if not notes:
        return "No notes yet."
    lines = []
    for n in notes:
        tag = f" [{n.category}]" if n.category else ""
        date_str = n.created_at.strftime("%Y-%m-%d")
        lines.append(f"- {n.title}{tag} — {date_str}\n  {n.content[:200]}")
    return "\n\n".join(lines)


@mcp.resource("reminders://pending")
async def resource_pending_reminders() -> str:
    """All incomplete reminders, soonest first."""
    from db.schemas import Reminders
    items = await Reminders.find(not Reminders.is_completed).sort(Reminders.reminder_datetime).to_list()
    if not items:
        return "No pending reminders."
    lines = []
    for r in items:
        ts = r.reminder_datetime.strftime("%Y-%m-%d %H:%M")
        lines.append(f"- {ts} — {r.description}")
    return "\n".join(lines)


@mcp.resource("reflections://recent")
async def resource_recent_reflections() -> str:
    """The 10 most recent self-reflections."""
    from db.schemas import Reflection
    items = await Reflection.find().sort(-Reflection.created_at).limit(10).to_list()
    if not items:
        return "No reflections stored yet. Run `reflect` to generate some."
    lines = []
    for r in items:
        ts = r.created_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"- [{ts}] ({r.category})\n  Observed: {r.observation}\n  Lesson: {r.lesson}")
    return "\n\n".join(lines)


# ===========================================================================
# Prompts
# ===========================================================================


@mcp.prompt()
def save_note_prompt(content: str) -> str:
    """Template for saving a note. Use when the user wants to remember something.

    Args:
        content: The content the user wants to save.
    """
    return f"""The user wants to save a note. Extract a clean title and well-formatted markdown body.

Content: {content}

Return the call to `save_note` with a concise plain-text title and a markdown body. Category (read/listen/watch/eat/visit) only if clearly applicable."""


@mcp.prompt()
def remind_me_prompt(what: str, when: str) -> str:
    """Template for setting a reminder.

    Args:
        what: What the user wants to be reminded about.
        when: When they want to be reminded (natural language).
    """
    return f"""The user wants a reminder.

What: {what}
When: {when}

Call `set_reminder` with the natural-language `when` (dateparser handles it) and the `what` as the description. Confirm back to the user with the parsed date and a friendly acknowledgement."""


@mcp.prompt()
def reflect_prompt(n: int = 3) -> str:
    """Template for self-reflecting on recent conversation exchanges.

    Args:
        n: Number of recent exchanges to review.
    """
    return f"""Run self-reflection on the most recent {n} exchanges.

Call `reflect` with n={n} and report back what was found — saved reflections and their categories. If no misfires, say so explicitly. Self-reflections feed the future self-correction mechanism."""


@mcp.prompt()
def consult_agent_prompt(question: str) -> str:
    """Template for delegating a question to jeanclawd's full intent classifier.

    Use this when you want the agent to handle something end-to-end (including
    persona-aware chat responses) rather than picking a specific tool yourself.

    Args:
        question: The free-form question or request.
    """
    return f"""Delegate this to jeanclawd — the agent's intent classifier will route it to the right skill, and the response will carry the active persona's voice:

{question}

Call `send_query` with the exact question and return the agent's response verbatim."""


# ===========================================================================
# Entry point
# ===========================================================================


_SENSITIVE_KEYS = {"api_key", "apikey", "token", "password", "secret", "authorization"}


def _redact_args(args: dict) -> str:
	"""Format tool args for logging, redacting likely secrets and truncating long strings."""
	if not args:
		return ""
	parts = []
	for k, v in args.items():
		if k.lower() in _SENSITIVE_KEYS:
			parts.append(f"{k}=***")
			continue
		s = repr(v)
		if len(s) > 80:
			s = s[:77] + "..."
		parts.append(f"{k}={s}")
	return "(" + ", ".join(parts) + ")"


def _install_tool_call_logging() -> None:
	"""Wrap mcp._tool_manager.call_tool to log every invocation to stderr.

	Format:  [tool] -> name(args)
	         [tool] <- name ok in 0.42s (1.2KB)  /  [tool] !! name failed: <error>
	"""
	import time
	from mcp.server.fastmcp.tools import ToolManager

	original = ToolManager.call_tool

	async def logged_call_tool(self, name, arguments, context=None, convert_result=False):
		ts = time.monotonic()
		logger.info("[tool] \u2192 %s%s", name, _redact_args(arguments or {}))
		try:
			result = await original(self, name, arguments or {}, context=context, convert_result=convert_result)
		except Exception as e:
			dt = time.monotonic() - ts
			logger.warning("[tool] \u2717 %s failed in %.2fs: %s: %s", name, dt, type(e).__name__, str(e)[:200])
			raise
		dt = time.monotonic() - ts
		logger.info("[tool] \u2190 %s ok in %.2fs [%s]%s", name, dt, _fmt_size(result), _fmt_preview(_extract_text(result)))
		return result

	ToolManager.call_tool = logged_call_tool
	logger.debug("tool-call logging installed")


def _result_blocks(result) -> list | None:
	"""Extract the list of TextContent blocks from any FastMCP tool result shape.

	FastMCP tool.run() returns:
	  - str (when convert_result=False and tool returns str)
	  - (content_blocks, structured_content) tuple (when convert_result=True)
	  - list (some tools return lists directly)
	  - object with .content (CallToolResult)
	"""
	try:
		if isinstance(result, str):
			return None
		if isinstance(result, tuple) and result:
			first = result[0]
			if isinstance(first, list):
				return first
			return None
		if isinstance(result, list):
			return result
		if hasattr(result, "content") and result.content:
			return result.content
	except Exception:
		pass
	return None


def _fmt_size(result) -> str:
	"""Best-effort size of a tool result, in bytes. Returns '?' on any miss."""
	try:
		if isinstance(result, str):
			return _human_size(len(result.encode("utf-8")))
		blocks = _result_blocks(result)
		if blocks is None:
			return "?"
		total = sum(len(getattr(b, "text", "") or "") for b in blocks)
		return _human_size(total)
	except Exception:
		return "?"


def _human_size(n: int) -> str:
	if n < 1024:
		return f"{n}B"
	if n < 1024 * 1024:
		return f"{n / 1024:.1f}KB"
	return f"{n / (1024 * 1024):.1f}MB"


def _extract_text(result, n: int = 80) -> str:
	"""Pull the first text snippet out of a tool result for log previews."""
	try:
		if isinstance(result, str):
			return result[:n].replace("\n", " ")
		blocks = _result_blocks(result)
		if not blocks:
			return ""
		for b in blocks:
			t = getattr(b, "text", None)
			if t:
				return t[:n].replace("\n", " ")
	except Exception:
		pass
	return ""


def _fmt_preview(preview: str) -> str:
	return f"  '{preview}'" if preview else ""


def main() -> None:
    """Run the MCP server over stdio."""
    _redirect_rich_console_to_stderr()
    _print_banner()
    _install_tool_call_logging()
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("MCP server stopped (Ctrl-C)")
    except Exception:
        logger.exception("MCP server crashed")
        raise


if __name__ == "__main__":
    main()
