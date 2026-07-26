"""Built-in slash commands for the assistant CLI."""

import asyncio

from utils.log import logger

from skills.conversation.skill import _history

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_history() -> str:
	"""Return the current session history as a plain-text string."""
	messages = list(_history)
	if not messages:
		return ""
	lines = []
	for msg in messages:
		role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
		lines.append(f"{role}: {msg.content}")
	return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


async def cmd_clear() -> None:
	_history.clear()
	logger.info("Conversation history cleared.")


async def cmd_history() -> None:
	text = _format_history()
	if not text:
		logger.info("No conversation history yet.")
	else:
		logger.info(f"Session history\n{text}")


async def cmd_session(args: str) -> None:
	"""Create, list, resume, or close resumable conversation sessions."""
	from utils.session_service import (
		close_session,
		create_session,
		get_active_session_id,
		list_sessions,
		resume_session,
	)

	parts = args.strip().split(maxsplit=1)
	action = parts[0].lower() if parts else "list"
	value = parts[1].strip() if len(parts) > 1 else ""
	if action == "new":
		session_id = await create_session(title=value or None)
		logger.info(f"Started new session: {session_id}")
	elif action == "resume":
		if not value:
			logger.info("Usage: /session resume <uuid>")
			return
		try:
			session_id = await resume_session(value)
		except ValueError:
			logger.info("Session not found. Check the UUID and try again.")
			return
		logger.info(f"Resumed session: {session_id}")
	elif action == "close":
		await close_session(value or get_active_session_id())
		session_id = await create_session()
		logger.info(f"Closed session and started: {session_id}")
	else:
		sessions = await list_sessions()
		active = get_active_session_id()
		if not sessions:
			logger.info("No sessions found.")
			return
		lines = []
		for session in sessions:
			marker = "*" if session.session_id == active else " "
			label = session.title or "Untitled session"
			lines.append(f"{marker} {session.session_id}  {label}  [{session.status}]")
		logger.info("Sessions:\n" + "\n".join(lines))


async def cmd_planner(args: str) -> None:
	from agents.planner import run_planner
	from utils.inference_gate import foreground_inference
	from utils.memory_service import record_interaction

	task = args.strip()
	if not task:
		logger.info("Please provide a task. Usage: /planner <task>")
		return

	logger.info(f"Running agent on: {task}")
	async with foreground_inference():
		response = await run_planner(task)
	await record_interaction(task, response, intent="PLANNER")
	logger.info(response)


async def cmd_ctx() -> None:
	import os

	import tiktoken

	try:
		from providers import get_provider

		n_ctx = get_provider().context_length()
	except Exception:
		n_ctx = int(os.environ.get("MODEL_CTX", 8192))
	history_text = _format_history()
	enc = tiktoken.get_encoding("cl100k_base")
	used = len(enc.encode(history_text)) if history_text else 0
	pct = used / n_ctx * 100

	bar_width = 40
	filled = int(bar_width * used / n_ctx)
	bar = "█" * filled + "░" * (bar_width - filled)

	logger.info(f"Context usage: {used:,} / {n_ctx:,} tokens ({pct:.1f}%)\n[{bar}]")


async def cmd_note(args: str) -> None:
	"""Create a note from the current conversation context.

	Usage:
	  /note                  — auto-generate title + content from recent exchange
	  /note <title>          — use provided title, generate content from context
	"""
	import os
	from typing import Optional
	from pydantic import BaseModel, Field
	from skills.conversation.skill import get_current_topic
	from skills.notes.create import handle_create_note_from_context
	from db.schemas import NoteCategory
	from utils.llm_structured_output import generate_structured_output

	recent = _format_history()
	if not recent and not args:
		logger.info("Nothing to note — no conversation history and no title provided.")
		return

	topic = get_current_topic() or ""

	class _NoteFromContext(BaseModel):
		title: str = Field(description="Short plain text title for this note")
		content: str = Field(
			description="Markdown content summarising what should be remembered"
		)
		category: Optional[NoteCategory] = Field(
			default=None,
			description="Category: read, listen, watch, eat, visit. Omit if none applies.",
		)

	user_title_hint = f"The user wants to title it: {args}\n\n" if args else ""
	prompt = (
		f"{user_title_hint}"
		f"Current topic: {topic}\n\n"
		f"Recent conversation:\n{recent[-1500:]}"  # last 1500 chars of history
	)

	extracted = generate_structured_output(
		model_name=os.environ["MODEL_GENERALIST"],
		user_prompt=prompt,
		system_prompt=(
			"Extract a note from this conversation. "
			"Title: plain text, concise. "
			"Content: markdown, capture what the user should remember. "
			"Category: read/listen/watch/eat/visit only if clearly applicable."
		),
		pydantic_model=_NoteFromContext,
		max_tokens=512,
	)

	result = await handle_create_note_from_context(
		title=extracted.title,
		content=extracted.content,
		category=extracted.category,
	)
	logger.info(result)


async def cmd_memories(args: str) -> None:
	"""List or search canonical user memories.

	Usage:
	  /memories           — list all stored memories
	  /memories <query>   — semantic search for relevant memories
	"""
	try:
		from utils.memory_service import list_memories

		query = args.strip()
		results = await list_memories(query)
		if not results:
			logger.info(f"No memories found{f' for {query!r}' if query else ''}.")
			return
		lines = "\n".join(
			f"  {str(item.id)[:8]}  {item.key}: {item.value} [{item.namespace}]"
			for item in results
		)
		logger.info(f"{len(results)} active memory fact(s):\n{lines}")
	except Exception as e:
		logger.error(f"memories failed: {e}")


async def cmd_process_memory(args: str) -> None:
	"""Run deferred memory extraction immediately for testing."""
	try:
		from utils.memory_service import process_pending_memory

		include_complete = args.strip().lower() == "all"
		logger.info("Running deferred memory extraction now…")
		if include_complete:
			logger.info("Including previously completed work items.")
		count = await process_pending_memory(force=True, include_complete=include_complete)
		logger.info(f"Memory extraction complete: stored {count} fact(s).")
	except Exception as e:
		logger.error(f"memory extraction failed: {e}")


async def cmd_forget(args: str) -> None:
	"""Mark one canonical fact as forgotten using its ID prefix."""
	try:
		from db.schemas import MemoryFact

		prefix = args.strip()
		if len(prefix) < 4:
			logger.info("Usage: /forget <memory-id-prefix>  (use /memories to list IDs)")
			return
		matches = [
			fact
			for fact in await MemoryFact.find(MemoryFact.status == "active").to_list()
			if str(fact.id).startswith(prefix)
		]
		if not matches:
			logger.info("No active memory matches that ID.")
			return
		if len(matches) > 1:
			logger.info("That prefix is ambiguous; use more characters from /memories.")
			return
		fact = matches[0]
		fact.status = "forgotten"
		await fact.save()
		from utils.memory_index import index_fact

		await asyncio.to_thread(index_fact, fact)
		logger.info(f"Forgot: {fact.key}: {fact.value}")
	except Exception as e:
		logger.error(f"forget failed: {e}")


async def cmd_apocalypse() -> None:
	"""Wipe interactions and learned memory facts; preserve explicit settings."""
	try:
		from db.schemas import Interaction, MemoryFact, MemoryWorkItem

		interaction_count = await Interaction.delete_all()
		fact_count = await MemoryFact.delete_all()
		await MemoryWorkItem.delete_all()
		from utils.memory_index import reset_index

		await asyncio.to_thread(reset_index)
		from skills.conversation.skill import _history

		_history.clear()
		logger.info(
			f"☢  Apocalypse: wiped {interaction_count} interaction(s) and {fact_count} memory fact(s)."
		)
	except Exception as e:
		logger.error(f"apocalypse failed: {e}")


async def cmd_model(args: str) -> None:
	"""Switch the active generalist model.

	Usage:
	  /model          — interactive fuzzy picker from loaded LM Studio models
	  /model <name>   — switch directly by model id
	"""
	import os

	import aiohttp

	host = os.getenv("MODEL_HOST", "http://localhost:1234/v1").rstrip("/")

	try:
		async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
			async with session.get(f"{host}/models") as resp:
				if resp.status != 200:
					logger.error(f"Failed to fetch models (HTTP {resp.status}).")
					return
				data = await resp.json()
	except Exception as e:
		logger.error(f"Could not reach inference server: {e}")
		return

	models = [m["id"] for m in data.get("data", []) if "embed" not in m["id"].lower()]

	if not models:
		logger.info("No non-embedding models found.")
		return

	target = args.strip()
	if target:
		if target not in models:
			available = "\n".join(f"  {m}" for m in models)
			logger.info(f"'{target}' not in loaded models. Available:\n{available}")
			return
		selected = target
	else:
		try:
			from prompt_toolkit import PromptSession
			from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

			picker = PromptSession(
				completer=FuzzyCompleter(WordCompleter(models, sentence=True))
			)
			logger.info(
				f"{len(models)} model(s) loaded. Tab to autocomplete, Enter to confirm, Ctrl-C to cancel."
			)
			selected = (await picker.prompt_async("  model ❯ ")).strip()
		except (ImportError, KeyboardInterrupt, EOFError):
			logger.info("Cancelled.")
			return

		if not selected:
			logger.info("No model selected.")
			return
		if selected not in models:
			logger.error(f"'{selected}' not recognised.")
			return

	os.environ["MODEL_GENERALIST"] = selected

	import providers

	providers._provider = None

	logger.info(f"Active model → {selected}")


async def cmd_connect(args: str) -> None:
	"""Connect a service that requires an auth flow.

	Usage:
	  /connect spotify   — OAuth2 browser flow (opens browser, catches redirect on :8888)
	  /connect hue       — Hue link-button flow (press the button on the bridge first)
	"""
	service = args.strip().lower()

	if service == "spotify":
		from utils.auth.spotify import SpotifyAuthProvider

		auth = SpotifyAuthProvider()
		already = await auth.is_connected()
		if already:
			logger.info(
				"Spotify is already connected. Reconnecting will overwrite the stored tokens."
			)
		logger.info(
			"Opening Spotify auth in browser — you can keep chatting while you complete the flow."
		)

		async def _spotify():
			result = await auth.connect()
			logger.info(result)

		asyncio.create_task(_spotify())

	elif service == "hue":
		from utils.auth.hue import HueAuthProvider

		auth = HueAuthProvider()
		already = await auth.is_connected()
		if already:
			logger.info(
				"Hue is already paired. Reconnecting will overwrite the stored credentials."
			)
		logger.info(
			"Press the link button on your Hue bridge now — connecting in the background."
		)

		async def _hue():
			result = await auth.connect()
			logger.info(result)

		asyncio.create_task(_hue())

	elif not service:
		logger.info("Usage: /connect <service>  (spotify | hue)")
	else:
		logger.info(f"Unknown service '{service}'. Available: spotify, hue")


async def cmd_voice(args: str) -> None:
	"""Set the active voice ID for the agent.

	Usage:
	  /voice              — interactive fuzzy picker
	  /voice <id>         — switch voice directly
	"""
	from prompt_toolkit import PromptSession
	from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

	from tts.tts import voice_ids, set_active_voice_id
	from db.preferences import update_preferences
	from agents.persona import get_active_persona_id

	# Get available voices
	voices = voice_ids

	# If no args provided, show interactive picker
	if not args.strip():
		try:
			picker = PromptSession(
				completer=FuzzyCompleter(WordCompleter(voices, sentence=True))
			)
			logger.info(f"Available voices: {', '.join(voices)}")
			raw_voice = (await picker.prompt_async("  voice ❯ ")).strip().lower()
		except (KeyboardInterrupt, EOFError):
			logger.info("Cancelled.")
			return
		if not raw_voice:
			logger.info("No voice selected.")
			return
		selected_voice = raw_voice
	else:
		# Use provided voice ID
		selected_voice = args.strip()

	# Validate voice exists
	if selected_voice not in voices:
		logger.info(f"Unknown voice '{selected_voice}'. Available: {', '.join(voices)}")
		return

	# Apply voice change
	set_active_voice_id(selected_voice)
	await update_preferences(agent_voice_id=selected_voice)

	# Get current persona for display
	from agents.persona import PERSONAS

	current_persona = get_active_persona_id()

	logger.info(
		f"Voice → {selected_voice} (saved)  |  Persona → {PERSONAS[current_persona]['name']}"
	)


async def cmd_persona(args: str) -> None:
	"""Set the active persona and optionally the voice.

	Usage:
	  /persona              — interactive fuzzy picker
	  /persona <id>         — switch persona directly
	  /persona <id> <voice> — switch persona and voice in one command
	"""
	from prompt_toolkit import PromptSession
	from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

	from agents.persona import PERSONAS, set_active_persona, get_active_voice_id
	from tts.tts import voice_ids, set_active_voice_id
	from db.preferences import update_preferences

	persona_ids = list(PERSONAS.keys())
	parts = args.strip().split()

	# --- Resolve persona ---
	if parts and parts[0] in persona_ids:
		selected_persona = parts[0]
		voice_arg = parts[1] if len(parts) > 1 else None
	elif parts:
		logger.info(f"Unknown persona '{parts[0]}'. Available: {', '.join(persona_ids)}")
		return
	else:
		# Interactive picker
		lines = [f"  {pid:<10} — {PERSONAS[pid]['name']}" for pid in persona_ids]
		logger.info("Available personas:\n" + "\n".join(lines))
		try:
			picker = PromptSession(
				completer=FuzzyCompleter(WordCompleter(persona_ids, sentence=True))
			)
			raw = (await picker.prompt_async("  persona ❯ ")).strip().lower()
		except (KeyboardInterrupt, EOFError):
			logger.info("Cancelled.")
			return
		if not raw:
			logger.info("No persona selected.")
			return
		raw_parts = raw.split()
		selected_persona = raw_parts[0]
		voice_arg = raw_parts[1] if len(raw_parts) > 1 else None
		if selected_persona not in persona_ids:
			logger.info(f"Unknown persona '{selected_persona}'.")
			return

	# --- Resolve voice ---
	current_voice = get_active_voice_id() or "thomas"
	persona_default_voice = PERSONAS[selected_persona].get("voice_id") or current_voice

	if voice_arg and voice_arg in voice_ids:
		selected_voice = voice_arg
	elif voice_arg:
		logger.info(f"Unknown voice '{voice_arg}'. Available: {', '.join(voice_ids)}")
		return
	else:
		try:
			picker = PromptSession(
				completer=FuzzyCompleter(WordCompleter(voice_ids, sentence=True))
			)
			logger.info(f"Voices: {', '.join(voice_ids)}")
			raw_voice = (
				(await picker.prompt_async(f"  voice ❯ [{persona_default_voice}] "))
				.strip()
				.lower()
			)
			selected_voice = raw_voice if raw_voice in voice_ids else persona_default_voice
			if raw_voice and raw_voice not in voice_ids:
				logger.info(f"Unknown voice '{raw_voice}', using '{persona_default_voice}'.")
		except (KeyboardInterrupt, EOFError):
			logger.info("Cancelled.")
			return

	# --- Apply + persist ---
	set_active_persona(selected_persona)
	set_active_voice_id(selected_voice)
	await update_preferences(
		agent_persona=selected_persona, agent_voice_id=selected_voice
	)
	logger.info(
		f"Persona → {PERSONAS[selected_persona]['name']}  |  Voice → {selected_voice}  (saved)"
	)


_NEWS_LENGTHS: tuple[str, ...] = ("short", "medium", "long")


async def cmd_news_length(args: str) -> None:
	"""Set how detailed the news briefing should be.

	Usage:
	  /news-length              — show current setting and available options
	  /news-length <short|medium|long>
	"""
	from db.preferences import get_preferences, update_preferences

	target = args.strip().lower()
	if not target:
		prefs = await get_preferences()
		options = " | ".join(_NEWS_LENGTHS)
		logger.info(f"News length: {prefs.news_length}  (options: {options})")
		return

	if target not in _NEWS_LENGTHS:
		options = " | ".join(_NEWS_LENGTHS)
		logger.info(f"Unknown length '{target}'. Options: {options}")
		return

	await update_preferences(news_length=target)
	logger.info(f"News length → {target}  (saved)")


async def cmd_look(args: str) -> None:
	"""Describe an image via the VLM and inject the description into conversation context."""
	from utils.vlm import describe_image
	from skills.conversation.skill import append_to_history

	parts = args.strip().split(None, 1)
	if not parts:
		logger.info("Usage: /look <image_path> [optional prompt]")
		return

	image_path = parts[0]
	prompt = parts[1] if len(parts) > 1 else None

	try:
		with open(image_path, "rb") as f:
			image_bytes = f.read()
	except FileNotFoundError:
		logger.info(f"File not found: {image_path}")
		return
	except OSError as e:
		logger.info(f"Could not read file: {e}")
		return

	logger.info(f"Analysing image: {image_path}")
	kwargs = {"image_bytes": image_bytes}
	if prompt:
		kwargs["prompt"] = prompt

	description = await describe_image(**kwargs)

	# Inject into conversation history so the CHAT model has visual context
	append_to_history(
		f"[Image: {image_path}]",
		f"I can see the following in the image: {description}",
	)

	logger.info(f"Image description:\n{description}")


async def cmd_help() -> None:
	logger.info(
		"Available commands:\n"
		"  /clear             — wipe the in-memory conversation history\n"
		"  /history           — print session history to the terminal\n"
		"  /session [list|new|resume|close] — manage resumable sessions\n"
		"  /persona [id] [v]  — switch persona and voice\n"
		"  /voice [id]        — switch voice only (interactive picker or direct ID)\n"
		"  /news-length [s|m|l] — set news detail level (short | medium | long)\n"
		"  /model [name]      — switch the active model (fuzzy picker, or pass name directly)\n"
		"  /connect <service> — authenticate a service (spotify | hue)\n"
		"  /memories [q]      — list/search canonical user memory facts\n"
		"  /process-memory [all] — run deferred extraction (all retries history)\n"
		"  /forget-memory <id>       — forget one fact (ID prefix from /memories)\n"
		"  /apocalypse        — wipe interactions and learned memory facts\n"
		"  /ctx               — show context window usage for the current session\n"
		"  /note [title]      — save a note from the current conversation context\n"
		"  /planner <task>    — run an autonomous multi-step planning agent\n"
		"  /look <path> [q]   — describe an image via the VLM and inject into context\n"
		"  /help              — show this message"
	)


_COMMANDS = {
	"/clear": cmd_clear,
	"/history": cmd_history,
	"/session": cmd_session,
	"/persona": cmd_persona,
	"/voice": cmd_voice,
	"/news-length": cmd_news_length,
	"/model": cmd_model,
	"/connect": cmd_connect,
	"/memories": cmd_memories,
	"/process-memory": cmd_process_memory,
	"/forget-memory": cmd_forget,
	"/apocalypse": cmd_apocalypse,
	"/ctx": cmd_ctx,
	"/note": cmd_note,
	"/planner": cmd_planner,
	"/look": cmd_look,
	"/help": cmd_help,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def handle_command(user_input: str) -> bool:
	"""Execute a slash command if recognised. Returns True if handled."""
	import inspect

	parts = user_input.strip().split(None, 1)
	cmd = parts[0].lower()
	args = parts[1] if len(parts) > 1 else ""

	if cmd not in _COMMANDS:
		logger.info(f"Unknown command '{cmd}'. Type /help for available commands.")
		return True  # still consumed — don't pass to classifier
	try:
		fn = _COMMANDS[cmd]
		if inspect.signature(fn).parameters:
			await fn(args)
		else:
			await fn()
	except Exception as e:
		logger.error(f"Command {cmd} failed: {e}")
	return True
