"""Built-in slash commands for the assistant CLI."""

import asyncio
import os

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


async def cmd_memories(args: str) -> None:
	"""List or search canonical user memories.

	Usage:
	  /memories           — list all stored memories
	  /memories <query>   — search stored LangMem profile fields
	"""
	try:
		from utils.langmem_service import list_memories

		query = args.strip()
		results = await list_memories(query)
		if not results:
			logger.info(f"No memories found{f' for {query!r}' if query else ''}.")
			return
		lines = "\n".join(
			f"  {item.key:14}  {item.value} [{'/'.join(item.namespace)}]"
			for item in results
		)
		logger.info(f"{len(results)} stored memory field(s):\n{lines}")
	except Exception as e:
		logger.error(f"memories failed: {e}")


async def cmd_forget(args: str) -> None:
	"""Clear one LangMem profile field."""
	try:
		from utils.langmem_service import forget_memory

		field = args.strip()
		if not field:
			logger.info("Usage: /forget-memory <name|location|likes|dislikes|interests|relationships|context>")
			return
		if await forget_memory(field):
			logger.info(f"Cleared memory field: {field}")
		else:
			logger.info(f"Unknown or empty memory field: {field}")
	except Exception as e:
		logger.error(f"forget failed: {e}")


async def cmd_apocalypse() -> None:
	"""Wipe interactions and learned memory facts; preserve explicit settings."""
	try:
		from db.schemas import Interaction
		from utils.langmem_service import clear_memories

		interaction_count = await Interaction.delete_all()
		memory_count = await clear_memories()
		from skills.conversation.skill import _history

		_history.clear()
		logger.info(
			f"☢  Apocalypse: wiped {interaction_count} interaction(s) and {memory_count} LangMem profile record(s)."
		)
	except Exception as e:
		logger.error(f"apocalypse failed: {e}")


_MODEL_PURPOSES: dict[str, tuple[str, str]] = {
	"generalist": ("Generalist chat and reasoning", "MODEL_GENERALIST"),
	"intent": ("Intent classification and query rewriting", "MODEL_INTENT"),
	"vision": ("Vision and image description", "MODEL_VLM"),
	"embedding": ("Memory and semantic embeddings", "MODEL_EMBEDDING"),
	"planning": ("Multi-step planning", "MODEL_PLANNING"),
	"research": ("Deep research", "MODEL_RESEARCH"),
}


_EMBEDDING_MODEL_HINTS = (
	"embed",
	"bge-",
	"e5-",
	"gte-",
	"jina-embeddings",
	"minilm",
	"mxbai-embed",
	"nomic-embed",
	"snowflake-arctic-embed",
	"text-embedding",
)


def _is_embedding_model(model: dict) -> bool:
	"""Classify embedding models from endpoint metadata and conservative ID hints.

	The OpenAI /models schema does not require a capability field, so the ID
	patterns are a fallback for providers such as LM Studio that return only IDs.
	"""
	metadata = [
		model.get("id", ""),
		model.get("type", ""),
		model.get("task", ""),
		model.get("architecture", ""),
		model.get("capabilities", ""),
	]
	haystack = " ".join(str(value).lower() for value in metadata)
	return any(hint in haystack for hint in _EMBEDDING_MODEL_HINTS)


async def _inference_models() -> list[dict] | None:
	"""Return model records advertised by the configured inference endpoint."""
	import aiohttp

	host = os.getenv("MODEL_HOST", "http://localhost:1234/v1").rstrip("/")
	try:
		async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
			async with session.get(f"{host}/models") as resp:
				if resp.status != 200:
					logger.error(f"Failed to fetch models (HTTP {resp.status}).")
					return None
				data = await resp.json()
	except Exception as exc:
		logger.error(f"Could not reach inference server: {exc}")
		return None

	records: list[dict] = []
	seen: set[str] = set()
	for model in data.get("data", []):
		if not isinstance(model, dict) or not isinstance(model.get("id"), str):
			continue
		if model["id"] not in seen:
			records.append(model)
			seen.add(model["id"])
	return records


async def cmd_change_model(args: str) -> None:
	"""Choose a model purpose, then select an available endpoint model for it."""
	model_records = await _inference_models()
	if not model_records:
		logger.info("No models found at the inference endpoint.")
		return

	purpose = args.strip().lower()
	if not purpose:
		try:
			from prompt_toolkit import PromptSession
			from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

			purpose_picker = PromptSession(
				completer=FuzzyCompleter(
					WordCompleter(list(_MODEL_PURPOSES), sentence=True)
				)
			)
			logger.info(
				"Model purpose:\n" + "\n".join(
					f"  {key:<10} — {description} (current: {os.getenv(env_key, 'unset')})"
					for key, (description, env_key) in _MODEL_PURPOSES.items()
				)
			)
			purpose = (await purpose_picker.prompt_async("  purpose ❯ ")).strip().lower()
		except (ImportError, KeyboardInterrupt, EOFError):
			logger.info("Cancelled.")
			return

	if purpose not in _MODEL_PURPOSES:
		logger.info(f"Unknown purpose '{purpose}'. Choose: {', '.join(_MODEL_PURPOSES)}")
		return

	description, env_key = _MODEL_PURPOSES[purpose]
	if purpose == "embedding":
		candidate_records = [model for model in model_records if _is_embedding_model(model)]
	else:
		candidate_records = [model for model in model_records if not _is_embedding_model(model)]
	candidates = [model["id"] for model in candidate_records]
	if not candidates:
		if purpose == "embedding":
			logger.info(
				"No embedding model was identifiable from the inference endpoint metadata or model IDs."
			)
		else:
			logger.info("No non-embedding models were found for this purpose.")
		return

	try:
		from prompt_toolkit import PromptSession
		from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

		model_picker = PromptSession(
			completer=FuzzyCompleter(WordCompleter(candidates, sentence=True))
		)
		logger.info(
			f"{description}. {len(candidates)} compatible model(s) available. "
			"Tab to autocomplete, Enter to confirm, Ctrl-C to cancel."
		)
		selected = (await model_picker.prompt_async("  model ❯ ")).strip()
	except (ImportError, KeyboardInterrupt, EOFError):
		logger.info("Cancelled.")
		return

	if not selected:
		logger.info("No model selected.")
		return
	if selected not in candidates:
		logger.error(f"'{selected}' is not advertised by the inference endpoint.")
		return

	os.environ[env_key] = selected
	if env_key in {"MODEL_GENERALIST", "MODEL_INTENT"}:
		import providers

		providers._provider = None
		providers._intent_provider = None

	logger.info(f"{description} model → {selected} ({env_key})")


async def cmd_connect(args: str) -> None:
	"""Connect a service that requires an auth flow.

	Usage:
	  /connect <service> — choose from the registered connector flows
	"""
	from connectors.registry import connector_ids, get_connector

	service = args.strip().lower()
	if not service:
		try:
			from prompt_toolkit import PromptSession
			from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

			picker = PromptSession(
				completer=FuzzyCompleter(WordCompleter(connector_ids(), sentence=True))
			)
			logger.info("Select a connector (Tab to autocomplete, Ctrl-C to cancel).")
			service = (await picker.prompt_async("  connector ❯ ")).strip().lower()
		except (ImportError, KeyboardInterrupt, EOFError):
			logger.info("Cancelled.")
			return

	connector = get_connector(service)
	if connector is None:
		available = ", ".join(connector_ids()) or "none"
		logger.info(f"Unknown connector '{service}'. Available: {available}")
		return

	if await connector.is_connected():
		logger.info(
			f"{connector.display_name} is already connected. "
			"Reconnecting may overwrite stored credentials."
		)
	if connector.service_id == "spotify":
		logger.info("Opening Spotify auth in browser — you can keep chatting while you complete the flow.")
	elif connector.service_id == "hue":
		logger.info("Press the link button on your Hue bridge now — connecting in the background.")
	else:
		logger.info(f"Starting {connector.display_name} connection in the background.")

	async def _connect() -> None:
		logger.info(await connector.connect())

	asyncio.create_task(_connect())


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
				completer=FuzzyCompleter(WordCompleter(persona_ids, sentence=True)),
				complete_while_typing=True,
			)
			logger.info("Type a persona or press Tab to autocomplete.")
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


async def cmd_introspect() -> None:
	"""Inspect the safe codebase in bounded batches and persist assessments."""
	import os
	from collections import defaultdict
	from pathlib import Path

	from pydantic import BaseModel, Field
	from rich.console import Console
	from rich.live import Live
	from rich.spinner import Spinner

	from utils.self_inspector import (
		architecture,
		exposed_files,
		list_reflections,
		read_file,
		read_reflection,
		record_reflection,
	)

	class _Introspection(BaseModel):
		purpose_and_capabilities: str | list[str] = Field(
			description="What this application is for and what capabilities the inspected code exposes"
		)
		inspected: str | list[str] = Field(description="Files, modules, and architectural areas actually inspected")
		findings: str | list[str] = Field(description="Evidence-based findings; distinguish facts from uncertainty")
		recommendations: str | list[str] = Field(
			default="No recommendations recorded.",
			description="Recommendations only when purpose_and_capabilities supports them; otherwise use the default",
		)

	def _as_markdown(value: str | list[str]) -> str:
		"""Normalize model prose or list-shaped output for the reflection file."""
		if isinstance(value, list):
			return "\n".join(f"- {item}" for item in value if str(item).strip())
		return value

	def _scope(path: str) -> str:
		return Path(path).parts[0] if len(Path(path).parts) > 1 else "root"

	def _chunks(paths: list[str], max_chars: int = 28_000) -> list[tuple[str, str]]:
		"""Read every text file and split it into bounded model-input batches."""
		batches: list[tuple[str, str]] = []
		current_scope = ""
		current: list[str] = []
		current_chars = 0
		for path in paths:
			try:
				content = read_file(path, max_bytes=64_000)
			except (OSError, ValueError, PermissionError) as exc:
				logger.event("introspection_file_skipped", file=path, error_type=type(exc).__name__)
				continue
			parts = [content[i : i + max_chars] for i in range(0, len(content), max_chars)] or [""]
			for part_index, part in enumerate(parts):
				label = f"--- {path}"
				if len(parts) > 1:
					label += f" (part {part_index + 1}/{len(parts)})"
				label += " ---\n"
				block = label + part
				path_scope = _scope(path)
				if current and (path_scope != current_scope or current_chars + len(block) > max_chars):
					batches.append((current_scope, "\n\n".join(current)))
					current = []
					current_chars = 0
				current_scope = path_scope
				current.append(block)
				current_chars += len(block)
		if current:
			batches.append((current_scope, "\n\n".join(current)))
		return batches

	try:
		logger.event("introspection_progress", phase="started")
		inventory = exposed_files()
		text_files = [
			path
			for path in inventory
			if path.startswith("introspections/") is False
			and Path(path).suffix.lower() in {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt", ".ini", ".cfg", ".sql"}
		]
		logger.event("introspection_progress", phase="inventory", exposed_files=len(inventory), text_files=len(text_files))
		logger.event("introspection_progress", phase="reading_architecture", file="CODEBASE.md")
		architecture_map = architecture()
		memory_parts = []
		for reflection_path in list_reflections(limit=10):
			try:
				memory_parts.append(read_reflection(Path(reflection_path).name, max_bytes=4_000))
			except (OSError, ValueError, PermissionError):
				continue
		prior_memory = "\n\n".join(memory_parts[-5:]) or "No prior introspection records available."
		batches = _chunks(text_files)
		logger.event("introspection_progress", phase="batches_prepared", batches=len(batches))

		from utils.llm_structured_output import generate_structured_output

		saved = 0
		for index, (scope, source_batch) in enumerate(batches, 1):
			logger.event("introspection_batch_started", batch=index, total_batches=len(batches), scope=scope)
			prompt = (
				"Perform a read-only architectural introspection of this codebase batch. "
				"Use only the supplied facade output as evidence. Do not invent files, behavior, "
				"or capabilities. State uncertainty explicitly. Recommendations are allowed only "
				"after identifying the application's purpose and the capabilities actually observed.\n\n"
				f"Complete safe inventory: {len(inventory)} files; this batch contains {scope} sources.\n"
				f"Architecture map:\n{architecture_map}\n\n"
				f"Prior introspection memory:\n{prior_memory}\n\n"
				f"Current source batch:\n{source_batch}"
			)
			try:
				with Live(
					Spinner("dots", text=f"Introspecting {scope} ({index}/{len(batches)})…"),
					console=Console(stderr=True),
					refresh_per_second=10,
					transient=True,
				):
					result = await asyncio.to_thread(
						generate_structured_output,
						model_name=os.environ["MODEL_GENERALIST"],
						user_prompt=prompt,
						system_prompt=(
							"You are performing a private developer-only codebase introspection. "
							"Return concise Markdown-compatible text in the requested fields. "
							"Never suggest changing behavior merely to make self-modification easier."
						),
						pydantic_model=_Introspection,
						max_tokens=2_048,
						temperature=0.1,
					)
				purpose = _as_markdown(result.purpose_and_capabilities)
				inspected = _as_markdown(result.inspected)
				findings = _as_markdown(result.findings)
				recommendations = _as_markdown(result.recommendations)
				path = record_reflection(
					f"Inventory: {len(inventory)} exposed files.\n\n{inspected}",
					findings,
					recommendations,
					purpose_and_capabilities=purpose,
					scope=f"{scope}-{index}",
				)
				saved += 1
				logger.event("introspection_batch_completed", batch=index, scope=scope, reflection_path=path)
			except Exception as exc:
				logger.event("introspection_batch_failed", level="ERROR", batch=index, scope=scope, error_type=type(exc).__name__)
		logger.event("introspection_completed", saved=saved, total_batches=len(batches))
	except Exception as exc:
		logger.error(f"Introspection failed: {exc}")


async def cmd_introspection(args: str) -> None:
	"""Answer a developer question using the persisted introspection records."""
	import os

	from pydantic import BaseModel, Field
	from rich.console import Console
	from rich.live import Live
	from rich.spinner import Spinner

	from utils.self_inspector import list_reflections, read_reflection

	question = args.strip()
	if not question:
		logger.info("Usage: /introspection <question>")
		return

	class _Answer(BaseModel):
		answer: str = Field(description="A concise evidence-based answer to the developer's question")

	try:
		paths = list_reflections(limit=100)
		if not paths:
			logger.info("No introspection records found. Run /introspect first.")
			return

		records = []
		for path in paths:
			try:
				records.append((path, read_reflection(path.rsplit("/", 1)[-1], max_bytes=64_000)))
			except (OSError, ValueError, PermissionError) as exc:
				logger.info(f"[introspection] Skipping {path}: {exc}")

		batches: list[str] = []
		current: list[str] = []
		current_chars = 0
		for path, record in records:
			block = f"\n--- {path} ---\n{record}"
			if current and current_chars + len(block) > 24_000:
				batches.append("\n".join(current))
				current = []
				current_chars = 0
			current.append(block)
			current_chars += len(block)
		if current:
			batches.append("\n".join(current))

		logger.info(f"[introspection] Asking about {len(records)} records in {len(batches)} batches")
		from utils.llm_structured_output import generate_structured_output

		batch_answers = []
		for index, batch in enumerate(batches, 1):
			logger.info(f"[introspection] Analyzing memory batch {index}/{len(batches)}")
			with Live(
				Spinner("dots", text=f"Reading introspection memory ({index}/{len(batches)})…"),
				console=Console(stderr=True),
				refresh_per_second=10,
				transient=True,
			):
				result = await asyncio.to_thread(
					generate_structured_output,
					model_name=os.environ["MODEL_GENERALIST"],
					user_prompt=(
						f"Developer question: {question}\n\n"
						"Answer using only the following persisted introspection records. "
						"Separate evidence from uncertainty and do not invent capabilities.\n"
						+ batch
					),
					system_prompt="You are reviewing the agent's private introspection memory for a developer.",
					pydantic_model=_Answer,
					max_tokens=1_024,
					temperature=0.1,
				)
			batch_answers.append(result.answer)

		logger.info("[introspection] Synthesizing the answer")
		with Live(
			Spinner("dots", text="Synthesizing introspection answer…"),
			console=Console(stderr=True),
			refresh_per_second=10,
			transient=True,
		):
			final = await asyncio.to_thread(
				generate_structured_output,
				model_name=os.environ["MODEL_GENERALIST"],
				user_prompt=(
					f"Developer question: {question}\n\n"
					"Synthesize a direct answer from these batch analyses. "
					"Mention uncertainty where the records do not establish a fact.\n\n"
					+ "\n\n".join(batch_answers)
				),
				system_prompt="You are the agent's introspection-memory question-answering layer.",
				pydantic_model=_Answer,
				max_tokens=1_024,
				temperature=0.1,
			)
		logger.info(final.answer)
	except Exception as exc:
		logger.error(f"Introspection query failed: {exc}")


async def cmd_help() -> None:
	logger.info(
		"Available commands:\n"
		"  /clear             — wipe the in-memory conversation history\n"
		"  /session [list|new|resume|close] — manage resumable sessions\n"
		"  /persona [id] [v]  — switch persona and voice\n"
		"  /voice [id]        — switch voice only (interactive picker or direct ID)\n"
		"  /news-length [s|m|l] — set news detail level (short | medium | long)\n"
		"  /change-model      — choose an available model by purpose\n"
		"  /connect <service> — authenticate a registered service\n"
		"  /memories [q]      — list/search LangMem profile fields\n"
		"  /forget-memory <field> — clear one LangMem profile field\n"
		"  /apocalypse        — wipe interactions and LangMem profile memory\n"
		"  /ctx               — show context window usage for the current session\n"
		"  /look <path> [q]   — describe an image via the VLM and inject into context\n"
		"  /introspect        — inspect the safe codebase facade and save findings\n"
		"  /introspection <q> — ask a developer question about saved introspections\n"
		"  /help              — show this message"
	)


_COMMANDS = {
	"/clear": cmd_clear,
	"/session": cmd_session,
	"/persona": cmd_persona,
	"/voice": cmd_voice,
	"/news-length": cmd_news_length,
	"/change-model": cmd_change_model,
	"/connect": cmd_connect,
	"/memories": cmd_memories,
	"/forget-memory": cmd_forget,
	"/apocalypse": cmd_apocalypse,
	"/ctx": cmd_ctx,
	"/look": cmd_look,
	"/introspect": cmd_introspect,
	"/introspection": cmd_introspection,
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
