"""Built-in slash commands for the assistant CLI.

Available commands:
  /save           — summarise the current session
  /clear          — wipe the in-memory conversation history
  /history        — print the current session history to the terminal
  /analyze        — extract personal facts from this session into your persona profile
  /planner <task> — run an autonomous multi-step planning agent
  /help           — list available commands
"""

import asyncio

from utils.log import logger

from agents.persona import get_active_identity
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

async def cmd_save() -> None:
    if not _history:
        logger.info("Nothing to save — conversation history is empty.")
        return

    history_text = _format_history()

    from langchain_core.messages import HumanMessage, SystemMessage
    from skills.conversation.skill import _get_llm

    llm = _get_llm()
    summary_prompt = (
        f"{get_active_identity()} Summarise the following conversation between you and the user. "
        "Write in first person — 'I told the user...', 'the user asked me...'. "
        "Capture the key topics discussed and any conclusions or answers given. Be concise."
    )
    response = await llm.ainvoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content=history_text),
    ])
    summary = response.content.strip()
    logger.info(f"Summary:\n{summary}")


async def cmd_clear() -> None:
    _history.clear()
    logger.info("Conversation history cleared.")


async def cmd_history() -> None:
    text = _format_history()
    if not text:
        logger.info("No conversation history yet.")
    else:
        logger.info(f"Session history\n{text}")


async def cmd_planner(args: str) -> None:
    from skills.planner import run_planner

    task = args.strip()
    if not task:
        logger.info("Please provide a task. Usage: /planner <task>")
        return

    logger.info(f"Running agent on: {task}")
    response = await run_planner(task)
    logger.info(response)


async def cmd_analyze() -> None:
    from skills.personalizer.skill import analyze_persona_from_log, analyze_persona_from_memories, get_persona_context
    from skills.journal.compact import compact_conversations
    from skills.journal.journal_writer import write_journal_entry
    from db.schemas import Conversations, Journal

    all_docs = await Conversations.find_all().to_list()

    # Step 1: build user persona from all raw conversation exchanges
    all_exchanges = [ex for doc in all_docs if doc.exchanges for ex in doc.exchanges]
    if all_exchanges:
        logger.info(f"Building user persona from {len(all_exchanges)} exchange(s) across {len(all_docs)} day(s)…")
        await analyze_persona_from_log(all_exchanges)
    else:
        logger.info("No conversation exchanges found.")

    # Step 2: build user persona from mem0 memories (fills remaining gaps)
    logger.info("Analysing mem0 memories for persona gaps…")
    mem_updates = await analyze_persona_from_memories()
    if mem_updates:
        logger.info(f"Memory pass filled {mem_updates} additional field(s).")
    else:
        logger.info("Memory pass found nothing new.")

    # Step 3: compact any uncompacted days
    uncompacted = [d for d in all_docs if d.exchanges and d.compact_status != "complete"]
    if uncompacted:
        logger.info(f"Compacting {len(uncompacted)} day(s)…")
        for doc in uncompacted:
            try:
                await compact_conversations(doc.date)
                logger.info(f"  ✓ {doc.date} ({len(doc.exchanges)} exchange(s))")
            except Exception as e:
                logger.warning(f"  ✗ {doc.date}: {e}")
    else:
        logger.info("All days already compacted.")

    # Step 4: write journal entries for compacted days that don't have one yet
    all_docs = await Conversations.find_all().to_list()
    existing_journal_dates = {e.datestamp for e in await Journal.find_all().to_list()}
    needs_journal = [d for d in all_docs if d.compacted and d.date not in existing_journal_dates]
    if needs_journal:
        logger.info(f"Writing {len(needs_journal)} journal entry/entries…")
        for doc in needs_journal:
            try:
                await write_journal_entry(doc.date)
                logger.info(f"  ✓ Journal written for {doc.date}")
            except Exception as e:
                logger.warning(f"  ✗ Journal failed for {doc.date}: {e}")
    else:
        logger.info("All compacted days already have journal entries.")

    context = get_persona_context()
    if context:
        logger.info(f"Done.\n{context}")
    else:
        logger.info("Done. No personal facts found.")


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
        content: str = Field(description="Markdown content summarising what should be remembered")
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
    """List or search mem0 memories.

    Usage:
      /memories           — list all stored memories
      /memories <query>   — semantic search for relevant memories
    """
    try:
        from utils.memory_client import _get_memory, search_memories

        query = args.strip()
        if query:
            results = search_memories(query, limit=10)
            if not results:
                logger.info(f"No memories found for '{query}'.")
                return
            lines = "\n".join(f"  {i}. {text}" for i, text in enumerate(results, 1))
            logger.info(f"Top memories for '{query}':\n{lines}")
        else:
            mem = _get_memory()
            raw = mem.get_all(filters={"user_id": "default"}, top_k=1000)
            results = raw.get("results", raw) if isinstance(raw, dict) else raw
            if not results:
                logger.info("No memories stored yet.")
                return
            lines = "\n".join(
                f"  {i}. {r.get('memory') or r.get('text') if isinstance(r, dict) else str(r)}"
                for i, r in enumerate(results, 1)
            )
            logger.info(f"{len(results)} stored memory/memories:\n{lines}")
    except Exception as e:
        logger.error(f"memories failed: {e}")


async def cmd_pluck_memory(args: str) -> None:
    """Delete a single memory by its list number.

    Usage:
      /pluck-memory 13   — delete memory #13 from /memories output
    """
    try:
        from utils.memory_client import _get_memory

        arg = args.strip()
        if not arg.isdigit():
            logger.info("Usage: /pluck-memory <number>")
            return

        index = int(arg)
        if index < 1:
            logger.info("Number must be 1 or greater.")
            return

        mem = _get_memory()
        raw = mem.get_all(filters={"user_id": "default"}, top_k=1000)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw

        if not results:
            logger.info("No memories stored.")
            return

        if index > len(results):
            logger.info(f"Only {len(results)} memories exist.")
            return

        entry = results[index - 1]
        memory_id = entry.get("id") if isinstance(entry, dict) else None
        text = entry.get("memory") or entry.get("text") if isinstance(entry, dict) else str(entry)

        if not memory_id:
            logger.error(f"Could not find ID for memory #{index}.")
            return

        mem.delete(memory_id)
        logger.info(f"Deleted #{index}: {text}")
    except Exception as e:
        logger.error(f"pluck-memory failed: {e}")


async def cmd_compact_memory() -> None:
    """Deduplicate and merge mem0 memories using the LLM, then replace the store."""
    from rich.console import Console

    console = Console()
    try:
        from utils.memory_client import compact_memories

        with console.status("[bold cyan]Compacting memories…[/bold cyan]", spinner="dots"):
            before, after = compact_memories()

        if before == 0:
            logger.info("No memories stored yet.")
        else:
            removed = before - after
            logger.info(f"Done. {before} → {after} memories ({removed} removed).")
    except Exception as e:
        logger.error(f"compact-memory failed: {e}")


async def cmd_flush_memory() -> None:
    """Delete all mem0 memories by dropping the Qdrant collection (mem0 recreates it on next use)."""
    import os

    try:
        from qdrant_client import QdrantClient

        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))
        client = QdrantClient(host=host, port=port)

        existing = {c.name for c in client.get_collections().collections}
        count = 0
        if "langbox_memories" in existing:
            info = client.get_collection("langbox_memories")
            count = info.points_count or 0
            client.delete_collection("langbox_memories")

        import utils.memory_client as _mc
        _mc._memory = None

        if count:
            logger.info(f"{count} memories deleted.")
        else:
            logger.info("No memories to delete.")
    except Exception as e:
        logger.error(f"flush-memory failed: {e}")


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

    models = [
        m["id"] for m in data.get("data", [])
        if "embed" not in m["id"].lower()
    ]

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
            logger.info(f"{len(models)} model(s) loaded. Tab to autocomplete, Enter to confirm, Ctrl-C to cancel.")
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
            logger.info("Spotify is already connected. Reconnecting will overwrite the stored tokens.")
        logger.info("Opening Spotify auth in browser — you can keep chatting while you complete the flow.")

        async def _spotify():
            result = await auth.connect()
            logger.info(result)

        asyncio.create_task(_spotify())

    elif service == "hue":
        from utils.auth.hue import HueAuthProvider
        auth = HueAuthProvider()
        already = await auth.is_connected()
        if already:
            logger.info("Hue is already paired. Reconnecting will overwrite the stored credentials.")
        logger.info("Press the link button on your Hue bridge now — connecting in the background.")

        async def _hue():
            result = await auth.connect()
            logger.info(result)

        asyncio.create_task(_hue())

    elif not service:
        logger.info("Usage: /connect <service>  (spotify | hue)")
    else:
        logger.info(f"Unknown service '{service}'. Available: spotify, hue")


async def cmd_persona(args: str) -> None:
    """Set the active persona and optionally the voice.

    Usage:
      /persona              — interactive fuzzy picker
      /persona <id>         — switch persona directly
      /persona <id> <voice> — switch persona and voice in one command
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

    from agents.persona import PERSONAS, set_active_persona, get_active_persona_id, get_active_voice_id
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
    current_voice = get_active_voice_id() or "azelma"
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
                await picker.prompt_async(f"  voice ❯ [{persona_default_voice}] ")
            ).strip().lower()
            selected_voice = raw_voice if raw_voice in voice_ids else persona_default_voice
            if raw_voice and raw_voice not in voice_ids:
                logger.info(f"Unknown voice '{raw_voice}', using '{persona_default_voice}'.")
        except (KeyboardInterrupt, EOFError):
            logger.info("Cancelled.")
            return

    # --- Apply + persist ---
    set_active_persona(selected_persona)
    set_active_voice_id(selected_voice)
    await update_preferences(agentPersona=selected_persona, agentVoiceId=selected_voice)
    logger.info(
        f"Persona → {PERSONAS[selected_persona]['name']}  |  Voice → {selected_voice}  (saved)"
    )


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
        "  /save              — summarise and save the current session to the database\n"
        "  /clear             — wipe the in-memory conversation history\n"
        "  /history           — print session history to the terminal\n"
        "  /analyze           — extract personal facts from this session into your persona profile\n"
        "  /persona [id] [v]  — switch persona and voice (fuzzy picker, or pass args directly)\n"
        "  /model [name]      — switch the active model (fuzzy picker, or pass name directly)\n"
        "  /connect <service> — authenticate a service (spotify | hue)\n"
        "  /memories [q]      — list all mem0 memories, or search with a query\n"
        "  /compact-memory    — deduplicate and merge memories using the LLM\n"
        "  /pluck-memory <n>  — delete memory #n from the /memories list\n"
        "  /flush-memory      — delete all mem0 memories\n"
        "  /ctx               — show context window usage for the current session\n"
        "  /note [title]      — save a note from the current conversation context\n"
        "  /planner <task>    — run an autonomous multi-step planning agent\n"
        "  /look <path> [q]   — describe an image via the VLM and inject into context\n"
        "  /help              — show this message"
    )


_COMMANDS = {
    "/save": cmd_save,
    "/clear": cmd_clear,
    "/history": cmd_history,
    "/analyze": cmd_analyze,
    "/persona": cmd_persona,
    "/model": cmd_model,
    "/connect": cmd_connect,
    "/memories": cmd_memories,
    "/compact-memory": cmd_compact_memory,
    "/pluck-memory": cmd_pluck_memory,
    "/flush-memory": cmd_flush_memory,
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
