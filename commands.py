"""Built-in slash commands for the assistant CLI.

Available commands:
  /save           — summarise the current session
  /clear          — wipe the in-memory conversation history
  /history        — print the current session history to the terminal
  /analyze        — extract personal facts from this session into your persona profile
  /planner <task> — run an autonomous multi-step planning agent
  /help           — list available commands
"""

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
        print("[/save] Nothing to save — conversation history is empty.")
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
    print(f"[/save] Summary:\n{summary}")


async def cmd_clear() -> None:
    _history.clear()
    print("[/clear] Conversation history cleared.")


async def cmd_history() -> None:
    text = _format_history()
    if not text:
        print("[/history] No conversation history yet.")
    else:
        print(f"\n--- Session history ---\n{text}\n-----------------------")


async def cmd_planner(args: str) -> None:
    from skills.planner import run_planner

    task = args.strip()
    if not task:
        print("[/planner] Please provide a task. Usage: /planner <task>")
        return

    print(f"[/planner] Running agent on: {task}\n")
    response = await run_planner(task)
    print(response)


async def cmd_analyze() -> None:
    from skills.personalizer.skill import analyze_persona_from_log, get_persona_context
    from db.schemas import Conversations
    from datetime import date

    today = date.today()
    doc = await Conversations.find_one(Conversations.date == today)
    if not doc or not doc.exchanges:
        print("[/analyze] No conversations from today to analyse.")
        return

    await analyze_persona_from_log(doc.exchanges)
    context = get_persona_context()
    if context:
        print(f"[/analyze] Done ({len(doc.exchanges)} exchange(s) scanned).\n{context}")
    else:
        print(f"[/analyze] Done ({len(doc.exchanges)} exchange(s) scanned). No new personal facts found.")


async def cmd_ctx() -> None:
    import os
    from utils.llm_structured_output_llamacpp import _get_or_load_llama, _model_path

    model_name = os.environ.get("MODEL_GENERALIST")
    n_ctx = int(os.environ.get("MODEL_CTX", 8192))
    full_path = _model_path(model_name)
    llama = _get_or_load_llama(model_name, full_path, -1, {})

    history_text = _format_history()
    tokens = llama.tokenize(history_text.encode()) if history_text else []
    used = len(tokens)
    pct = used / n_ctx * 100

    bar_width = 40
    filled = int(bar_width * used / n_ctx)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\n[/ctx] Context usage: {used:,} / {n_ctx:,} tokens ({pct:.1f}%)")
    print(f"       [{bar}]\n")


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
        print("[/note] Nothing to note — no conversation history and no title provided.")
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
    print(f"[/note] {result}")


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
                print(f"[/memories] No memories found for '{query}'.")
                return
            print(f"\n[/memories] Top memories for '{query}':")
            for i, text in enumerate(results, 1):
                print(f"  {i}. {text}")
        else:
            mem = _get_memory()
            raw = mem.get_all(filters={"user_id": "default"}, top_k=1000)
            results = raw.get("results", raw) if isinstance(raw, dict) else raw
            if not results:
                print("[/memories] No memories stored yet.")
                return
            print(f"\n[/memories] {len(results)} stored memory/memories:")
            for i, r in enumerate(results, 1):
                text = r.get("memory") or r.get("text") if isinstance(r, dict) else str(r)
                print(f"  {i}. {text}")
        print()
    except Exception as e:
        print(f"[/memories] Failed: {e}")


async def cmd_pluck_memory(args: str) -> None:
    """Delete a single memory by its list number.

    Usage:
      /pluck-memory 13   — delete memory #13 from /memories output
    """
    try:
        from utils.memory_client import _get_memory

        arg = args.strip()
        if not arg.isdigit():
            print("[/pluck-memory] Usage: /pluck-memory <number>")
            return

        index = int(arg)
        if index < 1:
            print("[/pluck-memory] Number must be 1 or greater.")
            return

        mem = _get_memory()
        raw = mem.get_all(filters={"user_id": "default"}, top_k=1000)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw

        if not results:
            print("[/pluck-memory] No memories stored.")
            return

        if index > len(results):
            print(f"[/pluck-memory] Only {len(results)} memories exist.")
            return

        entry = results[index - 1]
        memory_id = entry.get("id") if isinstance(entry, dict) else None
        text = entry.get("memory") or entry.get("text") if isinstance(entry, dict) else str(entry)

        if not memory_id:
            print(f"[/pluck-memory] Could not find ID for memory #{index}.")
            return

        mem.delete(memory_id)
        print(f"[/pluck-memory] Deleted #{index}: {text}")
    except Exception as e:
        print(f"[/pluck-memory] Failed: {e}")


async def cmd_compact_memory() -> None:
    """Deduplicate and merge mem0 memories using the LLM, then replace the store."""
    from rich.console import Console
    from rich.spinner import Spinner

    console = Console()
    try:
        from utils.memory_client import compact_memories

        with console.status("[bold cyan]Compacting memories…[/bold cyan]", spinner="dots"):
            before, after = compact_memories()

        if before == 0:
            print("[/compact-memory] No memories stored yet.")
        else:
            removed = before - after
            print(f"[/compact-memory] Done. {before} → {after} memories ({removed} removed).")
    except Exception as e:
        print(f"[/compact-memory] Failed: {e}")


async def cmd_flush_memory() -> None:
    """Delete all mem0 memories with a progress indicator."""
    import os
    from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn

    try:
        from qdrant_client import QdrantClient

        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))
        client = QdrantClient(host=host, port=port)

        # Collect all point IDs first so we know the total
        all_ids = []
        offset = None
        while True:
            result, next_offset = client.scroll(
                "langbox_memories",
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            all_ids.extend(p.id for p in result)
            if next_offset is None:
                break
            offset = next_offset

        if not all_ids:
            print("[/flush-memory] Nothing to delete.")
            return

        with Progress(
            TextColumn("[[]/flush-memory]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total} memories cleared"),
            transient=True,
        ) as progress:
            task = progress.add_task("Deleting", total=len(all_ids))
            batch_size = 50
            for i in range(0, len(all_ids), batch_size):
                batch = all_ids[i : i + batch_size]
                client.delete(
                    "langbox_memories",
                    points_selector=batch,
                )
                progress.advance(task, len(batch))

        # Reset mem0 singleton so next call reconnects to the now-empty collection
        import utils.memory_client as _mc
        _mc._memory = None
        print(f"[/flush-memory] {len(all_ids)} memories deleted.")
    except Exception as e:
        print(f"[/flush-memory] Failed: {e}")


async def cmd_help() -> None:
    print(
        "\nAvailable commands:\n"
        "  /save              — summarise and save the current session to the database\n"
        "  /clear             — wipe the in-memory conversation history\n"
        "  /history           — print session history to the terminal\n"
        "  /analyze           — extract personal facts from this session into your persona profile\n"
        "  /memories [q]      — list all mem0 memories, or search with a query\n"
        "  /compact-memory    — deduplicate and merge memories using the LLM\n"
        "  /pluck-memory <n>  — delete memory #n from the /memories list\n"
        "  /flush-memory      — delete all mem0 memories\n"
        "  /ctx               — show context window usage for the current session\n"
        "  /note [title]      — save a note from the current conversation context\n"
        "  /planner <task>    — run an autonomous multi-step planning agent\n"
        "  /help              — show this message\n"
    )


_COMMANDS = {
    "/save": cmd_save,
    "/clear": cmd_clear,
    "/history": cmd_history,
    "/analyze": cmd_analyze,
    "/memories": cmd_memories,
    "/compact-memory": cmd_compact_memory,
    "/pluck-memory": cmd_pluck_memory,
    "/flush-memory": cmd_flush_memory,
    "/ctx": cmd_ctx,
    "/note": cmd_note,
    "/planner": cmd_planner,
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
        print(f"Unknown command '{cmd}'. Type /help for available commands.")
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
