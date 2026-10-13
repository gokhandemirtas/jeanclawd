"""Advanced terminal UI for jeanclawd.

Replaces the plain Rich REPL in main.py with a full Textual application:

  Tab 1  Dashboard — scrolling chat (left) + models / live resources /
                     pipeline subway graph (right)
  Tab 2  Monitor   — log tail, event stream, SearXNG response viewer

Usage:
    uv run tui.py [--session <uuid>] [--speak] [--listen] [--server]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
from dotenv import load_dotenv
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker

load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent / "logs" / "app.log"
from utils.resource_monitor import CONTEXT_FILE  # noqa: E402

LEVEL_STYLE = {
    "DEBUG": "dim white",
    "INFO": "bold cyan",
    "WARN": "bold yellow",
    "ERROR": "bold red",
}
ALL_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")

# Subway pipeline node definitions (id, display label)
PIPELINE_NODES: list[tuple[str, str]] = [
    ("input",    "INPUT"),
    ("routing",  "ROUTING"),
    ("skill",    "SKILL"),
    ("respond",  "RESPOND"),
]

# ── resource helpers (shared with monitor.py) ──────────────────────────────────


def _rbar(used: float, total: float, width: int = 16) -> str:
    pct = used / total if total else 0
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


def _fmt_mb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def _fmt_metric(value, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _resource_markup() -> str:
    lines: list[str] = []

    # CPU
    pct = psutil.cpu_percent(interval=None)
    bar = _rbar(pct, 100)
    lines.append("[bold cyan]CPU[/bold cyan]")
    lines.append(f"[dim]{bar}[/dim] [bold]{pct:.1f}%[/bold]")
    lines.append("")

    # RAM
    vm = psutil.virtual_memory()
    used_mb = vm.used / 1024 ** 2
    total_mb = vm.total / 1024 ** 2
    bar = _rbar(used_mb, total_mb)
    lines.append("[bold cyan]RAM[/bold cyan]")
    lines.append(f"[dim]{bar}[/dim]")
    lines.append(f"[bold]{_fmt_mb(used_mb)}[/bold] / {_fmt_mb(total_mb)}")
    lines.append("")

    # Swap
    try:
        sw = psutil.swap_memory()
        if sw.total > 0:
            sw_used = sw.used / 1024 ** 2
            sw_total = sw.total / 1024 ** 2
            bar = _rbar(sw_used, sw_total)
            lines.append("[bold cyan]Swap[/bold cyan]")
            lines.append(f"[dim]{bar}[/dim]")
            lines.append(f"[bold]{_fmt_mb(sw_used)}[/bold] / {_fmt_mb(sw_total)}")
            lines.append("")
    except OSError:
        pass

    # Disk
    try:
        du = psutil.disk_usage("/")
        free_gb = du.free / 1024 ** 3
        total_gb = du.total / 1024 ** 3
        used_gb = du.used / 1024 ** 3
        bar = _rbar(du.used, du.total)
        lines.append("[bold cyan]Disk[/bold cyan]")
        lines.append(f"[dim]{bar}[/dim]")
        lines.append(f"[bold]{free_gb:.1f} GB[/bold] free / {total_gb:.0f} GB")
        lines.append("")
    except Exception:
        pass

    # NVIDIA VRAM
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
        )
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        name = nvmlDeviceGetName(h)
        vram_used = info.used / 1024 ** 2
        vram_total = info.total / 1024 ** 2
        bar = _rbar(vram_used, vram_total)
        lines.append(f"[bold cyan]VRAM[/bold cyan] [dim]{name}[/dim]")
        lines.append(f"[dim]{bar}[/dim]")
        lines.append(f"[bold]{_fmt_mb(vram_used)}[/bold] / {_fmt_mb(vram_total)}")
        lines.append("")
    except Exception:
        pass

    # LLM context window
    used_ctx, max_ctx = 0, 0
    has_ctx = False
    try:
        if CONTEXT_FILE.exists():
            ctx = json.loads(CONTEXT_FILE.read_text())
            used_ctx = int(ctx.get("used", 0))
            max_ctx = int(ctx.get("max", 0)) or 0
            has_ctx = True
    except Exception:
        pass

    lines.append("[bold cyan]Context[/bold cyan] [dim]window[/dim]")
    if not has_ctx or max_ctx <= 0:
        bar = _rbar(0, 1)
        lines.append(f"[dim]{bar}[/dim]")
        lines.append("[dim]— waiting for first request[/dim]")
    else:
        bar = _rbar(used_ctx, max_ctx)
        pct_ctx = used_ctx / max_ctx * 100
        lines.append(f"[dim]{bar}[/dim]")
        lines.append(f"[bold]{used_ctx:,}[/bold] / {max_ctx:,} ({pct_ctx:.0f}%)")

    return "\n".join(lines)


def _models_markup() -> str:
    """One-shot markup for the models panel (env-based, static)."""
    rows = [
        ("Generalist", os.getenv("MODEL_GENERALIST", "—")),
        ("Intent",     os.getenv("MODEL_INTENT", "—")),
        ("Research",   os.getenv("MODEL_RESEARCH", "—")),
        ("Extraction", os.getenv("MODEL_EXTRACTION", "—")),
        ("Embedding",  os.getenv("MODEL_EMBEDDING", "—")),
        ("Host",       os.getenv("MODEL_HOST", "—")),
    ]
    lines: list[str] = ["[bold cyan]Models[/bold cyan]", "[dim]─────────────────────[/dim]"]
    for label, val in rows:
        # Truncate long model names for the narrow panel
        val_display = val if len(val) <= 22 else val[:19] + "…"
        lines.append(f"[dim]{label:<11}[/dim] [cyan]{val_display}[/cyan]")
    return "\n".join(lines)


def _subway_markup(current: str, label: str, completed: set[str], error: bool) -> str:
    lines: list[str] = [
        "[bold cyan]Pipeline[/bold cyan]",
        "[dim]─────────────────────[/dim]",
    ]
    for i, (node_id, default_label) in enumerate(PIPELINE_NODES):
        display = label.upper() if node_id == "skill" and label else default_label

        if node_id == current and error:
            icon, style, suffix = "◎", "bold red", " ✗"
        elif node_id == current and current != "idle":
            icon, style, suffix = "◎", "bold yellow", " ◄"
        elif node_id in completed:
            icon, style, suffix = "●", "dim green", ""
        else:
            icon, style, suffix = "○", "dim", ""

        lines.append(f"[{style}]{icon} {display}{suffix}[/{style}]")
        if i < len(PIPELINE_NODES) - 1:
            connector_style = "dim green" if node_id in completed else "dim"
            lines.append(f"[{connector_style}]│[/{connector_style}]")

    return "\n".join(lines)


# ── log helpers (from monitor.py) ─────────────────────────────────────────────


def _performance_text(entry: dict) -> str:
    model = escape(str(entry.get("model", "unknown")))
    prompt = _fmt_metric(entry.get("prompt_tokens"))
    completion = _fmt_metric(entry.get("completion_tokens"))
    server_tps = _fmt_metric(entry.get("server_tokens_per_second"), " tok/s")
    measured_tps = _fmt_metric(entry.get("measured_tokens_per_second"), " tok/s")
    server_ttft = _fmt_metric(entry.get("server_ttft_ms"), " ms")
    client_ttft = _fmt_metric(entry.get("client_ttft_ms"), " ms")
    return (
        f"[dim]{entry.get('ts', '')}[/dim] [bold yellow]LLM PERF[/bold yellow] "
        f"[bold]{model}[/bold]\n"
        f"  tokens: prompt={prompt}, completion={completion}  "
        f"throughput: server={server_tps}, measured={measured_tps}  "
        f"TTFT: server={server_ttft}, client={client_ttft}"
    )


# ── TUI app ────────────────────────────────────────────────────────────────────


class TuiApp(App):
    TITLE = "jeanclawd"
    SUB_TITLE = "personal AI"

    CSS = """
    /* ── shared ── */
    Header { height: 1; }

    /* ── dashboard tab ── */
    #dash-layout {
        height: 1fr;
    }
    #chat-panel {
        width: 3fr;
        height: 1fr;
    }
    #chat-log {
        height: 1fr;
        border: none;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    #streaming-label {
        height: auto;
        min-height: 1;
        padding: 0 1 1 1;
        border-top: solid $panel;
        background: $surface;
        color: $text;
    }
    #chat-input {
        height: 3;
        border-top: solid $primary;
        border-bottom: none;
        border-left: none;
        border-right: none;
    }
    #right-panel {
        width: 32;
        border-left: solid $panel;
        padding: 1 1;
        background: $surface;
        overflow-y: auto;
    }
    #models-panel {
        height: auto;
        padding-bottom: 1;
    }
    #resource-panel {
        height: auto;
        padding-bottom: 1;
    }
    #subway-panel {
        height: auto;
    }

    /* ── monitor tab ── */
    #monitor-layout {
        height: 1fr;
    }
    #log-pane {
        width: 1fr;
        height: 1fr;
    }
    #monitor-filter-bar {
        height: 2;
        background: $surface;
        padding: 0 1;
    }
    #monitor-filter-bar Button {
        margin: 0 1;
        min-width: 9;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }
    #monitor-filter-bar Button.-active {
        background: $accent;
        color: $text;
    }
    #monitor-stats {
        height: 1;
        padding: 0 2;
        background: $panel;
        color: $text-muted;
    }
    #monitor-log-tabs {
        height: 1fr;
    }
    #monitor-log-view, #monitor-event-view, #monitor-sear-view {
        height: 1fr;
        border: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("a", "toggle_autoscroll", "Auto-scroll", show=False),
    ]

    def __init__(
        self,
        session_id: str | None = None,
        speak: bool = False,
        listen: bool = False,
        emote: bool = False,
        server: bool = False,
        telegram: bool = False,
    ) -> None:
        super().__init__()
        self._session_id = session_id
        self._speak = speak
        self._listen = listen
        self._emote = emote
        self._server = server
        self._telegram = telegram

        # Pipeline state
        self._pipeline_current = "idle"
        self._pipeline_label = ""
        self._pipeline_completed: set[str] = set()
        self._pipeline_error = False

        # Streaming
        self._stream_tokens: list[str] = []
        self._streaming = False

        # Log monitor state
        self._log_pos = 0
        self._log_filter = "ALL"
        self._log_autoscroll = True
        self._log_counts: dict[str, int] = {lvl: 0 for lvl in ALL_LEVELS}
        self._log_entries: list[dict] = []
        self._event_entries: list[dict] = []
        self._sear_mtime: int = -1

        # Whether the pipeline worker is busy
        self._busy = False

    # ── compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="main-tabs"):
            with TabPane("Dashboard", id="tab-dash"):
                with Horizontal(id="dash-layout"):
                    with Vertical(id="chat-panel"):
                        yield RichLog(
                            id="chat-log",
                            highlight=False,
                            markup=True,
                            wrap=True,
                        )
                        yield Label("", id="streaming-label", markup=True)
                        yield Input(placeholder="❯  message…", id="chat-input")
                    with Vertical(id="right-panel"):
                        yield Static(_models_markup(), id="models-panel", markup=True)
                        yield Static("", id="resource-panel", markup=True)
                        yield Static(
                            _subway_markup("idle", "", set(), False),
                            id="subway-panel",
                            markup=True,
                        )
            with TabPane("Monitor", id="tab-monitor"):
                with Vertical(id="log-pane"):
                    with TabbedContent(initial="tab-logs", id="monitor-log-tabs"):
                        with TabPane("Logs", id="tab-logs"):
                            with Horizontal(id="monitor-filter-bar"):
                                yield Button("ALL", id="btn-ALL", classes="-active")
                                for lvl in ALL_LEVELS:
                                    yield Button(lvl, id=f"btn-{lvl}")
                                yield Button("LLM PERF", id="btn-LLM_PERF")
                                yield Button("CLEAR", id="btn-clear", variant="warning")
                            yield Label("", id="monitor-stats")
                            yield RichLog(
                                id="monitor-log-view",
                                highlight=False,
                                markup=True,
                                wrap=True,
                            )
                        with TabPane("Events", id="tab-events"):
                            yield RichLog(
                                id="monitor-event-view",
                                highlight=False,
                                markup=True,
                                wrap=True,
                            )
                        with TabPane("SearXNG", id="tab-sear"):
                            yield RichLog(
                                id="monitor-sear-view",
                                highlight=False,
                                markup=False,
                                wrap=False,
                            )
        yield Footer()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        psutil.cpu_percent(interval=None)  # warm up first call
        if LOG_FILE.exists():
            self._log_pos = LOG_FILE.stat().st_size  # tail from now

        self.set_interval(1.0, self._refresh_resources)
        self.set_interval(0.5, self._poll_logs)
        self.set_interval(0.1, self._flush_stream)
        self.set_interval(0.2, self._refresh_subway)

        self._refresh_resources()
        self.run_worker(self._startup(), exclusive=False, name="startup")

    async def _startup(self) -> None:
        """Initialize DB, session, and preferences, then show greeting."""
        from pathlib import Path as _Path

        for d in ("logs", "memory", "storage", "introspections"):
            _Path(d).mkdir(parents=True, exist_ok=True)

        from utils import logger
        logger.init_file_logging("logs/app.log")

        from db.init import db_init
        from agents.persona import set_io_capabilities
        from utils.task_queue import worker

        asyncio.create_task(worker())
        set_io_capabilities(speak=self._speak, listen=self._listen)

        if self._emote:
            from skills.conversation.skill import enable_emote
            enable_emote()

        if self._speak:
            from tts.tts import enable_tts
            enable_tts()

        chat_log = self.query_one("#chat-log", RichLog)

        try:
            await db_init()
        except Exception as exc:
            chat_log.write(f"[bold red]Database error:[/bold red] {exc}")
            return

        if self._server:
            from api.server import start_api_server
            await start_api_server(host="0.0.0.0", port=8000)

        if self._telegram:
            from skills.telegram_skill import start_telegram_bot
            await start_telegram_bot()

        from db.preferences import apply_preferences
        await apply_preferences()

        from utils.session_service import initialize_session, resume_session
        try:
            if self._session_id:
                await resume_session(self._session_id)
            else:
                await initialize_session()
        except ValueError:
            chat_log.write(
                f"[bold red]Could not resume session:[/bold red] {self._session_id}"
            )
            return

        from skills.conversation.skill import generate_greeting
        greeting = await generate_greeting()
        chat_log.write(f"[bold green]Jean[/bold green]  {greeting}")
        chat_log.write("")

        self.query_one("#chat-input", Input).focus()

    # ── resource refresh ──────────────────────────────────────────────────────

    def _refresh_resources(self) -> None:
        try:
            self.query_one("#resource-panel", Static).update(_resource_markup())
        except Exception:
            pass

    # ── subway refresh ────────────────────────────────────────────────────────

    def _refresh_subway(self) -> None:
        from utils.agent_state import snapshot
        s = snapshot()
        try:
            self.query_one("#subway-panel", Static).update(
                _subway_markup(s.current, s.label, s.completed, s.error)
            )
        except Exception:
            pass

    # ── streaming ─────────────────────────────────────────────────────────────

    def _flush_stream(self) -> None:
        if not self._stream_tokens:
            return
        text = "".join(self._stream_tokens)
        label = self.query_one("#streaming-label", Label)
        label.update(f"[bold green]Jean[/bold green]  {text}[dim]▌[/dim]")

    def _on_token(self, token: str) -> None:
        self._stream_tokens.append(token)

    def _finalize_stream(self, response: str) -> None:
        self._stream_tokens.clear()
        self._streaming = False
        label = self.query_one("#streaming-label", Label)
        label.update("")
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[bold green]Jean[/bold green]  {response}")
        chat_log.write("")
        chat_log.scroll_end(animate=False)

    # ── chat input ────────────────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        event.input.clear()

        if self._busy:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write("[dim]Still processing previous request…[/dim]")
            return

        # Handle slash commands
        if user_text.startswith("/"):
            if user_text in ("/quit", "/exit"):
                await self._graceful_quit()
                return
            await self._run_command(user_text)
            return

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[bold cyan]You[/bold cyan]  {escape(user_text)}")
        self._stream_tokens.clear()
        self._streaming = True
        self._busy = True

        self.run_worker(
            self._process_message(user_text),
            exclusive=False,
            name="pipeline",
        )

    async def _run_command(self, cmd: str) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        try:
            from commands import handle_command
            await handle_command(cmd)
        except Exception as exc:
            chat_log.write(f"[bold red]Command error:[/bold red] {exc}")

    # ── pipeline worker ───────────────────────────────────────────────────────

    async def _process_message(self, user_input: str) -> None:
        from utils import agent_state as _as

        chat_log = self.query_one("#chat-log", RichLog)
        response: str = ""
        resolved_intent: str | None = None
        start = time.time()

        try:
            _as.reset()

            from skills.conversation.skill import get_recent_history, consume_pending_intent, resolve_query
            from agents.action_resolver.agent import resolve_action_query
            from utils.inference_gate import foreground_inference
            from utils.pending_plan import (
                clear_pending, is_cancellation, is_confirmation,
                set_pending, take_pending,
            )

            async with foreground_inference():
                _as.set_node("routing")
                effective_query = await resolve_query(user_input)
                effective_query = await resolve_action_query(effective_query)

                planned = take_pending()
                if planned is not None:
                    if is_cancellation(user_input):
                        response = "Cancelled the pending requests."
                        _as.set_node("respond")
                    elif is_confirmation(user_input):
                        _as.set_node("skill", "BATCH")
                        parts: list[str] = []
                        for index, task in enumerate(planned, 1):
                            try:
                                from agents.intent_router.intent_router import route
                                resolved_intent = task.intent
                                answer = await route(task.intent, task.query)
                                parts.append(f"{index}. {answer}")
                            except Exception as exc:
                                parts.append(f"{index}. Failed: {exc}")
                        response = "\n\n".join(parts)
                        _as.set_node("respond")
                    else:
                        clear_pending()
                        response = "I cancelled the pending requests because the confirmation response was unclear."
                        _as.set_node("respond")
                else:
                    from agents.decomposer import decompose_request, extract_batch_request
                    from agents.generalist.router import generalist_route
                    from agents.intent_router.intent_router import route

                    batch_query = extract_batch_request(effective_query)
                    explicit_sear = effective_query.strip().lower().startswith("sear:")
                    decomposition = (
                        await decompose_request(batch_query)
                        if batch_query and not explicit_sear
                        else None
                    )

                    if decomposition is not None and decomposition.is_compound:
                        set_pending(decomposition.tasks)
                        lines = [f"I found {len(decomposition.tasks)} requests:"]
                        lines.extend(
                            f"{i}. {task.description}"
                            for i, task in enumerate(decomposition.tasks, 1)
                        )
                        lines.append("\nProcess these in order? [y/n]")
                        response = "\n".join(lines)
                        _as.set_node("respond")
                    else:
                        from agents.intent_classifier.intent_classifier import _explicit_command_classify

                        shortcut = _explicit_command_classify(effective_query)
                        pending = consume_pending_intent()

                        if shortcut:
                            intent = shortcut
                        elif pending:
                            intent = pending
                        else:
                            _as.set_node("routing")
                            intent, confidence = await generalist_route(
                                effective_query,
                                get_recent_history(n=3, include_summary=True),
                            )
                            if confidence < 0.75 and intent != "CHAT":
                                intent = "CHAT"

                        resolved_intent = intent
                        _as.set_node("skill", intent)

                        response = await route(
                            intent, effective_query, on_token=self._on_token
                        )
                        _as.set_node("respond")

            _as.set_node("idle")

        except Exception as exc:
            from utils.agent_state import set_error
            set_error()
            response = f"Sorry, something went wrong: {exc}"
            chat_log.write(f"[bold red]Error:[/bold red] {escape(str(exc))}")

        finally:
            elapsed = time.time() - start
            self._finalize_stream(response)
            self._busy = False

            from utils.observability import emit
            from utils.langmem_service import record_interaction
            await record_interaction(user_input, response, intent=resolved_intent)

            from utils import logger
            logger.timing(elapsed)

    # ── actions ───────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        try:
            self.query_one("#chat-log", RichLog).clear()
            self.query_one("#streaming-label", Label).update("")
        except Exception:
            pass

    def action_toggle_autoscroll(self) -> None:
        self._log_autoscroll = not self._log_autoscroll
        self._update_monitor_stats()

    async def _graceful_quit(self) -> None:
        from utils.langmem_service import close_memory_store
        from utils.session_service import get_active_session_id
        session_id = get_active_session_id()
        await close_memory_store()
        if session_id:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(
                f"\n[dim]Resume with:[/dim] [cyan]uv run tui.py --session {session_id}[/cyan]"
            )
            await asyncio.sleep(0.5)
        self.exit()

    async def action_quit(self) -> None:
        await self._graceful_quit()

    # ── monitor log polling (adapted from monitor.py) ─────────────────────────

    def _poll_logs(self) -> None:
        self._refresh_sear_response()
        if not LOG_FILE.exists():
            return
        try:
            if LOG_FILE.stat().st_size < self._log_pos:
                self._log_pos = 0
            with open(LOG_FILE) as f:
                f.seek(self._log_pos)
                new_lines = f.readlines()
                self._log_pos = f.tell()
        except OSError:
            return

        if not new_lines:
            return

        try:
            log_view = self.query_one("#monitor-log-view", RichLog)
        except Exception:
            return

        for raw in new_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                entry = {"ts": "", "level": "INFO", "msg": raw}

            level = entry.get("level", "INFO")
            self._log_counts[level] = self._log_counts.get(level, 0) + 1
            self._log_entries.append(entry)

            if self._entry_matches_filter(entry):
                self._render_log_entry(log_view, entry)

            if entry.get("event"):
                self._event_entries.append(entry)
                self._render_event_entry(entry)

        self._update_monitor_stats()

    def _entry_matches_filter(self, entry: dict) -> bool:
        if self._log_filter == "ALL":
            return True
        if self._log_filter == "LLM_PERF":
            return entry.get("event") == "llm_generation_completed"
        return entry.get("level", "INFO") == self._log_filter

    def _render_log_entry(self, log_view: RichLog, entry: dict) -> None:
        if self._log_filter == "LLM_PERF" and entry.get("event") == "llm_generation_completed":
            log_view.write(_performance_text(entry))
            if self._log_autoscroll:
                log_view.scroll_end(animate=False)
            return
        ts = entry.get("ts", "")
        level = entry.get("level", "INFO")
        msg = entry.get("msg", "")
        style = LEVEL_STYLE.get(level, "")
        log_view.write(f"[dim]{ts}[/dim] [{style}]{level:<5}[/{style}] {escape(msg)}")
        if self._log_autoscroll:
            log_view.scroll_end(animate=False)

    def _render_event_entry(self, entry: dict) -> None:
        try:
            event_view = self.query_one("#monitor-event-view", RichLog)
        except Exception:
            return
        ts = entry.get("ts", "")
        level = entry.get("level", "INFO")
        name = entry.get("event", entry.get("msg", "event"))
        if name.startswith("skill_execution_") and entry.get("skill"):
            name = f"{name} [{entry['skill']}]"
        fields = {
            k: v for k, v in entry.items()
            if k not in {"ts", "level", "msg", "event"}
        }
        field_text = (
            escape(json.dumps(fields, ensure_ascii=False, default=str)) if fields else ""
        )
        style = LEVEL_STYLE.get(level, "")
        event_view.write(
            f"[dim]{ts}[/dim] [{style}]{level:<5}[/{style}] "
            f"[bold yellow]{escape(str(name))}[/bold yellow] {field_text}"
        )
        if self._log_autoscroll:
            event_view.scroll_end(animate=False)

    def _update_monitor_stats(self) -> None:
        try:
            parts = [
                f"[{LEVEL_STYLE[lvl]}]{lvl}: {self._log_counts[lvl]}[/{LEVEL_STYLE[lvl]}]"
                for lvl in ALL_LEVELS
            ]
            scroll_indicator = (
                "  [bold green]↓ auto[/bold green]"
                if self._log_autoscroll
                else "  [dim]scroll: off[/dim]"
            )
            self.query_one("#monitor-stats", Label).update(
                "  ".join(parts) + scroll_indicator
            )
        except Exception:
            pass

    def _refresh_sear_response(self) -> None:
        path = Path(os.environ.get("SEARXNG_RAW_RESPONSE_PATH", "logs/searxng-latest.json"))
        try:
            mtime = path.stat().st_mtime_ns
            if mtime == self._sear_mtime:
                return
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return
        self._sear_mtime = mtime
        try:
            sear_view = self.query_one("#monitor-sear-view", RichLog)
            sear_view.clear()
            sear_view.write(raw)
        except Exception:
            pass

    # ── monitor filter buttons ────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-clear":
            try:
                self.query_one("#monitor-log-view", RichLog).clear()
                self.query_one("#monitor-event-view", RichLog).clear()
            except Exception:
                pass
            self._log_counts = {lvl: 0 for lvl in ALL_LEVELS}
            self._log_entries.clear()
            self._event_entries.clear()
            self._update_monitor_stats()
            return

        level = btn_id.replace("btn-", "") if btn_id else None
        if level not in ("ALL", *ALL_LEVELS, "LLM_PERF"):
            return

        try:
            for btn in self.query("#monitor-filter-bar Button"):
                btn.remove_class("-active")
            event.button.add_class("-active")
        except Exception:
            pass

        self._log_filter = level
        try:
            log_view = self.query_one("#monitor-log-view", RichLog)
            log_view.clear()
            for entry in self._log_entries:
                if self._entry_matches_filter(entry):
                    self._render_log_entry(log_view, entry)
        except Exception:
            pass


# ── entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="jeanclawd TUI")
    parser.add_argument("--session", help="Resume an existing session by UUID")
    parser.add_argument("--speak", action="store_true", help="Enable TTS")
    parser.add_argument("--listen", action="store_true", help="Enable voice input")
    parser.add_argument("--emote", action="store_true", help="Enable emote mode")
    parser.add_argument("--server", action="store_true", help="Start API server")
    parser.add_argument("--telegram", action="store_true", help="Enable Telegram bot")
    args = parser.parse_args()

    TuiApp(
        session_id=args.session,
        speak=args.speak,
        listen=args.listen,
        emote=args.emote,
        server=args.server,
        telegram=args.telegram,
    ).run()


if __name__ == "__main__":
    main()
