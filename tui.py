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
import colorsys
from datetime import datetime
import json
import math
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import psutil
from dotenv import load_dotenv
from rich.color import Color
from rich.markup import escape
from rich.style import Style
from rich.text import Text
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

from utils.monitor_helpers import (  # noqa: E402
    ALL_LEVELS,
    LEVEL_STYLE,
    performance_text as _performance_text,
    resource_markup as _resource_markup,
)

# Subway pipeline node definitions (id, display label)
PIPELINE_NODES: list[tuple[str, str]] = [
    ("input",    "INPUT"),
    ("routing",  "ROUTING"),
    ("skill",    "SKILL"),
    ("respond",  "RESPOND"),
]


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


def _fmt_dur(secs: float) -> str:
    if secs < 10:
        return f"{secs:.1f}s"
    return f"{secs:.0f}s"


def _subway_markup(
    current: str,
    label: str,
    completed: set[str],
    error: bool,
    node_times: dict[str, float] | None = None,
    total_time: float = 0.0,
) -> str:
    node_times = node_times or {}
    NODE_W = 18  # chars reserved for "icon label suffix"

    lines: list[str] = [
        "[bold cyan]Pipeline[/bold cyan]",
        "[dim]──────────────────────────[/dim]",
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

        node_text = f"{icon} {display}{suffix}"
        pad = max(1, NODE_W - len(node_text))

        if node_id in node_times:
            time_part = f"[dim]{' ' * pad}{_fmt_dur(node_times[node_id])}[/dim]"
        elif node_id == current and current != "idle":
            time_part = f"[dim]{' ' * pad}…[/dim]"
        else:
            time_part = ""

        lines.append(f"[{style}]{node_text}[/{style}]{time_part}")
        if i < len(PIPELINE_NODES) - 1:
            connector_style = "dim green" if node_id in completed else "dim"
            lines.append(f"[{connector_style}]│[/{connector_style}]")

    if total_time > 0:
        lines.append("[dim]──────────────────────────[/dim]")
        lines.append(f"[dim]Total{' ' * (NODE_W - 5)}[/dim][bold]{_fmt_dur(total_time)}[/bold]")

    return "\n".join(lines)


# ── log helpers (from monitor.py) ─────────────────────────────────────────────


# ── splash screen ─────────────────────────────────────────────────────────────

_SPLASH_ART = r"""

                                          _ ___   _   _  _      ___ _    _   _ _ _ ___
                                         | | __| /_\ | \| |    / __| |  /_\ | | | |   \
                                       _ | | _| / _ \| .` |   | (__| |_/ _ \| |_| | |) |
                                       \__/|___/_/ \_|_|\_|    \___|____/_/ \_|\___/|___/

                                    #++***                                                 %*#***
                                =-:=@@@@@                                                 @%@@@@%-:=+
                              =:.=@@@@                                                         @@@%:.:=
                            +=::-*@@     %*#%                                           @##%     @%*:..=#
                           *=-::=#@ *===*@@@                                             @@%=-=*  %*-:.==*
                          +=::==#%*=:.:-*%                                                @@#-..:=#%*=-.-=+
                        **+=:::-. =.::-**                                                   *-::..-.:=::-+**
                        *#*=---.   :-=*                                                      *=--:.  .===****
                       *#**+=-....::-+                  %=%#           @*%%                   *=:::...:=**##*
                       *%#**+:::::-=#                  %@@@*%         *@@@%                    **=-::::-#*#%*%
                       *#%%*==--==**               %#*++#@@#%%+==+++*%*%@@++*%#                  *+=---=-%%%*%
                       ##%%#*=+*###           *+*****==%@%**%@@%#*%%@%%=#@@*+****#*#              %#*+=+*%%%%*
                       #%%#*#=:-=%@#+      *++=%%%%@@@@%***+***++**+%===+**#%@@@%@@***#         *%%*=-:#@%#%%%
                        %@%%*===*%*-+****   %@@@%%%#*+=--++*%*=*%%%**%*=-==++*#%%%@@@@#    ***++=*#%=--=+#%%%
                         %%%*+#%*+=::==+**#%%%%%%%%#*+=:::=*+  :**=  =*+:::=+**#%%%%%%@ %#**===::+*=*%#**%%%
                            %%#*=::::..:=**%%@@#**#%#+==-=:*=. :*%*..-*====+*#%%##%%@@%%#*+=:...::::-*%%%
                             @%%**=-::::.:=#+*@@@%%#*****-:*+:-:*%=-=+==--*****#%@@@@#*#*=::.:::-==**%%
                                %%**+==--*:::+***%@##%*====*+-:-*%+--*+*==+***#%@%*#++:..:*:-==++*#%%
                                    #%#****=#*=.:.:.*:::*@%%#-=%%%%*-*%%%%-.:*-....:-+%*=%****%%
                                         %%%*+*==----.::+:**++..:::.:*=%=+=.::=:---=*=*%%%%
                                           @%%%%###+=*#*::+...::::-:::.--:*%*==****#%%@@
                                             @#%%%@%@%*#%*:......:.....:=##**%#%%%%%%
                                              @@%%%@#=:::*=-:::-::-::---+=:::=%@%%%%
                                      *#%%%%%%%%@@@%%**==*::::::-=:::::-=*==*%%%@@@%@%%%%%%*#
                                 #*%%%#***++==+===*:=-:****===-=+*======#+@::=::*+=+=-+****##%%***
                         #*##**#%***=:..:=====:=:-*-:*:-*#%====*#%#*-===%%=-=+-=*:-::=++==:..:=***%*******
                 ***#**#***=+*+=*#=+*******+==::=:=#*%-+@*%*+******+***#*%@==%##=-=::==+*****++==#*++**=***#*#*##*#
      *%%####**#*+**+++=--+**%##*=*%*****+--==+#%#%##%%%%%%@%##%%%%###%@%%%@%##%%#%*+=--=+****%#*=**##*+=---====*#*+%%#*******
   %%*##=+=====+%*=*+***%%%%%%#**%%*%%%%%%%%%%@@                   @@%%%%%%%@%#%@%+*#%%%%%%###***==%*+====**+%**%%
%%#%@@@%@##%%%%%%%%%%%%%@@        @@@%%@                                  %@@%@%@       @@@@%%%%%#@@%%%%%%*#@%@@@%%%%
@@@@@@@@@@@@@@@@    @@@                                                                        @@@   @@@@@@@@@@@ @@@@@@

"""

_SPLASH_DURATION = 2.5
_SPLASH_FPS = 20


def _hue_rgb(hue: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _gradient_art(art: str, start_rgb: tuple[int, int, int], end_rgb: tuple[int, int, int]) -> Text:
    lines = [line.rstrip() for line in textwrap.dedent(art).split("\n")]
    max_width = max((len(line) for line in lines), default=1)
    rich_text = Text(no_wrap=True)
    for line in lines:
        for i, char in enumerate(line):
            t = i / max_width
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
            rich_text.append(char, style=Style(color=Color.from_rgb(r, g, b)))
        rich_text.append("\n")
    return rich_text


class SplashPanel(Static):
    """Animated ASCII splash rendered as a full-screen overlay.

    Uses ``overlay: screen`` so it sits above all other widgets without
    affecting the layout underneath.  After _SPLASH_DURATION seconds (or on
    any key press) it removes itself and refocuses the chat input.
    """

    def on_mount(self) -> None:
        self._start = time.monotonic()
        self.set_interval(1 / _SPLASH_FPS, self._tick)

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._start
        if elapsed >= _SPLASH_DURATION:
            self._dismiss()
            return
        t = elapsed / _SPLASH_DURATION
        start_hue = 0.13 + 0.5 * math.sin(t * math.pi * 2)
        self.update(_gradient_art(_SPLASH_ART, _hue_rgb(start_hue), _hue_rgb(start_hue + 0.15)))

    def on_key(self) -> None:
        if time.monotonic() - self._start > 0.3:  # avoid accidental skip
            self._dismiss()

    def _dismiss(self) -> None:
        self.remove()
        try:
            self.app.query_one("#chat-input", Input).focus()
        except Exception:
            pass


# ── TUI app ────────────────────────────────────────────────────────────────────


class TuiApp(App):
    TITLE = "jeanclawd"
    SUB_TITLE = "personal AI"

    CSS = """
    /* ── splash ── */
    SplashPanel {
        overlay: screen;
        background: $background;
        align: center middle;
        content-align: center middle;
        padding: 2 4;
    }

    /* ── shared ── */
    Header { height: 1; }

    /* extra breathing room on both tab bars */
    TabbedContent > Tabs {
        padding-top: 1;
    }
    TabbedContent > Tabs > Tab {
        padding: 0 2;
    }

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
    #chat-input:focus {
        border-top: solid $accent;
    }
    #right-panel {
        width: 36;
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
        Binding("y", "copy_log", "Copy log", show=False),
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

        # Suppress leading blank line before the very first message
        self._first_message = True

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
                            _subway_markup("idle", "", set(), False, {}, 0.0),
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
                                yield Button("COPY", id="btn-copy", variant="default")
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
        yield SplashPanel(id="splash")
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
        # Focus input immediately so the cursor blinks before startup completes
        self.call_after_refresh(lambda: self.query_one("#chat-input", Input).focus())
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
        self._write_jean(greeting)

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
                _subway_markup(s.current, s.label, s.completed, s.error,
                               s.node_times, s.total_time)
            )
        except Exception:
            pass

    # ── chat message helpers ──────────────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write_user(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        if not self._first_message:
            chat_log.write("")
        self._first_message = False
        chat_log.write(f"[dim]{self._ts()}[/dim]  [bold cyan]You[/bold cyan]  {escape(text)}")

    def _write_jean(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[dim]{self._ts()}[/dim]  [bold green]Jean[/bold green]  {text}")
        chat_log.scroll_end(animate=False)

    # ── streaming ─────────────────────────────────────────────────────────────

    def _flush_stream(self) -> None:
        if not self._stream_tokens:
            return
        text = "".join(self._stream_tokens)
        label = self.query_one("#streaming-label", Label)
        label.update(
            f"[dim]{self._ts()}[/dim]  [bold green]Jean[/bold green]  {text}[dim]▌[/dim]"
        )

    def _on_token(self, token: str) -> None:
        self._stream_tokens.append(token)

    def _finalize_stream(self, response: str) -> None:
        self._stream_tokens.clear()
        self._streaming = False
        self.query_one("#streaming-label", Label).update("")
        self._write_jean(response)

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

        self._write_user(user_text)
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

    # ── monitor copy ──────────────────────────────────────────────────────────

    def _entry_text(self, entry: dict, event_view: bool = False) -> str:
        if not event_view:
            return f"{entry.get('ts', '')} {entry.get('level', 'INFO'):<5} {entry.get('msg', '')}"
        fields = {k: v for k, v in entry.items() if k not in {"ts", "level", "msg", "event"}}
        return (
            f"{entry.get('ts', '')} {entry.get('level', 'INFO'):<5} "
            f"{entry.get('event', entry.get('msg', 'event'))} "
            f"{json.dumps(fields, ensure_ascii=False, default=str)}"
        ).rstrip()

    def action_copy_log(self) -> None:
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self.notify("Copied selection to clipboard")
            return
        try:
            tabs = self.query_one("#monitor-log-tabs", TabbedContent)
            is_events = tabs.active == "tab-events"
        except Exception:
            is_events = False
        entries = self._event_entries if is_events else self._log_entries
        text = "\n".join(
            self._entry_text(e, event_view=is_events)
            for e in entries
            if self._entry_matches_filter(e)
        )
        if not text:
            self.notify("No matching log entries to copy", severity="warning")
            return
        self.copy_to_clipboard(text)
        self.notify(f"Copied {len(text.splitlines())} filtered result(s) to clipboard")

    # ── monitor filter buttons ────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-copy":
            self.action_copy_log()
            return
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
