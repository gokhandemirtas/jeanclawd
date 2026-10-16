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
from textual.suggester import Suggester
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option
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
TTS_PIPELINE_NODE = ("tts", "TTS")

_SPINNER_FRAMES = ("◐", "◓", "◑", "◒")


def _slash_command_matches(value: str) -> list[str]:
    """Return every completion that should be shown for a slash prefix."""
    if not value.startswith("/"):
        return []

    if value.startswith("/connect "):
        from connectors.registry import connector_ids

        prefix = value.removeprefix("/connect ").casefold()
        return [
            f"/connect {service_id}"
            for service_id in sorted(connector_ids())
            if service_id.casefold().startswith(prefix)
        ]

    if value.startswith("/persona "):
        try:
            from agents.persona import list_personas
            prefix = value.removeprefix("/persona ").casefold()
            return [
                f"/persona {p.id}"
                for p in list_personas()
                if p.id.casefold().startswith(prefix)
            ]
        except Exception:
            return []

    if value.startswith("/voice "):
        from voicebox.client import voice_ids
        prefix = value.removeprefix("/voice ").casefold()
        return [f"/voice {v}" for v in voice_ids if v.casefold().startswith(prefix)]

    if " " in value:
        return []

    from commands import _COMMANDS

    prefix = value.casefold()
    return [
        command
        for command in sorted((*_COMMANDS, "/quit", "/exit"))
        if command.casefold().startswith(prefix)
    ]


class SlashCommandSuggester(Suggester):
    """Offer the same slash-command completions as the regular CLI."""

    async def get_suggestion(self, value: str) -> str | None:
        matches = _slash_command_matches(value)
        return matches[0] if matches else None


class ChatInput(Input):
    """Chat input with slash-command completion on Tab or Right Arrow."""

    def on_key(self, event) -> None:
        if event.key in ("ctrl+c", "ctrl+q"):
            self.app.run_worker(self.app._graceful_quit(), exclusive=True, name="quit")
            event.stop()


def _sub_options(cmd: str) -> list[str] | None:
    """Return sub-option strings for commands that take a fixed argument, or None."""
    if cmd == "/news-length":
        return ["short", "medium", "long"]
    if cmd == "/change-model":
        return ["generalist", "intent", "vision", "embedding", "planning", "research"]
    if cmd == "/connect":
        try:
            from connectors.registry import connector_ids
            return sorted(connector_ids())
        except Exception:
            return []
    if cmd == "/voice":
        try:
            from voicebox.client import voice_ids
            return list(voice_ids)
        except Exception:
            return []
    if cmd == "/persona":
        try:
            from agents.persona import PERSONAS
            return sorted(PERSONAS.keys())
        except Exception:
            return []
    return None


class CommandMenu(OptionList):
    can_focus = False
    _parent_cmd: str | None = None  # set when showing sub-options


def _models_markup(value_width: int = 18) -> str:
    """Markup for the models panel sized to its available terminal column."""
    rows = [
        ("Generalist", os.getenv("MODEL_GENERALIST", "—")),
        ("Intent",     os.getenv("MODEL_INTENT", "—")),
        ("Research",   os.getenv("MODEL_RESEARCH", "—")),
        ("Extraction", os.getenv("MODEL_EXTRACTION", "—")),
        ("Embedding",  os.getenv("MODEL_EMBEDDING", "—")),
        ("Host",       os.getenv("MODEL_HOST", "—")),
    ]
    value_width = max(6, value_width)
    lines: list[str] = [
        "[bold #22d3ee]MODELS[/bold #22d3ee]  [dim]ACTIVE CLUSTER[/dim]",
        f"[dim]{'─' * (11 + value_width)}[/dim]",
    ]
    for label, val in rows:
        val_display = val if len(val) <= value_width else val[:value_width - 1] + "…"
        lines.append(
            f"[dim]{label:<11}[/dim][#22d3ee]{val_display:>{value_width}}[/#22d3ee]"
        )
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
    width: int = 26,
    include_tts: bool = False,
) -> str:
    node_times = node_times or {}
    node_width = max(12, width)
    nodes = [*PIPELINE_NODES, *([TTS_PIPELINE_NODE] if include_tts else [])]

    exec_indicator = f"  [bold #facc15]● EXEC[/bold #facc15]" if current != "idle" else ""
    lines: list[str] = [
        f"[bold #22d3ee]PIPELINE[/bold #22d3ee]{exec_indicator}",
        f"[dim]{'─' * node_width}[/dim]",
    ]
    for i, (node_id, default_label) in enumerate(nodes):
        display = label.upper() if node_id == "skill" and label else default_label

        if node_id == current and error:
            icon, style, conn = "◎", "bold #f87171", "#f87171"
            suffix = " ✗"
        elif node_id == current and current != "idle":
            icon = _SPINNER_FRAMES[int(time.monotonic() * 5) % len(_SPINNER_FRAMES)]
            style, conn, suffix = "bold #facc15", "#facc15", ""
        elif node_id in completed:
            icon, style, conn = "●", "#4ade80", "#4ade80"
            suffix = ""
        else:
            icon, style, conn = "○", "dim", "dim"
            suffix = ""

        node_text = f"{icon} {display}{suffix}"
        pad = max(1, node_width - len(node_text))

        if node_id in node_times:
            elapsed = _fmt_dur(node_times[node_id])
            time_part = f"[dim]{' ' * max(1, pad - len(elapsed))}{elapsed}[/dim]"
        elif node_id == current and current != "idle":
            time_part = f"[dim]{' ' * max(1, pad - 1)}…[/dim]"
        else:
            time_part = f"[dim]{' ' * max(1, pad - 1)}—[/dim]" if node_id not in completed else ""

        lines.append(f"[{style}]{node_text}[/{style}]{time_part}")
        if i < len(nodes) - 1:
            lines.append(f"[{conn}]│[/{conn}]")

    if total_time > 0:
        lines.append(f"[dim]{'─' * node_width}[/dim]")
        lines.append(f"[dim]PREV TOTAL{' ' * max(1, node_width - 10)}[/dim][bold]{_fmt_dur(total_time)}[/bold]")

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
        self._tick()  # paint first frame synchronously — no blank flash
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
    Screen {
        background: #0f111a;
        color: #f1f5f9;
    }

    /* ── splash ── */
    SplashPanel {
        overlay: screen;
        width: 100%;
        height: 100%;
        background: $background;
        align: center middle;
        content-align: center middle;
    }

    /* ── shared ── */
    Header {
        height: 1;
        background: #141622;
        color: #f1f5f9;
        border-bottom: solid #24283b;
    }

    Footer {
        height: 1;
        background: #141622;
        color: #94a3b8;
        border-top: solid #24283b;
    }

    TabbedContent > Tabs {
        padding-top: 1;
        background: #12141f;
        border-bottom: solid #24283b;
    }
    TabbedContent > Tabs ContentTab {
        padding: 0 3;
        color: #a78bfa;
        background: #1b1d2a;
    }

    /* ID-based selectors beat Textual 8's Tabs:focus .-active rule (0,2,1)
       which would otherwise override background with $block-cursor-background. */
    #main-tabs > Tabs {
        height: 3;
        padding: 0 2;
    }
    #main-tabs #tabs-list {
        height: 3;
        align: left middle;
    }
    #main-tabs ContentTab {
        min-width: 18;
        height: 1;
        padding: 0 5;
        color: #a78bfa;
        background: #1b1d2a;
    }
    #main-tabs ContentTab.-active {
        color: #ffffff;
        background: #7c3aed;
        text-style: bold;
    }

    /* ── dashboard ── */
    #dash-layout {
        height: 1fr;
    }

    /* chat panel */
    #chat-panel {
        width: 3fr;
        height: 1fr;
    }
    #chat-log {
        height: 1fr;
        border: none;
        padding: 0 2;
        scrollbar-gutter: stable;
        background: #0f111a;
    }
    #streaming-label {
        height: auto;
        min-height: 2;
        padding: 0 2 1 2;
        border-top: tall #23273a;
        background: #0a0c12;
        color: #cbd5e1;
    }
    #chat-input-bar {
        height: 6;
        background: #12141f;
        border-top: tall #24283b;
        padding: 1 1;
        align: left middle;
    }
    #chat-input-bar:focus-within { border-top: tall #7c3aed; }
    #command-menu {
        width: 1fr;
        height: 1fr;
        background: #0d0f1a;
        border: none;
        padding: 0;
        scrollbar-size: 1 1;
        display: none;
    }
    #command-menu.-visible {
        display: block;
    }
    #command-menu > .option-list--option {
        color: #22d3ee;
        padding: 0 2;
    }
    #command-menu > .option-list--option-highlighted {
        background: #7c3aed;
        color: white;
    }
    #research-log {
        display: none;
        height: auto;
        max-height: 12;
        padding: 1 0 0 0;
        border-top: solid #262b40;
        color: #94a3b8;
    }
    #research-log.-visible { display: block; }
    #chat-input {
        width: 1fr;
        height: 3;
        border: round #2b3046;
        background: #171a28;
        color: #f1f5f9;
        padding: 0 1;
    }
    #chat-input:focus {
        border: round #7c3aed;
    }
    #btn-send {
        height: 3;
        min-width: 9;
        margin: 0 0 0 1;
        background: #7c3aed;
        color: white;
        border: none;
        border-top: none;
        border-right: none;
        border-bottom: none;
        border-left: none;
    }
    #btn-send:hover {
        background: #6d28d9;
    }

    /* right info panel */
    #right-panel {
        width: 25%;
        height: 1fr;
        border-left: tall #24283b;
        background: #141624;
    }
    #right-info {
        width: 1fr;
        height: 4fr;
        padding: 1 2;
    }
    #models-panel, #resource-panel, #subway-panel {
        width: 100%;
    }
    #models-panel {
        height: 1fr;
        padding: 0 0 1 0;
        border-bottom: solid #262b40;
    }
    #resource-panel {
        height: 3fr;
        padding: 1 0;
        border-bottom: solid #262b40;
    }
    #subway-panel {
        height: auto;
        padding: 1 0 0 0;
    }

    /* slash-command terminal */
    #slash-terminal {
        width: 1fr;
        height: 1fr;
        border-top: tall #24283b;
        background: #0d0f1a;
    }
    #slash-log {
        height: auto;
        border: none;
        padding: 0 1;
        background: #0d0f1a;
        scrollbar-gutter: stable;
    }
    #slash-input {
        width: 1fr;
        height: 3;
        border: tall #1e2235;
        background: #10121d;
        color: #f1f5f9;
        padding: 0 1;
        margin: 0;
    }
    #slash-input:focus {
        border: tall #7c3aed;
    }

    /* ── monitor tab ── */
    #log-pane {
        width: 1fr;
        height: 1fr;
    }
    #monitor-filter-bar {
        height: 3;
        background: #141624;
        padding: 1 1 0 1;
        border-bottom: solid #24283b;
    }
    #monitor-filter-bar Button {
        margin: 0 1;
        min-width: 9;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        background: #1e2235;
        color: #94a3b8;
    }
    #monitor-filter-bar Button:hover {
        background: #282e47;
        color: #cbd5e1;
    }
    #monitor-filter-bar Button.-active {
        background: #7c3aed;
        color: white;
        text-style: bold;
    }
    #monitor-stats {
        height: 1;
        padding: 0 2;
        background: #10121d;
        color: #64748b;
        border-bottom: solid #202436;
    }
    #monitor-log-tabs {
        height: 1fr;
        background: #0b0d14;
    }
    #monitor-log-tabs > Tabs {
        padding-top: 1;
        padding-left: 2;
        background: #131522;
        border-bottom: solid #24283b;
    }
    #monitor-log-tabs > Tabs ContentTab {
        padding: 0 2;
        color: #94a3b8;
        background: #1e2233;
    }
    #monitor-log-tabs > Tabs ContentTab.-active {
        color: white;
        background: #7c3aed;
        border: none;
    }
    #monitor-log-view, #monitor-event-view, #monitor-sear-view {
        height: 1fr;
        border: none;
        padding: 0 1;
        background: #0b0d14;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "exit_tui", "Quit", priority=True),
        Binding("ctrl+q", "exit_tui", "Quit", priority=True),
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

        # Queued chat requests (max 3) — processed one after another
        self._request_queue: list[str] = []

        # Suppress leading blank line before the very first message
        self._first_message = True

    # ── compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        # Overlay widgets retain their compose position in a vertical layout.
        # Mount this first so `overlay: screen` starts at the screen origin.
        yield SplashPanel(id="splash")
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
                        with Horizontal(id="chat-input-bar"):
                            yield ChatInput(
                                placeholder="Send prompt or /help for commands…",
                                suggester=SlashCommandSuggester(case_sensitive=False),
                                id="chat-input",
                            )
                            yield Button("SEND", id="btn-send")
                    with Vertical(id="right-panel"):
                        with Vertical(id="right-info"):
                            yield Static(_models_markup(), id="models-panel", markup=True)
                            yield Static("", id="resource-panel", markup=True)
                            yield Static(
                                _subway_markup(
                                    "idle", "", set(), False, {}, 0.0,
                                    include_tts=self._speak,
                                ),
                                id="subway-panel",
                                markup=True,
                            )
                            yield Static("", id="research-log", markup=True)
                        with Vertical(id="slash-terminal"):
                            yield CommandMenu(id="command-menu")
                            yield RichLog(
                                id="slash-log",
                                highlight=False,
                                markup=True,
                                wrap=True,
                            )
                            yield Input(
                                placeholder="/help",
                                suggester=SlashCommandSuggester(case_sensitive=False),
                                id="slash-input",
                            )
            with TabPane("Monitor", id="tab-monitor"):
                with Vertical(id="log-pane"):
                    with TabbedContent(initial="tab-logs", id="monitor-log-tabs"):
                        with TabPane("Logs (Main)", id="tab-logs"):
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
                        with TabPane("Events (Dispatcher)", id="tab-events"):
                            yield RichLog(
                                id="monitor-event-view",
                                highlight=False,
                                markup=True,
                                wrap=True,
                            )
                        with TabPane("SearXNG Bridge", id="tab-sear"):
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
        self.set_interval(0.2, self._refresh_research_log)

        self._refresh_resources()
        # The right rail has no final width until the first layout pass.
        self.call_after_refresh(self._refresh_right_rail)
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
            from voicebox.client import enable_tts
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

        from utils.resource_monitor import push_context_to_file, set_context_used
        set_context_used(0)
        push_context_to_file()

        from skills.conversation.skill import generate_greeting
        greeting = await generate_greeting()
        self._write_jean(greeting)

        self.query_one("#chat-input", Input).focus()

    # ── resource refresh ──────────────────────────────────────────────────────

    def _refresh_resources(self) -> None:
        try:
            self.query_one("#resource-panel", Static).update(
                _resource_markup(
                    compact=True,
                    bar_width=self._right_panel_content_width(),
                )
            )
        except Exception:
            pass

    def _right_panel_content_width(self) -> int:
        """Return usable cells inside the right rail after its horizontal padding."""
        return max(12, self.query_one("#right-panel").size.width - 4)

    def _refresh_right_rail(self) -> None:
        self._refresh_models_panel()
        self._refresh_resources()
        self._refresh_subway()

    def _refresh_models_panel(self) -> None:
        """Use the right rail's full width without allowing model rows to wrap."""
        try:
            # Model labels consume eleven cells; the remaining cells hold the value.
            value_width = max(6, self._right_panel_content_width() - 11)
            self.query_one("#models-panel", Static).update(_models_markup(value_width))
        except Exception:
            pass

    def on_resize(self) -> None:
        self._refresh_right_rail()

    # ── subway refresh ────────────────────────────────────────────────────────

    def _refresh_subway(self) -> None:
        from utils.agent_state import snapshot
        s = snapshot()
        try:
            self.query_one("#subway-panel", Static).update(
                _subway_markup(s.current, s.label, s.completed, s.error,
                               s.node_times, s.total_time,
                               self._right_panel_content_width(),
                               include_tts=self._speak)
            )
        except Exception:
            pass

    def _refresh_research_log(self) -> None:
        try:
            from utils.agent_state import snapshot
            steps = snapshot().step_log
            widget = self.query_one("#research-log", Static)
            if not steps:
                widget.remove_class("-visible")
                return
            lines = ["[dim bold]RESEARCH[/dim bold]"]
            for msg in steps[-8:]:  # show last 8 steps to fit in the panel
                lines.append(f"[dim]○ {escape(msg)}[/dim]")
            widget.update("\n".join(lines))
            widget.add_class("-visible")
        except Exception:
            pass

    def _update_command_menu(self, value: str) -> None:
        try:
            ol = self.query_one("#command-menu", CommandMenu)
        except Exception:
            return
        if ol._parent_cmd is not None:
            return  # sub-menu is active; don't interfere
        if not value.startswith("/"):
            ol.clear_options()
            ol.remove_class("-visible")
            return
        from commands import _COMMANDS
        all_cmds = sorted((*_COMMANDS, "/quit", "/exit"))
        matches = [c for c in all_cmds if c.casefold().startswith(value.casefold())]
        ol.clear_options()
        for cmd in (matches if matches else all_cmds):
            ol.add_option(Option(cmd))
        ol.add_class("-visible")

    # ── chat message helpers ──────────────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write_user(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        if not self._first_message:
            chat_log.write("")
        self._first_message = False
        chat_log.write(f"[dim]{self._ts()}[/dim]  [bold #22d3ee]You:[/bold #22d3ee]  {escape(text)}")

    def _write_jean(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[dim]{self._ts()}[/dim]  [bold #4ade80]Jean:[/bold #4ade80]  {text}")
        chat_log.scroll_end(animate=False)

    # ── streaming ─────────────────────────────────────────────────────────────

    def _flush_stream(self) -> None:
        if not self._stream_tokens:
            return
        text = "".join(self._stream_tokens)
        label = self.query_one("#streaming-label", Label)
        label.update(
            f"[dim]{self._ts()}[/dim]  [bold #4ade80]Jean:[/bold #4ade80]  {text}[dim]▌[/dim]"
        )

    def _on_token(self, token: str) -> None:
        self._stream_tokens.append(token)

    def _finalize_stream(self, response: str) -> None:
        self._stream_tokens.clear()
        self._streaming = False
        self.query_one("#streaming-label", Label).update("")
        self._write_jean(response)

    # ── chat input ────────────────────────────────────────────────────────────

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "command-menu":
            return
        ol = self.query_one("#command-menu", CommandMenu)
        selected = str(event.option.prompt)

        if ol._parent_cmd is not None:
            # Sub-option selected — execute the full command
            full_cmd = f"{ol._parent_cmd} {selected}"
            ol._parent_cmd = None
            ol.clear_options()
            ol.remove_class("-visible")
            self.run_worker(
                self._run_command(full_cmd, log_id="slash-log"),
                exclusive=False,
                name="slash-command",
            )
            return

        sub = _sub_options(selected)
        if sub:
            # Show sub-menu in place — don't touch slash_input to avoid re-triggering on_input_changed
            ol._parent_cmd = selected
            ol.clear_options()
            for opt in sub:
                ol.add_option(Option(opt))
            ol.add_class("-visible")
        else:
            # No sub-options — fill input ready to run
            ol.clear_options()
            ol.remove_class("-visible")
            slash_input = self.query_one("#slash-input", Input)
            slash_input.value = selected + " "
            slash_input.cursor_position = len(slash_input.value)
            slash_input.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "slash-input":
            self._update_command_menu(event.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        event.input.clear()

        # Slash terminal — always runs as a command
        if event.input.id == "slash-input":
            parts = user_text.split(None, 1)
            cmd = parts[0] if user_text.startswith("/") else f"/{parts[0]}"
            args = parts[1] if len(parts) > 1 else ""

            # If command has sub-options and no arg yet, show sub-menu instead of executing
            if not args:
                sub = _sub_options(cmd)
                if sub:
                    ol = self.query_one("#command-menu", CommandMenu)
                    ol._parent_cmd = cmd
                    ol.clear_options()
                    for opt in sub:
                        ol.add_option(Option(opt))
                    ol.add_class("-visible")
                    return

            self._update_command_menu("")
            if cmd in ("/quit", "/exit"):
                await self._graceful_quit()
                return
            self.run_worker(
                self._run_command(user_text if user_text.startswith("/") else f"/{user_text}", log_id="slash-log"),
                exclusive=False,
                name="slash-command",
            )
            return

        if self._busy:
            chat_log = self.query_one("#chat-log", RichLog)
            if len(self._request_queue) >= 3:
                chat_log.write("[dim]Queue full (3/3) — please wait before sending more.[/dim]")
            else:
                self._request_queue.append(user_text)
                n = len(self._request_queue)
                chat_log.write(f"[dim]Queued ({n}/3) — I'll get to it right after.[/dim]")
            return

        # Handle slash commands from main chat input
        if user_text.startswith("/"):
            if user_text in ("/quit", "/exit"):
                await self._graceful_quit()
                return
            # Run in a worker so commands that block (tiktoken, prompt_toolkit
            # pickers, network calls) don't freeze the Textual event loop.
            self.run_worker(
                self._run_command(user_text),
                exclusive=False,
                name="command",
            )
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

    async def _run_command(self, cmd: str, log_id: str = "chat-log") -> None:
        log = self.query_one(f"#{log_id}", RichLog)
        parts = cmd.strip().split(None, 1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        # Intercept commands that use prompt_toolkit interactive pickers —
        # those pickers corrupt Textual's terminal state and freeze the TUI.
        if command == "/persona":
            await self._cmd_persona_tui(args, log)
            return
        if command == "/voice" and not args:
            await self._cmd_show_voices(log)
            return
        if command in ("/change-model", "/connect") and not args:
            await self._cmd_show_picker_options(command, log)
            return

        try:
            from commands import handle_command
            from utils import logger

            with logger.capture_messages() as messages:
                await handle_command(cmd)
            if messages:
                for level, message in messages:
                    style = LEVEL_STYLE.get(level, "dim")
                    log.write(
                        f"[dim]{self._ts()}[/dim]  "
                        f"[bold #4ade80]Jean:[/bold #4ade80] "
                        f"[{style}]{escape(message)}[/{style}]"
                    )
                log.scroll_end(animate=False)
            else:
                log.write(f"[dim]{self._ts()}[/dim]  [bold #4ade80]Jean:[/bold #4ade80] [dim]Command completed.[/dim]")
        except Exception as exc:
            log.write(f"[bold red]Command error:[/bold red] {escape(str(exc))}")

    def _write_cmd_output(self, chat_log: RichLog, messages: list) -> None:
        for level, message in messages:
            style = LEVEL_STYLE.get(level, "dim")
            chat_log.write(
                f"[dim]{self._ts()}[/dim]  [bold #4ade80]Jean:[/bold #4ade80] "
                f"[{style}]{escape(message)}[/{style}]"
            )
        if messages:
            chat_log.scroll_end(animate=False)

    async def _cmd_persona_tui(self, args: str, chat_log: RichLog) -> None:
        try:
            from agents.persona import PERSONAS
        except Exception as exc:
            chat_log.write(f"[bold red]Error:[/bold red] {exc}")
            return

        persona_ids = list(PERSONAS.keys())
        arg_parts = args.strip().split()

        if not arg_parts:
            lines = ["[bold #22d3ee]Personas[/bold #22d3ee]  [dim]usage: /persona <id> [voice][/dim]"]
            for pid in persona_ids:
                info = PERSONAS[pid]
                default_voice = info.get("voice_id") or "—"
                lines.append(
                    f"  [bold #22d3ee]{pid}[/bold #22d3ee]"
                    f"  [dim]{escape(info.get('name', pid))}  voice={default_voice}[/dim]"
                )
            chat_log.write("\n".join(lines))
            chat_log.scroll_end(animate=False)
            return

        persona_id = arg_parts[0]
        if persona_id not in persona_ids:
            chat_log.write(
                f"[bold red]Unknown persona '{escape(persona_id)}'.[/bold red]"
                f"  Available: {', '.join(persona_ids)}"
            )
            return

        voice_arg = (
            arg_parts[1] if len(arg_parts) > 1
            else PERSONAS[persona_id].get("voice_id") or "thomas"
        )

        try:
            from commands import handle_command
            from utils import logger
            with logger.capture_messages() as messages:
                await handle_command(f"/persona {persona_id} {voice_arg}")
            self._write_cmd_output(chat_log, messages)
            if not messages:
                self._write_jean(f"Persona → {PERSONAS[persona_id].get('name', persona_id)}  |  Voice → {voice_arg}")
        except Exception as exc:
            chat_log.write(f"[bold red]Command error:[/bold red] {exc}")

    async def _cmd_show_voices(self, chat_log: RichLog) -> None:
        try:
            from voicebox.client import voice_ids
            ids = list(voice_ids)
        except Exception:
            ids = []
        body = ("  " + "  ".join(f"[#22d3ee]{escape(v)}[/#22d3ee]" for v in ids)) if ids else "  [dim](none)[/dim]"
        chat_log.write(
            "[bold #22d3ee]Voices[/bold #22d3ee]  [dim]usage: /voice <id>[/dim]\n" + body
        )
        chat_log.scroll_end(animate=False)

    async def _cmd_show_picker_options(self, command: str, chat_log: RichLog) -> None:
        if command == "/change-model":
            try:
                from commands import _MODEL_PURPOSES
            except Exception as exc:
                chat_log.write(f"[bold red]Error:[/bold red] {exc}")
                return
            lines = ["[bold #22d3ee]Model purposes[/bold #22d3ee]  [dim]usage: /change-model <purpose>[/dim]"]
            for key, (desc, env_key) in _MODEL_PURPOSES.items():
                current = os.getenv(env_key, "—")
                lines.append(
                    f"  [#22d3ee]{key:<10}[/#22d3ee]"
                    f"  [dim]{escape(desc)}  current: {escape(current)}[/dim]"
                )
            chat_log.write("\n".join(lines))
        elif command == "/connect":
            try:
                from connectors.registry import connector_ids
                ids = list(connector_ids())
            except Exception:
                ids = []
            if not ids:
                chat_log.write("[dim]No connectors registered.[/dim]")
            else:
                body = "  " + "  ".join(f"[#22d3ee]{escape(cid)}[/#22d3ee]" for cid in ids)
                chat_log.write(
                    "[bold #22d3ee]Connectors[/bold #22d3ee]  [dim]usage: /connect <service>[/dim]\n" + body
                )
        chat_log.scroll_end(animate=False)

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

        except Exception as exc:
            from utils.agent_state import set_error
            set_error()
            response = f"Sorry, something went wrong: {exc}"
            chat_log.write(f"[bold red]Error:[/bold red] {escape(str(exc))}")

        finally:
            elapsed = time.time() - start
            self._finalize_stream(response)

            if self._speak and response:
                _as.set_node("tts")
                from voicebox.client import active_voice_id, speak as vb_speak
                from agents.persona import get_active_voice_id

                await asyncio.to_thread(
                    vb_speak,
                    response,
                    get_active_voice_id() or active_voice_id,
                )

            _as.set_node("idle")
            self._busy = False

            # Process the next queued request if any
            if self._request_queue:
                next_query = self._request_queue.pop(0)
                chat_log = self.query_one("#chat-log", RichLog)
                remaining = len(self._request_queue)
                if remaining:
                    chat_log.write(f"[dim]Processing queued request ({remaining} still waiting)…[/dim]")
                self._write_user(next_query)
                self._stream_tokens.clear()
                self._streaming = True
                self._busy = True
                self.run_worker(
                    self._process_message(next_query),
                    exclusive=False,
                    name="pipeline",
                )

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

    def action_exit_tui(self) -> None:
        """Exit immediately from a global shortcut, even while input is focused."""
        self.exit()

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
        logger = entry.get("logger")
        style = LEVEL_STYLE.get(level, "")
        source_prefix = f" [dim]{escape(str(logger))}:[/dim]" if logger else ""
        level_token = escape(f"[{level}]")
        log_view.write(
            f"[dim]{escape(str(ts))}[/dim] "
            f"[{style}]{level_token}[/{style}]{source_prefix} {escape(str(msg))}"
        )
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
            total = sum(self._log_counts.values())
            parts = [f"[bold]TOTAL: {total}[/bold]"]
            parts += [
                f"[{LEVEL_STYLE[lvl]}]{lvl}: {self._log_counts[lvl]}[/{LEVEL_STYLE[lvl]}]"
                for lvl in ALL_LEVELS
            ]
            scroll_indicator = (
                "  [bold #4ade80]↓ auto[/bold #4ade80]"
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
        if btn_id == "btn-send":
            self.query_one("#chat-input", Input).action_submit()
            return
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
