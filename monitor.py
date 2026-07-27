"""Live log + resource monitor — run in a separate terminal tab while the main app is running.

Usage:
    python monitor.py
    python monitor.py --log logs/app.log
"""

import argparse
import json
import math
import time
from pathlib import Path

from rich.markup import escape

import psutil
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, RichLog, Static

LOG_FILE_DEFAULT = Path(__file__).parent / "logs" / "app.log"
# Imported from utils.resource_monitor so the writer (chat process) and reader
# (this monitor) always agree on the path, regardless of CWD.
from utils.resource_monitor import CONTEXT_FILE  # noqa: E402

# ------------------------------------------------------------------ resource helpers

def _rbar(used: float, total: float, width: int = 18) -> str:
	pct = used / total if total else 0
	filled = int(width * pct)
	return "█" * filled + "░" * (width - filled)


def _fmt_mb(mb: float) -> str:
	return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def _resource_markup() -> str:
	lines: list[str] = []

	# CPU
	pct = psutil.cpu_percent(interval=None)
	bar = _rbar(pct, 100)
	lines.append(f"[bold cyan]CPU[/bold cyan]")
	lines.append(f"[{bar}] [bold]{pct:.1f}%[/bold]")
	lines.append("")

	# RAM
	vm = psutil.virtual_memory()
	used_mb = vm.used / 1024 ** 2
	total_mb = vm.total / 1024 ** 2
	bar = _rbar(used_mb, total_mb)
	lines.append(f"[bold cyan]Memory[/bold cyan]")
	lines.append(f"[{bar}]")
	lines.append(f"[bold]{_fmt_mb(used_mb)}[/bold] / {_fmt_mb(total_mb)}")
	lines.append("")

	# Swap
	sw = psutil.swap_memory()
	if sw.total > 0:
		sw_used = sw.used / 1024 ** 2
		sw_total = sw.total / 1024 ** 2
		bar = _rbar(sw_used, sw_total)
		lines.append(f"[bold cyan]Swap[/bold cyan]")
		lines.append(f"[{bar}]")
		lines.append(f"[bold]{_fmt_mb(sw_used)}[/bold] / {_fmt_mb(sw_total)}")
		lines.append("")

	# NVIDIA VRAM
	try:
		from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlInit
		nvmlInit()
		h = nvmlDeviceGetHandleByIndex(0)
		info = nvmlDeviceGetMemoryInfo(h)
		name = nvmlDeviceGetName(h)
		vram_used = info.used / 1024 ** 2
		vram_total = info.total / 1024 ** 2
		bar = _rbar(vram_used, vram_total)
		lines.append(f"[bold cyan]VRAM[/bold cyan] [dim]{name}[/dim]")
		lines.append(f"[{bar}]")
		lines.append(f"[bold]{_fmt_mb(vram_used)}[/bold] / {_fmt_mb(vram_total)}")
		lines.append("")
	except Exception:
		pass

	# LLM context — written by utils/resource_monitor.py:push_context_to_file
	# from the chat process. ALWAYS rendered so the meter has a stable position
	# in the layout. Shows last-known values (even if stale); only shows a
	# placeholder when the file has never been written.
	used, max_ = 0, 0
	has_data = False
	try:
		if CONTEXT_FILE.exists():
			with open(CONTEXT_FILE) as f:
				ctx = json.loads(f.read())
			used = int(ctx.get("used", 0))
			max_ = int(ctx.get("max", 0)) or 0
			has_data = True
	except Exception:
		pass

	lines.append("[bold cyan]Context[/bold cyan] [dim]LLM[/dim]")
	if not has_data or max_ <= 0:
		# Never written yet, or max unknown — empty bar with dim placeholder
		bar = _rbar(0, 1)
		lines.append(f"[{bar}]")
		lines.append("[dim]— / — (waiting for chat)[/dim]")
	else:
		bar = _rbar(used, max_)
		pct = used / max_ * 100
		lines.append(f"[{bar}]")
		lines.append(f"[bold]{used:,}[/bold] / {max_:,} ({pct:.0f}%)")
	lines.append("")

	return "\n".join(lines)


# ------------------------------------------------------------------ log helpers

LEVEL_STYLE = {
	"DEBUG": "dim white",
	"INFO": "bold cyan",
	"WARN": "bold yellow",
	"ERROR": "bold red",
}

ALL_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")


class MonitorApp(App):
	TITLE = "jeanclawd — live monitor"
	SUB_TITLE = ""

	CSS = """
	#filter-bar {
		height: 3;
		dock: top;
		background: $surface;
		padding: 0 1;
		margin-top: 1;
	}
	#filter-bar Button {
		margin: 0 1;
		min-width: 9;
	}
	#filter-bar Button.-active {
		background: $accent;
		color: $text;
	}
	#stats {
		height: 1;
		dock: top;
		padding: 0 2;
		background: $panel;
		color: $text-muted;
	}
	#body {
		height: 1fr;
	}
	#log-pane {
		width: 3fr;
	}
	#log-view {
		border: none;
		height: 1fr;
	}
	#resource-pane {
		width: 26;
		border-left: solid $panel;
		padding: 1 2;
		background: $surface;
	}
	"""

	BINDINGS = [
		Binding("a", "toggle_autoscroll", "auto-scroll"),
		Binding("c", "clear", "clear"),
		Binding("q", "quit", "quit"),
	]

	def __init__(self, log_path: Path) -> None:
		super().__init__()
		self._log_path = log_path
		self._file_pos = 0
		self._active_filter = "ALL"
		self._auto_scroll = True
		self._counts: dict[str, int] = {lvl: 0 for lvl in ALL_LEVELS}
		self._entries: list[dict] = []

	def compose(self) -> ComposeResult:
		yield Header()
		with Horizontal(id="filter-bar"):
			yield Button("ALL", id="btn-ALL", classes="-active")
			for lvl in ALL_LEVELS:
				yield Button(lvl, id=f"btn-{lvl}")
			yield Button("CLEAR", id="btn-clear", variant="warning")
		yield Label(self._stats_text(), id="stats")
		with Horizontal(id="body"):
			with Vertical(id="log-pane"):
				yield RichLog(id="log-view", highlight=False, markup=True, wrap=True)
			with Vertical(id="resource-pane"):
				yield Static(_resource_markup(), id="resources")
		yield Footer()

	def on_mount(self) -> None:
		psutil.cpu_percent(interval=None)  # warm up — first call always returns 0.0
		if self._log_path.exists():
			self._file_pos = self._log_path.stat().st_size  # start from end (live tail)
		self.set_interval(0.5, self._poll)
		self.set_interval(1.0, self._poll_resources)

	def _poll_resources(self) -> None:
		self.query_one("#resources", Static).update(_resource_markup())

	# ------------------------------------------------------------------ polling

	def _poll(self) -> None:
		if not self._log_path.exists():
			return
		try:
			# The application keeps a bounded log and may truncate it between polls.
			if self._log_path.stat().st_size < self._file_pos:
				self._file_pos = 0
			with open(self._log_path) as f:
				f.seek(self._file_pos)
				new_lines = f.readlines()
				self._file_pos = f.tell()
		except OSError:
			return

		if not new_lines:
			return

		log_view = self.query_one("#log-view", RichLog)
		for raw in new_lines:
			raw = raw.strip()
			if not raw:
				continue
			try:
				entry = json.loads(raw)
			except json.JSONDecodeError:
				entry = {"ts": "", "level": "INFO", "msg": raw}

			level = entry.get("level", "INFO")
			self._counts[level] = self._counts.get(level, 0) + 1
			self._entries.append(entry)

			if self._active_filter in ("ALL", level):
				self._render_entry(log_view, entry)

		self.query_one("#stats", Label).update(self._stats_text())

	def _render_entry(self, log_view: RichLog, entry: dict) -> None:
		ts = entry.get("ts", "")
		level = entry.get("level", "INFO")
		msg = entry.get("msg", "")
		style = LEVEL_STYLE.get(level, "")
		log_view.write(f"[dim]{ts}[/dim] [{style}]{level:<5}[/{style}] {escape(msg)}")
		if self._auto_scroll:
			log_view.scroll_end(animate=False)

	def _stats_text(self) -> str:
		parts = [f"[{LEVEL_STYLE[lvl]}]{lvl}: {self._counts[lvl]}[/{LEVEL_STYLE[lvl]}]" for lvl in ALL_LEVELS]
		scroll_indicator = "  [bold green]↓ auto[/bold green]" if self._auto_scroll else "  [dim]scroll: off[/dim]"
		return "  ".join(parts) + scroll_indicator

	# ------------------------------------------------------------------ actions

	def action_toggle_autoscroll(self) -> None:
		self._auto_scroll = not self._auto_scroll
		self.query_one("#stats", Label).update(self._stats_text())

	def action_clear(self) -> None:
		self.query_one("#log-view", RichLog).clear()
		self._counts = {lvl: 0 for lvl in ALL_LEVELS}
		self._entries.clear()
		self.query_one("#stats", Label).update(self._stats_text())

	def on_button_pressed(self, event: Button.Pressed) -> None:
		btn_id = event.button.id

		if btn_id == "btn-clear":
			self.action_clear()
			return

		level = btn_id.replace("btn-", "") if btn_id else None
		if level not in ("ALL", *ALL_LEVELS):
			return

		for btn in self.query("#filter-bar Button"):
			btn.remove_class("-active")
		event.button.add_class("-active")

		self._active_filter = level
		log_view = self.query_one("#log-view", RichLog)
		log_view.clear()

		for entry in self._entries:
			if level == "ALL" or entry.get("level") == level:
				self._render_entry(log_view, entry)


def main() -> None:
	parser = argparse.ArgumentParser(description="jeanclawd live log monitor")
	parser.add_argument("--log", type=Path, default=LOG_FILE_DEFAULT, help="Path to log file")
	args = parser.parse_args()
	MonitorApp(log_path=args.log).run()


if __name__ == "__main__":
	main()
