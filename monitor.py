"""Live log monitor — run in a separate terminal tab while the main app is running.

Usage:
    python monitor.py
    python monitor.py --log logs/app.log
"""

import argparse
import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Label, RichLog

LOG_FILE_DEFAULT = Path(__file__).parent / "logs" / "app.log"

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
	#log-view {
		border: none;
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
		yield RichLog(id="log-view", highlight=False, markup=True, wrap=True)
		yield Footer()

	def on_mount(self) -> None:
		if self._log_path.exists():
			self._file_pos = self._log_path.stat().st_size  # start from end (live tail)
		self.set_interval(0.5, self._poll)

	# ------------------------------------------------------------------ polling

	def _poll(self) -> None:
		if not self._log_path.exists():
			return
		try:
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
		log_view.write(f"[dim]{ts}[/dim] [{style}]{level:<5}[/{style}] {msg}")
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
