import json
import os
import threading
import traceback
from datetime import datetime

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

_console = Console()
_debug_enabled = False
_log_file = None
_file_lock = threading.Lock()


def init_file_logging(path: str) -> None:
	global _log_file
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	_log_file = open(path, "a", buffering=1)


def _write(level: str, msg: str) -> None:
	if _log_file:
		entry = json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"), "level": level, "msg": msg})
		with _file_lock:
			_log_file.write(entry + "\n")


def set_debug(enabled: bool) -> None:
	global _debug_enabled
	_debug_enabled = enabled


def debug(msg: str) -> None:
	_write("DEBUG", msg)
	if _debug_enabled:
		_console.print(f"[dim][DEBUG] {msg}[/dim]")


def info(msg: str) -> None:
	_write("INFO", msg)
	_console.print(f"[bold cyan]>[/bold cyan] {msg}")


def warning(msg: str) -> None:
	_write("WARN", msg)
	if _debug_enabled:
		_console.print(f"[bold yellow][WARN][/bold yellow] {msg}")


def error(msg: str) -> None:
	_write("ERROR", msg)
	_console.print(f"[bold red][ERROR][/bold red] {escape(msg)}")


def exception(msg: str) -> None:
	_write("ERROR", msg + "\n" + traceback.format_exc())
	_console.print(f"[bold red][ERROR][/bold red] {escape(msg)}")
	_console.print(f"[dim]{escape(traceback.format_exc())}[/dim]")


def stream(content: str) -> None:
	_console.print(content, end="", highlight=False)


def newline() -> None:
	_console.print()


def timing(elapsed: float) -> None:
	if _debug_enabled:
		_console.print(f"[dim]{elapsed:.2f}s[/dim]")


def result(
	query: str, expected: str, actual: str, passed: bool, reasoning: str = ""
) -> None:
	color = "green" if passed else "red"
	status = "PASS" if passed else "FAIL"
	label = f'[{status}] "{query}"'
	detail = f"expected={expected}, got={actual}"
	line = Text()
	line.append(f"{label:<50}", style=color)
	line.append(f" {detail}")
	_console.print(line)
	if not passed and reasoning:
		_console.print(f"  [dim italic]{reasoning}[/dim italic]")


def summary(passed: int, total: int, failed: int, elapsed: float) -> None:
	_console.print()
	_console.print(Rule())
	_console.print(
		f"[bold green]{passed}/{total} passed[/bold green]  |  "
		f"[bold red]{failed} failed[/bold red]  |  "
		f"[dim]{elapsed:.1f}s total[/dim]"
	)
	_console.print(Rule())
