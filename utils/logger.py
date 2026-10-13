import json
import os
import threading
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

_console = Console()
_debug_enabled = False
_log_file = None
_log_path = None
_file_lock = threading.Lock()
_MAX_LOG_BYTES = 500 * 1024
_message_capture: ContextVar[list[tuple[str, str]] | None] = ContextVar(
	"message_capture", default=None
)


@contextmanager
def capture_messages():
	"""Capture log output for an interactive surface without printing to stdout."""
	messages: list[tuple[str, str]] = []
	token = _message_capture.set(messages)
	try:
		yield messages
	finally:
		_message_capture.reset(token)


def init_file_logging(path: str) -> None:
	global _log_file, _log_path
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	_log_path = path
	# The application log is deliberately a short operational buffer. Long-term
	# conversational history is handled by reflections, not by this file.
	_log_file = open(path, "a", buffering=1)
	if os.path.getsize(path) > _MAX_LOG_BYTES:
		_log_file.close()
		_log_file = open(path, "w", buffering=1)


def _write(level: str, msg: str, fields: dict | None = None) -> None:
	global _log_file
	capture = _message_capture.get()
	if capture is not None:
		capture.append((level, msg))
	if _log_file:
		with _file_lock:
			entry_data = {"ts": datetime.now().isoformat(timespec="milliseconds"), "level": level, "msg": msg}
			if fields:
				entry_data.update(fields)
			entry = json.dumps(entry_data, ensure_ascii=False, default=str)
			# Keep even an unexpectedly large traceback from exceeding the file cap.
			if len(entry.encode("utf-8")) >= _MAX_LOG_BYTES:
				entry = json.dumps({
					"ts": datetime.now().isoformat(timespec="milliseconds"),
					"level": level,
					"msg": msg[:2000] + " …[message truncated]",
				})
			entry_bytes = len((entry + "\n").encode("utf-8"))
			if _log_file.tell() + entry_bytes > _MAX_LOG_BYTES:
				_log_file.close()
				_log_file = open(_log_path, "w", buffering=1)
			_log_file.write(entry + "\n")


def set_debug(enabled: bool) -> None:
	global _debug_enabled
	_debug_enabled = enabled


def debug(msg: str, agent: str | None = None) -> None:
	_write("DEBUG", f"[{agent}] {msg}" if agent else msg)
	if _message_capture.get() is not None:
		return
	if not _debug_enabled:
		return
	# Extract leading [tag] from message if no explicit agent given
	tag = agent
	body = msg
	if not tag and msg.startswith("["):
		end = msg.find("]")
		if end != -1:
			tag = msg[1:end]
			body = msg[end + 1:].lstrip()
	if tag:
		_console.print(f"[dim][DEBUG] [/dim][bold magenta]{escape(tag)}[/bold magenta][dim] {escape(body)}[/dim]")
	else:
		_console.print(f"[dim][DEBUG] {escape(msg)}[/dim]")


def info(msg: str) -> None:
	_write("INFO", msg)
	if _message_capture.get() is not None:
		return
	_console.print(f"[bold cyan]>[/bold cyan] {msg}")


def warning(msg: str) -> None:
	_write("WARN", msg)
	if _message_capture.get() is not None:
		return
	if _debug_enabled:
		_console.print(f"[bold yellow][WARN][/bold yellow] {msg}")


def error(msg: str) -> None:
	_write("ERROR", msg)
	if _message_capture.get() is not None:
		return
	if _debug_enabled:
		_console.print(f"[bold red][ERROR][/bold red] {escape(msg)}")


def event(name: str, *, level: str = "INFO", **fields) -> None:
	"""Write a structured operational event without changing the console UX."""
	display_name = fields.pop("_display_name", name)
	_safe_fields = {key: value for key, value in fields.items() if value is not None}
	_write(level, display_name, {"event": name, **_safe_fields})
	if _debug_enabled and level != "DEBUG":
		_console.print(f"[dim][{level}] event={escape(name)}[/dim]")


def exception(msg: str) -> None:
	tb = traceback.format_exc()
	_write("ERROR", msg + "\n" + tb)
	if _debug_enabled:
		_console.print(f"[bold red][ERROR][/bold red] {escape(msg)}")
		_console.print(f"[dim]{escape(tb)}[/dim]")


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
