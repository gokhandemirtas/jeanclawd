import traceback

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

_console = Console()
_debug_enabled = False


def set_debug(enabled: bool) -> None:
	global _debug_enabled
	_debug_enabled = enabled


def debug(msg: str) -> None:
	if _debug_enabled:
		_console.print(f"[dim][DEBUG] {msg}[/dim]")


def info(msg: str) -> None:
	_console.print(f"[bold cyan]>[/bold cyan] {msg}")


def warning(msg: str) -> None:
	if _debug_enabled:
		_console.print(f"[bold yellow][WARN][/bold yellow] {msg}")


def error(msg: str) -> None:
	_console.print(f"[bold red][ERROR][/bold red] {escape(msg)}")


def exception(msg: str) -> None:
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
