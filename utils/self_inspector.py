"""Narrow, read-only facade for inspecting the jeanclawd source tree.

This module is deliberately the only filesystem surface intended for an
agent that needs to understand the application.  Keep the policy here
explicit: callers receive repository-relative paths, bounded text, and no
filesystem write or process-execution primitive.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_REFLECTIONS_DIR = _ROOT / "introspections"
_MAX_READ_BYTES = 64 * 1024
_MAX_RESULTS = 100
_MAX_REFLECTION_FIELD = 12_000

# Deny by name, not only by extension.  This protects secrets even if a new
# secret-bearing file uses an unfamiliar suffix.
_DENIED_NAMES = {
	".env",
	".env.*",
	".claude/*",
	".git",
	".venv",
	"venv",
	"__pycache__",
	".pytest_cache",
	".mypy_cache",
	".ruff_cache",
	"node_modules",
	"credentials",
	"secrets",
	"private",
	"*.secret",
	"*.credentials",
}
_DENIED_SUFFIXES = {".pyc", ".pem", ".key", ".crt", ".p12", ".pfx"}
_DENIED_RELATIVE_PATHS = {"utils/self_inspector.py"}
_TEXT_SUFFIXES = {
	".md",
	".py",
	".toml",
	".yaml",
	".yml",
	".json",
	".txt",
	".ini",
	".cfg",
	".sql",
}


def _relative(path: Path) -> str:
	return path.relative_to(_ROOT).as_posix()


def _is_denied(path: Path) -> bool:
	try:
		relative = path.resolve().relative_to(_ROOT)
	except ValueError:
		return True

	for part in relative.parts:
		if any(fnmatch.fnmatch(part.lower(), pattern) for pattern in _DENIED_NAMES):
			return True
	if relative.as_posix() in _DENIED_RELATIVE_PATHS:
		return True
	return path.suffix.lower() in _DENIED_SUFFIXES


def _resolve(relative_path: str) -> Path:
	if not relative_path or Path(relative_path).is_absolute():
		raise ValueError("path must be a non-empty repository-relative path")
	path = (_ROOT / relative_path).resolve()
	try:
		path.relative_to(_ROOT)
	except ValueError as exc:
		raise ValueError("path escapes the repository") from exc
	if _is_denied(path):
		raise PermissionError("that path is not exposed through the self-inspector")
	return path


def architecture() -> str:
	"""Return the curated architecture map, if present."""
	path = _resolve("CODEBASE.md")
	return path.read_text(encoding="utf-8")[:_MAX_READ_BYTES]


def list_files(prefix: str = "", limit: int = 5_000) -> str:
	"""List exposed repository files below an optional relative prefix."""
	limit = max(1, min(limit, 10_000))
	base = _resolve(prefix) if prefix else _ROOT
	if not base.is_dir():
		raise ValueError("prefix must refer to a directory")

	files = []
	for path in sorted(base.rglob("*")):
		if path.is_file() and not _is_denied(path):
			files.append(_relative(path))
			if len(files) >= limit:
				break
	result = "\n".join(files) or "No exposed files found."
	if len(files) >= limit:
		result += f"\n\n[listing capped at {limit} files]"
	return result


def exposed_files(prefix: str = "") -> list[str]:
	"""Return the complete exposed inventory for direct Python callers."""
	base = _resolve(prefix) if prefix else _ROOT
	if not base.is_dir():
		raise ValueError("prefix must refer to a directory")
	return [
		_relative(path)
		for path in sorted(base.rglob("*"))
		if path.is_file() and not _is_denied(path)
	]


def read_file(relative_path: str, max_bytes: int = 32_000) -> str:
	"""Read one bounded, text-like source or documentation file."""
	path = _resolve(relative_path)
	if not path.is_file():
		raise ValueError("path must refer to a file")
	if path.suffix.lower() not in _TEXT_SUFFIXES:
		raise PermissionError("only text source and documentation files are exposed")

	max_bytes = max(1, min(max_bytes, _MAX_READ_BYTES))
	data = path.read_bytes()[:max_bytes]
	text = data.decode("utf-8", errors="replace")
	if path.stat().st_size > max_bytes:
		text += f"\n\n[truncated at {max_bytes} bytes]"
	return text


def list_reflections(limit: int = 20) -> list[str]:
	"""Return recent reflection paths, newest first."""
	limit = max(1, min(limit, 100))
	if not _REFLECTIONS_DIR.is_dir():
		return []
	return [
		_relative(path)
		for path in sorted(_REFLECTIONS_DIR.glob("*.md"), reverse=True)[:limit]
		if not _is_denied(path)
	]


def read_reflection(filename: str, max_bytes: int = 64_000) -> str:
	"""Read one previously recorded reflection, never an arbitrary file."""
	if Path(filename).name != filename or not filename.endswith(".md"):
		raise ValueError("reflection must be a markdown filename")
	return read_file(f"introspections/{filename}", max_bytes=max_bytes)


def record_reflection(
	inspected: str,
	findings: str,
	recommendations: str = "No recommendations recorded.",
	*,
	purpose_and_capabilities: str = "",
	scope: str = "",
) -> str:
	"""Persist a dated inspection using the facade's only write operation.

	Recommendations are accepted only when the caller states what the agent
	understands about the system's purpose and available capabilities.  This is
	not a truth guarantee, but it prevents unsupported recommendations from
	being recorded accidentally and makes their basis auditable.
	"""
	if not inspected.strip() or not findings.strip():
		raise ValueError("inspected and findings must not be empty")
	if (
		recommendations.strip()
		and recommendations.strip() != "No recommendations recorded."
		and not purpose_and_capabilities.strip()
	):
		raise ValueError("recommendations require a purpose_and_capabilities basis")

	fields = {
		"inspected": inspected.strip(),
		"findings": findings.strip(),
		"recommendations": recommendations.strip() or "No recommendations recorded.",
	}
	for name, value in fields.items():
		if len(value.encode("utf-8")) > _MAX_REFLECTION_FIELD:
			raise ValueError(f"{name} exceeds the {_MAX_REFLECTION_FIELD}-byte limit")

	stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
	scope_slug = re.sub(r"[^a-z0-9]+", "-", scope.casefold()).strip("-")[:48]
	filename = f"{stamp}{f'-{scope_slug}' if scope_slug else ''}.md"
	path = _REFLECTIONS_DIR / filename
	_REFLECTIONS_DIR.mkdir(mode=0o700, exist_ok=True)
	content = (
		f"# Self-inspection — {stamp}\n\n"
		"## What it inspected\n\n"
		f"{fields['inspected']}\n\n"
		"## Findings\n\n"
		f"{fields['findings']}\n\n"
		"## Recommendations\n\n"
		f"{fields['recommendations']}\n"
	)
	# A generated filename and exclusive creation prevent overwriting an earlier
	# reflection if two inspections happen within the same second.
	for attempt in range(10):
		candidate = path if attempt == 0 else _REFLECTIONS_DIR / f"{stamp}-{attempt}.md"
		try:
			with candidate.open("x", encoding="utf-8") as handle:
				handle.write(content)
			return _relative(candidate)
		except FileExistsError:
			continue
	raise RuntimeError("could not create a unique reflection filename")


def search_code(query: str, prefix: str = "", limit: int = 50) -> str:
	"""Search exposed text files with ripgrep; return repository-relative matches."""
	query = query.strip()
	if not query:
		raise ValueError("query must not be empty")
	limit = max(1, min(limit, _MAX_RESULTS))
	base = _resolve(prefix) if prefix else _ROOT
	if not base.exists():
		raise ValueError("prefix does not exist")

	command = [
		"rg",
		"--no-heading",
		"--line-number",
		"--color=never",
		"--max-count",
		"20",
		"--glob",
		"!.env*",
		"--glob",
		"!.git/**",
		"--glob",
		"!**/__pycache__/**",
		"--glob",
		"!**/node_modules/**",
		"--glob",
		"!*.pem",
		"--glob",
		"!*.key",
		query,
		str(base),
	]
	result = subprocess.run(
		command, capture_output=True, text=True, timeout=5, check=False
	)
	if result.returncode not in (0, 1):
		raise RuntimeError("code search failed")

	lines = []
	for line in result.stdout.splitlines():
		if ":" not in line:
			continue
		absolute, rest = line.split(":", 1)
		path = Path(absolute)
		if _is_denied(path):
			continue
		lines.append(f"{_relative(path)}:{rest}")
		if len(lines) >= limit:
			break
	return "\n".join(lines) or "No matches found."


def policy_summary() -> str:
	"""Describe the facade's access policy without revealing host paths."""
	return (
		"Self-inspector policy:\n"
		"- read-only access to curated text source and documentation files\n"
		"- repository-relative paths only; path traversal and absolute paths denied\n"
		"- .env files, VCS metadata, virtual environments, caches, credentials, and key material denied\n"
		"- file reads capped at 64 KiB; searches capped at 100 results and 5 seconds\n"
		"- no write, delete, shell, package-install, network, or database mutation operations"
	)
