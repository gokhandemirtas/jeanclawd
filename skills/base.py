from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Skill:
	id: str
	description: str
	handle: Callable[..., Any]
	system_prompt: str | None = None
	needs_wrapping: bool = True
	auth_provider: Any = None
	routable: bool = True
