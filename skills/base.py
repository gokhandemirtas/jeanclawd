from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Skill:
	id: str
	description: str
	handle: Callable
	system_prompt: Optional[str] = None
	needs_wrapping: bool = True
	auth_provider: Any = None
