from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def generate_structured_output(
	model_name: str,
	user_prompt: str,
	system_prompt: str,
	pydantic_model: Type[T],
	max_tokens: int = 1024,
	temperature: float = 0.1,
	repeat_penalty: float = 1.0,
	top_p: float = 0.9,
	top_k: int = 40,
) -> T:
	from providers import get_provider

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
	return get_provider().complete(messages, pydantic_model)
