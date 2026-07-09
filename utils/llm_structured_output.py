from typing import Type, TypeVar
from pydantic import BaseModel
import os

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
	from providers import get_intent_provider, get_provider

	intent_model = os.getenv("MODEL_INTENT", "").strip()
	provider = (
		get_intent_provider()
		if intent_model and model_name.strip() == intent_model
		else get_provider()
	)

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
	return provider.complete(messages, pydantic_model)
