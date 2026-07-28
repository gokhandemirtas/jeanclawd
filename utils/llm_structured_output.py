import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _update_context_meter(
	system_prompt: str,
	user_prompt: str,
	pydantic_model: Type[BaseModel],
) -> None:
	"""Publish the estimated structured-request size for the live monitor."""
	try:
		from utils.resource_monitor import count_tokens, push_context_to_file, set_context_used

		# Providers append a JSON schema/template to the system prompt. Include
		# its size so structured requests are not understated in the meter.
		schema_text = json.dumps(pydantic_model.model_json_schema(), ensure_ascii=False)
		used = count_tokens(system_prompt) + count_tokens(user_prompt) + count_tokens(schema_text)
		set_context_used(used)
		push_context_to_file()
	except Exception:
		# Observability must never prevent a model request from running.
		pass


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
	_update_context_meter(system_prompt, user_prompt, pydantic_model)
	from utils.observability import span
	from utils.resource_monitor import count_tokens

	input_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]
	with span(
		"llm_request",
		provider=type(provider).__name__,
		model=model_name,
		input_tokens=input_tokens,
		max_tokens=max_tokens,
		structured=True,
	):
		return provider.complete(messages, pydantic_model)
