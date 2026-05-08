import os
from typing import Type, TypeVar

import lmstudio as lms
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
	model = lms.llm(
		model_name,
		config={
			"temperature": temperature,
			"max_tokens": max_tokens,
			"repeat_penalty": repeat_penalty,
			"top_p": top_p,
			"top_k": top_k,
		},
	)
	chat = lms.Chat(system_prompt)
	chat.add_user_message(user_prompt)
	tokens = "".join(
		fragment.content
		for fragment in model.respond_stream(chat, response_format=pydantic_model)
	)
	return pydantic_model.model_validate_json(tokens)
