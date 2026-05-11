import json
import os
from typing import Iterator, Type
from pydantic import BaseModel
import lmstudio as lms
from json_repair import repair_json

from .base import InferenceProvider

_JSON_INSTRUCTION = (
	"Respond with a single JSON object that strictly follows the schema below. "
	"Output only the JSON — no explanation, no markdown, no code fences.\n\n"
	"Schema:\n{schema}"
)


class LMStudioProvider(InferenceProvider):
	def __init__(self, model_key: str, temperature: float = 0, max_tokens: int = -1):
		self._model = lms.llm(
			model_key,
			config={
				"gpu_offload": "max",
				"temperature": temperature,
				"max_tokens": max_tokens,
			},
		)
		self._embedding_model = lms.embedding_model(os.environ["MODEL_EMBEDDING"])

	def _build_chat(self, messages: list[dict], system_override: str | None = None) -> lms.Chat:
		system = system_override or next((m["content"] for m in messages if m["role"] == "system"), "")
		chat = lms.Chat(system)
		for msg in messages:
			if msg["role"] == "user":
				chat.add_user_message(msg["content"])
			elif msg["role"] == "assistant":
				chat.add_assistant_message(msg["content"])
		return chat

	def complete_stream(
		self,
		messages: list[dict],
		response_format: Type[BaseModel],
	) -> Iterator[str]:
		# LM Studio's grammar-constrained generation (response_format=) crashes with
		# 'tree_reduce' on some models. Inject the schema into the system prompt instead
		# and parse JSON from free-form output.
		schema = json.dumps(response_format.model_json_schema(), indent=2)
		original_system = next((m["content"] for m in messages if m["role"] == "system"), "")
		system = original_system + "\n\n" + _JSON_INSTRUCTION.format(schema=schema)
		chat = self._build_chat(messages, system_override=system)
		for fragment in self._model.respond_stream(chat):
			yield fragment.content

	def complete(
		self,
		messages: list[dict],
		response_format: Type[BaseModel],
	) -> BaseModel:
		raw = "".join(self.complete_stream(messages, response_format))
		# Strip <think>...</think> blocks before parsing
		import re
		raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
		# Extract first JSON object/array if the model added surrounding text
		match = re.search(r"\{.*\}", raw, re.DOTALL)
		text = match.group(0) if match else raw
		return response_format.model_validate_json(repair_json(text))

	def respond(self, messages: list[dict]) -> str:
		chat = self._build_chat(messages)
		return "".join(f.content for f in self._model.respond_stream(chat))

	def embed(self, text: str) -> list[float]:
		return self._embedding_model.embed(text)

	def context_length(self) -> int:
		return self._model.get_context_length()
