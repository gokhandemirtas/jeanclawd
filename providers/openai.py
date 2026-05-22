import json
import os
import re
from typing import Iterator, Type

from json_repair import repair_json
from pydantic import BaseModel

from .base import InferenceProvider

_JSON_INSTRUCTION = (
	"Respond with a single JSON object that strictly follows the schema below. "
	"Output only the JSON — no explanation, no markdown, no code fences.\n\n"
	"Schema:\n{schema}"
)

_MODEL_CONTEXT: dict[str, int] = {
	"gpt-4o": 128000,
	"gpt-4o-mini": 128000,
	"gpt-4-turbo": 128000,
	"gpt-4": 8192,
	"gpt-3.5-turbo": 16385,
	"o1": 200000,
	"o3": 200000,
	"o4-mini": 200000,
}


class OpenAIProvider(InferenceProvider):
	"""Native OpenAI provider using the official SDK.

	Configured via env vars:
	  OPENAI_API_KEY   — required
	  MODEL_GENERALIST — model id, e.g. gpt-4o
	"""

	def __init__(self, model_key: str, **_kwargs):
		from openai import OpenAI
		self._model = model_key
		self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

	def _inject_schema(self, messages: list[dict], response_format: Type[BaseModel]) -> list[dict]:
		schema = json.dumps(response_format.model_json_schema(), indent=2)
		instruction = _JSON_INSTRUCTION.format(schema=schema)
		patched = list(messages)
		if patched and patched[0]["role"] == "system":
			patched[0] = {**patched[0], "content": patched[0]["content"] + "\n\n" + instruction}
		else:
			patched.insert(0, {"role": "system", "content": instruction})
		return patched

	def complete_stream(
		self,
		messages: list[dict],
		response_format: Type[BaseModel],
	) -> Iterator[str]:
		patched = self._inject_schema(messages, response_format)
		stream = self._client.chat.completions.create(
			model=self._model,
			messages=patched,
			stream=True,
			max_completion_tokens=2048,
		)
		for chunk in stream:
			content = chunk.choices[0].delta.content
			if content:
				yield content

	def complete(
		self,
		messages: list[dict],
		response_format: Type[BaseModel],
	) -> BaseModel:
		raw = "".join(self.complete_stream(messages, response_format))
		raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
		match = re.search(r"\{.*\}", raw, re.DOTALL)
		text = match.group(0) if match else raw
		return response_format.model_validate_json(repair_json(text))

	def respond(self, messages: list[dict]) -> str:
		resp = self._client.chat.completions.create(
			model=self._model,
			messages=messages,
			stream=False,
		)
		return resp.choices[0].message.content

	def embed(self, text: str) -> list[float]:
		resp = self._client.embeddings.create(
			model="text-embedding-3-small",
			input=text,
		)
		return resp.data[0].embedding

	def context_length(self) -> int:
		for prefix, ctx in _MODEL_CONTEXT.items():
			if self._model.startswith(prefix):
				return ctx
		return int(os.environ.get("MODEL_CTX", 128000))
