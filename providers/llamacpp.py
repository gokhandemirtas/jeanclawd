from typing import Iterator, Type
from pydantic import BaseModel

from .base import InferenceProvider


class LlamaCppProvider(InferenceProvider):
	def __init__(
		self,
		model_path: str,
		n_gpu_layers: int = -1,
		temperature: float = 0,
		max_tokens: int = -1,
	):
		from llama_cpp import Llama

		self._llm = Llama(
			model_path=model_path,
			n_gpu_layers=n_gpu_layers,
			n_ctx=4096,
			verbose=False,
		)
		self._temperature = temperature
		self._max_tokens = max_tokens if max_tokens != -1 else None

	def complete_stream(
		self,
		messages: list[dict],
		response_format: Type[BaseModel],
	) -> Iterator[str]:
		schema = response_format.model_json_schema()
		response = self._llm.create_chat_completion(
			messages=messages,
			response_format={"type": "json_object", "schema": schema},
			temperature=self._temperature,
			max_tokens=self._max_tokens,
			stream=True,
		)
		for chunk in response:
			content = chunk["choices"][0]["delta"].get("content")
			if content:
				yield content

	def respond(self, messages: list[dict]) -> str:
		response = self._llm.create_chat_completion(
			messages=messages,
			temperature=self._temperature,
			max_tokens=self._max_tokens,
			stream=False,
		)
		return response["choices"][0]["message"]["content"]

	def embed(self, text: str) -> list[float]:
		raise NotImplementedError("LlamaCppProvider does not support embeddings")
