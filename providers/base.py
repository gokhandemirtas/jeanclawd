from abc import ABC, abstractmethod
from typing import Iterator, Type
from pydantic import BaseModel


class InferenceProvider(ABC):
	@abstractmethod
	def complete_stream(
		self,
		messages: list[dict],
		response_format: Type[BaseModel],
	) -> Iterator[str]: ...

	def complete(
		self,
		messages: list[dict],
		response_format: Type[BaseModel],
	) -> BaseModel:
		tokens = "".join(self.complete_stream(messages, response_format))
		return response_format.model_validate_json(tokens)

	@abstractmethod
	def respond(self, messages: list[dict]) -> str:
		"""Raw text completion without structured output."""
		...

	@abstractmethod
	def embed(self, text: str) -> list[float]:
		"""Return an embedding vector for the given text."""
		...

	@abstractmethod
	def context_length(self) -> int:
		"""Return the actual context window size (tokens) for the loaded model."""
		...
