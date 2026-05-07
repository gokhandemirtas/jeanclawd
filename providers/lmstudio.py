from typing import Iterator, Type
from pydantic import BaseModel
import lmstudio as lms

from .base import InferenceProvider


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

    def complete_stream(
        self,
        messages: list[dict],
        response_format: Type[BaseModel],
    ) -> Iterator[str]:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        chat = lms.Chat(system)
        for msg in messages:
            if msg["role"] == "user":
                chat.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                chat.add_assistant_message(msg["content"])

        for fragment in self._model.respond_stream(chat, response_format=response_format):
            yield fragment.content
