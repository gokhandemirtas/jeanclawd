import asyncio
import os
from typing import Optional

import lmstudio as lms

from utils.messages import AIMessage, HumanMessage, SystemMessage


def _build_chat(messages: list) -> lms.Chat:
	system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
	chat = lms.Chat(system)
	for msg in messages:
		if isinstance(msg, HumanMessage):
			chat.add_user_message(msg.content)
		elif isinstance(msg, AIMessage):
			chat.add_assistant_message(msg.content)
	return chat


class _StreamChunk:
	def __init__(self, content: str):
		self.content = content


class _LMStudioLLM:
	def __init__(
		self,
		model_name: str,
		temperature: float = 0.7,
		max_tokens: int = 1024,
		top_p: float = 0.9,
		top_k: int = 40,
		repeat_penalty: float = 1.0,
	):
		self._model_name = model_name
		self._temperature = temperature
		self._max_tokens = max_tokens
		self._top_p = top_p
		self._top_k = top_k
		self._repeat_penalty = repeat_penalty

	def _get_model(self) -> lms.LLM:
		return lms.llm(
			self._model_name,
			config={
				"temperature": self._temperature,
				"max_tokens": self._max_tokens,
				"top_p": self._top_p,
				"top_k": self._top_k,
				"repeat_penalty": self._repeat_penalty,
			},
		)

	def bind(self, max_tokens: Optional[int] = None) -> "_LMStudioLLM":
		return _LMStudioLLM(
			model_name=self._model_name,
			temperature=self._temperature,
			max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
			top_p=self._top_p,
			top_k=self._top_k,
			repeat_penalty=self._repeat_penalty,
		)

	async def ainvoke(self, messages: list) -> AIMessage:
		chat = _build_chat(messages)
		model = self._get_model()

		def _run() -> str:
			return "".join(f.content for f in model.respond_stream(chat))

		text = await asyncio.to_thread(_run)
		return AIMessage(content=text)

	async def astream(self, messages: list):
		chat = _build_chat(messages)
		model = self._get_model()
		loop = asyncio.get_running_loop()
		queue: asyncio.Queue = asyncio.Queue()

		async def _producer():
			def _run():
				for fragment in model.respond_stream(chat):
					loop.call_soon_threadsafe(queue.put_nowait, fragment.content)
				loop.call_soon_threadsafe(queue.put_nowait, None)

			await asyncio.to_thread(_run)

		task = asyncio.create_task(_producer())
		try:
			while True:
				token = await queue.get()
				if token is None:
					break
				yield _StreamChunk(content=token)
		finally:
			await task


class _LMStudioAgent:
	def __init__(self, model_name: str):
		self._model_name = model_name

	def invoke(self, inputs: dict) -> dict:
		messages = inputs.get("messages", [])
		chat = _build_chat(messages)
		model = lms.llm(self._model_name)
		response_text = "".join(f.content for f in model.respond_stream(chat))
		return {"messages": [*messages, AIMessage(content=response_text)]}


def create_llm(
	model_name: Optional[str] = None,
	temperature: float = 0.7,
	max_tokens: int = 1024,
	top_p: float = 0.9,
	top_k: int = 40,
	repeat_penalty: float = 1.0,
) -> _LMStudioLLM:
	name = model_name or os.environ.get("MODEL_GENERALIST", "")
	return _LMStudioLLM(
		model_name=name,
		temperature=temperature,
		max_tokens=max_tokens,
		top_p=top_p,
		top_k=top_k,
		repeat_penalty=repeat_penalty,
	)


def create_llm_agent(model_name: Optional[str] = None) -> _LMStudioAgent:
	name = model_name or os.environ.get("MODEL_GENERALIST", "")
	return _LMStudioAgent(name)
