import json
import os
import re
from typing import Iterator, Type

import requests
from json_repair import repair_json
from pydantic import BaseModel

from .base import InferenceProvider

_JSON_INSTRUCTION = (
	"Respond with a single JSON object that strictly follows the schema below. "
	"Output only the JSON — no explanation, no markdown, no code fences.\n\n"
	"Schema:\n{schema}"
)


class OpenAICompatibleProvider(InferenceProvider):
	"""REST-only provider for any OpenAI-compatible inference server.

	Configured entirely via env vars:
	  MODEL_HOST       — base URL, e.g. http://localhost:11434/v1
	  MODEL_GENERALIST — model id for completions
	  MODEL_EMBEDDING  — model id for embeddings
	  OPENAI_API_KEY   — bearer token (default: "lm-studio")
	  MODEL_CTX        — fallback context length when /models doesn't report one
	"""

	def __init__(self, model_key: str, embedding_model: str | None = None):
		self._base = os.environ.get("MODEL_HOST", "http://localhost:1234/v1").rstrip("/")
		self._model = model_key
		self._embedding_model = embedding_model or os.environ.get("MODEL_EMBEDDING", "")
		self._api_key = os.environ.get("OPENAI_API_KEY", "lm-studio")
		self._headers = {
			"Authorization": f"Bearer {self._api_key}",
			"Content-Type": "application/json",
		}

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
		resp = requests.post(
			f"{self._base}/chat/completions",
			headers=self._headers,
			json={"model": self._model, "messages": patched, "stream": True},
			stream=True,
			timeout=120,
		)
		resp.raise_for_status()
		for raw in resp.iter_lines():
			if not raw:
				continue
			line = raw.decode() if isinstance(raw, bytes) else raw
			if line.startswith("data: "):
				line = line[6:]
			if line == "[DONE]":
				break
			try:
				content = json.loads(line)["choices"][0]["delta"].get("content")
				if content:
					yield content
			except Exception:
				pass

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
		resp = requests.post(
			f"{self._base}/chat/completions",
			headers=self._headers,
			json={"model": self._model, "messages": messages, "stream": False},
			timeout=120,
		)
		resp.raise_for_status()
		return resp.json()["choices"][0]["message"]["content"]

	def embed(self, text: str) -> list[float]:
		resp = requests.post(
			f"{self._base}/embeddings",
			headers=self._headers,
			json={"model": self._embedding_model, "input": text},
			timeout=30,
		)
		resp.raise_for_status()
		return resp.json()["data"][0]["embedding"]

	def context_length(self) -> int:
		# Try per-model endpoint first (supported by some servers)
		try:
			resp = requests.get(
				f"{self._base}/models/{self._model}",
				headers=self._headers,
				timeout=5,
			)
			if resp.ok:
				data = resp.json()
				ctx = data.get("context_length") or data.get("max_context_length")
				if ctx:
					return int(ctx)
		except Exception:
			pass

		# Fall back to model list
		try:
			resp = requests.get(f"{self._base}/models", headers=self._headers, timeout=5)
			if resp.ok:
				for m in resp.json().get("data", []):
					if m.get("id") == self._model:
						ctx = m.get("context_length") or m.get("max_context_length")
						if ctx:
							return int(ctx)
		except Exception:
			pass

		return int(os.environ.get("MODEL_CTX", 8192))
