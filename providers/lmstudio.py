import json
from typing import Iterator, Type
from pydantic import BaseModel
import lmstudio as lms
from json_repair import repair_json

from .base import InferenceProvider

def _make_template(response_format: Type[BaseModel]) -> str:
	"""Build a fill-in-the-blank JSON template from a Pydantic model.

	Shows the model a concrete object with placeholder values rather than
	the raw JSON schema definition, which some models echo back verbatim.
	"""
	schema = response_format.model_json_schema()
	props = schema.get("properties", {})
	template: dict = {}
	for field_name, field_schema in props.items():
		field_type = field_schema.get("type", "string")
		desc = field_schema.get("description", "")
		hint = f"<{desc}>" if desc else f"<{field_type}>"
		if field_type == "array":
			template[field_name] = [hint]
		elif field_type == "integer":
			template[field_name] = 0
		elif field_type == "boolean":
			template[field_name] = False
		elif field_type == "object":
			template[field_name] = {}
		else:
			template[field_name] = hint
	return json.dumps(template, indent=2)


_JSON_INSTRUCTION = (
	"Output ONLY a JSON object with the placeholder values below replaced by your real response.\n"
	"No explanation, no markdown, no code fences — just the JSON.\n\n"
	"{template}"
)


class LMStudioProvider(InferenceProvider):
	def __init__(self, model_key: str, temperature: float = 0, max_tokens: int = -1):
		config: dict = {"temperature": temperature, "max_tokens": max_tokens}
		self._model = lms.llm(model_key, config=config)

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
		template = _make_template(response_format)
		original_system = next((m["content"] for m in messages if m["role"] == "system"), "")
		system = original_system + "\n\n" + _JSON_INSTRUCTION.format(template=template)
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
		if not raw:
			raise ValueError("Model returned an empty response")
		# Extract first JSON object/array if the model added surrounding text
		match = re.search(r"\{.*\}", raw, re.DOTALL)
		text = match.group(0) if match else raw
		return response_format.model_validate_json(repair_json(text))

	def respond(self, messages: list[dict]) -> str:
		chat = self._build_chat(messages)
		return "".join(f.content for f in self._model.respond_stream(chat))

	def context_length(self) -> int:
		return self._model.get_context_length()
