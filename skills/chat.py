from pydantic import BaseModel

from agents import persona
from providers import get_provider
from skills.base import Skill


class _ChatResponse(BaseModel):
	response: str


def _handle_chat(user_input: str) -> str:
	provider = get_provider()
	messages = [
		{"role": "system", "content": persona.DESCRIPTION},
		{"role": "user", "content": user_input},
	]
	result = provider.complete(messages, _ChatResponse)
	return result.response


chat_skill = Skill(
	id="CHAT",
	description="General conversation, greetings, and anything that doesn't fit another category",
	system_prompt=None,
	handle=_handle_chat,
	needs_wrapping=False,
)
