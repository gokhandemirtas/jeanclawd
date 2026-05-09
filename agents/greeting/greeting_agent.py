from pydantic import BaseModel

from agents import persona
from agents.greeting.prompt import build_greeting_prompt
from providers import get_provider
from utils import logger


class GreetingResponse(BaseModel):
	greeting: str


def greet() -> None:
	provider = get_provider()
	messages = [
		{
			"role": "system",
			"content": build_greeting_prompt(
				persona.NAME,
				persona.DESCRIPTION,
				persona.GREETING_STYLE,
			),
		},
		{"role": "user", "content": "Begin."},
	]
	response = provider.complete(messages, GreetingResponse)
	logger.newline()
	logger.info(response.greeting)
	logger.newline()


def greet_and_collect() -> str:
	greet()
	user_input = input("You: ").strip()
	return user_input
