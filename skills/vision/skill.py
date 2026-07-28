"""Shared image-understanding skill used by the API and Telegram interfaces."""

from utils.vlm import describe_image


async def handle_vision(image_bytes: bytes, prompt: str | None = None, filename: str = "image") -> str:
	"""Describe an uploaded image and add the result to conversation history."""
	from skills.conversation.skill import append_to_history

	description = await describe_image(image_bytes, prompt=prompt or _default_prompt())
	append_to_history(f"[Image: {filename}]", f"I can see the following in the image: {description}")
	return description


def _default_prompt() -> str:
	return (
		"Describe everything visible in this image. Be specific about objects, people, "
		"text, colours, and spatial relationships. Do not speculate. Keep it to 1-2 paragraphs."
	)
