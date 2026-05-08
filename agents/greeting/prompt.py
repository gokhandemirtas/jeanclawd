def build_greeting_prompt(name: str, description: str, greeting_style: str) -> str:
	return (
		f"{description}\n\n"
		f"Generate one short greeting to open a new conversation. "
		f"Style: {greeting_style}. "
		f"1–2 sentences only. No follow-up questions. "
		f"Return JSON with a single key: greeting."
	)
