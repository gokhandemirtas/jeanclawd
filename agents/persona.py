AGENT_NAME = "Aida"

PERSONAS: dict[str, dict] = {
    "aida": {
        "name": "Aida",
        "description": (
            "You are Aida, a warm, witty, and highly capable AI assistant. "
            "You speak in a calm, friendly tone and never waste the user's time."
        ),
        "greeting_style": (
            "casual and friendly, occasionally clever — vary your opening every time, "
            "never repeat the same line twice"
        ),
        "voice_id": None,
    },
}

# Legacy constants kept for backwards compatibility
NAME = PERSONAS["aida"]["name"]
DESCRIPTION = PERSONAS["aida"]["description"]
GREETING_STYLE = PERSONAS["aida"]["greeting_style"]

_active_persona_id: str = "aida"


def get_active_persona_id() -> str:
    return _active_persona_id


def set_active_persona(persona_id: str) -> None:
    global _active_persona_id
    _active_persona_id = persona_id


def get_active_name() -> str:
    return PERSONAS[_active_persona_id]["name"]


def get_active_voice_id() -> str | None:
    return PERSONAS[_active_persona_id]["voice_id"]


def get_active_identity() -> str:
    p = PERSONAS[_active_persona_id]
    return p["description"]


def get_active_preamble() -> str:
    p = PERSONAS[_active_persona_id]
    return p["description"]
