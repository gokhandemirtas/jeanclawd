"""Supertonic expression tags and persona-aware speech decoration."""

import os
import re

from pydantic import BaseModel, Field

from agents.persona import get_active_persona_id
from utils.llm_structured_output import generate_structured_output
from utils import logger

# Supported by Supertonic 3. Keep this list in one place so prompts and
# validation cannot drift apart when the model's expression vocabulary grows.
EXPRESSION_TAGS: dict[str, str] = {
	"laugh": "a short natural laugh",
	"breath": "an audible breath",
	"surprise": "a surprised reaction",
	"sigh": "a sigh",
	"scream": "a brief scream; use very sparingly",
	"throatclear": "a throat-clearing sound",
	"sad": "a sad vocal expression",
	"angry": "an angry vocal expression",
	"cough": "a cough",
	"yawn": "a yawn",
}
VALID_EXPRESSION_TAGS = frozenset(EXPRESSION_TAGS)
_TAG_RE = re.compile(r"<([a-z]+)>", re.IGNORECASE)

_PERSONA_GUIDANCE: dict[str, str] = {
	"jean": "warm, calm, friendly; use breath or an occasional laugh when it feels natural",
	"jerk": "sarcastic and superior; use sigh, yawn, throatclear, and laugh noticeably more often",
	"beth": "gloomy and melancholic; favor sigh, sad, breath, and occasional cough",
	"kimi": "bright and enthusiastic; favor laugh, surprise, and breath",
	"borg": "logical and emotionless; use no tags unless one is genuinely necessary",
	"fred": "paranoid and alarmed; favor surprise, throatclear, cough, angry, and occasional sigh",
}
_FALLBACK_TAGS = {
	"jean": "breath",
	"jerk": "sigh",
	"beth": "sad",
	"kimi": "laugh",
	"borg": None,
	"fred": "surprise",
}


class _DecoratedSpeech(BaseModel):
	text: str = Field(description="The original response with optional Supertonic tags inserted")


def _clean_expression_tags(text: str, original: str) -> str:
	"""Keep only supported tags and fall back if the model returned no text."""
	cleaned = _TAG_RE.sub(
		lambda match: f"<{match.group(1).lower()}>"
		if match.group(1).lower() in VALID_EXPRESSION_TAGS
		else "",
		text,
	)
	return " ".join(cleaned.split()) or original


def decorate_for_speech(text: str) -> str:
	"""Add sparse, persona-aware Supertonic tags without changing the response wording.

	This is deliberately best-effort: a failed second LLM pass must never prevent the
	assistant from speaking the original response.
	"""
	if not text.strip() or os.getenv("TTS_PROVIDER", "local") != "local":
		return text

	persona_id = get_active_persona_id()
	guidance = _PERSONA_GUIDANCE.get(persona_id, "natural, restrained, and conversational")
	tag_reference = ", ".join(f"<{tag}> ({description})" for tag, description in EXPRESSION_TAGS.items())
	system_prompt = f"""You are the final speech-expression editor for Supertonic TTS.
Insert sparse expression tags into the response where they improve natural delivery.
The active persona is {persona_id}: {guidance}.

Rules:
- Preserve every word, number, punctuation mark, and line break from the original.
- Only insert tags; never rewrite, summarize, explain, or remove content.
- Use zero to three tags for a short response and no more than one tag per sentence.
- Do not force an expression into factual or serious content.
- Use only these exact tags: {tag_reference}
- Return only the tagged response in the structured `text` field.
"""
	try:
		result = generate_structured_output(
			model_name=os.getenv("MODEL_GENERALIST", ""),
			user_prompt=text,
			system_prompt=system_prompt,
			pydantic_model=_DecoratedSpeech,
			max_tokens=max(128, min(1024, len(text) // 2 + 128)),
			temperature=0.25,
		)
		decorated = _clean_expression_tags(result.text, text)
		if decorated == text:
			fallback_tag = _FALLBACK_TAGS.get(persona_id, "breath")
			if fallback_tag:
				decorated = f"<{fallback_tag}> {text}"
				logger.debug(f"[tts] Expression editor returned no tag; using <{fallback_tag}>")
		return decorated
	except Exception as exc:
		fallback_tag = _FALLBACK_TAGS.get(persona_id, "breath")
		if fallback_tag:
			logger.warning(f"[tts] Expression editor failed ({exc}); using <{fallback_tag}>")
			return f"<{fallback_tag}> {text}"
		logger.warning(f"[tts] Expression editor failed ({exc}); speaking without a tag")
		return text
