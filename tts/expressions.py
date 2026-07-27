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
	"sigh": "a sigh",
	"scream": "a brief scream; use very sparingly",
	"throatclear": "a throat-clearing sound",
	"sad": "a sad vocal expression",
	"angry": "an angry vocal expression",
	"cough": "a cough",
	"yawn": "a yawn",
}
VALID_EXPRESSION_TAGS = frozenset(EXPRESSION_TAGS)
_TAG_RE = re.compile(r"<(laugh|breath|sigh|sad|cough|scream|throatclear|angry|yawn)>")

# Supertonic 3 can ignore a single expression tag. These counts are deliberately
# conservative: repeated tags are a model workaround, not extra spoken words.
_TAG_REPEAT_COUNTS = {
	"laugh": 3,
	"breath": 2,
	"sigh": 2,
	"sad": 3,
	"cough": 3,
}

_PERSONA_GUIDANCE: dict[str, str] = {
	"jean": "warm, calm, friendly; use breath or an occasional laugh when it feels natural",
	"jerk": "sarcastic and superior; use sigh, yawn, throatclear, and laugh noticeably more often",
	"beth": "gloomy and melancholic; favor sigh, sad, breath, and occasional cough",
	"kimi": "bright and enthusiastic; favor laugh and breath",
	"borg": "logical and emotionless; use no tags unless one is genuinely necessary",
	"fred": "paranoid and alarmed; favor throatclear, cough, angry, and occasional sigh",
}
class _DecoratedSpeech(BaseModel):
	text: str = Field(
		description="The original response with optional Supertonic tags inserted"
	)


def _repeat_expression_tags(text: str) -> str:
	"""Repeat only known tags to improve Supertonic's unreliable tag activation."""
	def replace(match: re.Match[str]) -> str:
		tag = match.group(1)
		count = _TAG_REPEAT_COUNTS.get(tag, 1)
		return " ".join(f"<{tag}>" for _ in range(count))

	return _TAG_RE.sub(replace, text)


def decorate_for_speech(text: str) -> str:
	"""Add sparse, persona-aware Supertonic tags without changing the response wording.

	This is deliberately best-effort: a failed second LLM pass must never prevent the
	assistant from speaking the original response.
	"""
	if not text.strip() or os.getenv("TTS_PROVIDER", "local") != "local":
		return text

	persona_id = get_active_persona_id()
	guidance = _PERSONA_GUIDANCE.get(
		persona_id, "natural, restrained, and conversational"
	)
	tag_reference = ", ".join(
		f"<{tag}> ({description})" for tag, description in EXPRESSION_TAGS.items()
	)
	system_prompt = f"""You are a mechanical speech-expression editor for Supertonic TTS.
You are not a conversational assistant. Do not answer the user, react to the content,
retell a joke, describe a punchline, or add any commentary. Copy the supplied response
verbatim and insert sparse expression tags only where they improve natural delivery.
The active persona is {persona_id}: {guidance}.

Rules:
- Preserve every word, number, punctuation mark, and line break from the original.
- Only insert tags; never rewrite, summarize, explain, or remove content.
- Use zero to three tags for a short response and no more than one tag per sentence.
- Do not force an expression into factual or serious content.
- Use natural sentence-boundary placement when possible. Put a reaction to the
  user's surprising or amusing statement near the beginning of the response;
  put laughter after a joke's punchline or at the end of the humorous sentence.
- Put sighs, breaths, coughs, and throat-clears at a natural pause or sentence
  boundary, not in the middle of a word or phrase. Choose the position based
  on the meaning and the active persona rather than applying a fixed placement.
- Use only these exact tags: {tag_reference}
- Return only the complete tagged response in the structured `text` field, with no
  preface, explanation, or replacement text.
"""
	try:
		result = generate_structured_output(
			model_name=os.getenv("MODEL_GENERALIST", ""),
			user_prompt=f"SOURCE RESPONSE (copy this text; insert tags only):\n{text}\nEND SOURCE RESPONSE",
			system_prompt=system_prompt,
			pydantic_model=_DecoratedSpeech,
			max_tokens=max(256, min(2048, len(text) + 256)),
			temperature=0.0,
		)
		# Preserve the editor's words exactly; only expand expression markers for
		# Supertonic's known tag-activation quirk.
		return _repeat_expression_tags(result.text)
	except Exception as exc:
		logger.warning(f"[tts] Expression editor failed ({exc}); speaking original response")
		return text
