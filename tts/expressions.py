"""Supertonic expression tags and persona-aware speech decoration."""

import hashlib
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

# The alien voice gets a separate speech script. Keeping this deterministic makes
# retries sound consistent while still giving each source word a varied shape.
_ALIEN_SYLLABLES = ALIEN_SYLLABLES = (
	# Lovecraft-style UH
	"cthulu",
	"bab ah ghoul",
	"ch peth peth",
	"hun bah",
	"hula mnah",
	"thulak",
	# Lovecraft-style OH
	"bha rab hab",
	"oghul",
	"cthom kah",
	"shoggoth",
	# Lovecraft-style EH
	"fhtagn",
	"fhehl",
	"bah mhoul",
	"cthehl",
	"gheh",
	"BA'AL",
	"mef tunh",
	# Hybrid guttural
	"kthuloh",
	"gaht hul",
	"BRRR KAP KAP KUPUMMM!!!!",
	"ghthuno",
	"rrlyeh",
	"ch ch ch AKH lul!",
	# Your original vowel-extended base
	"ph ph tchaah acha",
	"tam hap",
	"ak lak mor",
	"ctacha",
	"lap hok",
	"paka nuh tah",
	"eckannn",
	"ka pap mak",
	"tata nok",
	"epa kh khar",
	"klopa ph ch",
	"man horkum",
	"ack tak",
	"bal ghora",
	"can deh lah",
	"khara hap",
	"aniyyaa! jerrrrr shak shak shak shak shak shak BLUP",
	"khora toom",
	"khula tah",
	"braka hok",
	"rruka",
	"ek ek ek",
	"thaka akh!",
	"ahk toh poht",
	"pakh glohhk",
	"tkula",
	"ukh blukh",
	"beh malaph",
	"ckolo",
	"cpet pet?",
	"kloka",
	"jorrrrrrr KHUNH!",
	"teke li li",
	"pckolohk ohkolokh",
)

_ALIEN_TAG_RE = re.compile(
	r"<(?:laugh|breath|sigh|sad|cough|scream|throatclear|angry|yawn)>"
)
_ALIEN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def alienize_for_speech(text: str) -> str:
	"""Render text as cryptic alien phonemes for audio only.

	Expression tags are retained so Supertonic can still perform laughs, breaths,
	and similar effects. The visible assistant response is never passed here.
	"""
	if not text.strip():
		return text

	word_index = 0

	def replace_word(match: re.Match[str]) -> str:
		nonlocal word_index
		word = match.group(0)
		digest = hashlib.blake2b(f"{word_index}:{word}".encode(), digest_size=8).digest()
		word_index += 1
		count = 2 + digest[0] % 3
		parts = [
			re.sub(
				r"[^A-Za-z']+",
				"",
				_ALIEN_SYLLABLES[digest[offset] % len(_ALIEN_SYLLABLES)],
			)
			for offset in range(1, count + 1)
		]
		return " ".join(parts)

	# Split around tags first so they remain exact and are not spoken as words.
	segments = _ALIEN_TAG_RE.split(text)
	tags = _ALIEN_TAG_RE.findall(text)
	phonemes: list[str] = []
	for index, segment in enumerate(segments):
		phonemes.append(_ALIEN_WORD_RE.sub(replace_word, segment))
		if index < len(tags):
			phonemes.append(tags[index])
	return "".join(phonemes)


_CAT_SOUNDS = (
	"meow",
	"merrk",
	"meoooow?",
	"mrrrkk",
	"meaaawww!!!!",
	"mrrrow",
	"brk. brkew?meowmeow mrk",
)


def catize_for_speech(text: str) -> str:
	"""Render text as meows and mrrks for Mr Meowsky's audio only."""
	if not text.strip():
		return text

	word_index = 0

	def replace_word(match: re.Match[str]) -> str:
		nonlocal word_index
		digest = hashlib.blake2b(
			f"cat:{word_index}:{match.group(0)}".encode(), digest_size=8
		).digest()
		word_index += 1
		count = 2 + digest[0] % 3
		return " ".join(
			_CAT_SOUNDS[digest[offset] % len(_CAT_SOUNDS)] for offset in range(1, count + 1)
		)

	segments = _ALIEN_TAG_RE.split(text)
	tags = _ALIEN_TAG_RE.findall(text)
	meows: list[str] = []
	for index, segment in enumerate(segments):
		meows.append(_ALIEN_WORD_RE.sub(replace_word, segment))
		if index < len(tags):
			meows.append(tags[index])
	return "".join(meows)


_PERSONA_GUIDANCE: dict[str, str] = {
	"jean": "warm, calm, friendly; use breath or an occasional laugh when it feels natural",
	"jerk": "sarcastic and superior; use sigh, yawn, throatclear, and laugh noticeably more often",
	"beth": "gloomy and melancholic; favor sigh, sad, breath, and occasional cough",
	"kimi": "bright and enthusiastic; favor laugh and breath",
	"borg": "logical and emotionless; use no tags unless one is genuinely necessary",
	"fred": "paranoid and alarmed; favor throatclear, cough, angry, and occasional sigh",
	"ztyntchtch": "cryptic alien observer; use sparse breaths and throat-clears around translated utterances",
	"mr_meowsky": "an expressive cat; use occasional meow-related vocal color",
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
	if not text.strip():
		return text

	persona_id = get_active_persona_id()
	if persona_id in {"ztyntchtch", "mr_meowsky"}:
		# Persona-specific vocalizations are intended for audio only. This path
		# also applies to remote TTS providers.
		if os.getenv("TTS_PROVIDER", "local") != "local":
			return (
				alienize_for_speech(text)
				if persona_id == "ztyntchtch"
				else catize_for_speech(text)
			)

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
		if os.getenv("TTS_PROVIDER", "local") != "local":
			return text
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
		decorated = _repeat_expression_tags(result.text)
		if persona_id == "ztyntchtch":
			return alienize_for_speech(decorated)
		if persona_id == "mr_meowsky":
			return catize_for_speech(decorated)
		return decorated
	except Exception as exc:
		logger.warning(
			f"[tts] Expression editor failed ({exc}); speaking fallback response"
		)
		if persona_id == "ztyntchtch":
			return alienize_for_speech(text)
		if persona_id == "mr_meowsky":
			return catize_for_speech(text)
		return text
