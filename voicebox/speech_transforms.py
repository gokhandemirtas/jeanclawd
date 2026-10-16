"""Persona-specific text transforms applied before TTS synthesis.

The text passed here is for audio only — the user never sees it.
"""
from __future__ import annotations

import hashlib
import os
import re

_TAG_RE = re.compile(r"<(?:laugh|breath|sigh|sad|cough|scream|throatclear|angry|yawn)>")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

# ── Ztyntchtch ────────────────────────────────────────────────────────────────

_ALIEN_SYLLABLES: list[str] = [
    "tak",
    "rah",
    "hoom",
    "tch",
    "puch",
    "knik",
    "ssss",
    "pet",
    "ufhk",
    "pat",
    "kap",
    "kunh",
    "krap",
    "hon",
    "kem",
    "leh",
    "uch",
    "nah",
    "am",
    "om",
    "moh",
    "aro",
    "tep",
    "ten",
    "sop",
    "bam",
    "gon",
    "cop",
    "tsk tsk chp"
]

try:
    _ALIEN_RATIO = max(0.05, min(2.0, float(os.getenv("ALIEN_SPEECH_WORD_RATIO", "0.5"))))
except ValueError:
    _ALIEN_RATIO = 0.5


def alienize_for_speech(text: str) -> str:
    """Replace words with deterministic alien phonemes.

    The ratio of alien sound-words to source words is controlled by
    ALIEN_SPEECH_WORD_RATIO (default 0.5 — roughly half as many sounds as words).
    Expression tags are stripped so the alien voice uses only its own phonemes.
    """
    if not text.strip():
        return text

    clean = _TAG_RE.sub("", text)
    word_count = len(_WORD_RE.findall(clean))
    if word_count == 0:
        return text

    target_count = max(3, round(word_count * _ALIEN_RATIO))

    if target_count <= word_count:
        if target_count == 1:
            selected_words: set[int] = {word_count // 2}
        else:
            selected_words = {
                round(i * (word_count - 1) / (target_count - 1))
                for i in range(target_count)
            }
        words_per_source: list[int] | None = None
    else:
        selected_words = set(range(word_count))
        words_per_source = [
            target_count // word_count + (i < target_count % word_count)
            for i in range(word_count)
        ]

    word_index = 0

    def replace_word(match: re.Match[str]) -> str:
        nonlocal word_index
        word = match.group(0)
        source_index = word_index
        word_index += 1
        count = 1 if words_per_source is None and source_index in selected_words else (
            words_per_source[source_index] if words_per_source is not None else 0
        )
        if count == 0:
            return ""
        digest = hashlib.blake2b(f"{word_index}:{word}".encode(), digest_size=8).digest()
        parts = [
            re.sub(r"[^A-Za-z']+", "", _ALIEN_SYLLABLES[digest[o % len(digest)] % len(_ALIEN_SYLLABLES)])
            for o in range(1, count + 1)
        ]
        return " ".join(parts)

    return _WORD_RE.sub(replace_word, clean).strip()


# ── Mr Meowsky ────────────────────────────────────────────────────────────────

_CAT_SOUNDS: list[str] = [
    "meow", "mew", "mao", "maow", "miah", "miaow",
    "nyah", "nyao", "nyae",
    "purr", "pruh", "prrr",
    "raw", "raow", "yow", "yowl",
    "chirp", "chup", "mrr", "hrrm",
]


def catize_for_speech(text: str) -> str:
    """Replace words with deterministic cat vocalisations, preserving expression tags."""
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
        return " ".join(_CAT_SOUNDS[digest[o] % len(_CAT_SOUNDS)] for o in range(1, count + 1))

    segments = _TAG_RE.split(text)
    tags = _TAG_RE.findall(text)
    result: list[str] = []
    for i, segment in enumerate(segments):
        result.append(_WORD_RE.sub(replace_word, segment))
        if i < len(tags):
            result.append(tags[i])
    return "".join(result)


# ── dispatch ──────────────────────────────────────────────────────────────────

def transform_for_persona(text: str, persona_id: str) -> str:
    if persona_id == "ztyntchtch":
        return alienize_for_speech(text)
    if persona_id == "mr_meowsky":
        return catize_for_speech(text)
    return text
