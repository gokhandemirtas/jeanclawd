"""Thin client for the jeanclawd-voicebox API server.

All speech synthesis and transcription is delegated to the external voicebox
process; this module owns voice state (active voice, enable flag) and provides
the same public surface that the rest of core previously imported from tts/ and stt/.

Environment variables:
    VOICEBOX_HOST   base URL of the voicebox server (default: http://localhost:8000)
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import tempfile
import threading

import requests

from utils import logger

VOICEBOX_HOST: str = os.getenv("VOICEBOX_HOST", "http://localhost:8000").rstrip("/")
VOICEBOX_TIMEOUT: int = int(os.getenv("VOICEBOX_TIMEOUT", "60"))

# VOICEBOX_BACKEND selects which speaker-code table to use.
# Set to "supertonic" if voicebox falls back to the ONNX model.
VOICEBOX_BACKEND: str = os.getenv("VOICEBOX_BACKEND", "qwen").lower()

# Friendly voice names exposed to the rest of the app.
voice_ids: list[str] = [
    "thomas", "james", "michael", "oliver", "lucas", "ethan",
    "anneke", "sophia", "emma", "clara", "lily", "nora",
]
DEFAULT_VOICE = "thomas"

# Friendly name → Qwen speaker
_QWEN_VOICE_MAP: dict[str, str] = {
    "thomas": "ryan",     "james": "aiden",    "michael": "eric",
    "oliver": "dylan",    "lucas": "uncle_fu",  "ethan": "ryan",
    "anneke": "serena",   "sophia": "vivian",   "emma": "ono_anna",
    "clara":  "sohee",    "lily":  "serena",    "nora":  "vivian",
}

# Friendly name → Supertonic speaker (fallback backend)
_SUPERTONIC_VOICE_MAP: dict[str, str] = {
    "thomas": "M1", "james": "M2", "michael": "M3",
    "oliver": "M4", "lucas": "M5", "ethan":   "M6",
    "anneke": "F1", "sophia": "F2", "emma":    "F3",
    "clara":  "F4", "lily":  "F5", "nora":    "F6",
}


def _resolve_speaker(voice_id: str) -> str:
    table = _QWEN_VOICE_MAP if VOICEBOX_BACKEND == "qwen" else _SUPERTONIC_VOICE_MAP
    return table.get(voice_id.lower(), voice_id)

active_voice_id: str = DEFAULT_VOICE
_tts_enabled: bool = False

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002702-\U000027B0"
    "]+",
    flags=re.UNICODE,
)


# ── state management ──────────────────────────────────────────────────────────

def enable_tts(enabled: bool = True) -> None:
    global _tts_enabled
    _tts_enabled = enabled


def is_tts_enabled() -> bool:
    return _tts_enabled


def get_active_voice_id() -> str:
    return active_voice_id


def set_active_voice_id(voice_id: str) -> None:
    global active_voice_id
    if not voice_id:
        return
    if voice_id not in voice_ids:
        logger.warning(f"[voicebox] Unknown voice '{voice_id}', keeping '{active_voice_id}'")
        return
    active_voice_id = voice_id


# ── STT ───────────────────────────────────────────────────────────────────────

def transcribe(audio_path: str) -> str:
    """POST audio file to voicebox /transcribe and return the transcript text."""
    url = f"{VOICEBOX_HOST}/transcribe"
    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=VOICEBOX_TIMEOUT,
        )
    resp.raise_for_status()
    return resp.json()["text"].strip()


# ── TTS ───────────────────────────────────────────────────────────────────────

def synthesize(text: str, voice_id: str | None = None, stream: bool = False) -> bytes:
    """POST to voicebox /synthesize and return WAV bytes (stream=False) or raw PCM."""
    url = f"{VOICEBOX_HOST}/synthesize"
    payload = {
        "text": text,
        "persona": "overclocked_ai",
        "stream": stream,
    }
    resp = requests.post(url, json=payload, timeout=VOICEBOX_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def _sanitize(text: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = _EMOJI_RE.sub("", text)
    return " ".join(text.split())


def _play_wav_file(path: str, stop_event: threading.Event) -> None:
    """Play a WAV file via a system command; honours stop_event for cancellation."""
    system = platform.system()
    if system == "Darwin":
        cmd = ["afplay", path]
    elif system == "Linux":
        cmd = ["aplay", path]
    else:
        logger.warning("[voicebox] Audio playback not supported on this platform")
        return

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if stop_event.is_set():
            proc.terminate()
            proc.wait(timeout=1.0)
            return
        threading.Event().wait(0.05)
    proc.wait()


def speak(
    text: str,
    voice_id: str | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Synthesize text via voicebox and play locally. Cancellable via cancel_event."""
    text = _sanitize(text)
    if not text:
        logger.warning("[voicebox] speak() called with empty text, skipping")
        return

    stop_event = cancel_event or threading.Event()
    try:
        wav_bytes = synthesize(text, voice_id=voice_id, stream=False)
    except Exception as exc:
        logger.error(f"[voicebox] TTS synthesis failed: {exc}")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        _play_wav_file(tmp_path, stop_event)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
