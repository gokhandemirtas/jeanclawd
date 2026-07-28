import contextlib
import importlib.util
import io
import os
import re
import select
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from utils import logger

_console = Console(stderr=True, force_terminal=True)
_show_spinner = True

# TTS_PROVIDER: local | lmstudio | remote
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "local")
# TTS_MODEL: adapter filename (without .py) under tts/models/
TTS_MODEL = os.getenv("TTS_MODEL", "")
_lmstudio_host = os.getenv("MODEL_HOST", "http://localhost:1234/v1")
TTS_HOST = os.getenv(
	"TTS_HOST",
	_lmstudio_host if TTS_PROVIDER == "lmstudio" else "https://api.openai.com/v1",
)
TTS_API_KEY = os.getenv("TTS_API_KEY", os.getenv("OPENAI_API_KEY", "lm-studio"))
TTS_LANG = os.getenv("TTS_LANG", "en").split("-")[0]


def _load_model_adapter():
	"""Load tts/models/{TTS_MODEL}.py if it exists. Returns the module or None."""
	if not TTS_MODEL:
		return None
	adapter_path = Path(__file__).parent / "models" / f"{TTS_MODEL}.py"
	if not adapter_path.exists():
		return None
	spec = importlib.util.spec_from_file_location(f"tts_model_{TTS_MODEL}", adapter_path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


_adapter = _load_model_adapter()

# Resolve effective model ID and default voice from adapter (or env fallbacks)
_effective_model_id: str = getattr(_adapter, "MODEL_ID", TTS_MODEL or "supertonic-3")
_adapter_default_voice: str = getattr(_adapter, "DEFAULT_VOICE", "M1")


def set_spinner(enabled: bool) -> None:
	global _show_spinner
	_show_spinner = enabled


voice_ids = [
	"thomas",
	"james",
	"michael",
	"oliver",
	"lucas",
	"ethan",
	"anneke",
	"sophia",
	"emma",
	"clara",
	"lily",
	"nora",
]

# Friendly application voice names → Supertonic voice styles.
SUPERTONIC_VOICE_MAP: dict[str, str] = {
	"thomas": "M1",
	"james": "M2",
	"michael": "M3",
	"oliver": "M4",
	"lucas": "M5",
	"ethan": "M6",
	"anneke": "F1",
	"sophia": "F2",
	"emma": "F3",
	"clara": "F4",
	"lily": "F5",
	"nora": "F6",
}


def _resolve_voice(voice_id: str) -> str:
	return SUPERTONIC_VOICE_MAP.get(voice_id.lower(), voice_id)


active_voice_id = _adapter_default_voice if TTS_PROVIDER != "local" else "thomas"
_tts_enabled = False

_local_tts = None
_local_tts_lock = threading.RLock()


def _get_local_tts():
	"""Load Supertonic once; its model and voice styles are reused by all callers."""
	global _local_tts
	if _local_tts is None:
		with _local_tts_lock:
			if _local_tts is None:
				from supertonic import TTS

				_local_tts = TTS(auto_download=True)
	return _local_tts


def _synthesise_local(text: str, voice_id: str, cancel_event: threading.Event):
	if cancel_event.is_set():
		return None, None
	with _local_tts_lock:
		tts = _get_local_tts()
		style = tts.get_voice_style(voice_name=_resolve_voice(voice_id))
		wav, _duration = tts.synthesize(
			text=text,
			voice_style=style,
			total_steps=8,
			speed=1.05,
			lang=TTS_LANG,
			verbose=False,
		)
	return np.asarray(wav).squeeze(), 44100


def enable_tts(enabled: bool = True) -> None:
	global _tts_enabled
	_tts_enabled = enabled


def set_active_voice_id(voice_id: str) -> None:
	global active_voice_id
	if not voice_id:
		return
	adapter_voices = getattr(_adapter, "VOICES", {})
	if TTS_PROVIDER != "local" and adapter_voices and voice_id not in adapter_voices:
		logger.warning(
			f"[tts] Voice '{voice_id}' not in adapter voices, keeping '{active_voice_id}'"
		)
		return
	active_voice_id = voice_id


def is_tts_enabled() -> bool:
	return _tts_enabled


def _escape_listener(stop_event: threading.Event) -> None:
	"""Background thread: sets stop_event if Escape is pressed."""
	fd = 0  # stdin
	old_settings = termios.tcgetattr(fd)
	try:
		tty.setraw(fd)
		while not stop_event.is_set():
			if select.select([fd], [], [], 0.05)[0]:
				ch = os.read(fd, 1)
				if ch == b"\x1b":
					stop_event.set()
					break
	except Exception:
		pass
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _get_openai_client():
	import openai

	return openai.OpenAI(api_key=TTS_API_KEY, base_url=TTS_HOST)


def _remote_synthesise_bytes(text: str, voice_id: str) -> bytes:
	import requests

	start_time = time.time()
	payload = {
		"text": text,
		"voice": _resolve_voice(voice_id),
		"speed": 1.8,
		"lang": TTS_LANG,
	}
	response = requests.post(TTS_HOST, json=payload, timeout=60)
	response.raise_for_status()
	data = response.content
	if not data.startswith(b"RIFF"):
		raise RuntimeError(f"TTS endpoint returned unexpected data: {data[:200]}")
	logger.debug(f"Voice generation finished in {time.time() - start_time:.2f}s")
	return data


def _play_wav_bytes(wav_bytes: bytes, stop_event: threading.Event) -> None:
	buf = io.BytesIO(wav_bytes)
	sample_rate, data = scipy.io.wavfile.read(buf)
	if data.dtype != np.float32:
		data = data.astype(np.float32) / np.iinfo(data.dtype).max
	if data.ndim > 1:
		data = data[:, 0]
	chunk_size = 1024
	with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
		for i in range(0, len(data), chunk_size):
			if stop_event.is_set():
				break
			stream.write(data[i : i + chunk_size].reshape(-1, 1))


# ── Local (Supertonic) backend ────────────────────────────────────────────────


def _local_synthesise(
	text: str, output_path: str, voice_id: str, cancel_event: threading.Event
) -> str:
	start_time = time.time()
	ctx = (
		Live(
			Spinner("dots", text=Text("Synthesising audio…", style="dim")),
			console=_console,
			transient=True,
		)
		if _show_spinner
		else contextlib.nullcontext()
	)
	with ctx:
		combined, sample_rate = _synthesise_local(text, voice_id, cancel_event)
	logger.debug(f"Voice generation finished in {time.time() - start_time:.2f}s")

	if combined is None or cancel_event.is_set():
		return output_path

	scipy.io.wavfile.write(output_path, sample_rate, combined.astype(np.float32))
	return output_path


def _local_speak(text: str, voice_id: str, stop_event: threading.Event) -> None:
	start_time = time.time()
	spinner_ctx = (
		Live(
			Spinner("dots", text=Text("Synthesising audio…", style="dim")),
			console=_console,
			transient=True,
		)
		if _show_spinner
		else contextlib.nullcontext()
	)
	with spinner_ctx:
		wav, sample_rate = _synthesise_local(text, voice_id, stop_event)
	logger.debug(f"Voice generation finished in {time.time() - start_time:.2f}s")
	if wav is None or stop_event.is_set():
		return
	with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
		for i in range(0, len(wav), 1024):
			if stop_event.is_set():
				break
			stream.write(wav[i : i + 1024].astype(np.float32).reshape(-1, 1))


# ── Public API ────────────────────────────────────────────────────────────────


def synthesise(
	text: str,
	output_path: str,
	voice_id: str = active_voice_id,
	cancel_event: threading.Event | None = None,
) -> str:
	if not text:
		raise ValueError("No text provided for TTS")

	logger.debug(f"Synthesizing via provider={TTS_PROVIDER}: {text[:60]}")
	owned = cancel_event is None
	if owned:
		cancel_event = threading.Event()
		listener = threading.Thread(
			target=_escape_listener, args=(cancel_event,), daemon=True
		)
		listener.start()
	else:
		listener = None

	try:
		if TTS_PROVIDER == "local":
			_local_synthesise(text, output_path, voice_id, cancel_event)
		else:
			if not cancel_event.is_set():
				ctx = (
					Live(
						Spinner("dots", text=Text("Synthesising audio…", style="dim")),
						console=_console,
						transient=True,
					)
					if _show_spinner
					else contextlib.nullcontext()
				)
				with ctx:
					wav_bytes = _remote_synthesise_bytes(text, voice_id)
				if not cancel_event.is_set():
					with open(output_path, "wb") as f:
						f.write(wav_bytes)
	finally:
		if owned:
			cancel_event.set()
		if listener:
			listener.join(timeout=0.5)

	logger.debug(f"TTS audio saved to {output_path}")
	return output_path


def _remote_stream_speak(text: str, voice_id: str, stop_event: threading.Event) -> None:
	import requests

	start_time = time.time()
	payload = {
		"text": text,
		"voice": _resolve_voice(voice_id),
		"speed": 1.8,
		"lang": TTS_LANG,
	}
	response = requests.post(TTS_HOST, json=payload, timeout=60)
	response.raise_for_status()
	logger.debug(f"Voice generation finished in {time.time() - start_time:.2f}s")
	_play_wav_bytes(response.content, stop_event)


def _sanitize_for_speech(text: str) -> str:
	# markdown images: ![alt](url) → drop entirely
	text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
	# markdown links: [label](url) → keep label only
	text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
	# bare URLs
	text = re.sub(r"https?://\S+", "", text)
	# leftover punctuation from removed URLs (e.g. trailing commas, parens)
	text = re.sub(r"\s([,;:])", r"\1", text)
	return " ".join(text.split())


def speak(
	text: str,
	voice_id: str = active_voice_id,
	cancel_event: threading.Event | None = None,
):
	text = _sanitize_for_speech(text)
	if not text:
		logger.warning("No text provided for TTS, skipping")
		return

	stop_event = cancel_event or threading.Event()
	listener = threading.Thread(target=_escape_listener, args=(stop_event,), daemon=True)
	listener.start()

	try:
		if TTS_PROVIDER == "local":
			_local_speak(text, voice_id, stop_event)
		else:
			_remote_stream_speak(text, voice_id, stop_event)

		if stop_event.is_set():
			logger.debug("TTS interrupted by user")

	except Exception as e:
		logger.error(str(e))
	finally:
		stop_event.set()
		listener.join(timeout=0.5)


if __name__ == "__main__":
	speak(
		"""We offer commercial support for teams integrating Kitten TTS into their products. This includes integration assistance, custom voice development, and enterprise licensing."""
	)
