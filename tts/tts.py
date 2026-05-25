import contextlib
import importlib.util
import io
import os
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
TTS_LANG = os.getenv("TTS_LANG", "en-us")


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
_effective_model_id: str = getattr(_adapter, "MODEL_ID", TTS_MODEL or "kokoro")
_adapter_default_voice: str = getattr(_adapter, "DEFAULT_VOICE", "af_heart")


def set_spinner(enabled: bool) -> None:
	global _show_spinner
	_show_spinner = enabled


voice_ids = [
	"alba",
	"marius",
	"javert",
	"jean",
	"fantine",
	"cosette",
	"eponine",
	"azelma",
]

# Maps internal persona voice names → Kokoro voice IDs
_KOKORO_VOICE_MAP: dict[str, str] = {
	"alba": "bf_emma",  # British female, warm
	"marius": "am_michael",  # American male, young
	"javert": "bm_lewis",  # British male, stern
	"jean": "bm_george",  # British male, authoritative
	"fantine": "af_bella",  # American female, emotive
	"cosette": "af_nicole",  # American female, bright
	"eponine": "af_sky",  # American female, distinct
	"azelma": "af_heart",  # American female, default
}


def _resolve_kokoro_voice(voice_id: str) -> str:
	return _KOKORO_VOICE_MAP.get(voice_id, voice_id)


active_voice_id = _adapter_default_voice if TTS_PROVIDER != "local" else "azelma"
_tts_enabled = False


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

	payload = {
		"text": text,
		"voice": _resolve_kokoro_voice(voice_id),
		"speed": 1.8,
		"lang": TTS_LANG,
	}
	response = requests.post(TTS_HOST, json=payload, timeout=60)
	response.raise_for_status()
	data = response.content
	if not data.startswith(b"RIFF"):
		raise RuntimeError(f"TTS endpoint returned unexpected data: {data[:200]}")
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


# ── Local (pocket_tts) backend ────────────────────────────────────────────────


def _local_synthesise(
	text: str, output_path: str, voice_id: str, cancel_event: threading.Event
) -> str:
	from pocket_tts import TTSModel

	tts_model = TTSModel.load_model(temp=0.5, lsd_decode_steps=7, eos_threshold=-1.0)
	voice_state = tts_model.get_state_for_audio_prompt(voice_id)
	chunks = tts_model.generate_audio_stream(voice_state, text_to_generate=text)

	all_audio = []
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
		for chunk in chunks:
			if cancel_event.is_set():
				break
			all_audio.append(chunk.numpy())

	if not all_audio:
		return output_path

	combined = np.concatenate(all_audio)
	scipy.io.wavfile.write(output_path, tts_model.sample_rate, combined)
	return output_path


def _local_speak(text: str, voice_id: str, stop_event: threading.Event) -> None:
	from pocket_tts import TTSModel

	tts_model = TTSModel.load_model(temp=0.5, lsd_decode_steps=7, eos_threshold=-1.0)
	sample_rate = tts_model.sample_rate
	voice_state = tts_model.get_state_for_audio_prompt(voice_id)
	chunks = tts_model.generate_audio_stream(voice_state, text_to_generate=text)

	spinner_ctx = (
		Live(
			Spinner("dots", text=Text("Synthesising audio…", style="dim")),
			console=_console,
			transient=True,
		)
		if _show_spinner
		else contextlib.nullcontext()
	)
	with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
		with spinner_ctx:
			for chunk in chunks:
				if stop_event.is_set():
					break
				stream.write(chunk.numpy().reshape(-1, 1))


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

	payload = {
		"text": text,
		"voice": _resolve_kokoro_voice(voice_id),
		"speed": 1.0,
		"lang": TTS_LANG,
	}
	response = requests.post(TTS_HOST, json=payload, timeout=60)
	response.raise_for_status()
	_play_wav_bytes(response.content, stop_event)


def speak(text: str, voice_id: str = active_voice_id):
	if not text:
		logger.warning("No text provided for TTS, skipping")
		return

	stop_event = threading.Event()
	listener = threading.Thread(target=_escape_listener, args=(stop_event,), daemon=True)
	listener.start()

	try:
		start_time = time.time()

		if TTS_PROVIDER == "local":
			_local_speak(text, voice_id, stop_event)
		else:
			_remote_stream_speak(text, voice_id, stop_event)

		if stop_event.is_set():
			logger.debug("TTS interrupted by user")
		else:
			logger.debug(f"Voice generation finished in {time.time() - start_time:.2f}s")

	except Exception as e:
		logger.error(str(e))
	finally:
		stop_event.set()
		listener.join(timeout=0.5)


if __name__ == "__main__":
	speak(
		"""We offer commercial support for teams integrating Kitten TTS into their products. This includes integration assistance, custom voice development, and enterprise licensing."""
	)
