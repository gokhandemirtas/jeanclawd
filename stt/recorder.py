import math
import tempfile
import threading

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

_SAMPLE_RATE = 16000
_CHUNK_FRAMES = 1600  # 100ms per chunk
_SILENCE_RMS = 200  # below this → silence
_SILENCE_SECONDS = 1.5  # stop after this much continuous silence
_MIN_SPEECH_SECONDS = 0.3  # discard recordings shorter than this
_SILENCE_CHUNKS = int(_SILENCE_SECONDS * _SAMPLE_RATE / _CHUNK_FRAMES)
_MIN_SPEECH_CHUNKS = int(_MIN_SPEECH_SECONDS * _SAMPLE_RATE / _CHUNK_FRAMES)
_BAR_WIDTH = 20


def monitor_for_speech(
	stop_event: threading.Event,
	speech_event: threading.Event,
	*,
	cancel_event: threading.Event | None = None,
	start_chunks: int = 3,
) -> None:
	"""Set ``speech_event`` when microphone speech is detected.

	This lightweight monitor is used during TTS playback. It deliberately does
	not save audio; the normal recorder starts after playback is interrupted.
	Headset use keeps the assistant's output out of this signal in practice.
	"""
	consecutive_speech = 0
	try:
		with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
			while not stop_event.is_set():
				data, _ = stream.read(_CHUNK_FRAMES)
				rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
				if rms >= _SILENCE_RMS:
					consecutive_speech += 1
					if consecutive_speech >= start_chunks:
						speech_event.set()
						if cancel_event is not None:
							cancel_event.set()
						stop_event.set()
						return
				else:
					consecutive_speech = 0
	except Exception as e:
		# Barge-in is optional: a microphone failure must not break TTS.
		from utils.log import logger
		logger.warning(f"[stt] Barge-in microphone monitor unavailable: {e}")


def _vu_bar(rms: float) -> str:
	"""Render a VU bar with dBFS and RMS for threshold tuning."""
	db = 20 * math.log10(max(rms, 1) / 32768)  # dBFS, 0 = full scale
	# Map -60..0 dBFS to 0.._BAR_WIDTH
	filled = max(0, min(_BAR_WIDTH, int((db + 60) / 60 * _BAR_WIDTH)))
	bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
	state = "SPEECH" if rms >= _SILENCE_RMS else "silent"
	return f"\r🎙  │{bar}│ {db:+.1f} dBFS  rms={rms:5.0f}  threshold={_SILENCE_RMS}  [{state}]   "


def record_until_silence(
	*,
	wait_for_speech: bool = False,
	initial_silence_seconds: float = 8.0,
	max_recording_seconds: float = 120.0,
) -> tuple[str, bool]:
	"""Record from the default mic, stopping after sustained silence.

	When ``wait_for_speech`` is enabled, leading silence is not included in the
	recording and the function returns after ``initial_silence_seconds`` if no
	speech arrives. This is used by terminal conversation mode for follow-ups.

	Returns (wav_path, too_short). Caller is responsible for deleting the file.
	too_short=True means the recording was below the minimum speech threshold.
	"""
	chunks: list[np.ndarray] = []
	silent_chunks = 0
	speech_chunks = 0
	leading_silence_chunks = 0
	max_chunks = int(max_recording_seconds * _SAMPLE_RATE / _CHUNK_FRAMES)
	initial_silence_chunks = int(initial_silence_seconds * _SAMPLE_RATE / _CHUNK_FRAMES)

	with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
		for _ in range(max_chunks):
			data, _ = stream.read(_CHUNK_FRAMES)
			rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
			has_speech = rms >= _SILENCE_RMS
			if not wait_for_speech or speech_chunks or has_speech:
				chunks.append(data.copy())

			print(_vu_bar(rms), end="", flush=True)

			if not has_speech:
				if not speech_chunks:
					leading_silence_chunks += 1
					if wait_for_speech and leading_silence_chunks >= initial_silence_chunks:
						break
				silent_chunks += 1
				if speech_chunks >= _MIN_SPEECH_CHUNKS and silent_chunks >= _SILENCE_CHUNKS:
					break
			else:
				speech_chunks += 1
				silent_chunks = 0
				leading_silence_chunks = 0

	print()  # newline after VU bar
	if not chunks:
		chunks = [np.zeros((_CHUNK_FRAMES, 1), dtype=np.int16)]
	audio = np.concatenate(chunks, axis=0)
	too_short = speech_chunks < _MIN_SPEECH_CHUNKS

	tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
	wav.write(tmp.name, _SAMPLE_RATE, audio)
	return tmp.name, too_short
