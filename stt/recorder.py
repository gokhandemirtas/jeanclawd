import math
import tempfile

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


def _vu_bar(rms: float) -> str:
	"""Render a VU bar with dBFS and RMS for threshold tuning."""
	db = 20 * math.log10(max(rms, 1) / 32768)  # dBFS, 0 = full scale
	# Map -60..0 dBFS to 0.._BAR_WIDTH
	filled = max(0, min(_BAR_WIDTH, int((db + 60) / 60 * _BAR_WIDTH)))
	bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
	state = "SPEECH" if rms >= _SILENCE_RMS else "silent"
	return f"\r🎙  │{bar}│ {db:+.1f} dBFS  rms={rms:5.0f}  threshold={_SILENCE_RMS}  [{state}]   "


def record_until_silence() -> tuple[str, bool]:
	"""Record from the default mic, stopping after sustained silence.

	Returns (wav_path, too_short). Caller is responsible for deleting the file.
	too_short=True means the recording was below the minimum speech threshold.
	"""
	chunks: list[np.ndarray] = []
	silent_chunks = 0
	speech_chunks = 0

	with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
		while True:
			data, _ = stream.read(_CHUNK_FRAMES)
			rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
			chunks.append(data.copy())

			print(_vu_bar(rms), end="", flush=True)

			if rms < _SILENCE_RMS:
				silent_chunks += 1
				if speech_chunks >= _MIN_SPEECH_CHUNKS and silent_chunks >= _SILENCE_CHUNKS:
					break
			else:
				speech_chunks += 1
				silent_chunks = 0

	print()  # newline after VU bar
	audio = np.concatenate(chunks, axis=0)
	too_short = speech_chunks < _MIN_SPEECH_CHUNKS

	tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
	wav.write(tmp.name, _SAMPLE_RATE, audio)
	return tmp.name, too_short
