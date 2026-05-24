import tempfile
import threading

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

_SAMPLE_RATE = 16000  # Whisper expects 16kHz
_CHANNELS = 1
_SILENCE_THRESHOLD = 200  # RMS below this → treat as silence


def record_until_enter() -> tuple[str, bool]:
    """Record from the default microphone until the user presses Enter.

    Returns (wav_path, is_silent). Caller is responsible for deleting the file.
    is_silent=True means the audio is below the silence threshold — skip transcription.
    """
    chunks: list[np.ndarray] = []

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=_SAMPLE_RATE,
        channels=_CHANNELS,
        dtype="int16",
        callback=_callback,
    )

    with stream:
        input()  # blocks until Enter

    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((1, 1), dtype="int16")
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    is_silent = rms < _SILENCE_THRESHOLD

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, _SAMPLE_RATE, audio)
    return tmp.name, is_silent
