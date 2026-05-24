import os

import numpy as np
import sounddevice as sd

_SAMPLE_RATE = 16000
_CHUNK_FRAMES = 1280   # 80ms — required by openwakeword
_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
_MODEL_NAME = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")

_model = None


def _get_model():
    global _model
    if _model is None:
        from openwakeword.model import Model
        _model = Model(wakeword_models=[_MODEL_NAME], inference_framework="onnx")
    return _model


def preload() -> None:
    """Download (if needed) and load the wake word model at startup."""
    import openwakeword
    openwakeword.utils.download_models([_MODEL_NAME])
    _get_model()


def listen_for_wakeword() -> None:
    """Block until the configured wake word is detected."""
    model = _get_model()
    with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            data, _ = stream.read(_CHUNK_FRAMES)
            scores = model.predict(data.flatten())
            if scores.get(_MODEL_NAME, 0.0) >= _THRESHOLD:
                model.reset()
                return
