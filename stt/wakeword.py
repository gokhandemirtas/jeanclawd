import os
import time

import sounddevice as sd

from utils.log import logger

_SAMPLE_RATE = 16000
_CHUNK_FRAMES = 1280   # 80ms — required by openwakeword
_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
_MODEL_NAME = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
_RETRY_DELAY_S = 5.0   # back off between mic-error retries

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


def _diagnose_mic_error(err: Exception) -> str:
    """Map a sounddevice/PortAudio error to a one-line user-facing hint."""
    msg = str(err)
    # PaErrorCode -9986 = paInternalError, -9985 = paDeviceUnavailable,
    # -9996 = paInvalidDevice. All mean the input device can't be opened.
    return (
        f"{msg}\n"
        "  • macOS: System Settings → Privacy & Security → Microphone — enable for your terminal.\n"
        "  • Another app may be holding the mic (Zoom, browser, etc.) — close it and retry.\n"
        "  • Run `uv run python -m stt.diagnose` to inspect the input device and RMS level."
    )


def listen_for_wakeword() -> None:
    """Block until the configured wake word is detected.

    On mic error, logs once with a diagnosis, then retries every
    _RETRY_DELAY_S seconds. Spinning back to the caller on every error
    would cause main.py to spam 'An error occurred' forever.
    """
    model = _get_model()
    while True:
        try:
            stream = sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16")
        except Exception as e:
            logger.error(
                f"Could not open microphone for wake-word listening.\n{_diagnose_mic_error(e)}"
            )
            logger.info(f"Retrying mic in {_RETRY_DELAY_S:.0f}s… (Ctrl+C to quit)")
            time.sleep(_RETRY_DELAY_S)
            continue
        try:
            with stream:
                while True:
                    data, _ = stream.read(_CHUNK_FRAMES)
                    scores = model.predict(data.flatten())
                    if scores.get(_MODEL_NAME, 0.0) >= _THRESHOLD:
                        model.reset()
                        return
        except Exception as e:
            logger.error(
                f"Microphone stream error during wake-word listening.\n{_diagnose_mic_error(e)}"
            )
            logger.info(f"Retrying mic in {_RETRY_DELAY_S:.0f}s… (Ctrl+C to quit)")
            time.sleep(_RETRY_DELAY_S)
