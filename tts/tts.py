import os
import select
import termios
import threading
import time
import tty

import numpy as np
import scipy.io.wavfile
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from utils import logger
from pocket_tts import TTSModel

_console = Console(stderr=True, force_terminal=True)

voice_ids = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
active_voice_id = "azelma"
_tts_enabled = False


def enable_tts(enabled: bool = True) -> None:
    global _tts_enabled
    _tts_enabled = enabled


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
                if ch == b'\x1b':
                    stop_event.set()
                    break
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def synthesise(text: str, output_path: str, voice_id: str = active_voice_id, cancel_event: threading.Event | None = None) -> str:
    if not text:
        raise ValueError("No text provided for TTS")

    logger.debug(f"Synthesizing {text}")
    owned = cancel_event is None
    if owned:
        cancel_event = threading.Event()
        listener = threading.Thread(target=_escape_listener, args=(cancel_event,), daemon=True)
        listener.start()
    else:
        listener = None

    tts_model = TTSModel.load_model(temp=0.5, lsd_decode_steps=7, eos_threshold=-1.0)
    voice_state = tts_model.get_state_for_audio_prompt(voice_id)
    chunks = tts_model.generate_audio_stream(voice_state, text_to_generate=text)

    all_audio = []
    try:
        with Live(Spinner("dots", text=Text("Synthesising audio…", style="dim")), console=_console, transient=True):
            for chunk in chunks:
                if cancel_event.is_set():
                    break
                all_audio.append(chunk.numpy())
    finally:
        if owned:
            cancel_event.set()
        if listener:
            listener.join(timeout=0.5)

    if not all_audio:
        return output_path

    combined = np.concatenate(all_audio)
    scipy.io.wavfile.write(output_path, tts_model.sample_rate, combined)
    logger.debug(f"TTS audio saved to {output_path}")
    return output_path


def speak(text: str, voice_id: str = active_voice_id):
    if not text:
        logger.warning("No text provided for TTS, skipping")
        return

    stop_event = threading.Event()
    listener = threading.Thread(target=_escape_listener, args=(stop_event,), daemon=True)
    listener.start()

    tts_model = TTSModel.load_model(temp=0.5, lsd_decode_steps=7, eos_threshold=-1.0)
    sample_rate = tts_model.sample_rate

    try:
        start_time = time.time()
        voice_state = tts_model.get_state_for_audio_prompt(voice_id)
        chunks = tts_model.generate_audio_stream(voice_state, text_to_generate=text)

        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            with Live(Spinner("dots", text=Text("Synthesising audio…", style="dim")), console=_console, transient=True):
                for chunk in chunks:
                    if stop_event.is_set():
                        break
                    stream.write(chunk.numpy().reshape(-1, 1))

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
    speak("""We offer commercial support for teams integrating Kitten TTS into their products. This includes integration assistance, custom voice development, and enterprise licensing.""")
