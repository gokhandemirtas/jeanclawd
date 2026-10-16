"""Continuous mic listener: VAD → speech capture → voicebox transcription."""
from __future__ import annotations

import os
import tempfile
import threading
import wave
from collections import deque
from typing import Callable

import webrtcvad

from utils.log import logger

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
BYTES_PER_SAMPLE = 2
FRAME_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE

# Consecutive speech frames required to start recording (~180ms)
_SPEECH_START = 6
# Consecutive silent frames required to end an utterance (~600ms)
_SILENCE_END = 20
# Minimum utterance frames before transcribing (~300ms) — drops mouth noise bursts
_MIN_SPEECH_FRAMES = 10
# Hard cap: ~30 seconds
_MAX_FRAMES = 1000
# Minimum RMS to count a frame as speech (gates very quiet noise)
_RMS_GATE = 0.01


def _rms(frame_bytes: bytes) -> float:
    import struct
    count = len(frame_bytes) // 2
    if not count:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame_bytes)
    return min(1.0, (sum(s * s for s in samples) / count) ** 0.5 / 32768.0)


class ContinuousListener:
    """Record speech segments from the mic and fire on_transcript for each utterance."""

    def __init__(
        self,
        on_transcript: Callable[[str], None],
        on_level: Callable[[float], None] | None = None,
        on_state: Callable[[str], None] | None = None,
        on_speech_detected: Callable[[], None] | None = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_level = on_level
        self._on_state = on_state
        self._on_speech_detected = on_speech_detected
        self._vad = webrtcvad.Vad(3)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = "idle"

    # ── public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mic-listener")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._set_state("idle")

    # ── internals ─────────────────────────────────────────────────────────────

    def _set_state(self, state: str) -> None:
        self.state = state
        if self._on_state:
            self._on_state(state)

    def _run(self) -> None:
        import sounddevice as sd

        self._set_state("waiting")
        pre: deque[bytes] = deque(maxlen=10)  # rolling pre-speech buffer
        utterance: list[bytes] = []
        speech_count = 0
        silence_count = 0
        recording = False

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
            ) as stream:
                while not self._stop.is_set():
                    data, _ = stream.read(FRAME_SAMPLES)
                    frame = bytes(data)

                    if self._on_level:
                        self._on_level(_rms(frame))

                    try:
                        speech = self._vad.is_speech(frame, SAMPLE_RATE) and _rms(frame) >= _RMS_GATE
                    except Exception:
                        speech = False

                    if not recording:
                        pre.append(frame)
                        speech_count = (speech_count + 1) if speech else 0
                        if speech_count >= _SPEECH_START:
                            recording = True
                            silence_count = 0
                            utterance = list(pre)
                            self._set_state("speaking")
                            if self._on_speech_detected:
                                self._on_speech_detected()
                    else:
                        utterance.append(frame)
                        if speech:
                            silence_count = 0
                        else:
                            silence_count += 1

                        if silence_count >= _SILENCE_END or len(utterance) >= _MAX_FRAMES:
                            recording = False
                            speech_count = 0
                            pre.clear()
                            speech_frames = len(utterance) - silence_count
                            if speech_frames >= _MIN_SPEECH_FRAMES:
                                self._set_state("transcribing")
                                self._transcribe(utterance)
                            utterance = []
                            self._set_state("waiting")

        except Exception as exc:
            logger.error(f"[listener] mic error: {exc}")
        finally:
            self._set_state("idle")

    def _transcribe(self, frames: list[bytes]) -> None:
        from voicebox.client import transcribe

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(BYTES_PER_SAMPLE)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))

            text = transcribe(tmp_path)
            if text:
                self._on_transcript(text)
        except Exception as exc:
            logger.error(f"[listener] transcription error: {exc}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
