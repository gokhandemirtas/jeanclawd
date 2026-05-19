import os

from stt.base import STTProvider


class WhisperProvider(STTProvider):
    def __init__(self) -> None:
        import whisper
        model_name = os.getenv("STT_MODEL", "base")
        self._model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        return self._model.transcribe(audio_path)["text"].strip()
