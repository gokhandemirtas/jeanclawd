import os

from stt.base import STTProvider


class MLXWhisperProvider(STTProvider):
    def __init__(self) -> None:
        self._repo = os.getenv("STT_MODEL", "mlx-community/whisper-large-v3-turbo")

    def transcribe(self, audio_path: str) -> str:
        import mlx_whisper
        return mlx_whisper.transcribe(audio_path, path_or_hf_repo=self._repo)["text"].strip()
