import os

import requests

from stt.base import STTProvider


class RemoteSTTProvider(STTProvider):
    """Transcribes audio by POSTing to a remote OpenAI-compatible Whisper endpoint.

    Compatible with faster-whisper-server, whisper.cpp server, and any service
    that mirrors the OpenAI /v1/audio/transcriptions API.

    Required env var:
        STT_HOST   base URL of the remote server, e.g. http://192.168.1.50:8000

    Optional env vars:
        STT_MODEL      model name to request (default: "whisper-1")
        STT_API_KEY    bearer token if the server requires auth (default: none)
        STT_LANGUAGE   ISO-639-1 language hint, e.g. "en" (default: "en")
        STT_TIMEOUT    request timeout in seconds (default: 60)
    """

    def __init__(self) -> None:
        host = os.environ.get("STT_HOST", "").rstrip("/")
        if not host:
            raise ValueError(
                "STT_HOST is required for STT_PROVIDER=remote. "
                "Set it to the base URL of your Whisper server, "
                "e.g. http://192.168.1.50:8000"
            )
        self._url = f"{host}/v1/audio/transcriptions"
        self._model = os.environ.get("STT_MODEL", "whisper-1")
        self._language = os.environ.get("STT_LANGUAGE", "en")
        self._timeout = int(os.environ.get("STT_TIMEOUT", "60"))
        api_key = os.environ.get("STT_API_KEY", "")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                self._url,
                headers=self._headers,
                files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                data={"model": self._model, "language": self._language},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json()["text"].strip()
