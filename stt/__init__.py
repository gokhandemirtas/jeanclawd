import os

from stt.base import STTProvider

_provider: STTProvider | None = None

_BACKEND_LABELS: dict[str, str] = {
    "whisper": "OpenAI Whisper",
    "mlx": "MLX Whisper",
}


def backend_label() -> str:
    backend = os.getenv("STT_PROVIDER", "whisper").lower()
    return _BACKEND_LABELS.get(backend, backend)


def get_stt_provider() -> STTProvider:
    global _provider
    if _provider is None:
        backend = os.getenv("STT_PROVIDER", "whisper").lower()
        if backend == "whisper":
            from stt.whisper_provider import WhisperProvider
            _provider = WhisperProvider()
        elif backend == "mlx":
            from stt.mlx_provider import MLXWhisperProvider
            _provider = MLXWhisperProvider()
        else:
            raise ValueError(
                f"Unknown STT_PROVIDER: {backend!r}. Choose 'whisper' or 'mlx'."
            )
    return _provider
