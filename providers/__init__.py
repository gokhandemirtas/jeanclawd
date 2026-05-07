import os
from .base import InferenceProvider

_provider: InferenceProvider | None = None


def get_provider() -> InferenceProvider:
    global _provider
    if _provider is not None:
        return _provider

    backend = os.getenv("INFERENCE_PROVIDER", "lmstudio").lower()
    model = os.getenv("MODEL_GENERALIST", "")

    if backend == "lmstudio":
        from .lmstudio import LMStudioProvider
        _provider = LMStudioProvider(model)
    elif backend == "llamacpp":
        from .llamacpp import LlamaCppProvider
        _provider = LlamaCppProvider(model)
    else:
        raise ValueError(f"Unknown INFERENCE_PROVIDER: {backend!r}. Choose 'lmstudio' or 'llamacpp'.")

    return _provider
