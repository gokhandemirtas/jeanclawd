import os
from .base import InferenceProvider

_provider: InferenceProvider | None = None

_BACKEND_LABELS: dict[str, str] = {
	"lmstudio": "LM Studio",
	"llamacpp": "llama.cpp",
	"openai_compatible": "OpenAI-compatible",
}


def backend_label() -> str:
	"""Human-readable name for the active inference backend."""
	backend = os.getenv("INFERENCE_PROVIDER", "lmstudio").lower()
	return _BACKEND_LABELS.get(backend, backend)


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
	elif backend == "openai_compatible":
		from .openai_compatible import OpenAICompatibleProvider
		_provider = OpenAICompatibleProvider(model)
	else:
		raise ValueError(
			f"Unknown INFERENCE_PROVIDER: {backend!r}. "
			"Choose 'lmstudio', 'llamacpp', or 'openai_compatible'."
		)

	return _provider
