import os
from .base import InferenceProvider

_provider: InferenceProvider | None = None
_intent_provider: InferenceProvider | None = None

_BACKEND_LABELS: dict[str, str] = {
	"lmstudio": "LM Studio",
	"llamacpp": "llama.cpp",
	"openai_compatible": "OpenAI-compatible",
	"openai": "OpenAI",
}


def backend_label() -> str:
	"""Human-readable name for the active inference backend."""
	backend = os.getenv("INFERENCE_PROVIDER", "lmstudio").lower()
	return _BACKEND_LABELS.get(backend, backend)


def _make_provider(model: str) -> InferenceProvider:
	backend = os.getenv("INFERENCE_PROVIDER", "lmstudio").lower()
	if backend == "lmstudio":
		from .lmstudio import LMStudioProvider
		return LMStudioProvider(model)
	elif backend == "llamacpp":
		from .llamacpp import LlamaCppProvider
		return LlamaCppProvider(model)
	elif backend == "openai_compatible":
		from .openai_compatible import OpenAICompatibleProvider
		return OpenAICompatibleProvider(model)
	elif backend == "openai":
		from .openai import OpenAIProvider
		return OpenAIProvider(model)
	else:
		raise ValueError(
			f"Unknown INFERENCE_PROVIDER: {backend!r}. "
			"Choose 'lmstudio', 'llamacpp', 'openai_compatible', or 'openai'."
		)


def get_provider() -> InferenceProvider:
	global _provider
	if _provider is None:
		_provider = _make_provider(os.getenv("MODEL_GENERALIST", ""))
	return _provider


def get_intent_provider() -> InferenceProvider:
	"""Lean provider for the intent classifier.

	When MODEL_INTENT is set to a different model than MODEL_GENERALIST, creates
	a dedicated lean instance. Otherwise reuses the main provider to avoid
	duplicate handles to the same model.
	"""
	global _intent_provider
	if _intent_provider is not None:
		return _intent_provider

	intent_model = os.getenv("MODEL_INTENT", "").strip()
	generalist_model = os.getenv("MODEL_GENERALIST", "").strip()

	if intent_model and intent_model != generalist_model:
		_intent_provider = _make_provider(intent_model)
	else:
		_intent_provider = get_provider()

	return _intent_provider
