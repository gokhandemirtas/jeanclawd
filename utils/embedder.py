"""Embedding helper — delegates to the active inference provider."""

import os

_probed_dim: int | None = None


def embed(text: str) -> list[float]:
	from providers import get_provider

	return get_provider().embed(text)


def get_embedding_dim() -> int:
	"""Return the actual embedding dimension by probing the model once, then caching."""
	global _probed_dim
	if _probed_dim is None:
		env = os.environ.get("EMBEDDING_DIM")
		if env:
			_probed_dim = int(env)
		else:
			_probed_dim = len(embed("dim probe"))
	return _probed_dim


# Legacy alias — kept for any code that imported EMBEDDING_DIM directly.
# Uses the env var only; callers that need the real runtime dim should use get_embedding_dim().
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
