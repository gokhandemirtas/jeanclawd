"""Embedding helper — delegates to the active inference provider."""

import os

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))


def embed(text: str) -> list[float]:
	from providers import get_provider

	return get_provider().embed(text)
