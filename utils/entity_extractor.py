"""Local GLiNER2 entity/relation extraction for user-memory candidates."""

from __future__ import annotations

from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
import warnings

_MODEL_NAME = "fastino/gliner2-base-v1"
_ENTITY_LABELS = {
	"person_name": "a person's name",
	"birth_date": "a date of birth",
	"location": "a city, country, or place where someone lives",
	"interest": "a hobby, interest, or recurring activity",
	"food": "a food, cuisine, or dietary preference",
	"music_genre": "a music genre or style",
	"technology": "a technology, software, or hardware project",
}
_RELATION_LABELS = ["likes", "dislikes", "lives in", "works on"]


@lru_cache(maxsize=1)
def _model():
	from gliner2 import GLiNER2
	from transformers.utils import logging as transformers_logging

	# GLiNER2 intentionally wraps a DeBERTa encoder in its custom ``extractor``
	# model class. Transformers emits a compatibility warning for that expected
	# composition; keep other warnings/errors visible.
	previous_verbosity = transformers_logging.get_verbosity()
	try:
		transformers_logging.set_verbosity_error()
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				message=r"You are using a model of type `extractor`.*",
			)
			# GLiNER2 prints its configuration directly to stdout during
			# construction; keep that library banner out of the chat terminal.
			with redirect_stdout(StringIO()):
				return GLiNER2.from_pretrained(_MODEL_NAME)
	finally:
		transformers_logging.set_verbosity(previous_verbosity)


def extract_user_signals(text: str) -> dict:
	"""Return GLiNER2 entities and user-specific relations for one turn."""
	model = _model()
	return {
		"entities": model.extract_entities(
			text, _ENTITY_LABELS, threshold=0.35, include_confidence=True
		),
		"relations": model.extract_relations(
			text, _RELATION_LABELS, threshold=0.35, include_confidence=True
		),
	}
