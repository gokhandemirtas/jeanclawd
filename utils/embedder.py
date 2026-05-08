"""Shared llama.cpp embedding model singleton.

Loads a dedicated embedding GGUF (all-MiniLM-L6-v2-q8_0.gguf, 384-dim) once
and reuses it across memory_client and journal_index.
"""

from __future__ import annotations

import os
from typing import Optional

from utils.log import logger

_model: Optional["Llama"] = None

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))


def get_model():
    global _model
    if _model is None:
        from llama_cpp import Llama

        model_path = os.path.join(
            os.environ["MODEL_PATH"],
            os.environ["MODEL_EMBEDDING"],
        )
        _model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=512,
            n_gpu_layers=-1,
            verbose=False,
        )
        logger.info(f"[embedder] Loaded {os.environ['MODEL_EMBEDDING']}")
    return _model


def embed(text: str) -> list[float]:
    import os as _os
    old_err = _os.dup(2)
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(devnull, 2)
    try:
        return get_model().embed(text)
    finally:
        _os.dup2(old_err, 2)
        _os.close(old_err)
        _os.close(devnull)
