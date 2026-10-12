"""Shared fixtures and helpers for the integration eval suite.

Run the full suite:
    uv run pytest tests/eval/ -v -m eval

Run a single module:
    uv run pytest tests/eval/test_routing.py -v

NOTE: deepeval 2.5.5 imports `langchain.schema` which was removed in langchain 1.x.
A compatibility shim is required in the venv after a fresh `uv sync`:
    echo "from langchain_core.messages import *; from langchain_core.outputs import *" > \
        .venv/lib/python3.11/site-packages/langchain/schema.py
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# DeepEval judge — uses the same local generalist model so no OpenAI key
# is needed for LLM-as-judge scoring.
# ---------------------------------------------------------------------------

class _LocalJudge(DeepEvalBaseLLM):
	"""Thin wrapper that routes deepeval's judge calls to the local LLM."""

	def __init__(self) -> None:
		kwargs: dict = {
			"model": os.environ["MODEL_GENERALIST"],
			"api_key": os.environ.get("OPENAI_API_KEY", "lm-studio"),
			"temperature": 0,
		}
		if os.getenv("INFERENCE_PROVIDER", "lmstudio").lower() != "openai":
			kwargs["base_url"] = os.environ.get("MODEL_HOST", "http://localhost:1234/v1")
		self._llm = ChatOpenAI(**kwargs)

	def get_model_name(self) -> str:
		return os.environ["MODEL_GENERALIST"]

	def load_model(self):
		return self._llm

	def generate(self, prompt: str) -> str:
		return self._llm.invoke(prompt).content

	async def a_generate(self, prompt: str) -> str:
		result = await self._llm.ainvoke(prompt)
		return result.content


@pytest.fixture(scope="session")
def local_judge() -> _LocalJudge:
	from dotenv import load_dotenv
	load_dotenv()
	return _LocalJudge()


# ---------------------------------------------------------------------------
# DB + pipeline init — function-scoped to ensure every test runs in its own
# event loop, avoiding "Lock bound to a different event loop" errors that
# arise from asyncio module-level singletons (inference_gate._lock, etc.)
# when a session-scoped fixture's loop differs from the test's loop.
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def pipeline():
	"""Initialise the full JeanClawd pipeline for one test, then tear down."""
	from dotenv import load_dotenv
	load_dotenv()

	# Reset asyncio singletons so they rebind to this test's event loop.
	import utils.inference_gate as _gate
	_gate._lock = None
	_gate._foreground_count = 0

	from db.init import db_init
	from utils.session_service import initialize_session

	await db_init()
	await initialize_session()
	yield
	from utils.langmem_service import close_memory_store
	await close_memory_store()


@pytest_asyncio.fixture
async def clean_memories(pipeline):
	"""Wipe all memories before a test and restore nothing after."""
	from utils.langmem_service import clear_memories
	await clear_memories()
	yield
	await clear_memories()


def pytest_configure(config):
	config.addinivalue_line("markers", "eval: integration eval tests (require live inference + DB)")
