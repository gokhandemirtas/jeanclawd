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
import asyncio

import pytest
import pytest_asyncio
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Pre-flight health checks — run before any test is collected so a missing
# backend produces one clear error instead of a cascade of 400 / connection-
# refused failures mid-suite.
# ---------------------------------------------------------------------------

def _check_mongo() -> None:
	"""Verify MongoDB is reachable and auth succeeds."""
	import pymongo
	from dotenv import load_dotenv
	load_dotenv()

	uri = (
		f"mongodb://{os.environ['MONGODB_USER']}:{os.environ['MONGODB_PASSWORD']}@"
		f"{os.environ.get('MONGODB_HOST', 'localhost')}:{os.environ.get('MONGODB_PORT', '27017')}"
		"/?authSource=admin&directConnection=true"
		"&serverSelectionTimeoutMS=4000&connectTimeoutMS=4000"
	)
	try:
		client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=4000)
		client.admin.command("ping")
		client.close()
	except Exception as exc:
		pytest.exit(
			f"\n[preflight] MongoDB is not reachable at "
			f"{os.environ.get('MONGODB_HOST', 'localhost')}:"
			f"{os.environ.get('MONGODB_PORT', '27017')} — {exc}\n"
			"Start MongoDB before running the eval suite.",
			returncode=3,
		)


def _check_inference() -> None:
	"""Verify the inference backend is up and all required models are loaded."""
	import json
	import urllib.error
	import urllib.request
	from dotenv import load_dotenv
	load_dotenv()

	provider = os.getenv("INFERENCE_PROVIDER", "lmstudio").lower()
	if provider == "openai":
		return  # OpenAI is always assumed reachable; real API errors surface naturally

	base_url = os.environ.get("MODEL_HOST", "http://localhost:1234/v1").rstrip("/")
	models_url = f"{base_url}/models"

	try:
		req = urllib.request.Request(models_url, headers={"Authorization": "Bearer lm-studio"})
		with urllib.request.urlopen(req, timeout=5) as resp:
			data = json.loads(resp.read())
	except urllib.error.URLError as exc:
		pytest.exit(
			f"\n[preflight] LM Studio is not reachable at {base_url} — {exc}\n"
			"Start LM Studio and load at least the generalist model before running the eval suite.",
			returncode=3,
		)
	except Exception as exc:
		pytest.exit(f"\n[preflight] Inference backend check failed: {exc}", returncode=3)

	loaded = [m["id"] for m in data.get("data", [])]
	if not loaded:
		pytest.exit(
			f"\n[preflight] LM Studio is running but no models are loaded ({base_url}).\n"
			"Load at least the generalist model before running the eval suite.",
			returncode=3,
		)

	generalist = os.environ.get("MODEL_GENERALIST", "")
	embedding = os.environ.get("MODEL_EMBEDDING", "")
	missing: list[str] = []
	if generalist and not any(generalist.lower() in m.lower() for m in loaded):
		missing.append(f"generalist  →  {generalist}")
	if embedding and not any(embedding.lower() in m.lower() for m in loaded):
		missing.append(f"embedding   →  {embedding}")

	if missing:
		loaded_str = "\n    ".join(loaded)
		pytest.exit(
			f"\n[preflight] Required model(s) not loaded in LM Studio:\n"
			+ "".join(f"  • {m}\n" for m in missing)
			+ f"Loaded models:\n    {loaded_str}\n"
			"Load the missing model(s) in LM Studio before running the eval suite.",
			returncode=3,
		)


def pytest_sessionstart(session) -> None:
	"""Run all pre-flight checks before test collection begins."""
	_check_mongo()
	_check_inference()


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

@pytest_asyncio.fixture(scope="session")
async def pipeline():
	"""Initialise the full JeanClawd pipeline once for the session."""
	from dotenv import load_dotenv
	load_dotenv()

	from db.init import db_init
	from utils.session_service import initialize_session

	await db_init()
	await initialize_session()
	yield
	from utils.langmem_service import close_memory_store
	await close_memory_store()


@pytest_asyncio.fixture
async def clean_memories(pipeline):
	"""Isolate memories and their MongoDB connection for each eval test."""
	from db.schemas import Interaction, Session
	from utils.langmem_service import (
		_EXTRACTION_TASKS,
		_NAME_CACHE,
		clear_memories,
		close_memory_store,
	)

	# An extraction launched by an earlier test can otherwise write after its
	# database cleanup, contaminating this test's supposedly empty store.
	tasks = list(_EXTRACTION_TASKS)
	if tasks:
		await asyncio.gather(*tasks, return_exceptions=True)
	await close_memory_store()
	await clear_memories()
	await Interaction.find(Interaction.user_id == "default").delete()
	await Session.find(Session.user_id == "default").delete()
	_NAME_CACHE.clear()
	yield
	tasks = list(_EXTRACTION_TASKS)
	if tasks:
		await asyncio.gather(*tasks, return_exceptions=True)
	await clear_memories()
	await Interaction.find(Interaction.user_id == "default").delete()
	await Session.find(Session.user_id == "default").delete()
	_NAME_CACHE.clear()
	# Reopen on the next test so a long eval run never reuses a stale MongoDB
	# connection pool.
	await close_memory_store()


def pytest_configure(config) -> None:
	config.addinivalue_line("markers", "eval: integration eval tests (require live inference + DB)")
