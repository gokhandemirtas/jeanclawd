"""Memory retention and quality — evaluated through the full agent pipeline.

Facts are introduced via the agent, then recalled by asking the agent questions.
We judge the agent's recall responses, not the raw memory store contents.

Run:
    uv run pytest tests/eval/test_memory.py -v -m eval
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
import pytest_asyncio

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ---------------------------------------------------------------------------
# Facts introduced in setup turns
# ---------------------------------------------------------------------------

_SETUP_TURNS: list[str] = [
	"My name is Gokhan Demirtas. I'm a software engineer living in Istanbul.",
	"I've been doing backend development for 12 years, mainly Python and Go.",
	"I have a wife named Selin — she's a cardiologist.",
	"We have a 4-year-old daughter named Defne who loves dinosaurs.",
	"I also have a golden retriever called Boncuk.",
	"My birthday is March 15th, 1988. I studied at Boğaziçi University.",
	"I play bass guitar in a progressive rock band called Demir Kapı.",
	"I'm a big Ferrari fan and I drink black coffee — at least 3 cups a day.",
]


# ---------------------------------------------------------------------------
# Recall checks: (question, what the answer must contain)
# ---------------------------------------------------------------------------

@dataclass
class _RecallCase:
	question: str
	expected_fact: str
	note: str = ""


_RECALL_CASES: list[_RecallCase] = [
	_RecallCase("what's my name?", "Gokhan", note="name"),
	_RecallCase("what do I do for work?", "software engineer", note="occupation"),
	_RecallCase("where do I live?", "Istanbul", note="location"),
	_RecallCase("how many years of experience do I have?", "12", note="experience"),
	_RecallCase("what's my wife's name?", "Selin", note="spouse"),
	_RecallCase("what are my kids' names?", "Defne", note="child"),
	_RecallCase("what's my dog's name?", "Boncuk", note="pet"),
	_RecallCase("when is my birthday?", "March", note="birthday"),
	_RecallCase("what university did I go to?", "Boğaziçi", note="education"),
	_RecallCase("what instrument do I play?", "bass", note="instrument"),
	_RecallCase("what's my band called?", "Demir Kapı", note="band"),
	_RecallCase("what F1 team do I support?", "Ferrari", note="F1"),
	_RecallCase("how do I take my coffee?", "black", note="coffee preference"),
]


@pytest.mark.eval
class TestMemoryRetention:

	@pytest_asyncio.fixture(autouse=True)
	async def _setup(self, clean_memories, pipeline):
		"""Run setup turns then wait for background extraction to complete."""
		from agents.intent_classifier import run_intent_classifier
		from utils.langmem_service import _EXTRACTION_TASKS
		for turn in _SETUP_TURNS:
			await run_intent_classifier(turn)
		await asyncio.gather(*list(_EXTRACTION_TASKS), return_exceptions=True)

	@pytest.mark.parametrize(
		"case", _RECALL_CASES, ids=[c.note for c in _RECALL_CASES]
	)
	async def test_recall(self, pipeline, case: _RecallCase):
		"""Agent's response to a recall question must mention the expected fact."""
		from agents.intent_classifier import run_intent_classifier
		response = await run_intent_classifier(case.question)
		assert case.expected_fact.lower() in response.lower(), (
			f"[{case.note}] Expected '{case.expected_fact}' in response.\n"
			f"Question: {case.question}\nResponse: {response}"
		)

	async def test_retention_rate(self, pipeline):
		"""At least 80 % of recall questions must be answered correctly."""
		from agents.intent_classifier import run_intent_classifier
		passed = 0
		failures = []
		for case in _RECALL_CASES:
			response = await run_intent_classifier(case.question)
			if case.expected_fact.lower() in response.lower():
				passed += 1
			else:
				failures.append(f"  [{case.note}] missing '{case.expected_fact}': {response[:120]}")

		rate = passed / len(_RECALL_CASES)
		assert rate >= 0.80, (
			f"Retention rate {rate:.1%} < 80 %.\n" + "\n".join(failures)
		)


@pytest.mark.eval
class TestMemoryQuality:
	"""Checks on the raw memory store — no placeholders, no misattributions."""

	@pytest_asyncio.fixture(autouse=True)
	async def _setup(self, clean_memories, pipeline):
		from agents.intent_classifier import run_intent_classifier
		from utils.langmem_service import _EXTRACTION_TASKS
		for turn in _SETUP_TURNS:
			await run_intent_classifier(turn)
		await asyncio.gather(*list(_EXTRACTION_TASKS), return_exceptions=True)

	async def test_no_placeholder_records(self):
		"""No memory record should contain placeholder/unknown text."""
		from utils.langmem_service import list_memories
		_PLACEHOLDER_PHRASES = (
			"not specified", "not mentioned", "unknown",
			"not provided", "unspecified", "not stated", "n/a",
		)
		records = await list_memories()
		bad = [
			r.value for r in records
			if any(p in r.value.lower() for p in _PLACEHOLDER_PHRASES)
		]
		assert not bad, "Placeholder records:\n" + "\n".join(f"  - {v}" for v in bad)

	async def test_no_misattributed_third_party_facts(self, local_judge):
		"""Facts about Selin or Defne must not appear without attribution."""
		from utils.langmem_service import list_memories
		records = await list_memories()
		store_text = "\n".join(f"- {r.value}" for r in records)

		metric = GEval(
			name="no_misattribution",
			criteria=(
				"These are memory facts stored about a user named Gokhan. "
				"Score 1.0 if every fact that clearly belongs to a third party "
				"(Selin, Defne) names that person in the record. "
				"Score lower for each fact that omits the person's name, making "
				"it look like Gokhan's own fact (e.g. 'Occupation: cardiologist' "
				"with no mention of Selin is a misattribution)."
			),
			evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
			model=local_judge,
			threshold=0.75,
		)
		tc = LLMTestCase(
			input="Are third-party facts properly attributed?",
			actual_output=store_text,
		)
		metric.measure(tc)
		assert metric.score >= metric.threshold, (
			f"Misattribution score {metric.score:.2f} < {metric.threshold}.\n"
			f"Reason: {metric.reason}\n\nStore:\n{store_text}"
		)
