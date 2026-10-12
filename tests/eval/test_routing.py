"""Routing accuracy — evaluated through the full agent pipeline.

Each test sends a query through run_intent_classifier and judges whether the
agent's response is appropriate for the expected intent.  We never inspect
internal routing state — only the observable output.

Run:
    uv run pytest tests/eval/test_routing.py -v -m eval
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


@dataclass
class _Case:
	query: str
	# What the response MUST demonstrate to pass
	pass_criteria: str
	# Optional: what the response must NOT do
	fail_criteria: str = ""
	note: str = ""


_CASES: list[_Case] = [
	_Case(
		query="what's the weather in Istanbul?",
		pass_criteria="Response provides weather information for Istanbul — temperature, conditions, or forecast.",
		note="WEATHER skill",
	),
	_Case(
		query="what's in the news today?",
		pass_criteria="Response contains at least two distinct news headlines or current events.",
		note="NEWSFEED skill",
	),
	_Case(
		query="what's Apple's stock price?",
		pass_criteria="Response contains a numeric stock price or market value for Apple.",
		note="FINANCE_STOCKS skill",
	),
	_Case(
		query="set a timer for 5 minutes",
		pass_criteria="Response confirms that a timer has been set for 5 minutes.",
		note="REMINDER skill",
	),
	_Case(
		query="give me a recipe for pasta carbonara",
		pass_criteria="Response includes ingredients or preparation steps for pasta carbonara.",
		note="NUTRITION skill",
	),
	_Case(
		query="who was Shakespeare?",
		pass_criteria="Response explains who Shakespeare was — playwright, poet, or historical figure.",
		fail_criteria="Response says it cannot answer or needs to search the web.",
		note="CHAT — stable historical fact answered from knowledge",
	),
	_Case(
		query="how does your intent routing work?",
		pass_criteria="Response describes how the assistant classifies or routes requests internally.",
		note="INTROSPECTION skill",
	),
	_Case(
		query="you are a local assistant running on my machine",
		pass_criteria="Response acknowledges or engages with the statement conversationally.",
		fail_criteria="Response explains its own architecture or implementation details at length.",
		note="Declarative correction — must be CHAT not INTROSPECTION",
	),
	_Case(
		query="research the history of the Ottoman Empire",
		pass_criteria=(
			"Response either (a) provides historical content about the Ottoman Empire — "
			"founding, expansion, or decline — OR (b) clearly states that the research "
			"or search service is currently unavailable and asks the user to try again. "
			"Both are acceptable outcomes depending on whether the search service is reachable."
		),
		note="SEARCH — deep research trigger",
	),
]


@pytest.mark.eval
class TestRoutingBehavior:
	"""End-to-end routing — each query goes through the full pipeline."""

	@pytest.mark.parametrize("case", _CASES, ids=[c.note or c.query[:40] for c in _CASES])
	async def test_case(self, pipeline, local_judge, case: _Case):
		from agents.intent_classifier import run_intent_classifier

		response = await run_intent_classifier(case.query)
		assert response, "Agent returned an empty response"

		criteria = case.pass_criteria
		if case.fail_criteria:
			criteria += f"\n\nFail if: {case.fail_criteria}"

		metric = GEval(
			name="routing_quality",
			criteria=criteria,
			evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
			model=local_judge,
			threshold=0.7,
		)
		tc = LLMTestCase(input=case.query, actual_output=response)
		metric.measure(tc)
		assert metric.score >= metric.threshold, (
			f"[{case.note}]\n"
			f"Query: {case.query}\n"
			f"Response: {response[:300]}\n"
			f"Score: {metric.score:.2f} — {metric.reason}"
		)
