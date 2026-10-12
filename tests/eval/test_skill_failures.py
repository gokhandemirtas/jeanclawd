"""Graceful handling of deterministic-skill failures through the full pipeline."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


_THRESHOLD = 0.7
_FAILURE_RESPONSE = "The requested service is unavailable. Please try again later."
_UNRESOLVED_TICKER = "I couldn't resolve the ticker."
_UNABLE_TO_FETCH = "Unable to fetch current news."


async def _unavailable_weather(_: str) -> str:
	"""Simulate an unavailable weather service at the skill boundary."""
	raise RuntimeError("weather service unavailable")


async def _empty_newsfeed(_: str) -> str:
	return _UNABLE_TO_FETCH


def _unresolved_ticker(_: str) -> str:
	return _UNRESOLVED_TICKER


async def _assert_graceful_response(
	query: str,
	response: str,
	criteria: str,
	local_judge: Callable,
) -> None:
	assert response, "Agent returned an empty response"
	metric = GEval(
		name="skill_failure_handling",
		criteria=criteria,
		evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
		model=local_judge,
		threshold=_THRESHOLD,
	)
	metric.measure(LLMTestCase(input=query, actual_output=response))
	assert metric.score >= metric.threshold, (
		f"Query: {query}\nResponse: {response[:300]}\n"
		f"Score: {metric.score:.2f} — {metric.reason}"
	)


@pytest.mark.eval
class TestSkillFailureHandling:
	"""Deterministic skill failures must be useful and never fabricate data."""

	async def test_weather_service_unavailable(self, pipeline, local_judge) -> None:
		from agents.intent_classifier import run_intent_classifier

		query = "what's the weather in Istanbul?"
		with (
			patch("skills.weather.skill.handle_weather", new=_unavailable_weather),
			patch("skills.weather.weather_skill.handle", new=_unavailable_weather),
		):
			response = await run_intent_classifier(query)

		await _assert_graceful_response(
			query,
			response,
			"Response clearly acknowledges that weather data is unavailable or could not be retrieved. "
			"It must not state a temperature, conditions, forecast, or other invented weather data.",
			local_judge,
		)

	async def test_stock_ticker_not_found(self, pipeline, local_judge) -> None:
		from agents.intent_classifier import run_intent_classifier

		query = "what's the stock price of an unknown ticker?"
		with (
			patch("skills.finance.skill.handle_finance_stocks", new=_unresolved_ticker),
			patch("skills.finance.finance_skill.handle", new=_unresolved_ticker),
		):
			response = await run_intent_classifier(query)

		await _assert_graceful_response(
			query,
			response,
			"Response says the ticker could not be resolved or asks for a valid ticker. "
			"It must not invent a stock price, market value, or ticker result.",
			local_judge,
		)

	async def test_newsfeed_empty(self, pipeline, local_judge) -> None:
		from agents.intent_classifier import run_intent_classifier

		query = "what's in the news today?"
		with (
			patch("skills.newsfeed.skill.handle_newsfeed", new=_empty_newsfeed),
			patch("skills.newsfeed.newsfeed_skill.handle", new=_empty_newsfeed),
		):
			response = await run_intent_classifier(query)

		await _assert_graceful_response(
			query,
			response,
			"Response acknowledges that current news could not be fetched. "
			"It must not present headlines, current events, or other fabricated news content.",
			local_judge,
		)

	async def test_reminder_without_time(self, pipeline, local_judge) -> None:
		from agents.intent_classifier import run_intent_classifier

		query = "set a reminder"
		response = await run_intent_classifier(query)

		await _assert_graceful_response(
			query,
			response,
			"Response asks exactly one clarifying question for the missing reminder details or time. "
			"It must not claim that a reminder was created or guess a time or reminder content.",
			local_judge,
		)
