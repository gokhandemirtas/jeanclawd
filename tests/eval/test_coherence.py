"""Conversation coherence — evaluated through the full agent pipeline.

Each scenario runs a multi-turn conversation through run_intent_classifier and
judges the final response with deepeval metrics.

Dimensions tested:
  - No re-fetch: after skill data lands in history, follow-ups get conversational
    answers, not fresh skill invocations
  - Answer relevancy: responses address what was asked
  - Fluency: responses are natural and well-formed

Run:
    uv run pytest tests/eval/test_coherence.py -v -m eval
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ---------------------------------------------------------------------------
# Multi-turn helpers
# ---------------------------------------------------------------------------

async def _converse(turns: list[str]) -> list[str]:
	"""Run turns sequentially through the agent; return all responses."""
	from agents.intent_classifier import run_intent_classifier
	responses = []
	for turn in turns:
		response = await run_intent_classifier(turn)
		responses.append(response)
	return responses


# ---------------------------------------------------------------------------
# Coherence scenarios
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestNoReFetch:
	"""Follow-up questions after a skill response must be answered conversationally."""

	async def test_newsfeed_followup_is_conversational(self, pipeline, local_judge):
		"""Asking an opinion on a fetched headline must not trigger a second fetch."""
		turns = [
			"what's in the news today?",
			"what do you think about the top story?",
		]
		responses = await _converse(turns)
		followup_response = responses[1]

		# The follow-up response should be an opinion/discussion, not a new headline dump
		metric = GEval(
			name="no_refetch",
			criteria=(
				"The assistant was asked for its opinion on a previously fetched news story. "
				"Score 1.0 if the response is an opinion, analysis, or discussion of that story. "
				"Score 0.0 if the response is a fresh list of headlines (it fetched the news again instead of discussing)."
			),
			evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
			model=local_judge,
			threshold=0.7,
		)
		tc = LLMTestCase(input=turns[1], actual_output=followup_response)
		metric.measure(tc)
		assert metric.score >= metric.threshold, (
			f"Score {metric.score:.2f} — {metric.reason}\nResponse: {followup_response[:400]}"
		)

	async def test_weather_followup_is_conversational(self, pipeline, local_judge):
		"""Asking advice after a weather fetch must not re-fetch weather."""
		turns = [
			"what's the weather in Istanbul today?",
			"should I bring a jacket?",
		]
		responses = await _converse(turns)

		metric = GEval(
			name="no_refetch",
			criteria=(
				"The user asked for jacket advice after receiving weather data. "
				"Score 1.0 if the response gives clothing/practical advice based on the weather context. "
				"Score 0.0 if the response re-fetches weather data instead of answering the advice question."
			),
			evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
			model=local_judge,
			threshold=0.7,
		)
		tc = LLMTestCase(input=turns[1], actual_output=responses[1])
		metric.measure(tc)
		assert metric.score >= metric.threshold, (
			f"Score {metric.score:.2f} — {metric.reason}\nResponse: {responses[1][:400]}"
		)

	async def test_correction_changes_topic(self, pipeline, local_judge):
		"""A correction ('I meant X not Y') must redirect without repeating the original answer."""
		turns = [
			"research the history of Apple Inc",
			"no I meant apple the fruit, not the company",
		]
		responses = await _converse(turns)

		metric = GEval(
			name="correction_handled",
			criteria=(
				"The user corrected themselves: they originally asked about Apple the tech company "
				"but now want to know about apples the fruit. "
				"Score 1.0 if the response addresses apples the fruit or asks for clarification. "
				"Score 0.0 if the response continues discussing Apple Inc or ignores the correction."
			),
			evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
			model=local_judge,
			threshold=0.7,
		)
		tc = LLMTestCase(input=turns[1], actual_output=responses[1])
		metric.measure(tc)
		assert metric.score >= metric.threshold, (
			f"Score {metric.score:.2f} — {metric.reason}\nResponse: {responses[1][:400]}"
		)


@pytest.mark.eval
class TestAnswerRelevancy:
	"""Responses must directly address the question asked."""

	@pytest.mark.parametrize("query", [
		"who invented the telephone?",
		"what's the capital of Japan?",
		"explain what a neural network is in simple terms",
		"what are the main differences between Python and Go?",
	])
	async def test_relevancy(self, pipeline, local_judge, query: str):
		from agents.intent_classifier import run_intent_classifier
		response = await run_intent_classifier(query)

		metric = AnswerRelevancyMetric(threshold=0.7, model=local_judge, async_mode=False)
		tc = LLMTestCase(input=query, actual_output=response)
		metric.measure(tc)
		assert metric.score >= metric.threshold, (
			f"Relevancy {metric.score:.2f} < {metric.threshold} for: {query!r}\n"
			f"Response: {response[:300]}\nReason: {metric.reason}"
		)


@pytest.mark.eval
class TestFluency:
	"""Agent responses must be natural, fluent, and well-formed."""

	@pytest.mark.parametrize("query", [
		"what is the meaning of life?",
		"tell me something interesting about the ocean",
		"what would you recommend reading this weekend?",
	])
	async def test_fluency(self, pipeline, local_judge, query: str):
		from agents.intent_classifier import run_intent_classifier
		response = await run_intent_classifier(query)

		metric = GEval(
			name="fluency",
			criteria=(
				"Score how fluent, natural, and well-structured the response is. "
				"1.0 = reads like a knowledgeable, articulate person. "
				"Deduct for awkward phrasing, abrupt endings, repetition, or incoherence."
			),
			evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
			model=local_judge,
			threshold=0.7,
		)
		tc = LLMTestCase(input=query, actual_output=response)
		metric.measure(tc)
		assert metric.score >= metric.threshold, (
			f"Fluency {metric.score:.2f} < {metric.threshold} for: {query!r}\n"
			f"Response: {response[:300]}\nReason: {metric.reason}"
		)
