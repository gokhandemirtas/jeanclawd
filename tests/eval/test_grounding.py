"""Skill output grounding — the agent's response must faithfully reflect skill data.

When a skill (weather, finance, newsfeed) returns concrete facts, the agent must
not contradict, fabricate, or materially distort those facts in its response.

Each test patches the skill to return a controlled, known payload, then judges
whether the agent's response accurately reflects that payload.

Run:
    uv run pytest tests/eval/test_grounding.py -v -m eval
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geval(name: str, criteria: str, local_judge) -> GEval:
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=local_judge,
        threshold=0.7,
    )


# ---------------------------------------------------------------------------
# Weather grounding
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestWeatherGrounding:

    async def test_temperature_not_fabricated(self, pipeline, local_judge):
        """Agent must report the temperature returned by the skill, not invent one."""
        controlled = (
            "Current weather in Istanbul: 9°C, overcast. "
            "Low 6°C, high 11°C. Wind 20 km/h from the north."
        )
        with patch("skills.weather.weather_skill.handle", return_value=controlled):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's the weather in Istanbul?")

        metric = _geval(
            "weather_grounding",
            "The skill returned: 9°C, overcast, low 6°C, high 11°C. "
            "Score 1.0 if the response reflects this data accurately — temperature in the right range, "
            "conditions consistent with overcast. "
            "Score 0.0 if the response invents a different temperature or contradicts the data.",
            local_judge,
        )
        metric.measure(LLMTestCase(input="what's the weather in Istanbul?", actual_output=response))
        assert metric.score >= metric.threshold, (
            f"Score {metric.score:.2f} — {metric.reason}\nResponse: {response[:400]}"
        )

    async def test_conditions_not_contradicted(self, pipeline, local_judge):
        """Agent must not describe sunny conditions when the skill reports rain."""
        controlled = (
            "Current weather in London: 13°C, heavy rain. "
            "Low 10°C, high 15°C. Humidity 92%."
        )
        with patch("skills.weather.weather_skill.handle", return_value=controlled):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("how's the weather in London?")

        metric = _geval(
            "weather_conditions_grounding",
            "The skill returned heavy rain in London. "
            "Score 1.0 if the response mentions rain or wet conditions. "
            "Score 0.0 if the response describes clear, sunny, or dry weather — that would be a fabrication.",
            local_judge,
        )
        metric.measure(LLMTestCase(input="how's the weather in London?", actual_output=response))
        assert metric.score >= metric.threshold, (
            f"Score {metric.score:.2f} — {metric.reason}\nResponse: {response[:400]}"
        )


# ---------------------------------------------------------------------------
# Finance grounding
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestFinanceGrounding:

    async def test_stock_price_not_fabricated(self, pipeline, local_judge):
        """Agent must report the price returned by the skill, not invent one."""
        controlled = (
            "Apple Inc (AAPL) — Current price: $213.45 USD. "
            "Change: +1.23 (+0.58%). Market cap: 3.26T."
        )
        with patch("skills.finance.finance_skill.handle", return_value=controlled):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's Apple's stock price?")

        metric = _geval(
            "finance_grounding",
            "The skill returned Apple stock at $213.45. "
            "Score 1.0 if the response reflects a price around $213 (±$1). "
            "Score 0.0 if the response invents a clearly different price.",
            local_judge,
        )
        metric.measure(LLMTestCase(input="what's Apple's stock price?", actual_output=response))
        assert metric.score >= metric.threshold, (
            f"Score {metric.score:.2f} — {metric.reason}\nResponse: {response[:400]}"
        )

    async def test_stock_direction_not_flipped(self, pipeline, local_judge):
        """Agent must not report a gain when the skill returns a loss."""
        controlled = (
            "Tesla Inc (TSLA) — Current price: $172.10 USD. "
            "Change: -8.45 (-4.68%). Market cap: 551B."
        )
        with patch("skills.finance.finance_skill.handle", return_value=controlled):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("how is Tesla stock doing?")

        metric = _geval(
            "finance_direction_grounding",
            "The skill returned Tesla is DOWN 4.68% today. "
            "Score 1.0 if the response reflects a decline, drop, or negative movement. "
            "Score 0.0 if the response says Tesla is up or gaining — that contradicts the data.",
            local_judge,
        )
        metric.measure(LLMTestCase(input="how is Tesla stock doing?", actual_output=response))
        assert metric.score >= metric.threshold, (
            f"Score {metric.score:.2f} — {metric.reason}\nResponse: {response[:400]}"
        )


# ---------------------------------------------------------------------------
# Newsfeed grounding
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestNewsfeedGrounding:

    async def test_headlines_not_fabricated(self, pipeline, local_judge):
        """Agent must surface the headlines returned by the skill, not invent new ones."""
        controlled = (
            "1. Scientists discover water ice deposits near Mars south pole\n"
            "2. EU passes landmark AI liability directive\n"
            "3. Record-breaking heatwave sweeps across southern Europe\n"
        )
        with patch("skills.newsfeed.newsfeed_skill.handle", return_value=controlled):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's in the news today?")

        metric = _geval(
            "newsfeed_grounding",
            "The skill returned three specific headlines: Mars ice, EU AI directive, Europe heatwave. "
            "Score 1.0 if the response includes at least two of these topics. "
            "Score 0.0 if the response invents entirely different stories not present in the skill output.",
            local_judge,
        )
        metric.measure(LLMTestCase(input="what's in the news today?", actual_output=response))
        assert metric.score >= metric.threshold, (
            f"Score {metric.score:.2f} — {metric.reason}\nResponse: {response[:400]}"
        )
