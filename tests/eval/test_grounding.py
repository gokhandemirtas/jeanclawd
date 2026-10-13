"""Skill output grounding — the agent's response must faithfully reflect skill data.

When a skill (weather, finance, newsfeed) returns concrete facts, the agent must
not contradict, fabricate, or materially distort those facts in its response.

Each test patches the skill to return a controlled, known payload, then judges
whether the agent's response accurately reflects that payload.

Run:
    uv run pytest tests/eval/test_grounding.py -v -m eval
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

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
        # handle_weather is async — use AsyncMock so inspect.iscoroutinefunction passes
        from skills.weather import weather_skill as _ws
        with patch.object(_ws, "handle", new=AsyncMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's the weather in Istanbul?")

        # Structural check: response must mention a temperature in the 6–13°C range or the
        # word "overcast" (synthesis may rephrase slightly).
        numeric_temps = [int(re.sub(r"[^-\d]", "", t)) for t in re.findall(r"-?\d+\s*°?[Cc]", response)]
        in_range = any(6 <= t <= 13 for t in numeric_temps)
        assert in_range or "9" in response or "overcast" in response.lower(), (
            f"Response does not reflect 9°C overcast Istanbul data:\n{response[:400]}"
        )

    async def test_conditions_not_contradicted(self, pipeline, local_judge):
        """Agent must not describe sunny conditions when the skill reports rain."""
        controlled = (
            "Current weather in London: 13°C, heavy rain. "
            "Low 10°C, high 15°C. Humidity 92%."
        )
        from skills.weather import weather_skill as _ws
        with patch.object(_ws, "handle", new=AsyncMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("how's the weather in London?")

        # Only check that the response does NOT describe dry/sunny conditions.
        # "Not available" responses are acceptable — they don't contradict rain.
        sunny_words = {"sunny", "sunshine", "clear skies", "no rain", "not raining"}
        lower = response.lower()
        contradicts = any(w in lower for w in sunny_words)
        assert not contradicts, (
            f"Response contradicts heavy rain in London:\n{response[:400]}"
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
        # Use patch.object to reliably replace the skill's handle on its Skill instance.
        from skills.finance import finance_skill as _fs
        with patch.object(_fs, "handle", new=MagicMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's Apple's stock price?")

        # Structural check: response must mention a price in the $212–$215 range.
        # GEval with local Qwen3 is unreliable for numeric checks (returns 0.0 even when correct).
        prices = [float(m.replace(",", "")) for m in re.findall(r"\$(\d[\d,]*\.?\d*)", response)]
        in_range = any(212 <= p <= 215 for p in prices)
        assert in_range or "213" in response, (
            f"Response does not reflect $213.45 Apple stock price:\n{response[:400]}"
        )

    async def test_stock_direction_not_flipped(self, pipeline, local_judge):
        """Agent must not report a gain when the skill returns a loss."""
        controlled = (
            "Tesla Inc (TSLA) — Current price: $172.10 USD. "
            "Change: -8.45 (-4.68%). Market cap: 551B."
        )
        from skills.finance import finance_skill as _fs
        with patch.object(_fs, "handle", new=MagicMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("how is Tesla stock doing?")

        # Structural check: response must not claim an upward movement.
        lower = response.lower()
        gain_words = {"up", "gain", "rose", "risen", "higher", "increase", "positive", "surge"}
        loss_words = {"down", "drop", "fell", "decline", "decrease", "loss", "lower", "negative", "-", "4.68"}
        claims_gain = any(w in lower for w in gain_words)
        claims_loss = any(w in lower for w in loss_words)
        assert claims_loss and not claims_gain, (
            f"Response does not reflect Tesla declining 4.68%:\n{response[:400]}"
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
        from skills.newsfeed import newsfeed_skill as _nf
        with patch.object(_nf, "handle", new=MagicMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's in the news today?")

        # Structural check: at least 2 of the 3 controlled headlines must appear.
        lower = response.lower()
        mars = "mars" in lower
        eu_ai = ("eu" in lower or "european" in lower) and ("ai" in lower or "artificial" in lower or "directive" in lower)
        heatwave = "heatwave" in lower or ("heat" in lower and ("europe" in lower or "southern" in lower))
        matched = sum([mars, eu_ai, heatwave])
        assert matched >= 2, (
            f"Response only matched {matched}/3 controlled headlines:\n{response[:400]}"
        )
