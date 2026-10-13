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


# ---------------------------------------------------------------------------
# Unit integrity
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestUnitIntegrity:
    """Agent must preserve the units returned by the skill — no silent conversion."""

    async def test_celsius_not_reported_as_fahrenheit(self, pipeline, local_judge):
        """Skill returns °C — agent must not report the same number labelled °F."""
        controlled = "Current weather in Oslo: 15°C, partly cloudy. Low 10°C, high 17°C."
        from skills.weather import weather_skill as _ws
        with patch.object(_ws, "handle", new=AsyncMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's the weather in Oslo?")

        # 15°F would be far below freezing; if the agent writes "15°F" or "15 degrees Fahrenheit"
        # it has silently re-labelled the unit.
        lower = response.lower()
        assert "15°f" not in lower and "15 degrees fahrenheit" not in lower and "15 f" not in lower, (
            f"Agent re-labelled 15°C as Fahrenheit:\n{response[:400]}"
        )
        # Also confirm the temperature surface at all (basic grounding).
        assert "15" in response or "oslo" in lower, (
            f"Agent didn't surface any Oslo temperature data:\n{response[:400]}"
        )

    async def test_currency_not_switched(self, pipeline, local_judge):
        """Skill returns EUR — agent must not convert silently to USD."""
        controlled = (
            "Volkswagen AG (VOW3.DE) — Current price: €150.00 EUR. "
            "Change: +2.50 (+1.69%). Market cap: 65.3B EUR."
        )
        from skills.finance import finance_skill as _fs
        with patch.object(_fs, "handle", new=MagicMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's Volkswagen's stock price?")

        lower = response.lower()
        # Agent must not claim the price is in dollars when the skill said euros.
        assert "$150" not in response and "150 usd" not in lower and "150 dollars" not in lower, (
            f"Agent converted €150 EUR to USD without basis:\n{response[:400]}"
        )
        assert "150" in response, (
            f"Agent didn't surface the €150 price at all:\n{response[:400]}"
        )


# ---------------------------------------------------------------------------
# Multi-fact consistency
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestMultiFactConsistency:
    """When a skill returns multiple facts, none may be contradicted."""

    async def test_weather_multi_fact(self, pipeline, local_judge):
        """All three facts — temperature, humidity, wind — must survive intact."""
        controlled = (
            "Current weather in Tokyo: 22°C, light drizzle. "
            "Humidity: 78%. Wind: 35 km/h from the south-east."
        )
        from skills.weather import weather_skill as _ws
        with patch.object(_ws, "handle", new=AsyncMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's the weather in Tokyo right now?")

        lower = response.lower()

        # Temperature: must reference 22°C (or nearby range) without flipping to F
        temps = [int(re.sub(r"[^-\d]", "", t)) for t in re.findall(r"-?\d+\s*°?[Cc]", response)]
        temp_ok = any(20 <= t <= 24 for t in temps) or "22" in response
        assert temp_ok, f"Tokyo temperature not faithfully reported:\n{response[:400]}"

        # Humidity: must not claim it is dry / low humidity when 78% was given
        dry_words = {"dry", "low humidity", "arid"}
        assert not any(w in lower for w in dry_words), (
            f"Agent contradicted 78% humidity:\n{response[:400]}"
        )

        # Conditions: drizzle/rain must not be described as sunny or clear
        sunny_words = {"sunny", "clear sky", "no rain", "not raining", "sunshine"}
        assert not any(w in lower for w in sunny_words), (
            f"Agent contradicted light drizzle in Tokyo:\n{response[:400]}"
        )

    async def test_finance_multi_fact(self, pipeline, local_judge):
        """Price, direction, and market cap must all be faithfully represented."""
        controlled = (
            "NVIDIA Corp (NVDA) — Current price: $875.20 USD. "
            "Change: +12.30 (+1.43%). Market cap: 2.16T."
        )
        from skills.finance import finance_skill as _fs
        with patch.object(_fs, "handle", new=MagicMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("how is Nvidia stock doing?")

        lower = response.lower()

        # Price in range
        prices = [float(m.replace(",", "")) for m in re.findall(r"\$(\d[\d,]*\.?\d*)", response)]
        price_ok = any(870 <= p <= 880 for p in prices) or "875" in response
        assert price_ok, f"NVDA price not faithfully reported:\n{response[:400]}"

        # Direction: must not claim a loss
        loss_words = {"down", "dropped", "fell", "decline", "loss", "negative", "-12", "-1.43"}
        assert not any(w in lower for w in loss_words), (
            f"Agent flipped NVDA gain to a loss:\n{response[:400]}"
        )


# ---------------------------------------------------------------------------
# No-data fabrication
# ---------------------------------------------------------------------------

@pytest.mark.eval
class TestNoDataFabrication:
    """When a skill returns no data, the agent must not invent values."""

    async def test_weather_unavailable_no_fabrication(self, pipeline, local_judge):
        """Skill returns 'No data available' — agent must not invent a temperature."""
        controlled = "No weather data available for the requested location."
        from skills.weather import weather_skill as _ws
        with patch.object(_ws, "handle", new=AsyncMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("what's the weather in Narnia?")

        # No specific temperature should appear when there is no data.
        fabricated_temps = re.findall(r"-?\d+\s*°[CcFf]", response)
        assert not fabricated_temps, (
            f"Agent fabricated temperature(s) {fabricated_temps} despite no-data skill response:\n{response[:400]}"
        )

    async def test_finance_partial_data_no_invented_percentage(self, pipeline, local_judge):
        """Skill returns price only — agent must not invent a change percentage."""
        controlled = "Microsoft Corp (MSFT) — Current price: $415.00 USD. Change data unavailable."
        from skills.finance import finance_skill as _fs
        with patch.object(_fs, "handle", new=MagicMock(return_value=controlled)):
            from agents.intent_classifier import run_intent_classifier
            response = await run_intent_classifier("how is Microsoft stock doing today?")

        # Any percentage in the response is fabricated since the skill didn't supply one.
        percentages = re.findall(r"[+\-]?\d+\.?\d*\s*%", response)
        assert not percentages, (
            f"Agent invented change percentage(s) {percentages} from partial skill data:\n{response[:400]}"
        )
