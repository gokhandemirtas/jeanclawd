import unittest

from agents.intent_classifier.intent_classifier import _keyword_classify


class ApproximateCommandTests(unittest.TestCase):
    def test_deep_research_typo(self) -> None:
        self.assertEqual(
            _keyword_classify("deep resarch: Gokhan Demirtas"), "SEARCH"
        )

    def test_weather_typo(self) -> None:
        self.assertEqual(_keyword_classify("wether in London"), "WEATHER")

    def test_ordinary_research_conversation_is_not_rewritten(self) -> None:
        self.assertIsNone(_keyword_classify("ordinary conversation about research"))


if __name__ == "__main__":
    unittest.main()
