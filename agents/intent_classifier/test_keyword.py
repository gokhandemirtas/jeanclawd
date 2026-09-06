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

    def test_research_history_question_is_chat(self) -> None:
        self.assertEqual(
            _keyword_classify("which metal genres did you research earlier?"), "CHAT"
        )

    def test_research_we_did_in_topic_is_chat(self) -> None:
        self.assertEqual(
            _keyword_classify("tell me about the research we did in this topic"),
            "CHAT",
        )

    def test_standalone_research_command_remains_search(self) -> None:
        self.assertEqual(_keyword_classify("research heavy metal"), "SEARCH")


if __name__ == "__main__":
    unittest.main()
