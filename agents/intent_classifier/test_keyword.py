import unittest

from agents.intent_classifier.intent_classifier import _explicit_command_classify


class ExplicitCommandTests(unittest.TestCase):
    def test_command_typo_is_left_to_context_router(self) -> None:
        self.assertIsNone(
            _explicit_command_classify("deep resarch: Gokhan Demirtas")
        )

    def test_non_command_is_left_to_context_router(self) -> None:
        self.assertIsNone(_explicit_command_classify("wether in London"))

    def test_research_mention_is_left_to_context_router(self) -> None:
        self.assertIsNone(_explicit_command_classify("ordinary conversation about research"))

    def test_research_history_question_is_left_to_context_router(self) -> None:
        self.assertIsNone(
            _explicit_command_classify("which metal genres did you research earlier?")
        )

    def test_research_we_did_in_topic_is_left_to_context_router(self) -> None:
        self.assertIsNone(
            _explicit_command_classify("tell me about the research we did in this topic")
        )

    def test_standalone_research_command_remains_search(self) -> None:
        self.assertEqual(_explicit_command_classify("research heavy metal"), "SEARCH")

    def test_anchored_search_command_remains_search(self) -> None:
        self.assertEqual(_explicit_command_classify("search for black metal"), "SEARCH")


if __name__ == "__main__":
    unittest.main()
