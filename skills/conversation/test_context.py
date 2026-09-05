import unittest

from skills.conversation import skill
from utils.resource_monitor import count_tokens


class ConversationContextTests(unittest.TestCase):
    def setUp(self) -> None:
        skill._history.clear()
        skill._rolling_summary = None

    def tearDown(self) -> None:
        skill._history.clear()
        skill._rolling_summary = None

    def test_history_entries_are_compacted(self) -> None:
        skill.append_to_history("user " + "word " * 300, "assistant " + "answer " * 500)
        self.assertLessEqual(len(skill._history[0].content), skill.HISTORY_USER_MAX_CHARS)
        self.assertLessEqual(len(skill._history[1].content), skill.HISTORY_ASSISTANT_MAX_CHARS)

    def test_summary_is_bounded_when_restored(self) -> None:
        skill.restore_session_context("summary " * 1000, [])
        self.assertLessEqual(len(skill._rolling_summary or ""), skill.SUMMARY_MAX_CHARS)

    def test_context_stays_within_budget(self) -> None:
        skill._rolling_summary = "older " * 500
        for index in range(10):
            skill.append_to_history(f"question {index} " + "detail " * 100, "answer " + "detail " * 200)
        context = skill._fit_context_to_budget(
            skill._chat_context_sections(), max_tokens=300
        )
        self.assertLessEqual(count_tokens(context), 300)


if __name__ == "__main__":
    unittest.main()
