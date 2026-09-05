import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skills.conversation import skill
from utils.resource_monitor import count_tokens


class ConversationContextTests(unittest.TestCase):
    def setUp(self) -> None:
        skill._history.clear()
        skill._rolling_summary = None
        skill._last_research_report_path = None

    def tearDown(self) -> None:
        skill._history.clear()
        skill._rolling_summary = None
        skill._last_research_report_path = None

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

    def test_research_followup_uses_report_findings(self) -> None:
        with TemporaryDirectory() as directory:
            report = Path(directory) / "research.md"
            report.write_text(
                "# Research\n\n## Final answer\n\nBurzum and Mayhem are relevant names.\n\n"
                "## Research notes\n\nPrivate detail\n",
                encoding="utf-8",
            )
            skill._last_research_report_path = str(report)
            context = skill._research_followup_context(
                "Can you name artists involved in these activities?"
            )
        self.assertIn("Burzum and Mayhem", context)
        self.assertNotIn("Private detail", context)


if __name__ == "__main__":
    unittest.main()
