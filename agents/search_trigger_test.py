import unittest

# Initialize the registry first, matching the application's normal import
# order and avoiding the package's legacy direct-import cycle.
import skills.registry  # noqa: F401

from agents.search.agent import (
	_canonicalize_deep_research_trigger,
	_is_deep_research,
	_should_use_topic_hint,
	_strip_trigger,
	_topic_hint_for_search,
)


class SearchTriggerTests(unittest.TestCase):
	def test_typo_keeps_deep_research_mode(self) -> None:
		query = "deep researh: Gokhan Demirtas"
		self.assertEqual(
			_canonicalize_deep_research_trigger(query),
			"deep research: Gokhan Demirtas",
		)
		self.assertTrue(_is_deep_research(query))

	def test_deep_research_colon_is_stripped(self) -> None:
		self.assertEqual(_strip_trigger("deep research: Gokhan Demirtas"), "Gokhan Demirtas")

	def test_regular_search_is_not_deep_research(self) -> None:
		self.assertFalse(_is_deep_research("search Gokhan Demirtas"))

	def test_new_research_subject_does_not_inherit_old_topic(self) -> None:
		self.assertFalse(_should_use_topic_hint("deep research: apple pie", "blue sky"))
		self.assertTrue(_should_use_topic_hint("deep research: it", "blue sky"))
		self.assertTrue(_should_use_topic_hint("look it up", "blue sky"))

	def test_quick_search_keeps_existing_topic_lock(self) -> None:
		self.assertEqual(
			_topic_hint_for_search("search: apple pie", "blue sky"), "blue sky"
		)
		self.assertEqual(
			_topic_hint_for_search("deep research: apple pie", "blue sky"), ""
		)


if __name__ == "__main__":
	unittest.main()
