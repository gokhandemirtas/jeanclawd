"""Fast unit checks for the request-decomposition contract."""

import unittest

from agents.decomposer.agent import DecomposedTask, DecompositionResult, _normalize, extract_batch_request
from utils.pending_plan import is_cancellation, is_confirmation, set_pending, take_pending


class DecomposerTests(unittest.TestCase):
	def test_task_limit_and_order(self) -> None:
		tasks = [
			DecomposedTask(description=str(i), intent="SEARCH", query=str(i))
			for i in range(10)
		]
		result = DecompositionResult(is_compound=True, tasks=tasks)
		self.assertEqual([task.description for task in result.tasks], [str(i) for i in range(8)])

	def test_internal_intent_aliases_are_normalized(self) -> None:
		result = DecompositionResult(
			is_compound=True,
			tasks=[
				{"description": "turn lights off", "intent": "turn_light", "query": "turn lights off"},
				{"description": "set reminder", "intent": "set_reminder", "query": "remind me tomorrow"},
				{"description": "research", "intent": "research_topic", "query": "research Buffy"},
			],
		)
		self.assertEqual([task.intent for task in result.tasks], ["HOME_CONTROL", "REMINDER", "SEARCH"])

	def test_confirmation_and_cancellation(self) -> None:
		self.assertTrue(is_confirmation("yes"))
		self.assertTrue(is_confirmation("Y"))
		self.assertTrue(is_cancellation("no"))
		self.assertTrue(is_cancellation("cancel"))

	def test_pending_plan_is_consumed(self) -> None:
		task = DecomposedTask(description="Find it", intent="SEARCH", query="it")
		set_pending([task])
		self.assertEqual(take_pending(), [task])
		self.assertIsNone(take_pending())

	def test_model_cannot_split_single_sentence_without_boundary(self) -> None:
		result = DecompositionResult(
			is_compound=True,
			tasks=[
				{"description": "greeting", "intent": "CHAT", "query": "hey there"},
				{"description": "dialogue", "intent": "CHAT", "query": "glum chum"},
			],
		)
		normalized = _normalize(result, "hey there glum chum,")
		self.assertFalse(normalized.is_compound)
		self.assertEqual(normalized.tasks[0].query, "hey there glum chum,")

	def test_comma_in_one_conversational_request_does_not_enable_split(self) -> None:
		result = DecompositionResult(
			is_compound=True,
			tasks=[
				{"description": "book a flight", "intent": "TRANSPORTATION", "query": "book a flight"},
				{"description": "get the weather", "intent": "WEATHER", "query": "get the weather"},
				{"description": "check a stock price", "intent": "FINANCE_STOCKS", "query": "check a stock price"},
			],
		)
		normalized = _normalize(result, "no I'm testing your speech abilities, so help me test it")
		self.assertFalse(normalized.is_compound)
		self.assertEqual(normalized.tasks[0].query, "no I'm testing your speech abilities, so help me test it")

	def test_batch_request_requires_explicit_prefix(self) -> None:
		self.assertEqual(extract_batch_request("batch request: book a flight; get weather"), "book a flight; get weather")
		self.assertEqual(extract_batch_request("  BATCH REQUEST: check stocks"), "check stocks")
		self.assertIsNone(extract_batch_request("okay?"))
		self.assertIsNone(extract_batch_request("please batch request: check stocks"))


if __name__ == "__main__":
	unittest.main()
