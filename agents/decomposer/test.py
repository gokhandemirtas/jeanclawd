"""Fast unit checks for the request-decomposition contract."""

import unittest

from agents.decomposer.agent import DecomposedTask, DecompositionResult
from utils.pending_plan import is_cancellation, is_confirmation, set_pending, take_pending


class DecomposerTests(unittest.TestCase):
	def test_task_limit_and_order(self) -> None:
		tasks = [
			DecomposedTask(description=str(i), intent="SEARCH", query=str(i))
			for i in range(10)
		]
		result = DecompositionResult(is_compound=True, tasks=tasks)
		self.assertEqual([task.description for task in result.tasks], [str(i) for i in range(8)])

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


if __name__ == "__main__":
	unittest.main()
