import unittest
from unittest.mock import MagicMock, patch

from agents.generalist.router import _RouteDecision, generalist_route


class GeneralistRouterContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_router_receives_session_summary_for_semantic_followup(self) -> None:
        provider = MagicMock()
        provider.complete.return_value = _RouteDecision(intent="CHAT", confidence=0.99)

        with patch("agents.generalist.router.get_intent_provider", return_value=provider):
            intent, confidence = await generalist_route(
                "tell me about the research we did in this topic",
                [
                    (
                        "Earlier session summary",
                        "Deep research completed on thrash metal. "
                        "Deep research completed on black metal.",
                    )
                ],
            )

        self.assertEqual((intent, confidence), ("CHAT", 0.99))
        routed_prompt = provider.complete.call_args.args[0][1]["content"]
        self.assertIn("Deep research completed on thrash metal", routed_prompt)
        self.assertIn("Deep research completed on black metal", routed_prompt)
        self.assertIn("tell me about the research we did in this topic", routed_prompt)

    async def test_router_keeps_summary_when_recent_history_is_full(self) -> None:
        provider = MagicMock()
        provider.complete.return_value = _RouteDecision(intent="CHAT", confidence=0.99)

        history = [("Earlier session summary", "Deep research completed on thrash metal.")]
        history.extend((f"question {i}", f"answer {i}") for i in range(4))
        with patch("agents.generalist.router.get_intent_provider", return_value=provider):
            await generalist_route("what do you think about it?", history)

        routed_prompt = provider.complete.call_args.args[0][1]["content"]
        self.assertIn("Deep research completed on thrash metal", routed_prompt)
        self.assertIn("question 3", routed_prompt)
        self.assertNotIn("question 0", routed_prompt)

    async def test_declarative_assistant_statement_is_chat(self) -> None:
        intent, confidence = await generalist_route(
            "you are a local assistant running on my machine",
            [("how does your intent routing work?", "Here is the architecture...")],
        )

        self.assertEqual((intent, confidence), ("CHAT", 1.0))

    async def test_question_about_assistant_remains_model_routed(self) -> None:
        provider = MagicMock()
        provider.complete.return_value = _RouteDecision(intent="INTROSPECTION", confidence=0.99)

        with patch("agents.generalist.router.get_intent_provider", return_value=provider):
            intent, confidence = await generalist_route("are you a local assistant?", [])

        self.assertEqual((intent, confidence), ("INTROSPECTION", 0.99))


if __name__ == "__main__":
    unittest.main()
