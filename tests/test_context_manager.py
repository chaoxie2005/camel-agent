import unittest

from camel_agent.agent_components.memory_context import (
    ContextBudget,
    ContextInput,
    ContextManager,
)


class MessageCountTokenCounter:
    """测试用计数器：每条消息计为一个 token。"""

    def count_tokens_from_messages(self, messages):
        return len(messages)


class ContentLengthTokenCounter:
    """测试用计数器：消息内容的字符数视为 token 数。"""

    def count_tokens_from_messages(self, messages):
        return sum(len(str(message.get("content", ""))) for message in messages)


class ContextManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = ContextManager(
            token_counter=MessageCountTokenCounter(),  # type: ignore[arg-type]
            budget=ContextBudget(max_tokens=6, reserved_output_tokens=1),
        )

    def test_keeps_required_messages_and_recent_history(self):
        result = self.manager.build_context(
            ContextInput(
                system_messages=[{"role": "system", "content": "system"}],
                history_messages=[
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "new question"},
                    {"role": "assistant", "content": "new answer"},
                ],
                current_messages=[{"role": "user", "content": "current"}],
                retrieved_messages=[
                    {"role": "assistant", "content": "memory"}
                ],
            )
        )

        self.assertEqual(
            [message["content"] for message in result.messages],
            ["system", "memory", "new question", "new answer", "current"],
        )
        self.assertEqual(result.token_count, 5)
        self.assertEqual(result.dropped_count, 2)
        self.assertFalse(result.summarized)

    def test_keeps_tool_call_unit_together(self):
        manager = ContextManager(
            token_counter=MessageCountTokenCounter(),  # type: ignore[arg-type]
            budget=ContextBudget(max_tokens=6, reserved_output_tokens=1),
        )
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        }
        result = manager.build_context(
            ContextInput(
                system_messages=[{"role": "system", "content": "system"}],
                history_messages=[
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call],
                    },
                    {
                        "role": "tool",
                        "content": "result",
                        "tool_call_id": "call-1",
                    },
                    {"role": "assistant", "content": "answer"},
                ],
                current_messages=[{"role": "user", "content": "current"}],
                retrieved_messages=[],
            )
        )

        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.dropped_count, 4)

    def test_raises_when_required_messages_exceed_budget(self):
        manager = ContextManager(
            token_counter=MessageCountTokenCounter(),  # type: ignore[arg-type]
            budget=ContextBudget(max_tokens=2, reserved_output_tokens=1),
        )
        with self.assertRaisesRegex(ValueError, "超过输入 token 预算"):
            manager.build_context(
                ContextInput(
                    system_messages=[{"role": "system", "content": "system"}],
                    history_messages=[],
                    current_messages=[{"role": "user", "content": "current"}],
                    retrieved_messages=[],
                )
            )

    def test_validates_budget(self):
        with self.assertRaises(ValueError):
            ContextBudget(max_tokens=100, reserved_output_tokens=100)

    def test_history_stops_at_first_oversized_unit(self):
        manager = ContextManager(
            token_counter=ContentLengthTokenCounter(),  # type: ignore[arg-type]
            budget=ContextBudget(max_tokens=9, reserved_output_tokens=1),
        )
        result = manager.build_context(
            ContextInput(
                system_messages=[{"role": "system", "content": "s"}],
                history_messages=[
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "xxxxx"},
                    {"role": "assistant", "content": "xxxxx"},
                    {"role": "user", "content": "b"},
                    {"role": "assistant", "content": "b"},
                ],
                current_messages=[{"role": "user", "content": "c"}],
                retrieved_messages=[],
            )
        )

        self.assertEqual(
            [message["content"] for message in result.messages],
            ["s", "b", "b", "c"],
        )

    def test_retrieved_messages_skip_oversized_unit(self):
        manager = ContextManager(
            token_counter=ContentLengthTokenCounter(),  # type: ignore[arg-type]
            budget=ContextBudget(max_tokens=6, reserved_output_tokens=1),
        )
        result = manager.build_context(
            ContextInput(
                system_messages=[{"role": "system", "content": "s"}],
                history_messages=[],
                current_messages=[{"role": "user", "content": "c"}],
                retrieved_messages=[
                    {"role": "assistant", "content": "AA"},
                    {"role": "assistant", "content": "xxxxxxxxxx"},
                    {"role": "assistant", "content": "B"},
                ],
            )
        )

        self.assertEqual(
            [message["content"] for message in result.messages],
            ["s", "AA", "B", "c"],
        )

    def test_truncates_oversized_retrieved_message(self):
        manager = ContextManager(
            token_counter=ContentLengthTokenCounter(),  # type: ignore[arg-type]
            budget=ContextBudget(max_tokens=23, reserved_output_tokens=1),
        )
        original_content = "abcdefghijklmnopqrstuvwxyz0123456789"
        original = {"role": "assistant", "content": original_content}

        result = manager.build_context(
            ContextInput(
                system_messages=[{"role": "system", "content": "s"}],
                history_messages=[],
                current_messages=[{"role": "user", "content": "c"}],
                retrieved_messages=[original],
            )
        )

        retrieved_content = result.messages[1]["content"]
        self.assertIsInstance(retrieved_content, str)
        self.assertTrue(retrieved_content.endswith("...[内容已截断]"))
        self.assertLess(len(retrieved_content), len(original_content))
        self.assertLessEqual(
            result.token_count,
            manager.budget.input_token_limit,
        )
        self.assertEqual(original["content"], original_content)

    def test_returns_none_when_truncation_marker_does_not_fit(self):
        manager = ContextManager(
            token_counter=ContentLengthTokenCounter(),  # type: ignore[arg-type]
            budget=ContextBudget(max_tokens=4, reserved_output_tokens=1),
        )

        truncated = manager._truncate_message_to_fit(
            {"role": "assistant", "content": "long message"},
            prefix=[{"role": "system", "content": "s"}],
            suffix=[{"role": "user", "content": "c"}],
        )

        self.assertIsNone(truncated)


if __name__ == "__main__":
    unittest.main()
