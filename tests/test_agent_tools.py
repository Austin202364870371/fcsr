import unittest

from agent.models import ToolCall
from agent.tools import ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def test_executes_registered_tool(self) -> None:
        registry = ToolRegistry()
        registry.register("add", lambda left, right: left + right)

        result = registry.execute(
            ToolCall(tool_name="add", arguments={"left": 2, "right": 3})
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.output, 5)
        self.assertIsNone(result.error)

    def test_unknown_tool_is_a_failed_result(self) -> None:
        result = ToolRegistry().execute(
            ToolCall(tool_name="missing", arguments={})
        )

        self.assertFalse(result.ok)
        self.assertIn("unknown tool", result.error)

    def test_tool_exception_becomes_failed_result(self) -> None:
        registry = ToolRegistry()

        def fail() -> None:
            raise ValueError("bad input")

        registry.register("fail", fail)
        result = registry.execute(ToolCall(tool_name="fail", arguments={}))

        self.assertFalse(result.ok)
        self.assertIn("bad input", result.error)

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register("add", lambda: 1)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("add", lambda: 2)


if __name__ == "__main__":
    unittest.main()
