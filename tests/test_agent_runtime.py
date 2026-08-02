import unittest

from agent.models import SkillCandidate
from agent.organizers import FlatOrganizer
from agent.runtime import FlatSkillAgent
from agent.selectors import FirstRankedSelector
from agent.tools import ToolRegistry
from agent.verifiers import VerifierRegistry


def uppercase_candidate() -> SkillCandidate:
    return SkillCandidate(
        skill_id="text/uppercase",
        name="uppercase",
        description="Uppercase text",
        rank=1,
        score=1.0,
        tool_name="uppercase",
    )


class FlatSkillAgentTests(unittest.TestCase):
    def build_agent(self, tools: ToolRegistry) -> FlatSkillAgent:
        return FlatSkillAgent(
            organizer=FlatOrganizer(max_skills=5),
            selector=FirstRankedSelector(
                argument_builder=lambda task, skill: {"text": task}
            ),
            tools=tools,
            verifiers=VerifierRegistry.with_defaults(),
        )

    def test_executes_selected_skill_and_verifies_result(self) -> None:
        tools = ToolRegistry()
        tools.register("uppercase", lambda text: text.upper())
        agent = self.build_agent(tools)

        result = agent.run(
            task_id="uppercase-1",
            task="hello",
            candidates=[uppercase_candidate()],
            verifier_id="exact",
            expected="HELLO",
        )

        self.assertEqual(result.termination_reason, "verified")
        self.assertTrue(result.verification.passed)
        self.assertEqual(
            [event["event"] for event in result.trace],
            ["organized", "selected", "tool_executed", "verified"],
        )

    def test_empty_bundle_stops_before_tool_execution(self) -> None:
        agent = self.build_agent(ToolRegistry())

        result = agent.run(
            task_id="empty-1",
            task="hello",
            candidates=[],
            verifier_id="exact",
            expected="HELLO",
        )

        self.assertEqual(result.termination_reason, "selection_failed")
        self.assertIsNone(result.selected_skill_id)
        self.assertIsNone(result.tool_result)

    def test_tool_failure_stops_before_verification(self) -> None:
        agent = self.build_agent(ToolRegistry())

        result = agent.run(
            task_id="missing-tool-1",
            task="hello",
            candidates=[uppercase_candidate()],
            verifier_id="exact",
            expected="HELLO",
        )

        self.assertEqual(result.termination_reason, "execution_failed")
        self.assertFalse(result.tool_result.ok)
        self.assertFalse(result.verification.passed)

    def test_failed_verifier_has_distinct_termination_reason(self) -> None:
        tools = ToolRegistry()
        tools.register("uppercase", lambda text: text.upper())
        agent = self.build_agent(tools)

        result = agent.run(
            task_id="uppercase-wrong-1",
            task="hello",
            candidates=[uppercase_candidate()],
            verifier_id="exact",
            expected="WRONG",
        )

        self.assertEqual(result.termination_reason, "verification_failed")
        self.assertFalse(result.verification.passed)


if __name__ == "__main__":
    unittest.main()
