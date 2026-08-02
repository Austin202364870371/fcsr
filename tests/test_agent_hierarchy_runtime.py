import unittest

from agent.hierarchy import HierarchyOrganizer
from agent.models import SkillCandidate
from agent.runtime import FlatSkillAgent
from agent.selectors import FirstRankedSelector
from agent.tools import ToolRegistry
from agent.verifiers import VerifierRegistry


class HierarchyRuntimeTests(unittest.TestCase):
    def test_existing_runtime_executes_hierarchy_bundle(self) -> None:
        tools = ToolRegistry()
        tools.register("uppercase", lambda text: text.upper())
        agent = FlatSkillAgent(
            organizer=HierarchyOrganizer(max_groups=2, max_skills=2),
            selector=FirstRankedSelector(
                argument_builder=lambda task, skill: {"text": task}
            ),
            tools=tools,
            verifiers=VerifierRegistry.with_defaults(),
        )
        candidates = [
            SkillCandidate(
                skill_id="text/uppercase",
                name="uppercase",
                description="Uppercase text.",
                rank=1,
                score=1.0,
                tool_name="uppercase",
                category_path=("text", "transform"),
            ),
            SkillCandidate(
                skill_id="data/json",
                name="json",
                description="Parse JSON.",
                rank=2,
                score=0.5,
                tool_name="parse_json",
                category_path=("data", "json"),
            ),
        ]

        result = agent.run(
            task_id="hierarchy-uppercase-1",
            task="hello",
            candidates=candidates,
            verifier_id="exact",
            expected="HELLO",
        )

        self.assertEqual(result.termination_reason, "verified")
        self.assertEqual(result.trace[0]["event"], "organized")
        self.assertEqual(result.trace[0]["strategy"], "hierarchy")
        self.assertEqual(
            result.trace[0]["skill_ids"],
            ["text/uppercase", "data/json"],
        )


if __name__ == "__main__":
    unittest.main()
