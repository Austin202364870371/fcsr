import unittest

from agent.models import SkillBundle, SkillCandidate, ToolCall


class AgentModelTests(unittest.TestCase):
    def test_bundle_rejects_duplicate_skill_ids(self) -> None:
        skill = SkillCandidate(
            skill_id="data/json-parse",
            name="json-parse",
            description="Parse JSON input.",
            rank=1,
            score=0.9,
            tool_name="parse_json",
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            SkillBundle(strategy="flat", skills=[skill, skill])

    def test_tool_call_requires_object_arguments(self) -> None:
        with self.assertRaises(ValueError):
            ToolCall(tool_name="parse_json", arguments=["bad"])


if __name__ == "__main__":
    unittest.main()
