import unittest

from agent.candidates import adapt_ranked_candidates
from agent.models import SkillBundle, SkillCandidate
from agent.presentation import render_skill_bundle
from agent.selectors import FirstRankedSelector


class InstructionSkillTests(unittest.TestCase):
    def test_candidate_without_tool_binding_retains_instruction_body(self) -> None:
        candidates = adapt_ranked_candidates(
            {"ranked_skill_ids": ["S01"]},
            {
                "S01": {
                    "name": "Mesh analysis",
                    "description": "Analyze meshes",
                    "body": "Parse the STL, then verify volume.",
                }
            },
        )

        self.assertIsNone(candidates[0].tool_name)
        self.assertEqual(
            candidates[0].body,
            "Parse the STL, then verify volume.",
        )

    def test_instruction_only_skill_is_not_treated_as_executable_tool(self) -> None:
        skill = SkillCandidate(
            skill_id="S01",
            name="Mesh analysis",
            description="Analyze meshes",
            body="Parse an STL.",
            rank=1,
            score=1.0,
            tool_name=None,
        )
        bundle = SkillBundle(strategy="flat", skills=(skill,))

        self.assertIsNone(FirstRankedSelector().select("task", bundle))
        rendered = render_skill_bundle(bundle)
        self.assertNotIn("tool=None", rendered)
        self.assertIn("instruction-only", rendered)


if __name__ == "__main__":
    unittest.main()
