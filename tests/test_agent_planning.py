import json
import unittest

from agent.hard_pilot import PublicInstructionSkill, PublicPilotTask
from agent.planning import LLMReply, plan_task


class FakePlanningClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def complete(self, *, model, messages):
        self.messages = messages
        return LLMReply(
            content=self.content,
            prompt_tokens=100,
            completion_tokens=30,
        )


class PlanningTests(unittest.TestCase):
    def test_generates_valid_plan_from_anonymous_skill_cards(self) -> None:
        client = FakePlanningClient(
            json.dumps(
                {
                    "selected_skill_aliases": ["S01"],
                    "steps": [
                        {
                            "id": "step-1",
                            "objective": "Analyze the input",
                            "skill_aliases": ["S01"],
                            "expected_output": "A validated result",
                        }
                    ],
                    "final_output": "Write the requested artifact",
                }
            )
        )

        result = plan_task(self.task(), client=client)

        self.assertEqual(result.plan.selected_skill_aliases, ("S01",))
        self.assertEqual(result.prompt_tokens, 100)
        prompt = json.dumps(client.messages)
        self.assertNotIn("gt/", prompt)
        self.assertNotIn("distractor/", prompt)
        self.assertIn("S01", prompt)
        self.assertIn("Analyze meshes", prompt)

    def test_rejects_alias_not_present_in_candidates(self) -> None:
        client = FakePlanningClient(
            json.dumps(
                {
                    "selected_skill_aliases": ["S99"],
                    "steps": [
                        {
                            "id": "step-1",
                            "objective": "Do work",
                            "skill_aliases": ["S99"],
                            "expected_output": "result",
                        }
                    ],
                    "final_output": "result",
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "unknown Skill aliases"):
            plan_task(self.task(), client=client)

    @staticmethod
    def task() -> PublicPilotTask:
        return PublicPilotTask(
            task_id="task-1",
            task="Analyze a mesh",
            domain="engineering",
            skills=(
                PublicInstructionSkill(
                    alias="S01",
                    name="Mesh analysis",
                    description="Analyze meshes",
                    body="Parse the mesh and verify its volume.",
                    rank=1,
                    reranker_score=3.0,
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
