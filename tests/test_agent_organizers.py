import unittest

from agent.models import SkillCandidate
from agent.organizers import FlatOrganizer


class FlatOrganizerTests(unittest.TestCase):
    def test_flat_organizer_keeps_rank_and_budget(self) -> None:
        candidates = [
            SkillCandidate(
                skill_id=f"s{index}",
                name=f"s{index}",
                description="x",
                rank=index,
                score=1 / index,
                tool_name=f"tool_{index}",
            )
            for index in range(1, 4)
        ]

        bundle = FlatOrganizer(max_skills=2).organize(candidates)

        self.assertEqual(bundle.strategy, "flat")
        self.assertEqual([item.skill_id for item in bundle.skills], ["s1", "s2"])

    def test_flat_organizer_restores_rank_order(self) -> None:
        first = SkillCandidate(
            skill_id="s1",
            name="s1",
            rank=1,
            score=1.0,
            tool_name="tool_1",
        )
        second = SkillCandidate(
            skill_id="s2",
            name="s2",
            rank=2,
            score=0.5,
            tool_name="tool_2",
        )

        bundle = FlatOrganizer(max_skills=2).organize([second, first])

        self.assertEqual([item.skill_id for item in bundle.skills], ["s1", "s2"])

    def test_flat_organizer_rejects_non_positive_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_skills"):
            FlatOrganizer(max_skills=0)


if __name__ == "__main__":
    unittest.main()
