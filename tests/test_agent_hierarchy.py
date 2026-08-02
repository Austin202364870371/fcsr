import unittest

from agent.hierarchy import HierarchyOrganizer, HierarchySkillBundle
from agent.models import SkillCandidate


def candidate(
    skill_id: str,
    rank: int,
    category_path: tuple[str, ...],
) -> SkillCandidate:
    return SkillCandidate(
        skill_id=skill_id,
        name=skill_id,
        description=f"Skill {skill_id}",
        rank=rank,
        score=1.0 / rank,
        tool_name=f"tool_{skill_id}",
        category_path=category_path,
    )


class HierarchyOrganizerTests(unittest.TestCase):
    def test_selects_multiple_groups_and_preserves_global_skill_rank(self) -> None:
        candidates = [
            candidate("json", 1, ("data", "json")),
            candidate("upper", 2, ("text", "transform")),
            candidate("csv", 3, ("data", "csv")),
        ]

        bundle = HierarchyOrganizer(
            max_groups=2,
            max_skills=3,
            category_depth=1,
        ).organize(candidates)

        self.assertIsInstance(bundle, HierarchySkillBundle)
        self.assertEqual(bundle.strategy, "hierarchy")
        self.assertEqual([group.label for group in bundle.groups], ["data", "text"])
        self.assertEqual(
            [skill.skill_id for skill in bundle.skills],
            ["json", "upper", "csv"],
        )

    def test_top_group_gate_keeps_all_selected_group_members_within_budget(self) -> None:
        candidates = [
            candidate("json", 1, ("data", "json")),
            candidate("upper", 2, ("text", "transform")),
            candidate("csv", 3, ("data", "csv")),
        ]

        bundle = HierarchyOrganizer(
            max_groups=1,
            max_skills=3,
            category_depth=1,
        ).organize(candidates)

        self.assertEqual([group.label for group in bundle.groups], ["data"])
        self.assertEqual(
            [skill.skill_id for skill in bundle.skills],
            ["json", "csv"],
        )

    def test_missing_category_is_never_dropped(self) -> None:
        bundle = HierarchyOrganizer(max_groups=1, max_skills=1).organize(
            [candidate("unknown", 1, ())]
        )

        self.assertEqual(bundle.groups[0].category_path, ("uncategorized",))
        self.assertEqual(bundle.skills[0].skill_id, "unknown")

    def test_rejects_invalid_budgets(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_groups"):
            HierarchyOrganizer(max_groups=0, max_skills=1)
        with self.assertRaisesRegex(ValueError, "max_skills"):
            HierarchyOrganizer(max_groups=1, max_skills=0)
        with self.assertRaisesRegex(ValueError, "category_depth"):
            HierarchyOrganizer(max_groups=1, max_skills=1, category_depth=0)


if __name__ == "__main__":
    unittest.main()
