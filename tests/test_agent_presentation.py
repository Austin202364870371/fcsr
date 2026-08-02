import unittest

from agent.hierarchy import HierarchyOrganizer
from agent.models import SkillBundle, SkillCandidate
from agent.presentation import organization_stats, render_skill_bundle


def candidate(
    skill_id: str,
    rank: int,
    category: str,
) -> SkillCandidate:
    return SkillCandidate(
        skill_id=skill_id,
        name=skill_id.replace("/", "-"),
        description=f"Description for {skill_id}.",
        rank=rank,
        score=1.0 / rank,
        tool_name=f"tool_{rank}",
        category_path=(category,),
    )


class BundlePresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skills = [
            candidate("data/json", 1, "data"),
            candidate("text/upper", 2, "text"),
        ]

    def test_flat_bundle_renders_one_ranked_list(self) -> None:
        bundle = SkillBundle(strategy="flat", skills=self.skills)

        rendered = render_skill_bundle(bundle)
        stats = organization_stats(bundle)

        self.assertIn("# Skill bundle: flat", rendered)
        self.assertIn("1. `data/json`", rendered)
        self.assertIn("2. `text/upper`", rendered)
        self.assertNotIn("## Category:", rendered)
        self.assertEqual(stats.strategy, "flat")
        self.assertEqual(stats.group_count, 0)
        self.assertEqual(stats.skill_count, 2)

    def test_hierarchy_bundle_renders_category_headings(self) -> None:
        bundle = HierarchyOrganizer(max_groups=2, max_skills=2).organize(
            self.skills
        )

        rendered = render_skill_bundle(bundle)
        stats = organization_stats(bundle)

        self.assertIn("# Skill bundle: hierarchy", rendered)
        self.assertIn("## Category: data", rendered)
        self.assertIn("## Category: text", rendered)
        self.assertIn("`data/json`", rendered)
        self.assertIn("`text/upper`", rendered)
        self.assertEqual(stats.strategy, "hierarchy")
        self.assertEqual(stats.group_count, 2)
        self.assertEqual(stats.skill_count, 2)
        self.assertEqual(stats.rendered_characters, len(rendered))


if __name__ == "__main__":
    unittest.main()
