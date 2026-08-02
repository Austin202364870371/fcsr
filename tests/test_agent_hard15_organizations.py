import unittest

from agent.hard15_organizations import (
    build_evidence_graph,
    normalize_category_path,
    organize_task,
)
from agent.hard15_pilot import Hard15Skill, Hard15Task


def skill(alias, skill_id, name, rank, body=""):
    return Hard15Skill(
        alias=alias,
        skill_id=skill_id,
        name=name,
        description="",
        body=body,
        rank=rank,
        reranker_score=1.0 / rank,
        category_path=normalize_category_path({"skill_id": skill_id}),
    )


class Hard15OrganizationTests(unittest.TestCase):
    def setUp(self):
        self.skills = (
            skill("S01", "data/parser", "Parser", 1, "Use Formatter after parsing."),
            skill("S02", "data/formatter", "Formatter", 2, "format output"),
            skill("S03", "security/scanner", "Scanner", 3, "scan input"),
            skill("S04", "data/exporter", "Exporter", 4, "export data"),
        )
        self.task = Hard15Task(
            task_id="task-x",
            task="Produce an artifact",
            domain="test",
            stratum="multi_full",
            skills=self.skills,
        )

    def test_category_normalization_precedence_and_namespace_fallback(self):
        self.assertEqual(
            normalize_category_path({"skill_id": "x/y", "category_path": ["A", "B"]}),
            ("A", "B"),
        )
        self.assertEqual(
            normalize_category_path({"skill_id": "x/y", "category": "C/D"}),
            ("C", "D"),
        )
        self.assertEqual(normalize_category_path({"skill_id": "data/parser"}), ("data",))
        self.assertEqual(normalize_category_path({"skill_id": "standalone"}), ("uncategorized",))

    def test_graph_contains_only_evidence_supported_typed_edges(self):
        graph = build_evidence_graph(self.skills)
        explicit = [edge for edge in graph.edges if edge.edge_type == "explicit_reference"]
        namespace = [edge for edge in graph.edges if edge.edge_type == "same_namespace"]
        self.assertEqual([(edge.source, edge.target) for edge in explicit], [("S01", "S02")])
        self.assertEqual(len(namespace), 3)
        self.assertTrue(all(edge.evidence for edge in explicit))

    def test_all_methods_obey_shared_count_and_total_body_budget(self):
        for method in ("flat", "hierarchy", "graph"):
            with self.subTest(method=method):
                organized = organize_task(
                    self.task,
                    method=method,
                    max_skills=3,
                    body_char_budget=12,
                    max_groups=2,
                )
                self.assertLessEqual(len(organized.selected_aliases), 3)
                self.assertLessEqual(sum(len(item.body) for item in organized.skills), 12)
                self.assertEqual(
                    set(organized.selected_aliases) | set(organized.omitted_aliases),
                    {item.alias for item in self.skills},
                )
                self.assertNotIn("data/parser", organized.rendered_prompt)

    def test_graph_output_is_deterministic_and_complete(self):
        first = organize_task(self.task, method="graph", max_skills=4, body_char_budget=100)
        second = organize_task(self.task, method="graph", max_skills=4, body_char_budget=100)
        self.assertEqual(first, second)
        self.assertEqual(first.metadata["node_count"], 4)
        self.assertEqual(first.metadata["explicit_reference_count"], 1)
        self.assertEqual(set(first.metadata["reading_order"]), set(first.selected_aliases))


if __name__ == "__main__":
    unittest.main()
