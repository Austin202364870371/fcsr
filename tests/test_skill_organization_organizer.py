import json
import unittest

from skill_organization.models import FrozenSkill, SkillRecord, TaskInput
from skill_organization.organizer import (
    EvidenceEdge,
    EvidenceGraph,
    Hierarchy,
    HierarchyGroup,
    OrganizationBundle,
    build_organizer_messages,
    reading_order,
    validate_bundle,
)


def make_task_input() -> TaskInput:
    return TaskInput(
        task_key="T001",
        task_id="secret-task-id",
        skills=tuple(
            FrozenSkill(
                alias=f"S{index:02d}",
                rank=index,
                record=SkillRecord(
                    skill_id=f"gt/secret-{index}",
                    name=f"Skill {index}",
                    description=f"Description {index}",
                    body=f"Body {index} produces artifact {index}",
                    source="gt",
                ),
            )
            for index in range(1, 9)
        ),
    )


def valid_hierarchy() -> Hierarchy:
    return Hierarchy(
        roots=(
            HierarchyGroup(
                label="Procedures",
                children=(
                    HierarchyGroup(
                        label="Primary operations",
                        skills=tuple(f"S{index:02d}" for index in range(1, 9)),
                    ),
                ),
            ),
        )
    )


def empty_graph() -> EvidenceGraph:
    return EvidenceGraph(nodes=tuple(f"S{index:02d}" for index in range(1, 9)))


class OrganizerTests(unittest.TestCase):
    def test_hierarchy_must_cover_every_alias_once(self):
        hierarchy = Hierarchy(
            roots=(HierarchyGroup(label="Data", skills=("S01", "S01")),)
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_bundle(
                make_task_input(),
                OrganizationBundle(hierarchy=hierarchy, graph=empty_graph()),
            )

    def test_hierarchy_rejects_more_than_three_levels_including_alias(self):
        hierarchy = Hierarchy(
            roots=(
                HierarchyGroup(
                    label="One",
                    children=(
                        HierarchyGroup(
                            label="Two",
                            children=(
                                HierarchyGroup(
                                    label="Three",
                                    skills=tuple(
                                        f"S{index:02d}" for index in range(1, 9)
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "depth"):
            validate_bundle(
                make_task_input(),
                OrganizationBundle(hierarchy=hierarchy, graph=empty_graph()),
            )

    def test_graph_evidence_must_be_exact_source_and_target_substrings(self):
        graph = EvidenceGraph(
            nodes=tuple(f"S{index:02d}" for index in range(1, 9)),
            edges=(
                EvidenceEdge(
                    source="S01",
                    target="S02",
                    edge_type="produces_requires",
                    source_evidence="not in source",
                    target_evidence="Body 2",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "source evidence"):
            validate_bundle(
                make_task_input(),
                OrganizationBundle(hierarchy=valid_hierarchy(), graph=graph),
            )

    def test_graph_evidence_cannot_cross_visible_field_boundaries(self):
        graph = EvidenceGraph(
            nodes=tuple(f"S{index:02d}" for index in range(1, 9)),
            edges=(
                EvidenceEdge(
                    source="S01",
                    target="S02",
                    edge_type="setup_execute",
                    source_evidence="Skill 1\nDescription 1",
                    target_evidence="Body 2",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "source evidence"):
            validate_bundle(
                make_task_input(),
                OrganizationBundle(hierarchy=valid_hierarchy(), graph=graph),
            )

    def test_zero_edge_graph_falls_back_to_rank_order(self):
        bundle = OrganizationBundle(hierarchy=valid_hierarchy(), graph=empty_graph())
        validate_bundle(make_task_input(), bundle)
        self.assertEqual(
            reading_order(make_task_input(), bundle.graph),
            tuple(f"S{index:02d}" for index in range(1, 9)),
        )

    def test_cycles_are_condensed_and_ordered_by_rank(self):
        graph = EvidenceGraph(
            nodes=tuple(f"S{index:02d}" for index in range(1, 9)),
            edges=(
                EvidenceEdge(
                    source="S02",
                    target="S01",
                    edge_type="setup_execute",
                    source_evidence="Body 2",
                    target_evidence="Body 1",
                ),
                EvidenceEdge(
                    source="S01",
                    target="S02",
                    edge_type="execute_verify",
                    source_evidence="Body 1",
                    target_evidence="Body 2",
                ),
            ),
        )
        bundle = OrganizationBundle(hierarchy=valid_hierarchy(), graph=graph)
        validate_bundle(make_task_input(), bundle)
        self.assertEqual(reading_order(make_task_input(), graph)[:2], ("S01", "S02"))

    def test_organizer_messages_do_not_contain_provenance_or_task_id(self):
        messages = build_organizer_messages(
            task_key="T001", skills=make_task_input().organizer_view()
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertIn("T001", serialized)
        self.assertNotIn("secret-task-id", serialized)
        self.assertNotIn("gt/", serialized)
        self.assertNotIn('"skill_id"', serialized)
        self.assertNotIn('"source"', serialized)

    def test_organizer_messages_define_the_exact_output_schema(self):
        messages = build_organizer_messages(
            task_key="T001", skills=make_task_input().organizer_view()
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        for required in (
            "skill-hierarchy-v1",
            "skill-graph-v1",
            "roots",
            "edge_type",
            "source_evidence",
            "target_evidence",
        ):
            self.assertIn(required, serialized)

    def test_validation_feedback_requests_a_complete_replacement(self):
        messages = build_organizer_messages(
            task_key="T001",
            skills=make_task_input().organizer_view(),
            validation_feedback="hierarchy.roots: Field required",
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("hierarchy.roots: Field required", messages[-1]["content"])
        self.assertIn("complete JSON object", messages[-1]["content"])
        self.assertNotIn("secret-task-id", serialized)
        self.assertNotIn("gt/", serialized)


if __name__ == "__main__":
    unittest.main()
