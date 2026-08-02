import json
import unittest

from agent.hard15_experiment import compatible_completed, experiment_fingerprint
from agent.hard15_organizations import EvidenceEdge, _reading_order, build_evidence_graph
from agent.hard15_pilot import Hard15Skill, Hard15Task, prepare_fixed_hard15
from agent.hard15_planning import PlanningAttempt


def card(alias, skill_id, name, rank, body=""):
    return Hard15Skill(
        alias=alias,
        skill_id=skill_id,
        name=name,
        body=body,
        rank=rank,
        reranker_score=0.0,
        category_path=("x",),
    )


class Hard15ReviewConstraintTests(unittest.TestCase):
    def test_public_task_dump_excludes_gt_derived_stratum_and_private_ids(self):
        task = Hard15Task(
            task_id="x",
            task="do x",
            domain="test",
            stratum="single_full",
            skills=(card("S01", "gt/private", "private", 1),),
        )
        dumped = json.dumps(task.model_dump(mode="json"))
        self.assertNotIn("stratum", dumped)
        self.assertNotIn("gt/private", dumped)
        self.assertNotIn("skill_id", dumped)

    def test_fingerprint_binds_actual_input_digest(self):
        first = experiment_fingerprint("m", 8, 100, "r", input_digest="a")
        second = experiment_fingerprint("m", 8, 100, "r", input_digest="b")
        self.assertNotEqual(first, second)

    def test_incompatible_checkpoint_is_rejected(self):
        record = PlanningAttempt(
            task_id="x",
            method="flat",
            model="m",
            fingerprint="old",
            valid=False,
            error="failed",
            selected_candidate_aliases=("S01",),
            rendered_characters=1,
            omitted_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            organization_metadata={},
        )
        with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
            compatible_completed([record], "new")

    def test_main_experiment_rejects_non_top20_configuration(self):
        with self.assertRaisesRegex(ValueError, "exactly Top-20"):
            prepare_fixed_hard15(None, [], [], [], candidate_limit=19)  # type: ignore[arg-type]

    def test_skill_id_match_requires_a_complete_token(self):
        skills = (
            card("S01", "data/parser-extra", "Source", 1, "mentions data/parser-extra only"),
            card("S02", "data/parser", "Target", 2),
        )
        graph = build_evidence_graph(skills)
        explicit = [edge for edge in graph.edges if edge.edge_type == "explicit_reference"]
        self.assertEqual(explicit, [])

    def test_scc_order_preserves_edges_to_cycle_downstream(self):
        skills = (
            card("S01", "a/1", "one", 1),
            card("S02", "b/2", "two", 2),
            card("S03", "c/3", "three", 3),
        )
        edges = (
            EvidenceEdge(source="S01", target="S02", edge_type="explicit_reference", directed=True, evidence="x"),
            EvidenceEdge(source="S02", target="S01", edge_type="explicit_reference", directed=True, evidence="x"),
            EvidenceEdge(source="S02", target="S03", edge_type="explicit_reference", directed=True, evidence="x"),
        )
        order = _reading_order(skills, edges)
        self.assertLess(order.index("S02"), order.index("S03"))


if __name__ == "__main__":
    unittest.main()
