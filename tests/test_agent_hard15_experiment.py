import unittest

from agent.hard15_experiment import evaluate_attempts, experiment_fingerprint
from agent.hard15_pilot import Hard15Evaluation
from agent.hard15_planning import PlanningAttempt
from agent.llm import PlanStep, SkillPlan


class Hard15ExperimentTests(unittest.TestCase):
    def test_fingerprint_changes_with_fairness_configuration(self):
        first = experiment_fingerprint("m", 8, 100, "revision")
        self.assertEqual(first, experiment_fingerprint("m", 8, 100, "revision"))
        self.assertNotEqual(first, experiment_fingerprint("m", 9, 100, "revision"))

    def test_metrics_are_planning_metrics_and_resolve_aliases_privately(self):
        plan = SkillPlan(
            selected_skill_aliases=("S01",),
            steps=(PlanStep(id="1", objective="do", skill_aliases=("S01",), expected_output="x"),),
            final_output="x",
        )
        attempt = PlanningAttempt(
            task_id="x",
            method="graph",
            model="m",
            fingerprint="f",
            valid=True,
            plan=plan,
            selected_candidate_aliases=("S01", "S02"),
            rendered_characters=20,
            omitted_count=0,
            prompt_tokens=10,
            completion_tokens=5,
            organization_metadata={"node_count": 2, "edge_count": 1},
        )
        evaluation = Hard15Evaluation(
            task_id="x",
            gt_skill_ids=("gt/a", "gt/b"),
            alias_to_skill_id={"S01": "gt/a", "S02": "other/c"},
            full_candidate_coverage=False,
        )
        summary = evaluate_attempts([attempt], [evaluation])
        self.assertEqual(summary["methods"]["graph"]["valid_plan_rate"], 1.0)
        self.assertEqual(summary["methods"]["graph"]["mean_selected_gt_coverage"], 0.5)
        self.assertEqual(summary["methods"]["graph"]["complete_gt_coverage_rate"], 0.0)
        self.assertNotIn("task_success_rate", summary["methods"]["graph"])


if __name__ == "__main__":
    unittest.main()
