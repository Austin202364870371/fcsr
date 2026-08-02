import json
import unittest

from agent.hard_pilot import InsufficientEligibleTasks, prepare_hard_pilot


class HardPilotTests(unittest.TestCase):
    def test_prepares_deterministic_stratified_anonymous_sample(self) -> None:
        queries, rankings, skills = self.fixtures()

        first = prepare_hard_pilot(
            queries,
            rankings,
            skills,
            eligible_task_ids={record["query_id"] for record in queries},
            seed=42,
        )
        second = prepare_hard_pilot(
            queries,
            rankings,
            skills,
            eligible_task_ids={record["query_id"] for record in queries},
            seed=42,
        )

        self.assertEqual(first.public_tasks, second.public_tasks)
        self.assertEqual(len(first.public_tasks), 15)
        self.assertEqual(
            sum(len(item.gt_skill_ids) == 1 for item in first.evaluations), 5
        )
        self.assertEqual(
            sum(len(item.gt_skill_ids) > 1 for item in first.evaluations), 10
        )
        for task in first.public_tasks:
            self.assertEqual(
                [skill.alias for skill in task.skills],
                ["S01", "S02", "S03"],
            )
            serialized = json.dumps(task.model_dump(mode="json"))
            self.assertNotIn("gt/", serialized)
            self.assertNotIn("distractor/", serialized)
            self.assertNotIn("source", serialized)

    def test_refuses_to_fill_quota_with_ineligible_tasks(self) -> None:
        queries, rankings, skills = self.fixtures()

        with self.assertRaisesRegex(InsufficientEligibleTasks, "eligible tasks"):
            prepare_hard_pilot(
                queries,
                rankings,
                skills,
                eligible_task_ids=set(),
                seed=42,
            )

    @staticmethod
    def fixtures():
        queries = []
        rankings = []
        skills = []
        for index in range(3):
            skills.extend(
                [
                    {
                        "skill_id": f"gt/skill-{index}",
                        "name": f"Skill {index}",
                        "description": "Useful instructions",
                        "body": "Follow these steps.",
                        "source": "gt",
                    },
                    {
                        "skill_id": f"other/a-{index}",
                        "name": f"Other A {index}",
                        "description": "Alternative",
                        "body": "Alternative steps.",
                        "source": "pool",
                    },
                    {
                        "skill_id": f"distractor/b-{index}",
                        "name": f"Other B {index}",
                        "description": "Distractor",
                        "body": "Distracting steps.",
                        "source": "distractor",
                    },
                ]
            )

        bucket_specs = [
            ("single-covered", 4, 1, True),
            ("single-missing", 3, 1, False),
            ("multi-covered", 6, 2, True),
            ("multi-missing", 6, 2, False),
        ]
        serial = 0
        for label, count, gt_count, covered in bucket_specs:
            for _ in range(count):
                task_id = f"{label}-{serial}"
                gt_ids = ["gt/skill-0"]
                if gt_count == 2:
                    gt_ids.append("other/a-0" if covered else "gt/missing")
                elif not covered:
                    gt_ids = ["gt/missing"]
                queries.append(
                    {
                        "query_id": task_id,
                        "query": f"Solve task {serial}",
                        "gt_skill_ids": gt_ids,
                        "domain": "test",
                    }
                )
                rankings.append(
                    {
                        "query_id": task_id,
                        "reranked_candidates": [
                            {
                                "skill_id": "gt/skill-0",
                                "reranker_score": 3.0,
                            },
                            {
                                "skill_id": "other/a-0",
                                "reranker_score": 2.0,
                            },
                            {
                                "skill_id": "distractor/b-0",
                                "reranker_score": 1.0,
                            },
                        ],
                    }
                )
                serial += 1
        return queries, rankings, skills


if __name__ == "__main__":
    unittest.main()
