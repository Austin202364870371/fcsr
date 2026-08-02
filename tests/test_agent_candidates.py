import unittest

from agent.candidates import CandidateAdapterError, adapt_ranked_candidates


class CandidateAdapterTests(unittest.TestCase):
    def test_preserves_fcsr_order_and_caps_top_k(self) -> None:
        record = {"task_id": "q1", "ranked_skill_ids": ["s2", "s1"]}
        skill_index = {
            "s1": {
                "skill_id": "s1",
                "name": "one",
                "description": "first skill",
                "tool_name": "one_tool",
            },
            "s2": {
                "skill_id": "s2",
                "name": "two",
                "description": "second skill",
                "tool_name": "two_tool",
            },
        }

        result = adapt_ranked_candidates(record, skill_index, limit=1)

        self.assertEqual([item.skill_id for item in result], ["s2"])
        self.assertEqual(result[0].rank, 1)
        self.assertEqual(result[0].score, 1.0)

    def test_rejects_unknown_ranked_skill(self) -> None:
        with self.assertRaisesRegex(CandidateAdapterError, "missing"):
            adapt_ranked_candidates(
                {"task_id": "q1", "ranked_skill_ids": ["missing"]},
                {},
                limit=20,
            )

    def test_rejects_invalid_limit(self) -> None:
        with self.assertRaisesRegex(CandidateAdapterError, "limit"):
            adapt_ranked_candidates(
                {"task_id": "q1", "ranked_skill_ids": []},
                {},
                limit=0,
            )


if __name__ == "__main__":
    unittest.main()
