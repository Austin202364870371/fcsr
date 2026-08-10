"""Regression coverage for final-system and two-stage result tables."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation_tables import render_hard_tables


METRICS = {
    "Hit@1": 0.5,
    "MRR@10": 0.6,
    "nDCG@10": 0.55,
    "Recall@10": 0.58,
    "Recall@20": 0.65,
    "FullCoverage@10": 0.4,
}


class EvaluationTableTests(unittest.TestCase):
    def test_renders_separate_final_and_stage_ablation_tables(self) -> None:
        offsets = {
            ("retrieval", "bm25"): 0.01,
            ("retrieval", "dense"): 0.02,
            ("retrieval", "hybrid"): 0.03,
            ("reranker", "dense-base-reranker"): 0.04,
            ("retrieval", "skillrouter"): 0.05,
            ("reranker", "skillrouter"): 0.06,
            ("retrieval", "fcsr-single"): 0.07,
            ("reranker", "fcsr-single"): 0.08,
            ("retrieval", "fcsr-multiskill3x"): 0.09,
            ("reranker", "fcsr-multiskill3x"): 0.10,
            ("reranker", "fcsr-multiskill3x-rrf"): 0.11,
        }

        def load_summary(_: Path, stage: str, variant: str) -> dict[str, object]:
            offset = offsets[(stage, variant)]
            return {
                "stage": stage,
                "tier": "hard",
                "metrics": {
                    "all": {key: value + offset for key, value in METRICS.items()},
                    "multi": {"FullCoverage@10": 0.2 + offset},
                },
            }

        with (
            patch("evaluation_tables._load_summary", side_effect=load_summary),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text") as write_text,
        ):
            outputs = render_hard_tables(Path("reports"))

        self.assertEqual(outputs["final"], Path("reports/tables/hard-baselines.md"))
        final_table = write_text.call_args_list[0].args[0]
        ablation_table = write_text.call_args_list[1].args[0]

        self.assertIn("# Hard Pool Final System Comparison", final_table)
        self.assertIn("Ours: RRF + FCSR MultiSkill-3x", final_table)
        self.assertNotIn("| SkillRouter | Retrieval |", final_table)
        self.assertIn("**0.6100**", final_table)

        self.assertIn("# Hard Pool Two-Stage Ablation", ablation_table)
        self.assertIn("| SkillRouter | Retrieval |", ablation_table)
        self.assertIn("| SkillRouter | Rerank |", ablation_table)
        self.assertIn("| Ours: RRF + FCSR MultiSkill-3x | Retrieval |", ablation_table)
        self.assertIn("| Ours: RRF + FCSR MultiSkill-3x | Rerank |", ablation_table)


if __name__ == "__main__":
    unittest.main()
