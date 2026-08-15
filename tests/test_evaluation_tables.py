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
            ("retrieval", "fcsr-multiskill3x"): 0.085,
            ("retrieval", "rrf-fcsr-emb-multiskill3x"): 0.087,
            ("reranker", "fcsr-multiskill3x"): 0.088,
            ("reranker", "rrf-base-emb-multiskill3x"): 0.089,
            ("reranker", "rrf-fcsr-emb-multiskill3x-top20"): 0.091,
            ("retrieval", "fcsr-multiskill-weighted"): 0.09,
            ("reranker", "fcsr-multiskill-weighted"): 0.10,
            ("reranker", "rrf-base-emb-multiskill-weighted"): 0.11,
            ("retrieval", "rrf-fcsr-emb-multiskill-weighted"): 0.12,
            ("reranker", "rrf-fcsr-emb-multiskill-weighted"): 0.13,
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
            patch(
                "evaluation_tables._has_hard_summary",
            side_effect=lambda _, stage, variant: variant == "rrf-fcsr-emb-multiskill-weighted",
            ),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text") as write_text,
        ):
            outputs = render_hard_tables(Path("reports"))

        self.assertEqual(outputs["retrieval"], Path("reports/tables/hard-retrieval.md"))
        self.assertEqual(outputs["final"], Path("reports/tables/hard-final-systems.md"))
        self.assertEqual(outputs["ablation"], Path("reports/tables/hard-two-stage.md"))
        retrieval_table = write_text.call_args_list[0].args[0]
        final_table = write_text.call_args_list[1].args[0]
        ablation_table = write_text.call_args_list[2].args[0]

        self.assertIn("# Hard Pool Retrieval Comparison", retrieval_table)
        self.assertIn("RRF (FCSR Emb. MultiSkill-Weighted)", retrieval_table)

        self.assertIn("# Hard Pool Final System Comparison", final_table)
        self.assertIn("Ours: RRF (Base Emb.) + FCSR MultiSkill-Weighted", final_table)
        self.assertIn("Ours: RRF (FCSR Emb.) + FCSR MultiSkill-Weighted", final_table)
        self.assertNotIn("| SkillRouter | Retrieval |", final_table)
        self.assertIn("**0.6300**", final_table)

        self.assertIn("# Hard Pool Two-Stage Ablation", ablation_table)
        self.assertIn("| SkillRouter | Retrieval |", ablation_table)
        self.assertIn("| SkillRouter | Rerank |", ablation_table)
        self.assertIn(
            "| Ours: RRF (Base Emb.) + FCSR MultiSkill-Weighted | Retrieval |",
            ablation_table,
        )
        self.assertIn(
            "| Ours: RRF (Base Emb.) + FCSR MultiSkill-Weighted | Rerank |",
            ablation_table,
        )
        self.assertIn(
            "| Ours: RRF (FCSR Emb.) + FCSR MultiSkill-Weighted | Retrieval |",
            ablation_table,
        )
        self.assertIn(
            "| Ours: RRF (FCSR Emb.) + FCSR MultiSkill-Weighted | Rerank |",
            ablation_table,
        )


if __name__ == "__main__":
    unittest.main()
