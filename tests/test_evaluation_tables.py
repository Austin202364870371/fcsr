"""Regression coverage for canonical FCSR result tables."""

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
    def test_renders_canonical_fcsr_names_and_paths(self) -> None:
        offsets = {
            ("retrieval", "baselines/hard/bm25/retrieval"): 0.01,
            ("retrieval", "baselines/hard/base-dense/retrieval"): 0.02,
            ("retrieval", "baselines/hard/base-rrf/retrieval"): 0.03,
            ("reranker", "baselines/hard/base-dense/rerank"): 0.04,
            ("retrieval", "baselines/hard/skillrouter/retrieval"): 0.05,
            ("reranker", "baselines/hard/skillrouter/rerank"): 0.06,
            ("retrieval", "systems/fcsr-small/hard/dense"): 0.07,
            ("retrieval", "systems/fcsr-small/hard/rrf"): 0.08,
            ("reranker", "systems/fcsr-small/hard/dense-rerank"): 0.09,
            ("reranker", "systems/fcsr-small/hard/rrf-rerank"): 0.10,
            ("retrieval", "systems/fcsr/hard/dense"): 0.11,
            ("retrieval", "systems/fcsr/hard/rrf"): 0.12,
            ("reranker", "systems/fcsr/hard/dense-rerank"): 0.125,
            ("reranker", "systems/fcsr/hard/rrf-rerank"): 0.13,
        }

        def load_summary(_: Path, stage: str, location: str) -> dict[str, object]:
            offset = offsets[(stage, location)]
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

        self.assertEqual(outputs["retrieval"], Path("reports/tables/hard-retrieval.md"))
        self.assertEqual(outputs["final"], Path("reports/tables/hard-final.md"))
        self.assertEqual(outputs["ablation"], Path("reports/tables/hard-two-stage.md"))
        retrieval_table = write_text.call_args_list[0].args[0]
        final_table = write_text.call_args_list[1].args[0]
        ablation_table = write_text.call_args_list[2].args[0]

        self.assertIn("# Hard Pool Retrieval Comparison", retrieval_table)
        self.assertIn("FCSR-Small Retrieval (RRF)", retrieval_table)
        self.assertIn("FCSR Retrieval (RRF)", retrieval_table)

        self.assertIn("# Hard Pool Final System Comparison", final_table)
        self.assertIn("Ours: FCSR-Small", final_table)
        self.assertIn("Ours: FCSR", final_table)
        self.assertNotIn("MultiSkill", final_table)
        self.assertNotIn("| SkillRouter | Retrieval |", final_table)
        self.assertIn("**0.6300**", final_table)

        self.assertIn("# Hard Pool Two-Stage Ablation", ablation_table)
        self.assertIn("| SkillRouter | Retrieval |", ablation_table)
        self.assertIn("| SkillRouter | Rerank |", ablation_table)
        self.assertIn("| FCSR-Small | Retrieval |", ablation_table)
        self.assertIn("| FCSR-Small | Rerank |", ablation_table)
        self.assertIn("| FCSR | Retrieval |", ablation_table)
        self.assertIn("| FCSR | Rerank |", ablation_table)


if __name__ == "__main__":
    unittest.main()
