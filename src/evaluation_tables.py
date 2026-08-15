"""Render canonical Hard-pool result tables from organized report summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METRIC_COLUMNS = (
    ("Hit@1", "all", "Hit@1"),
    ("MRR@10", "all", "MRR@10"),
    ("nDCG@10", "all", "nDCG@10"),
    ("Recall@10", "all", "Recall@10"),
    ("Recall@20", "all", "Recall@20"),
    ("FullCoverage@10", "all", "FullCoverage@10"),
    ("Multi FullCoverage@10", "multi", "FullCoverage@10"),
)

FINAL_SYSTEMS = (
    ("Flat-BM25", "Retrieval", "retrieval", "baselines/hard/bm25/retrieval"),
    ("Flat-Dense", "Retrieval", "retrieval", "baselines/hard/base-dense/retrieval"),
    ("Flat-Hybrid (RRF)", "Retrieval", "retrieval", "baselines/hard/base-rrf/retrieval"),
    (
        "Flat-Dense + Base Reranker",
        "Rerank",
        "reranker",
        "baselines/hard/base-dense/rerank",
    ),
    ("SkillRouter", "Rerank", "reranker", "baselines/hard/skillrouter/rerank"),
    ("Ours: FCSR-Small", "Rerank", "reranker", "systems/fcsr-small/hard/rrf-rerank"),
    ("Ours: FCSR", "Rerank", "reranker", "systems/fcsr/hard/rrf-rerank"),
)

RETRIEVAL_SYSTEMS = (
    ("BM25", "Retrieval", "retrieval", "baselines/hard/bm25/retrieval"),
    ("Base Emb.", "Retrieval", "retrieval", "baselines/hard/base-dense/retrieval"),
    ("RRF (Base Emb.)", "Retrieval", "retrieval", "baselines/hard/base-rrf/retrieval"),
    ("SkillRouter", "Retrieval", "retrieval", "baselines/hard/skillrouter/retrieval"),
    ("FCSR-Small Retriever", "Retrieval", "retrieval", "systems/fcsr-small/hard/dense"),
    (
        "FCSR-Small Retrieval (RRF)",
        "Retrieval",
        "retrieval",
        "systems/fcsr-small/hard/rrf",
    ),
    ("FCSR Retriever", "Retrieval", "retrieval", "systems/fcsr/hard/dense"),
    ("FCSR Retrieval (RRF)", "Retrieval", "retrieval", "systems/fcsr/hard/rrf"),
)

TWO_STAGE_SYSTEMS = (
    (
        "Flat-Dense + Base Reranker",
        "Retrieval",
        "retrieval",
        "baselines/hard/base-dense/retrieval",
    ),
    (
        "Flat-Dense + Base Reranker",
        "Rerank",
        "reranker",
        "baselines/hard/base-dense/rerank",
    ),
    ("SkillRouter", "Retrieval", "retrieval", "baselines/hard/skillrouter/retrieval"),
    ("SkillRouter", "Rerank", "reranker", "baselines/hard/skillrouter/rerank"),
    ("FCSR-Small (Dense)", "Retrieval", "retrieval", "systems/fcsr-small/hard/dense"),
    (
        "FCSR-Small (Dense)",
        "Rerank",
        "reranker",
        "systems/fcsr-small/hard/dense-rerank",
    ),
    ("FCSR-Small", "Retrieval", "retrieval", "systems/fcsr-small/hard/rrf"),
    ("FCSR-Small", "Rerank", "reranker", "systems/fcsr-small/hard/rrf-rerank"),
    ("FCSR (Dense)", "Retrieval", "retrieval", "systems/fcsr/hard/dense"),
    ("FCSR (Dense)", "Rerank", "reranker", "systems/fcsr/hard/dense-rerank"),
    ("FCSR", "Retrieval", "retrieval", "systems/fcsr/hard/rrf"),
    ("FCSR", "Rerank", "reranker", "systems/fcsr/hard/rrf-rerank"),
)


def render_hard_tables(reports_dir: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """Write retrieval, final-system, and two-stage Hard-pool tables."""
    reports_dir = Path(reports_dir)
    output_dir = Path(output_dir) if output_dir is not None else reports_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_path = output_dir / "hard-retrieval.md"
    final_path = output_dir / "hard-final.md"
    ablation_path = output_dir / "hard-two-stage.md"
    retrieval_path.write_text(
        _render_table(
            "Hard Pool Retrieval Comparison",
            _load_rows(reports_dir, RETRIEVAL_SYSTEMS),
            stage_header="Stage",
            bold_best=True,
        ),
        encoding="utf-8",
    )
    final_path.write_text(
        _render_table(
            "Hard Pool Final System Comparison",
            _load_rows(reports_dir, FINAL_SYSTEMS),
            stage_header="Final Stage",
            bold_best=True,
        ),
        encoding="utf-8",
    )
    ablation_path.write_text(
        _render_table(
            "Hard Pool Two-Stage Ablation",
            _load_rows(reports_dir, TWO_STAGE_SYSTEMS),
            stage_header="Stage",
            bold_best=False,
        ),
        encoding="utf-8",
    )
    return {
        "retrieval": retrieval_path,
        "final": final_path,
        "ablation": ablation_path,
    }


def _load_rows(
    reports_dir: Path,
    specifications: tuple[tuple[str, str, str, str], ...],
) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "stage_label": stage_label,
            "metrics": _extract_metrics(
                _load_summary(reports_dir, stage, location), stage, location
            ),
        }
        for method, stage_label, stage, location in specifications
    ]


def _load_summary(reports_dir: Path, stage: str, location: str) -> dict[str, Any]:
    directory = reports_dir / location
    paths = _summary_paths(directory, stage)
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage") == stage and payload.get("tier") == "hard":
            return payload
    raise FileNotFoundError(
        f"missing Hard {stage} summary at {location}: "
        + ", ".join(str(path) for path in paths)
    )


def _summary_paths(directory: Path, stage: str) -> tuple[Path, Path]:
    return (directory / "summary.json", directory / f"{stage}_hard_summary.json")


def _extract_metrics(payload: dict[str, Any], stage: str, location: str) -> dict[str, float]:
    result: dict[str, float] = {}
    metric_groups = payload.get("metrics")
    if not isinstance(metric_groups, dict):
        raise ValueError(f"metrics missing from Hard {stage} summary at {location}")
    for label, group_name, metric_name in METRIC_COLUMNS:
        group = metric_groups.get(group_name)
        value = group.get(metric_name) if isinstance(group, dict) else None
        if not isinstance(value, (int, float)):
            raise ValueError(f"{label} missing from Hard {stage} summary at {location}")
        result[label] = float(value)
    return result


def _render_table(
    title: str,
    rows: list[dict[str, Any]],
    *,
    stage_header: str,
    bold_best: bool,
) -> str:
    headers = ["Method", stage_header, *(column[0] for column in METRIC_COLUMNS)]
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:|" + "---:|" * len(METRIC_COLUMNS),
    ]
    best = {
        label: max(row["metrics"][label] for row in rows)
        for label, _, _ in METRIC_COLUMNS
    }
    for row in rows:
        rendered_metrics = []
        for label, _, _ in METRIC_COLUMNS:
            value = row["metrics"][label]
            rendered = f"{value:.4f}"
            if bold_best and value == best[label]:
                rendered = f"**{rendered}**"
            rendered_metrics.append(rendered)
        lines.append(
            f"| {row['method']} | {row['stage_label']} | "
            + " | ".join(rendered_metrics)
            + " |"
        )
    if bold_best:
        lines.extend(
            [
                "",
                "> **Bold** marks the best numerical result among final system outputs; "
                "it does not indicate statistical significance.",
            ]
        )
    return "\n".join(lines) + "\n"
