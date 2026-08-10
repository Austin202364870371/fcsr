"""Render the canonical Hard-pool result tables from evaluation summaries."""

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
    ("Flat-BM25", "Retrieval", "retrieval", "bm25"),
    ("Flat-Dense", "Retrieval", "retrieval", "dense"),
    ("Flat-Hybrid (RRF)", "Retrieval", "retrieval", "hybrid"),
    ("Flat-Dense + Base Reranker", "Rerank", "reranker", "dense-base-reranker"),
    ("SkillRouter", "Rerank", "reranker", "skillrouter"),
    ("Ours: FCSR", "Rerank", "reranker", "fcsr-single"),
    ("Ours: FCSR + MultiSkill-3x", "Rerank", "reranker", "fcsr-multiskill3x"),
    (
        "Ours: RRF + FCSR MultiSkill-3x",
        "Rerank",
        "reranker",
        "fcsr-multiskill3x-rrf",
    ),
)

OPTIONAL_FINAL_SYSTEMS = (
    (
        "Ours: RRF (FCSR Emb.) + FCSR MultiSkill-3x",
        "Rerank",
        "reranker",
        "rrf-fcsr-multiskill3x",
    ),
)

TWO_STAGE_SYSTEMS = (
    ("Flat-Dense + Base Reranker", "Retrieval", "retrieval", "dense"),
    ("Flat-Dense + Base Reranker", "Rerank", "reranker", "dense-base-reranker"),
    ("SkillRouter", "Retrieval", "retrieval", "skillrouter"),
    ("SkillRouter", "Rerank", "reranker", "skillrouter"),
    ("Ours: FCSR", "Retrieval", "retrieval", "fcsr-single"),
    ("Ours: FCSR", "Rerank", "reranker", "fcsr-single"),
    ("Ours: FCSR + MultiSkill-3x", "Retrieval", "retrieval", "fcsr-multiskill3x"),
    ("Ours: FCSR + MultiSkill-3x", "Rerank", "reranker", "fcsr-multiskill3x"),
    ("Ours: RRF + FCSR MultiSkill-3x", "Retrieval", "retrieval", "hybrid"),
    (
        "Ours: RRF + FCSR MultiSkill-3x",
        "Rerank",
        "reranker",
        "fcsr-multiskill3x-rrf",
    ),
)

OPTIONAL_TWO_STAGE_SYSTEMS = (
    (
        "Ours: RRF (FCSR Emb.) + FCSR MultiSkill-3x",
        "Retrieval",
        "retrieval",
        "rrf-fcsr-multiskill3x",
    ),
    (
        "Ours: RRF (FCSR Emb.) + FCSR MultiSkill-3x",
        "Rerank",
        "reranker",
        "rrf-fcsr-multiskill3x",
    ),
)


def render_hard_tables(reports_dir: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """Write final-system and two-stage Hard-pool Markdown tables."""
    reports_dir = Path(reports_dir)
    output_dir = Path(output_dir) if output_dir is not None else reports_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / "hard-baselines.md"
    ablation_path = output_dir / "hard-two-stage-ablation.md"
    final_path.write_text(
        _render_table(
            "Hard Pool Final System Comparison",
            _load_rows(
                reports_dir,
                _available_systems(reports_dir, FINAL_SYSTEMS, OPTIONAL_FINAL_SYSTEMS),
            ),
            stage_header="Final Stage",
            bold_best=True,
        ),
        encoding="utf-8",
    )
    ablation_path.write_text(
        _render_table(
            "Hard Pool Two-Stage Ablation",
            _load_rows(
                reports_dir,
                _available_systems(reports_dir, TWO_STAGE_SYSTEMS, OPTIONAL_TWO_STAGE_SYSTEMS),
            ),
            stage_header="Stage",
            bold_best=False,
        ),
        encoding="utf-8",
    )
    return {"final": final_path, "ablation": ablation_path}


def _available_systems(
    reports_dir: Path,
    required: tuple[tuple[str, str, str, str], ...],
    optional: tuple[tuple[str, str, str, str], ...],
) -> tuple[tuple[str, str, str, str], ...]:
    """Keep optional systems out until every required stage has a Hard summary."""
    grouped: dict[str, list[tuple[str, str, str, str]]] = {}
    for specification in optional:
        grouped.setdefault(specification[0], []).append(specification)
    available = list(required)
    for specifications in grouped.values():
        if all(_has_hard_summary(reports_dir, stage, variant) for _, _, stage, variant in specifications):
            available.extend(specifications)
    return tuple(available)


def _load_rows(reports_dir: Path, specifications: tuple[tuple[str, str, str, str], ...]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "stage_label": stage_label,
            "metrics": _extract_metrics(_load_summary(reports_dir, stage, variant), stage, variant),
        }
        for method, stage_label, stage, variant in specifications
    ]


def _load_summary(reports_dir: Path, stage: str, variant: str) -> dict[str, Any]:
    directory = reports_dir / stage / "hard" / variant
    paths = _summary_paths(directory, stage)
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage") == stage and payload.get("tier") == "hard":
            return payload
    raise FileNotFoundError(
        f"missing Hard {stage} summary for {variant}: " + ", ".join(str(path) for path in paths)
    )


def _has_hard_summary(reports_dir: Path, stage: str, variant: str) -> bool:
    directory = reports_dir / stage / "hard" / variant
    for path in _summary_paths(directory, stage):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if payload.get("stage") == stage and payload.get("tier") == "hard":
            return True
    return False


def _summary_paths(directory: Path, stage: str) -> tuple[Path, Path]:
    return (directory / "summary.json", directory / f"{stage}_hard_summary.json")


def _extract_metrics(payload: dict[str, Any], stage: str, variant: str) -> dict[str, float]:
    result: dict[str, float] = {}
    metric_groups = payload.get("metrics")
    if not isinstance(metric_groups, dict):
        raise ValueError(f"metrics missing from Hard {stage} summary for {variant}")
    for label, group_name, metric_name in METRIC_COLUMNS:
        group = metric_groups.get(group_name)
        value = group.get(metric_name) if isinstance(group, dict) else None
        if not isinstance(value, (int, float)):
            raise ValueError(f"{label} missing from Hard {stage} summary for {variant}")
        result[label] = float(value)
    return result


def _render_table(title: str, rows: list[dict[str, Any]], *, stage_header: str, bold_best: bool) -> str:
    headers = ["Method", stage_header, *(column[0] for column in METRIC_COLUMNS)]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|---|---:|" + "---:|" * len(METRIC_COLUMNS)]
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
            f"| {row['method']} | {row['stage_label']} | " + " | ".join(rendered_metrics) + " |"
        )
    if bold_best:
        lines.extend(
            [
                "",
                "> **Bold** marks the best numerical result among final system outputs; it does not indicate statistical significance.",
            ]
        )
    return "\n".join(lines) + "\n"
