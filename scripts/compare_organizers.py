"""Run paired Flat and Hierarchy organization experiments locally."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.candidates import adapt_ranked_candidates
from agent.hierarchy import HierarchyOrganizer
from agent.organizers import FlatOrganizer
from agent.presentation import organization_stats, render_skill_bundle
from agent.runtime import FlatSkillAgent
from agent.selectors import FirstRankedSelector
from agent.verifiers import VerifierRegistry
from data_io import write_jsonl_atomic
from agent.local_tasks import (
    default_tools,
    index_skills,
    load_object_list,
    required_string,
)


OrganizerFactory = Callable[[], Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-skills", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--category-depth", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skills = load_object_list(args.skills, "skills")
    tasks = load_object_list(args.tasks, "tasks")
    skill_index = index_skills(skills)
    task_ids = [required_string(task, "task_id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique")

    method_factories: dict[str, OrganizerFactory] = {
        "flat": lambda: FlatOrganizer(max_skills=args.max_skills),
        "hierarchy": lambda: HierarchyOrganizer(
            max_groups=args.max_groups,
            max_skills=args.max_skills,
            category_depth=args.category_depth,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_summaries: dict[str, dict[str, Any]] = {}

    for method, organizer_factory in method_factories.items():
        rows = _run_method(
            tasks=tasks,
            skill_index=skill_index,
            organizer_factory=organizer_factory,
            top_k=args.top_k,
        )
        records_path = args.output_dir / f"{method}.jsonl"
        write_jsonl_atomic(records_path, rows)
        method_summaries[method] = _summarize(rows, records_path)

    summary = {
        "paired_task_ids": task_ids,
        "config": {
            "top_k": args.top_k,
            "max_skills": args.max_skills,
            "max_groups": args.max_groups,
            "category_depth": args.category_depth,
        },
        "methods": method_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Organizer comparison: "
        + ", ".join(
            f"{method}={values['verified']}/{values['tasks']}"
            for method, values in method_summaries.items()
        )
    )
    print(f"Summary: {summary_path}")
    all_verified = all(
        values["verified"] == values["tasks"]
        for values in method_summaries.values()
    )
    return 0 if all_verified else 1


def _run_method(
    tasks: list[dict[str, Any]],
    skill_index: dict[str, dict[str, Any]],
    organizer_factory: OrganizerFactory,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = default_tools()
    verifiers = VerifierRegistry.with_defaults()
    for task_record in tasks:
        candidates = adapt_ranked_candidates(task_record, skill_index, limit=top_k)
        organizer = organizer_factory()
        bundle = organizer.organize(candidates)
        rendered = render_skill_bundle(bundle)
        stats = organization_stats(bundle)
        selector_arguments = task_record.get("selector_arguments")
        if not isinstance(selector_arguments, dict):
            raise ValueError("task.selector_arguments must be an object")
        agent = FlatSkillAgent(
            organizer=organizer,
            selector=FirstRankedSelector(
                argument_builder=lambda task, skill, arguments=selector_arguments: arguments
            ),
            tools=tools,
            verifiers=verifiers,
        )
        result = agent.run(
            task_id=required_string(task_record, "task_id"),
            task=required_string(task_record, "task"),
            candidates=candidates,
            verifier_id=required_string(task_record, "verifier_id"),
            expected=task_record.get("expected"),
        )
        rows.append(
            {
                **result.model_dump(mode="json"),
                "candidate_skill_ids": [skill.skill_id for skill in candidates],
                "organization": stats.model_dump(mode="json"),
                "rendered_bundle": rendered,
            }
        )
    return rows


def _summarize(
    rows: list[dict[str, Any]],
    records_path: Path,
) -> dict[str, Any]:
    task_count = len(rows)
    divisor = task_count or 1
    return {
        "tasks": task_count,
        "verified": sum(row["termination_reason"] == "verified" for row in rows),
        "mean_skill_count": sum(
            row["organization"]["skill_count"] for row in rows
        )
        / divisor,
        "mean_group_count": sum(
            row["organization"]["group_count"] for row in rows
        )
        / divisor,
        "mean_rendered_characters": sum(
            row["organization"]["rendered_characters"] for row in rows
        )
        / divisor,
        "records": str(records_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
