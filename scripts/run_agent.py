"""Run the deterministic local Flat Skill Agent over JSON task records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agent.candidates import adapt_ranked_candidates
from agent.local_tasks import (
    default_tools,
    index_skills,
    load_object_list,
    required_string,
)
from agent.organizers import FlatOrganizer
from agent.runtime import FlatSkillAgent
from agent.selectors import FirstRankedSelector
from agent.verifiers import VerifierRegistry
from data_io import write_jsonl_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-skills", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skills = load_object_list(args.skills, "skills")
    tasks = load_object_list(args.tasks, "tasks")
    skill_index = index_skills(skills)
    tools = default_tools()
    verifiers = VerifierRegistry.with_defaults()
    records: list[dict[str, Any]] = []

    for task_record in tasks:
        candidates = adapt_ranked_candidates(
            task_record,
            skill_index,
            limit=args.top_k,
        )
        selector_arguments = task_record.get("selector_arguments")
        if not isinstance(selector_arguments, dict):
            raise ValueError("task.selector_arguments must be an object")
        selector = FirstRankedSelector(
            argument_builder=lambda task, skill, arguments=selector_arguments: arguments
        )
        agent = FlatSkillAgent(
            organizer=FlatOrganizer(max_skills=args.max_skills),
            selector=selector,
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
        records.append(result.model_dump(mode="json"))

    write_jsonl_atomic(args.output, records)
    verified = sum(record["termination_reason"] == "verified" for record in records)
    print(f"Agent tasks: {verified}/{len(records)} verified")
    print(f"Records: {args.output}")
    return 0 if verified == len(records) else 1

if __name__ == "__main__":
    raise SystemExit(main())
