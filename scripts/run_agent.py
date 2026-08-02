"""Run the deterministic local Flat Skill Agent over JSON task records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.candidates import adapt_ranked_candidates
from agent.organizers import FlatOrganizer
from agent.runtime import FlatSkillAgent
from agent.selectors import FirstRankedSelector
from agent.tools import ToolRegistry
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
    skills = _load_object_list(args.skills, "skills")
    tasks = _load_object_list(args.tasks, "tasks")
    skill_index = _index_skills(skills)
    tools = _default_tools()
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
            task_id=_required_string(task_record, "task_id"),
            task=_required_string(task_record, "task"),
            candidates=candidates,
            verifier_id=_required_string(task_record, "verifier_id"),
            expected=task_record.get("expected"),
        )
        records.append(result.model_dump(mode="json"))

    write_jsonl_atomic(args.output, records)
    verified = sum(record["termination_reason"] == "verified" for record in records)
    print(f"Agent tasks: {verified}/{len(records)} verified")
    print(f"Records: {args.output}")
    return 0 if verified == len(records) else 1


def _default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("uppercase", _uppercase)
    registry.register("extract_json_key", _extract_json_key)
    return registry


def _uppercase(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.upper()


def _extract_json_key(text: str, key: str) -> Any:
    if not isinstance(text, str) or not isinstance(key, str):
        raise TypeError("text and key must be strings")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    if key not in value:
        raise KeyError(key)
    return value[key]


def _load_object_list(path: Path, field: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or set(payload) != {field}:
        raise ValueError(f"{path} must contain exactly the top-level field {field!r}")
    records = payload[field]
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError(f"{path}:{field} must be a list of objects")
    return records


def _index_skills(skills: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for skill in skills:
        skill_id = _required_string(skill, "skill_id")
        if skill_id in index:
            raise ValueError(f"duplicate skill_id: {skill_id}")
        index[skill_id] = skill
    return index


def _required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
