"""Shared local task I/O and safe example tools for Agent experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.tools import ToolRegistry


def default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("uppercase", uppercase)
    registry.register("extract_json_key", extract_json_key)
    return registry


def uppercase(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.upper()


def extract_json_key(text: str, key: str) -> Any:
    if not isinstance(text, str) or not isinstance(key, str):
        raise TypeError("text and key must be strings")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    if key not in value:
        raise KeyError(key)
    return value[key]


def load_object_list(path: Path, field: str) -> list[dict[str, Any]]:
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


def index_skills(
    skills: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for skill in skills:
        skill_id = required_string(skill, "skill_id")
        if skill_id in index:
            raise ValueError(f"duplicate skill_id: {skill_id}")
        index[skill_id] = skill
    return index


def required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value
