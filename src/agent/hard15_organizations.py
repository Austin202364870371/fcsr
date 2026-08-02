"""Fair Flat, Hierarchy, and evidence-Graph presentations for Hard-15."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Literal, TYPE_CHECKING
import networkx as nx

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from agent.hard15_pilot import Hard15Skill, Hard15Task


Method = Literal["flat", "hierarchy", "graph"]


class EvidenceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    target: str
    edge_type: Literal["explicit_reference", "same_namespace"]
    directed: bool
    evidence: str


class EvidenceGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[str, ...]
    edges: tuple[EvidenceEdge, ...]
    reading_order: tuple[str, ...]
    connected_components: tuple[tuple[str, ...], ...]


class OrganizedTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    task: str
    method: Method
    skills: tuple[Any, ...]
    selected_aliases: tuple[str, ...]
    omitted_aliases: tuple[str, ...]
    rendered_prompt: str
    metadata: dict[str, Any]


def normalize_category_path(source: Mapping[str, Any]) -> tuple[str, ...]:
    """Use explicit metadata, then stable non-leaf Skill-ID namespaces."""
    for field in ("category_path", "category"):
        value = source.get(field)
        if isinstance(value, str) and value:
            parts = tuple(part for part in value.split("/") if part)
            if parts:
                return parts
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts = tuple(part for part in value if isinstance(part, str) and part)
            if parts:
                return parts
    skill_id = source.get("skill_id", "")
    if isinstance(skill_id, str):
        parts = tuple(part for part in skill_id.split("/") if part)
        if len(parts) > 1:
            return parts[:-1][:2]
    return ("uncategorized",)


def build_evidence_graph(skills: Sequence["Hard15Skill"]) -> EvidenceGraph:
    ranked = tuple(sorted(skills, key=lambda item: item.rank))
    edges: list[EvidenceEdge] = []
    name_counts: dict[str, int] = defaultdict(int)
    for item in ranked:
        name_counts[item.name.casefold()] += 1
    for source in ranked:
        text = f"{source.description}\n{source.body}".casefold()
        for target in ranked:
            if source.alias == target.alias:
                continue
            target_id = target.skill_id.casefold().strip()
            by_id = (
                bool(target_id)
                and re.search(rf"(?<![\w/.-]){re.escape(target_id)}(?![\w/.-])", text) is not None
            )
            target_name = target.name.casefold().strip()
            by_name = (
                len(target_name) >= 4
                and name_counts[target_name] == 1
                and re.search(rf"(?<!\w){re.escape(target_name)}(?!\w)", text) is not None
            )
            if by_id or by_name:
                edges.append(
                    EvidenceEdge(
                        source=source.alias,
                        target=target.alias,
                        edge_type="explicit_reference",
                        directed=True,
                        evidence=f"{source.alias} explicitly names {target.alias} ({target.name})",
                    )
                )
    for left, right in combinations(ranked, 2):
        if left.category_path == right.category_path:
            edges.append(
                EvidenceEdge(
                    source=left.alias,
                    target=right.alias,
                    edge_type="same_namespace",
                    directed=False,
                    evidence=f"{left.alias} and {right.alias} share an anonymized namespace",
                )
            )
    order = _reading_order(ranked, edges)
    components = _components(ranked, edges)
    return EvidenceGraph(
        nodes=tuple(item.alias for item in ranked),
        edges=tuple(edges),
        reading_order=order,
        connected_components=components,
    )


def organize_task(
    task: "Hard15Task",
    *,
    method: Method,
    max_skills: int,
    body_char_budget: int,
    max_groups: int = 4,
) -> OrganizedTask:
    if max_skills < 1 or body_char_budget < 0 or max_groups < 1:
        raise ValueError("invalid organization budget")
    ranked = tuple(sorted(task.skills, key=lambda item: item.rank))
    metadata: dict[str, Any] = {}
    if method == "flat":
        selected = ranked[:max_skills]
        order = tuple(item.alias for item in selected)
        metadata = {"group_count": 0}
    elif method == "hierarchy":
        grouped: dict[tuple[str, ...], list[Any]] = defaultdict(list)
        for item in ranked:
            grouped[item.category_path].append(item)
        chosen_groups = sorted(
            grouped,
            key=lambda path: (-sum(1.0 / item.rank for item in grouped[path]), min(item.rank for item in grouped[path])),
        )[:max_groups]
        selected = tuple(
            sorted(
                (item for path in chosen_groups for item in grouped[path]),
                key=lambda item: item.rank,
            )[:max_skills]
        )
        order = tuple(item.alias for item in selected)
        group_alias = {path: f"C{index:02d}" for index, path in enumerate(chosen_groups, 1)}
        metadata = {
            "group_count": len(chosen_groups),
            "groups": [
                {
                    "category_alias": group_alias[path],
                    "member_aliases": [item.alias for item in selected if item.category_path == path],
                }
                for path in chosen_groups
            ],
        }
    elif method == "graph":
        complete = build_evidence_graph(ranked)
        selected = _select_graph(ranked, complete, max_skills)
        chosen = {item.alias for item in selected}
        graph = build_evidence_graph(selected)
        by_alias = {item.alias: item for item in selected}
        order = graph.reading_order
        selected = tuple(by_alias[alias] for alias in order)
        explicit = sum(edge.edge_type == "explicit_reference" for edge in graph.edges)
        namespace = sum(edge.edge_type == "same_namespace" for edge in graph.edges)
        metadata = {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "explicit_reference_count": explicit,
            "namespace_edge_count": namespace,
            "connected_component_count": len(graph.connected_components),
            "reading_order": list(graph.reading_order),
            "edges": [edge.model_dump(mode="json") for edge in graph.edges],
        }
        assert chosen == set(order)
    else:
        raise ValueError(f"unknown method: {method}")

    selected = _truncate_bodies(selected, body_char_budget)
    selected_aliases = tuple(item.alias for item in selected)
    omitted = tuple(item.alias for item in ranked if item.alias not in set(selected_aliases))
    rendered = _render(method, selected, metadata)
    metadata = {
        **metadata,
        "selected_skill_count": len(selected),
        "omitted_count": len(omitted),
        "body_characters": sum(len(item.body) for item in selected),
        "rendered_characters": len(rendered),
    }
    return OrganizedTask(
        task_id=task.task_id,
        task=task.task,
        method=method,
        skills=selected,
        selected_aliases=selected_aliases,
        omitted_aliases=omitted,
        rendered_prompt=rendered,
        metadata=metadata,
    )


def _select_graph(skills, graph: EvidenceGraph, limit: int):
    by_alias = {item.alias: item for item in skills}
    selected = [min(skills, key=lambda item: item.rank)]
    while len(selected) < min(limit, len(skills)):
        aliases = {item.alias for item in selected}
        namespaces = {item.category_path for item in selected}
        scored = []
        for item in skills:
            if item.alias in aliases:
                continue
            explicit = sum(
                edge.edge_type == "explicit_reference"
                and item.alias in (edge.source, edge.target)
                and bool(aliases & {edge.source, edge.target})
                for edge in graph.edges
            )
            namespace = sum(
                edge.edge_type == "same_namespace"
                and item.alias in (edge.source, edge.target)
                and bool(aliases & {edge.source, edge.target})
                for edge in graph.edges
            )
            new_namespace = item.category_path not in namespaces
            score = 2.0 * explicit + namespace + (0.5 if new_namespace else 0.0) + 1.0 / item.rank
            scored.append((score, -item.rank, item.alias))
        _, _, alias = max(scored)
        selected.append(by_alias[alias])
    return tuple(selected)


def _truncate_bodies(skills, budget: int):
    remaining = budget
    bodies = {}
    for item in sorted(skills, key=lambda skill: skill.rank):
        bodies[item.alias] = item.body[:remaining]
        remaining -= len(bodies[item.alias])
    return tuple(item.model_copy(update={"body": bodies[item.alias]}) for item in skills)


def _render(method: Method, skills, metadata: Mapping[str, Any]) -> str:
    cards = [
        {
            "alias": item.alias,
            "name": item.name,
            "description": item.description,
            "instructions": item.body,
            "retrieval_rank": item.rank,
        }
        for item in skills
    ]
    organization = {key: value for key, value in metadata.items() if key not in {"rendered_characters"}}
    return json.dumps(
        {"organization": method, "structure": organization, "skills": cards},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _reading_order(skills, edges: Sequence[EvidenceEdge]) -> tuple[str, ...]:
    """Topologically order SCCs and break only cycles internally by FCSR rank."""
    rank = {item.alias: item.rank for item in skills}
    graph = nx.DiGraph()
    graph.add_nodes_from(rank)
    for edge in edges:
        if edge.edge_type == "explicit_reference":
            graph.add_edge(edge.source, edge.target)
    condensed = nx.condensation(graph)
    component_rank = {
        node: min(rank[member] for member in data["members"])
        for node, data in condensed.nodes(data=True)
    }
    component_order = nx.lexicographical_topological_sort(
        condensed,
        key=lambda node: component_rank[node],
    )
    return tuple(
        alias
        for component in component_order
        for alias in sorted(condensed.nodes[component]["members"], key=rank.get)
    )


def _components(skills, edges: Sequence[EvidenceEdge]) -> tuple[tuple[str, ...], ...]:
    rank = {item.alias: item.rank for item in skills}
    adjacency = {node: set() for node in rank}
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    unseen = set(rank)
    result = []
    while unseen:
        start = min(unseen, key=rank.get)
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        unseen -= component
        result.append(tuple(sorted(component, key=rank.get)))
    return tuple(result)
