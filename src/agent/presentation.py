"""Deterministic, measurable presentation of organized Skill bundles."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.hierarchy import HierarchySkillBundle
from agent.models import SkillBundle, SkillCandidate


class OrganizationStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str
    group_count: int = Field(ge=0)
    skill_count: int = Field(ge=0)
    rendered_characters: int = Field(ge=0)
    rendered_words: int = Field(ge=0)


def render_skill_bundle(bundle: SkillBundle) -> str:
    """Render only the selected bundle, preserving its organization strategy."""
    lines = [f"# Skill bundle: {bundle.strategy}"]
    if isinstance(bundle, HierarchySkillBundle):
        for group in bundle.groups:
            lines.append("")
            lines.append(f"## Category: {group.label}")
            lines.append(
                f"Category score: {group.score:.6f}; best rank: {group.best_rank}"
            )
            for skill in group.skills:
                lines.append(_render_skill(skill, prefix="-"))
    else:
        for index, skill in enumerate(bundle.skills, start=1):
            lines.append(_render_skill(skill, prefix=f"{index}."))
    return "\n".join(lines)


def organization_stats(bundle: SkillBundle) -> OrganizationStats:
    rendered = render_skill_bundle(bundle)
    group_count = len(bundle.groups) if isinstance(bundle, HierarchySkillBundle) else 0
    return OrganizationStats(
        strategy=bundle.strategy,
        group_count=group_count,
        skill_count=len(bundle.skills),
        rendered_characters=len(rendered),
        rendered_words=len(rendered.split()),
    )


def _render_skill(skill: SkillCandidate, prefix: str) -> str:
    return (
        f"{prefix} `{skill.skill_id}` ({skill.name}) — {skill.description} "
        f"[tool={skill.tool_name}; retrieval_rank={skill.rank}]"
    )
