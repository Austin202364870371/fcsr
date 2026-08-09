import json
import tempfile
import unittest
from pathlib import Path

from skill_organization.models import FrozenSkill, SkillRecord, TaskInput
from skill_organization.organizer import (
    EvidenceEdge,
    EvidenceGraph,
    Hierarchy,
    HierarchyGroup,
    OrganizationBundle,
)
from skill_organization.render import (
    ATOMIC_MARKER,
    render_context,
    write_skill_packages,
)
from skill_organization.validate import validate_rendered_task


def make_task_input() -> TaskInput:
    return TaskInput(
        task_key="T001",
        task_id="task-a",
        skills=tuple(
            FrozenSkill(
                alias=f"S{index:02d}",
                rank=index,
                record=SkillRecord(
                    skill_id=f"gt/private-{index}",
                    name=f"Skill {index}",
                    description=f"Description {index}",
                    body=f"Body {index}",
                    source="gt",
                ),
            )
            for index in range(1, 9)
        ),
    )


def make_bundle() -> OrganizationBundle:
    hierarchy = Hierarchy(
        roots=(
            HierarchyGroup(
                label="Operations",
                children=(
                    HierarchyGroup(
                        label="Core procedures",
                        skills=tuple(f"S{index:02d}" for index in range(1, 9)),
                    ),
                ),
            ),
        )
    )
    graph = EvidenceGraph(
        nodes=tuple(f"S{index:02d}" for index in range(1, 9)),
        edges=(
            EvidenceEdge(
                source="S01",
                target="S02",
                edge_type="setup_execute",
                source_evidence="Body 1",
                target_evidence="Body 2",
            ),
        ),
    )
    return OrganizationBundle(hierarchy=hierarchy, graph=graph)


class RenderTests(unittest.TestCase):
    def test_all_skill_conditions_share_identical_atomic_suffix(self):
        task = make_task_input()
        bundle = make_bundle()
        rendered = {
            method: render_context(task, bundle, method)
            for method in ("flat_top8", "hierarchy_top8", "graph_top8")
        }
        suffixes = {text.split(ATOMIC_MARKER, 1)[1] for text in rendered.values()}
        self.assertEqual(len(suffixes), 1)
        self.assertEqual(len(rendered), len(set(rendered.values())))

    def test_rendered_text_never_exposes_provenance(self):
        task = make_task_input()
        text = render_context(task, make_bundle(), "flat_top8")
        self.assertNotIn("gt/private-", text)
        self.assertNotIn("Source:", text)
        self.assertIn("Name: Skill 1", text)

    def test_writer_builds_benchflow_skill_layout_and_private_manifest(self):
        task = make_task_input()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bundle = make_bundle()
            paths = write_skill_packages(task, bundle, root)
            report = validate_rendered_task(task, bundle, root)

            self.assertEqual(set(paths), {"flat_top8", "hierarchy_top8", "graph_top8"})
            skill_path = (
                root
                / "task-a"
                / "flat_top8"
                / "skills"
                / "retrieved-skills"
                / "SKILL.md"
            )
            manifest_path = root / "task-a" / "flat_top8" / "context_manifest.json"
            self.assertTrue(skill_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(
                skill_path.read_text(encoding="utf-8").startswith(
                    "---\nname: retrieved-skills\n"
                )
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["skills"][0]["skill_id"], "gt/private-1")
            self.assertEqual(report["atomic_suffixes"], 1)

    def test_validator_rejects_changed_atomic_payload(self):
        task = make_task_input()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_skill_packages(task, make_bundle(), root)
            path = (
                root
                / "task-a"
                / "graph_top8"
                / "skills"
                / "retrieved-skills"
                / "SKILL.md"
            )
            path.write_text(
                path.read_text(encoding="utf-8").replace("Body 8", "Changed body", 1),
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(ValueError, "reviewed bundle|atomic payload"):
                validate_rendered_task(task, make_bundle(), root)

    def test_validator_rejects_a_rehashed_but_unreviewed_index(self):
        task = make_task_input()
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_skill_packages(task, bundle, root)
            condition_root = root / "task-a" / "hierarchy_top8"
            path = condition_root / "skills" / "retrieved-skills" / "SKILL.md"
            payload = path.read_text(encoding="utf-8").replace(
                "Core procedures", "Altered"
            )
            path.write_text(payload, encoding="utf-8", newline="\n")
            manifest_path = condition_root / "context_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            import hashlib

            manifest["rendered_context_sha256"] = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reviewed bundle"):
                validate_rendered_task(task, bundle, root)


if __name__ == "__main__":
    unittest.main()
