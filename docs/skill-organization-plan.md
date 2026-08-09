# Skill Organization Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Hard-15 pipeline that freezes FCSR Top-8 results, reconstructs the released SkillRouter text records, generates task-blind hierarchy/graph metadata, packages four SkillsBench conditions, runs OpenHands with DeepSeek-V4-Flash on Daytona, and aggregates verifier outcomes.

**Architecture:** Replace the planning-only `src/agent/` experiment with a focused `src/skill_organization/` package. The package separates frozen input loading, task-blind organization, byte-stable rendering, BenchFlow process orchestration, and result collection; `scripts/skill_organization.py` is the only user-facing CLI. Old planning code is removed only after the new unit tests and dry-run gates pass.

**Tech Stack:** Python 3.12, Pydantic 2, OpenAI-compatible DeepSeek client, NetworkX, BenchFlow 0.6.x CLI, OpenHands, Daytona, `unittest`.

## Global Constraints

- Retrieval input is only `reports/reranker/hard/fcsr-multiskill3x-rrf/predictions.json`.
- Skill input is only `data/raw/skills_hard.jsonl.gz`, expected SHA-256 `492BD8E7958434DEEAE97C91FBD6921AECEFB19EA16D4605F100B645BEC5AF31`.
- Hard-15 comes only from `data/agent/hard15/task_ids.txt` and `task_catalog.json`.
- SkillsBench is fixed to v1.1 commit `b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af`.
- Every task uses exactly FCSR rank 1--8; hierarchy and graph cannot select, remove, add, or rerank Skills.
- Organizer and Agent see aliases S01--S08, not `skill_id`, `source`, `gt/`, or `distractor/`.
- Organizer never receives task ID, task prompt, GT, oracle, verifier, Contract, trajectory, or reward.
- Agent payload is the frozen JSONL `name + description + body`; no Contract or current upstream Skill package is substituted.
- Flat, hierarchy, and graph atomic payload suffixes must be byte-identical.
- Tests use the standard library `unittest`; no new test framework dependency is added.
- Existing user deletions under `data/agent/examples/` are preserved and not restored.

---

### Task 1: Frozen input models and loader

**Files:**
- Create: `src/skill_organization/__init__.py`
- Create: `src/skill_organization/models.py`
- Create: `src/skill_organization/inputs.py`
- Create: `tests/test_skill_organization_inputs.py`

**Interfaces:**
- Consumes: `data_io.stream_jsonl`, frozen report, Hard JSONL, Hard-15 catalog.
- Produces: `SkillRecord`, `FrozenSkill`, `TaskInput`, `FrozenInputs`, `load_frozen_inputs(...)`, and `sha256_file(...)`.

- [ ] **Step 1: Write failing model and loader tests**

```python
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from skill_organization.inputs import load_frozen_inputs


class FrozenInputTests(unittest.TestCase):
    def test_loads_exact_top8_and_hides_provenance_from_organizer_view(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "task_ids.txt").write_text("task-a\n", encoding="utf-8")
            (root / "predictions.json").write_text(
                json.dumps({"task-a": [f"gt/s{i}" for i in range(1, 9)] + ["other/s9"]}),
                encoding="utf-8",
            )
            with gzip.open(root / "skills.jsonl.gz", "wt", encoding="utf-8") as handle:
                for index in range(1, 10):
                    handle.write(json.dumps({
                        "skill_id": f"gt/s{index}",
                        "name": f"Skill {index}",
                        "description": f"Description {index}",
                        "body": f"Body {index}",
                        "source": "gt",
                    }) + "\n")
            frozen = load_frozen_inputs(
                predictions_path=root / "predictions.json",
                skills_path=root / "skills.jsonl.gz",
                task_ids_path=root / "task_ids.txt",
                expected_skills_sha256=None,
                top_k=8,
            )
            task = frozen.tasks[0]
            self.assertEqual(tuple(item.rank for item in task.skills), tuple(range(1, 9)))
            self.assertEqual(task.skills[-1].record.skill_id, "gt/s8")
            view = task.skills[0].organizer_view()
            self.assertEqual(set(view), {"alias", "rank", "name", "description", "body"})
            self.assertNotIn("gt/", json.dumps(view))

    def test_fails_when_a_top8_record_is_missing(self):
        # Reuse a fixture with only seven matching records.
        with self.assertRaisesRegex(ValueError, "missing frozen Skill records"):
            load_frozen_inputs_from_fixture(record_count=7)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_inputs -v
```

Expected: import failure for `rq2.inputs`.

- [ ] **Step 3: Implement immutable models**

```python
# src/skill_organization/models.py
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    skill_id: str = Field(min_length=1)
    name: str
    description: str
    body: str
    source: Literal["pool", "gt", "distractor"]

    def canonical_hash(self) -> str:
        payload = json.dumps(
            {"name": self.name, "description": self.description, "body": self.body},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class FrozenSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    alias: str = Field(pattern=r"^S0[1-8]$")
    rank: int = Field(ge=1, le=8)
    record: SkillRecord

    def organizer_view(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "rank": self.rank,
            "name": self.record.name,
            "description": self.record.description,
            "body": self.record.body,
        }


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_key: str = Field(pattern=r"^T\d{3}$")
    task_id: str
    skills: tuple[FrozenSkill, ...] = Field(min_length=8, max_length=8)


class FrozenInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    predictions_sha256: str
    skills_sha256: str
    tasks: tuple[TaskInput, ...]
```

- [ ] **Step 4: Implement a streaming, fail-closed loader**

```python
# src/skill_organization/inputs.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_io import stream_jsonl
from skill_organization.models import FrozenInputs, FrozenSkill, SkillRecord, TaskInput


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_inputs(
    *,
    predictions_path: Path,
    skills_path: Path,
    task_ids_path: Path,
    expected_skills_sha256: str | None,
    top_k: int = 8,
) -> FrozenInputs:
    if top_k != 8:
        raise ValueError("Skill organization requires top_k=8")
    task_ids = tuple(line.strip() for line in task_ids_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(task_ids) != 15 or len(set(task_ids)) != 15:
        raise ValueError("Hard-15 must contain exactly 15 unique task IDs")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    selected = {task_id: tuple(predictions[task_id][:8]) for task_id in task_ids}
    if any(len(ids) != 8 for ids in selected.values()):
        raise ValueError("every Hard-15 task must have eight predictions")
    wanted = {skill_id for ids in selected.values() for skill_id in ids}
    records: dict[str, SkillRecord] = {}
    for raw in stream_jsonl(skills_path):
        skill_id = raw.get("skill_id")
        if skill_id not in wanted:
            continue
        record = SkillRecord.model_validate(raw)
        previous = records.get(record.skill_id)
        if previous is not None and previous != record:
            raise ValueError(f"conflicting duplicate Skill record: {record.skill_id}")
        records[record.skill_id] = record
    missing = sorted(wanted - records.keys())
    if missing:
        raise ValueError(f"missing frozen Skill records: {missing}")
    skills_hash = sha256_file(skills_path)
    if expected_skills_sha256 and skills_hash.lower() != expected_skills_sha256.lower():
        raise ValueError("Hard Skill pool SHA-256 mismatch")
    tasks = tuple(
        TaskInput(
            task_key=f"T{index:03d}",
            task_id=task_id,
            skills=tuple(
                FrozenSkill(alias=f"S{rank:02d}", rank=rank, record=records[skill_id])
                for rank, skill_id in enumerate(selected[task_id], start=1)
            ),
        )
        for index, task_id in enumerate(task_ids, start=1)
    )
    return FrozenInputs(
        predictions_sha256=sha256_file(predictions_path),
        skills_sha256=skills_hash,
        tasks=tasks,
    )
```

- [ ] **Step 5: Run the focused and existing data I/O tests**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_inputs tests.test_data_io -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the frozen loader**

```bash
git add src/skill_organization/__init__.py src/skill_organization/models.py src/skill_organization/inputs.py tests/test_skill_organization_inputs.py
git commit -m "feat(skill-org): freeze Hard-15 retrieval inputs"
```

---

### Task 2: Task-blind hierarchy and evidence graph organizer

**Files:**
- Create: `src/skill_organization/organizer.py`
- Create: `tests/test_skill_organization_organizer.py`

**Interfaces:**
- Consumes: `TaskInput.organizer_view()` records from Task 1.
- Produces: `Hierarchy`, `EvidenceGraph`, `OrganizationBundle`, `OrganizerClient`, `DeepSeekOrganizerClient`, `validate_bundle(...)`, and `reading_order(...)`.

- [ ] **Step 1: Write failing schema, leakage, and evidence tests**

```python
import unittest

from skill_organization.organizer import EvidenceEdge, EvidenceGraph, Hierarchy, HierarchyGroup, validate_bundle


class OrganizerTests(unittest.TestCase):
    def test_hierarchy_must_cover_every_alias_once(self):
        hierarchy = Hierarchy(roots=(HierarchyGroup(label="Data", skills=("S01", "S01")),))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_bundle(make_task_input(), hierarchy, empty_graph())

    def test_graph_evidence_must_be_exact_source_and_target_substrings(self):
        graph = EvidenceGraph(
            nodes=tuple(f"S{i:02d}" for i in range(1, 9)),
            edges=(EvidenceEdge(
                source="S01",
                target="S02",
                edge_type="produces_requires",
                source_evidence="not in source",
                target_evidence="Body 2",
            ),),
        )
        with self.assertRaisesRegex(ValueError, "source evidence"):
            validate_bundle(make_task_input(), valid_hierarchy(), graph)

    def test_zero_edge_graph_falls_back_to_rank_order(self):
        self.assertEqual(reading_order(make_task_input(), empty_graph()), tuple(f"S{i:02d}" for i in range(1, 9)))
```

- [ ] **Step 2: Run and verify the organizer test fails**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_organizer -v
```

Expected: import failure for `rq2.organizer`.

- [ ] **Step 3: Implement strict Pydantic schemas and validators**

```python
EdgeType = Literal[
    "produces_requires",
    "setup_execute",
    "execute_verify",
    "format_conversion",
    "explicit_reference",
]


class HierarchyGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1, max_length=80)
    skills: tuple[str, ...] = ()
    children: tuple["HierarchyGroup", ...] = ()


class Hierarchy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["skill-hierarchy-v1"] = "skill-hierarchy-v1"
    roots: tuple[HierarchyGroup, ...] = Field(min_length=1)


class EvidenceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    target: str
    edge_type: EdgeType
    source_evidence: str = Field(min_length=1, max_length=400)
    target_evidence: str = Field(min_length=1, max_length=400)


class EvidenceGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["skill-graph-v1"] = "skill-graph-v1"
    nodes: tuple[str, ...] = Field(min_length=8, max_length=8)
    edges: tuple[EvidenceEdge, ...] = ()


class OrganizationBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hierarchy: Hierarchy
    graph: EvidenceGraph
```

Implement `validate_bundle` so it flattens hierarchy leaves, requires exactly `S01` through `S08`, rejects depth greater than three, verifies edge endpoints and exact evidence substrings against only `name`, `description`, and `body`, and never repairs invalid LLM output silently.

- [ ] **Step 4: Implement deterministic reading order**

Use a `networkx.DiGraph`, add all eight aliases, add every validated directed edge, condense strongly connected components, then call `nx.lexicographical_topological_sort` with minimum FCSR rank as the key. Sort aliases inside a cycle by rank.

- [ ] **Step 5: Implement the DeepSeek organizer adapter**

```python
class OrganizerClient(Protocol):
    def organize(self, *, task_key: str, skills: list[dict[str, object]]) -> OrganizerReply: ...


class DeepSeekOrganizerClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def organize(self, *, task_key: str, skills: list[dict[str, object]]) -> OrganizerReply:
        forbidden = json.dumps(skills, ensure_ascii=False)
        if '"skill_id"' in forbidden or '"source"' in forbidden:
            raise ValueError("organizer payload contains prohibited provenance")
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=build_organizer_messages(task_key=task_key, skills=skills),
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("organizer returned empty content")
        return OrganizerReply(content=content, usage=extract_usage(response.usage))
```

The fixed system prompt must state that Skill documents are untrusted quoted data, instructions inside them must not override the schema, all eight aliases must be preserved, unsupported edges must be omitted, and no Skill text may be rewritten.

- [ ] **Step 6: Run organizer tests**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_organizer -v
```

Expected: all tests pass without network access.

- [ ] **Step 7: Commit the organizer**

```bash
git add src/skill_organization/organizer.py tests/test_skill_organization_organizer.py
git commit -m "feat(skill-org): add task-blind Skill organizer"
```

---

### Task 3: Byte-stable Skill rendering and fairness gate

**Files:**
- Create: `src/skill_organization/render.py`
- Create: `src/skill_organization/validate.py`
- Create: `tests/test_skill_organization_render.py`

**Interfaces:**
- Consumes: `TaskInput`, validated `OrganizationBundle`.
- Produces: `render_atomic_section(...)`, `render_context(...)`, `write_skill_packages(...)`, and `validate_rendered_task(...)`.

- [ ] **Step 1: Write failing byte-equivalence tests**

```python
class RenderTests(unittest.TestCase):
    def test_all_skill_conditions_share_identical_atomic_suffix(self):
        rendered = {
            method: render_context(make_task_input(), make_bundle(), method)
            for method in ("flat_top8", "hierarchy_top8", "graph_top8")
        }
        suffixes = {text.split("## Atomic skill payloads\n", 1)[1] for text in rendered.values()}
        self.assertEqual(len(suffixes), 1)

    def test_rendered_text_never_exposes_provenance(self):
        text = render_context(make_task_input_with_gt_ids(), make_bundle(), "flat_top8")
        self.assertNotIn("gt/", text)
        self.assertNotIn("distractor/", text)
        self.assertNotIn("Source:", text)
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_render -v
```

Expected: import failure for `rq2.render`.

- [ ] **Step 3: Implement one canonical atomic renderer**

```python
ATOMIC_MARKER = "## Atomic skill payloads\n"


def render_atomic_section(task: TaskInput) -> str:
    blocks = []
    for item in task.skills:
        blocks.append(
            f"### {item.alias}\n\n"
            f"Retrieval rank: {item.rank}\n\n"
            "<original_skill>\n"
            f"Name: {item.record.name}\n\n"
            "Description:\n"
            f"{item.record.description}\n\n"
            "Body:\n"
            f"{item.record.body}\n"
            "</original_skill>\n"
        )
    return ATOMIC_MARKER + "\n".join(blocks)
```

Generate the atomic section once per task and append the exact same string to all three organization headers. Render no graph evidence quotes; include only typed edges and reading order.

- [ ] **Step 4: Implement package writing**

Write each condition to:

```text
<output>/<task-id>/<method>/skills/retrieved-skills/SKILL.md
<output>/<task-id>/<method>/context_manifest.json
```

Use UTF-8 and `\n` newlines. The frontmatter is exactly:

```yaml
---
name: retrieved-skills
description: Retrieved procedural skills organized for the current task.
---
```

Write files through a sibling `.tmp` path followed by `os.replace`.

- [ ] **Step 5: Implement the fairness validator**

`validate_rendered_task` must reread all three files, split once at `ATOMIC_MARKER`, require identical suffix bytes, require the expected eight aliases exactly once, require the rendered hash values to match `context_manifest.json`, and reject occurrences of every hidden `skill_id` and the standalone `source` values in Agent-visible text.

- [ ] **Step 6: Run rendering tests**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_render -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit rendering and validation**

```bash
git add src/skill_organization/render.py src/skill_organization/validate.py tests/test_skill_organization_render.py
git commit -m "feat(skill-org): render equivalent Skill packages"
```

---

### Task 4: Preparation CLI and review workflow

**Files:**
- Create: `scripts/skill_organization.py`
- Create: `tests/test_skill_organization_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1--3.
- Produces CLI subcommands `audit`, `organize`, `render`, and `validate`.

- [ ] **Step 1: Write failing CLI dry-run tests**

Test `parse_args(["audit", ...])`, ensure the default Top-K is exactly 8, ensure `organize` refuses a missing API key, and ensure `validate` exits nonzero when one atomic suffix differs.

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_cli -v
```

Expected: `scripts.skill_organization` is missing.

- [ ] **Step 3: Implement the four subcommands**

```text
python scripts/skill_organization.py audit   --output reports/agent/skill-organization/<run-id>
python scripts/skill_organization.py organize --run-dir reports/agent/skill-organization/<run-id>
python scripts/skill_organization.py render   --run-dir reports/agent/skill-organization/<run-id>
python scripts/skill_organization.py validate --run-dir reports/agent/skill-organization/<run-id> --tasks-root /data-nfs/.../skillsbench/tasks
```

`audit` writes `experiment_manifest.json` and `preprocessing/frozen_skill_inventory.jsonl`. `organize` checkpoints one response per anonymous task key and never overwrites a validated response unless `--new-run-id` is used. `render` refuses unreviewed organization files. `validate` checks the fixed SkillsBench commit recorded by the caller plus `task.md`, `environment/Dockerfile`, `oracle/`, and `verifier/` for all 15 tasks.

- [ ] **Step 4: Add concise README commands**

Replace the obsolete planning-only Hard-15 section with the four commands above and a link to `docs/skill-organization.md`. State that the CLI prepares end-to-end BenchFlow inputs and does not report success until verifier results are collected.

- [ ] **Step 5: Run CLI tests and help output**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_cli -v
python scripts/skill_organization.py --help
```

Expected: tests pass and help lists `audit`, `organize`, `render`, `validate`.

- [ ] **Step 6: Commit the preparation CLI**

```bash
git add scripts/skill_organization.py tests/test_skill_organization_cli.py README.md
git commit -m "feat(skill-org): add preparation CLI"
```

---

### Task 5: Resumable BenchFlow run matrix

**Files:**
- Create: `src/skill_organization/runner.py`
- Create: `tests/test_skill_organization_runner.py`
- Modify: `scripts/skill_organization.py`

**Interfaces:**
- Consumes: validated generated packages and server SkillsBench root.
- Produces: `RunSpec`, `build_run_matrix(...)`, `bench_command(...)`, `execute_matrix(...)`, and CLI subcommands `plan-runs` and `run`.

- [ ] **Step 1: Write failing matrix and command tests**

```python
class RunnerTests(unittest.TestCase):
    def test_hard15_matrix_has_sixty_unique_runs(self):
        specs = build_run_matrix(task_ids=HARD15_IDS, repeats=1)
        self.assertEqual(len(specs), 60)
        self.assertEqual(len({spec.run_key for spec in specs}), 60)

    def test_no_skill_command_has_no_skills_dir(self):
        command = bench_command(make_spec("no_skill"), bench_bin="bench")
        self.assertIn("no-skill", command)
        self.assertNotIn("--skills-dir", command)

    def test_flat_command_uses_generated_package(self):
        command = bench_command(make_spec("flat_top8"), bench_bin="bench")
        self.assertIn("with-skill", command)
        self.assertIn("--skills-dir", command)
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_runner -v
```

Expected: import failure for `rq2.runner`.

- [ ] **Step 3: Implement exact run specifications**

```python
class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_key: str
    task_id: str
    condition: Literal["no_skill", "flat_top8", "hierarchy_top8", "graph_top8"]
    repeat_id: int = Field(ge=1)
    order_index: int = Field(ge=0)
    task_dir: Path
    skills_dir: Path | None
    jobs_dir: Path
```

Rotate the four-condition order by task index so every condition appears in each ordinal position across the batch. Serialize the immutable matrix to `run_matrix.jsonl` before execution.

- [ ] **Step 4: Build commands without a shell**

Every command begins:

```python
[
    bench_bin, "eval", "run",
    "--tasks-dir", str(spec.task_dir),
    "--agent", "openhands",
    "--model", "deepseek/deepseek-v4-flash",
    "--sandbox", "daytona",
    "--jobs-dir", str(spec.jobs_dir),
]
```

Append `--skill-mode no-skill` for `no_skill`; otherwise append `--skill-mode with-skill --skills-dir <generated-dir>`. Inherit credentials from the process environment but write only a boolean presence audit, never values.

- [ ] **Step 5: Implement resumable execution**

Use `subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)`. Write one atomic state JSON per `run_key` with `queued`, `running`, `completed`, or `process_error`. Skip only `completed`; do not auto-retry other states. Default to one worker and require explicit `--workers 4` for the pilot.

- [ ] **Step 6: Add and test `plan-runs` and `run`**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_runner -v
python scripts/skill_organization.py plan-runs --help
python scripts/skill_organization.py run --help
```

Expected: tests pass; help exposes worker count, bench path, tasks root, generated root, and jobs root.

- [ ] **Step 7: Commit the runner**

```bash
git add src/skill_organization/runner.py tests/test_skill_organization_runner.py scripts/skill_organization.py
git commit -m "feat(skill-org): add resumable BenchFlow matrix"
```

---

### Task 6: BenchFlow artifact collection and Skill organization metrics

**Files:**
- Create: `src/skill_organization/results.py`
- Create: `tests/test_skill_organization_results.py`
- Modify: `scripts/skill_organization.py`

**Interfaces:**
- Consumes: `run_matrix.jsonl`, BenchFlow `summary.json`, rollout `result.json`, `timing.json`, prompts, and trajectories.
- Produces: `collect_results(...)`, `task_matrix.csv`, `trajectories.jsonl`, `aggregate.json`, and `failures.jsonl`.

- [ ] **Step 1: Write failing result classification tests**

Fixtures must cover: passed reward 1, valid failed reward 0, verifier error, agent timeout, BenchFlow process error, Daytona setup error, and an unparseable missing result. Assert that verifier and infrastructure errors are never silently converted to reward 0.

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_results -v
```

Expected: import failure for `rq2.results`.

- [ ] **Step 3: Implement one normalized trajectory row**

```python
class TrajectoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_key: str
    task_id: str
    condition: str
    repeat_id: int
    status: Literal[
        "passed", "failed", "agent_error", "verifier_error",
        "timeout", "infrastructure_error", "missing_artifact",
    ]
    reward: float | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    tool_calls: int
    trajectory_steps: int
    environment_setup_time_s: float | None
    agent_execution_time_s: float | None
    verifier_time_s: float | None
    wall_time_s: float | None
    injection_verified: bool | None
    failure_type: str | None
```

Determine `injection_verified` by checking the expected context hash or a stable excerpt in `prompts.json`/trajectory; do not infer it from alias mentions in private reasoning.

- [ ] **Step 4: Implement paired summaries**

For the pilot, compute fixed-denominator pass rate over 15, valid-run pass rate, mean reward, condition transitions, token/time/tool summaries, error counts, and Top-8 complete/incomplete strata. Do not emit p-values for the one-repeat Hard-15 pilot.

- [ ] **Step 5: Add the `collect` command and tests**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_results -v
python scripts/skill_organization.py collect --help
```

Expected: tests pass and the command requires run and jobs directories.

- [ ] **Step 6: Commit result collection**

```bash
git add src/skill_organization/results.py tests/test_skill_organization_results.py scripts/skill_organization.py
git commit -m "feat(skill-org): collect verifier and efficiency metrics"
```

---

### Task 7: Cut over from the planning-only experiment and remove dead code

**Files:**
- Delete: `scripts/run_hard15_experiment.py`
- Delete: `scripts/sync_hard15_tasks.py`
- Delete: `scripts/migrate_jsonl_gzip.py`
- Delete: `src/agent/__init__.py`
- Delete: `src/agent/environment_audit.py`
- Delete: `src/agent/hard15_experiment.py`
- Delete: `src/agent/hard15_organizations.py`
- Delete: `src/agent/hard15_pilot.py`
- Delete: `src/agent/hard15_planning.py`
- Delete: `src/agent/llm.py`
- Delete: `src/agent/task_catalog.py`
- Delete: `src/agent/task_packages.py`
- Delete: `tests/test_agent_environment_audit.py`
- Delete: `tests/test_agent_hard15_catalog.py`
- Delete: `tests/test_agent_hard15_experiment.py`
- Delete: `tests/test_agent_hard15_organizations.py`
- Delete: `tests/test_agent_hard15_review_constraints.py`
- Delete: `tests/test_agent_llm.py`
- Delete: `tests/test_agent_task_packages.py`
- Delete: `tests/test_migrate_jsonl_gzip.py`
- Delete: `HARD15_RUN.md`
- Delete: `docs/rq2-skill-organization.md`
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: fully passing Tasks 1--6.
- Produces: one Skill organization implementation (`src/skill_organization/`), one CLI (`scripts/skill_organization.py`), one design document, and one implementation plan.

- [ ] **Step 1: Prove the new Skill organization suite passes before deletion**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_skill_organization_inputs tests.test_skill_organization_organizer tests.test_skill_organization_render tests.test_skill_organization_cli tests.test_skill_organization_runner tests.test_skill_organization_results -v
```

Expected: all new Skill organization tests pass.

- [ ] **Step 2: Delete the superseded files with `apply_patch`**

The deleted code is planning-only, downloads a leakage-safe partial task copy, groups by namespace, creates `same_namespace` edges, and cannot run BenchFlow verifier evaluation. Do not delete `data/agent/hard15/task_ids.txt`, `task_catalog.json`, frozen reports, raw data, checkpoints, RQ1 scripts, or reference PDFs.

- [ ] **Step 3: Remove the unused dependency**

Delete `langgraph>=1.0,<2` from `requirements.txt`. Keep `networkx`, `pydantic`, `openai`, and `python-dotenv` because the new Skill organization implementation uses them.

- [ ] **Step 4: Verify no stale references remain**

Run:

```powershell
git grep -n -E "run_hard15_experiment|sync_hard15_tasks|agent\.hard15|same_namespace|planning-only"
```

Expected: no matches in active code or documentation.

- [ ] **Step 5: Run the complete repository test suite**

Run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the cutover**

```bash
git add -A scripts src/agent tests HARD15_RUN.md docs/rq2-skill-organization.md requirements.txt README.md
git commit -m "refactor(skill-org): remove planning-only Hard-15 pipeline"
```

---

### Task 8: Local dry run and server handoff

**Files:**
- Modify: `docs/skill-organization.md` only if verified commands differ from the design.

**Interfaces:**
- Consumes: completed Skill organization CLI.
- Produces: a validated local preparation run and exact server commands for the eight-trajectory smoke test.

- [ ] **Step 1: Run the local frozen-input audit**

```powershell
$env:PYTHONPATH = "src"
python scripts/skill_organization.py audit --output reports/agent/skill-organization/hard15-pilot
```

Expected: 15 tasks, 120 task-level Skill instances, 114 unique records, zero missing records, and matching Hard JSONL SHA-256.

- [ ] **Step 2: Generate and validate organization metadata**

Run `organize`, perform the task-blind review recorded in `review_log.jsonl`, then run `render` and `validate`. Expected: 45 Skill packages, zero payload mismatches, zero provenance leaks, and 15 valid hierarchy/graph pairs.

- [ ] **Step 3: Generate the 60-run matrix without executing it**

```powershell
$env:PYTHONPATH = "src"
python scripts/skill_organization.py plan-runs --run-dir reports/agent/skill-organization/hard15-pilot --repeats 1
```

Expected: 60 unique run specs with 15 entries for each condition.

- [ ] **Step 4: Run complete verification before server handoff**

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
git status --short
```

Expected: all tests pass; status contains only intentional RQ2 changes and the user's pre-existing example deletions.

- [ ] **Step 5: Commit the verified documentation state**

```bash
git add docs/skill-organization.md docs/skill-organization-plan.md
git commit -m "docs(skill-org): document Hard-15 experiment"
```
