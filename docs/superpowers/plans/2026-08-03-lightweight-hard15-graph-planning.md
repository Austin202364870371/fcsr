# Lightweight Hard-15 Graph Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local planning-only experiment that synchronizes 15 lightweight SkillsBench tasks and compares Flat, Hierarchy, and evidence-based Graph Skill presentations with DeepSeek.

**Architecture:** A tracked catalog freezes task identity and source revision. A small synchronization boundary downloads only public task context, while pure modules handle readiness, metadata normalization, organization, prompt presentation, planning, resume, and private evaluation. The one-command experiment never invokes task tools or claims verifier success.

**Tech Stack:** Python 3.10, Pydantic 2, NetworkX, OpenAI-compatible DeepSeek client, huggingface_hub, unittest/pytest.

## Global Constraints

- Freeze SkillsBench to GitHub tag commit `b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af` and a pinned Hugging Face dataset revision.
- Use exactly the 15 task IDs and four coverage strata approved in the design.
- Never expose `gt_skill_ids`, bundled Skills, oracle files, verifier files, or ground-truth directories to the planner.
- Use the original FCSR query and final FCSR Top-20 candidates unchanged.
- Compare all methods with the same maximum Skill count and total body-character budget.
- Report planning metrics only; never label them task success, Pass@1, or verifier reward.
- Work directly on `main`, as explicitly requested by the repository owner.

---

### Task 1: Frozen Hard-15 catalog and report cleanup

**Files:**
- Create: `data/agent/hard15/task_catalog.json`
- Create: `data/agent/hard15/task_ids.txt`
- Delete generated outputs under: `reports/agent/flat_example.jsonl`, `reports/agent/flat_smoke.jsonl`, `reports/agent/organizer_comparison/`, `reports/agent/hard15/`
- Test: `tests/test_agent_hard15_catalog.py`

**Interfaces:**
- Produces: a JSON array with `task_id`, `source_task_id`, `stratum`, `estimated_context_bytes`, and `source_path`.
- Produces: the exact ordered 15 task IDs consumed by synchronization and pilot preparation.

- [ ] **Step 1: Write the failing catalog test**

```python
def test_catalog_has_exact_fixed_strata_and_unique_ids():
    catalog = load_catalog(CATALOG_PATH)
    assert len(catalog) == 15
    assert len({row.task_id for row in catalog}) == 15
    assert Counter(row.stratum for row in catalog) == {
        "single_full": 3,
        "single_missing": 2,
        "multi_full": 5,
        "multi_missing": 5,
    }
```

- [ ] **Step 2: Run test and verify RED**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_hard15_catalog.py -q`
Expected: FAIL because the catalog and loader do not exist.

- [ ] **Step 3: Add the minimal validated catalog and loader**

Implement a frozen Pydantic `PilotCatalogEntry` and `load_pilot_catalog(path)` that reject duplicate IDs, unknown strata, non-positive sizes, and non-`tasks/` source paths.

- [ ] **Step 4: Run test and verify GREEN**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_hard15_catalog.py -q`
Expected: PASS.

- [ ] **Step 5: Resolve and remove only the approved generated report paths**

Verify every resolved target is beneath `reports/agent`, then remove the four approved targets with PowerShell `Remove-Item -LiteralPath`.

- [ ] **Step 6: Commit**

```bash
git add data/agent/hard15 tests/test_agent_hard15_catalog.py src/agent/task_catalog.py
git commit -m "feat: freeze lightweight hard15 task catalog"
```

### Task 2: Selective package synchronization and readiness

**Files:**
- Create: `src/agent/task_packages.py`
- Create: `scripts/sync_hard15_tasks.py`
- Modify: `src/agent/environment_audit.py`
- Modify: `.gitignore`
- Modify: `requirements-agent.txt`
- Test: `tests/test_agent_task_packages.py`
- Test: `tests/test_agent_environment_audit.py`

**Interfaces:**
- Produces: `prohibited_relative_path(path: str) -> bool`.
- Produces: `sync_task_packages(catalog, destination, downloader) -> TaskSyncManifest`.
- Produces: `audit_planning_environment(task_id, packages_root) -> PlanningAudit`.
- Consumes: an injected downloader in tests; production wraps `huggingface_hub.snapshot_download`.

- [ ] **Step 1: Write failing filter and readiness tests**

```python
def test_prohibited_paths_cover_all_private_surfaces():
    assert prohibited_relative_path("x/environment/skills/a/SKILL.md")
    assert prohibited_relative_path("x/oracle/solve.sh")
    assert prohibited_relative_path("x/verifier/test_outputs.py")
    assert prohibited_relative_path("x/environment/groundtruth/answer.json")
    assert not prohibited_relative_path("x/environment/data/input.csv")

def test_planning_ready_requires_task_md_and_no_private_paths(tmp_path):
    task = tmp_path / "x"
    task.mkdir()
    (task / "task.md").write_text("task", encoding="utf-8")
    assert audit_planning_environment("x", tmp_path).planning_ready
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_task_packages.py tests/test_agent_environment_audit.py -q`
Expected: FAIL because the new interfaces do not exist.

- [ ] **Step 3: Implement minimal sync, manifest, and audit behavior**

Use exact allow patterns for the 15 source directories and ignore patterns for Skills, oracle, verifier, and ground-truth names. Validate the final filesystem recursively before marking a task ready. Preserve the existing execution audit unchanged.

- [ ] **Step 4: Add CLI and dependency**

`scripts/sync_hard15_tasks.py` accepts `--catalog`, `--packages-dir`, `--manifest`, and optional `--revision`; its default revision is the pinned catalog revision. Add `huggingface_hub>=0.34,<1` and ignore `data/agent/hard15/packages/`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_task_packages.py tests/test_agent_environment_audit.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements-agent.txt src/agent/task_packages.py src/agent/environment_audit.py scripts/sync_hard15_tasks.py tests/test_agent_task_packages.py tests/test_agent_environment_audit.py
git commit -m "feat: sync leakage-safe hard15 task context"
```

### Task 3: Deterministic fixed-pilot preparation

**Files:**
- Modify: `src/agent/hard_pilot.py`
- Modify: `scripts/prepare_hard_pilot.py`
- Modify: `tests/test_agent_hard_pilot.py`
- Create: `tests/test_prepare_hard_pilot_cli.py`

**Interfaces:**
- Changes: `prepare_hard_pilot(..., selected_task_ids: Sequence[str]) -> PreparedPilot` selects the catalog order and validates every stratum instead of hash sampling.
- CLI consumes the catalog and planning-package root; it produces `tasks.jsonl`, `evaluation.jsonl`, and a manifest-aware summary.

- [ ] **Step 1: Write failing fixed-order and missing-package tests**

```python
def test_fixed_task_ids_replace_random_sampling():
    pilot = prepare_hard_pilot(..., selected_task_ids=["b", "a"])
    assert [task.task_id for task in pilot.public_tasks] == ["b", "a"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_hard_pilot.py tests/test_prepare_hard_pilot_cli.py -q`
Expected: FAIL because selection still uses quotas and a hash seed.

- [ ] **Step 3: Implement fixed selection and planning audit**

Reject absent query/ranking/Skill records, duplicate catalog IDs, wrong coverage strata, and non-ready packages. Keep alias/GT separation unchanged.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_hard_pilot.py tests/test_prepare_hard_pilot_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/hard_pilot.py scripts/prepare_hard_pilot.py tests/test_agent_hard_pilot.py tests/test_prepare_hard_pilot_cli.py
git commit -m "feat: prepare fixed planning-ready hard15 pilot"
```

### Task 4: Skill namespace normalization

**Files:**
- Modify: `src/agent/candidates.py`
- Modify: `tests/test_agent_candidates.py`
- Modify: `tests/test_agent_hierarchy.py`

**Interfaces:**
- Produces: `normalized_category_path(source, skill_id) -> tuple[str, ...]` with explicit metadata precedence and ID namespace fallback.

- [ ] **Step 1: Write failing fallback tests**

```python
def test_skill_id_namespace_is_used_when_category_is_absent():
    result = adapt_ranked_candidates(record, {"design/pdf/edit": source})
    assert result[0].category_path == ("design", "pdf")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_candidates.py tests/test_agent_hierarchy.py -q`
Expected: FAIL because absent metadata currently becomes an empty path.

- [ ] **Step 3: Implement minimal normalization**

Use at most the first two non-empty ID path segments, while preserving explicit category metadata exactly.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_candidates.py tests/test_agent_hierarchy.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/candidates.py tests/test_agent_candidates.py tests/test_agent_hierarchy.py
git commit -m "fix: derive hierarchy namespaces from skill ids"
```

### Task 5: Evidence Graph organizer and presentation

**Files:**
- Create: `src/agent/graph.py`
- Modify: `src/agent/models.py`
- Modify: `src/agent/presentation.py`
- Create: `tests/test_agent_graph.py`
- Modify: `tests/test_agent_presentation.py`

**Interfaces:**
- Produces: `GraphEdge(source, target, relation, evidence)`.
- Produces: `GraphSkillBundle(strategy="graph", skills, edges, reading_order, component_count)`.
- Produces: `GraphOrganizer(max_skills).organize(candidates) -> GraphSkillBundle`.

- [ ] **Step 1: Write failing edge, selection, cycle, and rendering tests**

```python
def test_explicit_reference_becomes_directed_evidence_edge():
    a = candidate("a", 1, body="Use skill b before this step")
    b = candidate("b", 2)
    bundle = GraphOrganizer(max_skills=2).organize([a, b])
    assert [(e.source, e.target, e.relation) for e in bundle.edges] == [
        ("a", "b", "explicit_reference")
    ]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_graph.py tests/test_agent_presentation.py -q`
Expected: FAIL because no Graph organizer exists.

- [ ] **Step 3: Implement evidence extraction and graph selection**

Match normalized candidate IDs and unambiguous names with token boundaries. Add namespace edges deterministically. Seed by rank, then score remaining nodes by selected-neighbor count, unseen namespace bonus, and reciprocal rank. Use NetworkX for components and DAG ordering; resolve cycles by stable rank.

- [ ] **Step 4: Extend rendering and stats**

Render a node list, typed evidence edges, and reading order. Extend stats with optional graph fields while keeping Flat/Hierarchy serialization compatible.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_graph.py tests/test_agent_presentation.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agent/graph.py src/agent/models.py src/agent/presentation.py tests/test_agent_graph.py tests/test_agent_presentation.py
git commit -m "feat: add evidence graph skill organizer"
```

### Task 6: Shared-budget three-method planning experiment

**Files:**
- Create: `src/agent/pilot_presentations.py`
- Modify: `src/agent/planning.py`
- Create: `src/agent/planning_experiment.py`
- Create: `scripts/run_hard15_experiment.py`
- Modify: `tests/test_agent_planning.py`
- Create: `tests/test_agent_pilot_presentations.py`
- Create: `tests/test_agent_planning_experiment.py`

**Interfaces:**
- Produces: `present_pilot_task(task, method, max_skills, total_body_chars) -> PresentedPilotTask`.
- Changes: `plan_task` consumes the presented method-specific payload and records a configuration fingerprint.
- Produces: resumable `run_planning_experiment(...)` and private `summarize_plans(...)`.

- [ ] **Step 1: Write failing shared-budget and leakage tests**

```python
def test_all_methods_obey_the_same_total_body_budget():
    for method in ("flat", "hierarchy", "graph"):
        shown = present_pilot_task(task, method, max_skills=5, total_body_chars=100)
        assert sum(len(skill.instructions) for skill in shown.skills) <= 100
        assert "gt_skill_ids" not in shown.model_dump_json()
```

- [ ] **Step 2: Write failing resume and evaluation tests**

Verify that compatible records are skipped, incompatible fingerprints are rejected, and GT coverage is computed only after model calls from the private evaluation file.

- [ ] **Step 3: Run tests and verify RED**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_planning.py tests/test_agent_pilot_presentations.py tests/test_agent_planning_experiment.py -q`
Expected: FAIL because method presentations and experiment runner do not exist.

- [ ] **Step 4: Implement minimal method presentation and planning payload**

Allocate body characters in selected rank order until the shared total is exhausted. Put hierarchy groups or graph relations in a separate `organization` object. Preserve alias validation.

- [ ] **Step 5: Implement checkpointed runner and summary**

Write one atomic JSONL checkpoint after each successful task-method pair. Produce per-method and paired metrics without calling them task success.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_planning.py tests/test_agent_pilot_presentations.py tests/test_agent_planning_experiment.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agent/pilot_presentations.py src/agent/planning.py src/agent/planning_experiment.py scripts/run_hard15_experiment.py tests/test_agent_planning.py tests/test_agent_pilot_presentations.py tests/test_agent_planning_experiment.py
git commit -m "feat: run paired hard15 planning experiment"
```

### Task 7: One-command documentation and full verification

**Files:**
- Modify: `HARD15_AGENT_README.md`
- Modify: `src/agent/__init__.py` if public exports are needed

**Interfaces:**
- Documents: environment installation, task synchronization, pilot preparation, 45-call experiment, resume behavior, outputs, and the planning/verifier boundary.

- [ ] **Step 1: Update direct-run documentation**

Document these commands using the `agent-learn` environment:

```powershell
conda activate agent-learn
pip install -r requirements-agent.txt
$env:PYTHONPATH="src"
python -B scripts/sync_hard15_tasks.py
python -B scripts/prepare_hard_pilot.py --catalog data/agent/hard15/task_catalog.json --queries data/raw/evaluation_queries.jsonl --rankings reports/hard/fcsr/reranker_hard.jsonl --skills data/raw/skills_hard.jsonl --task-environments data/agent/hard15/packages --output-dir reports/agent/hard15
python -B scripts/run_hard15_experiment.py --tasks reports/agent/hard15/tasks.jsonl --evaluation reports/agent/hard15/evaluation.jsonl --output-dir reports/agent/hard15 --model deepseek-v4-flash
```

- [ ] **Step 2: Run the complete test suite**

Run: `conda run -n agent-learn python -m pytest -q`
Expected: all applicable tests pass; only documented optional PyTorch skips are allowed.

- [ ] **Step 3: Run static and repository checks**

Run: `git diff --check`
Expected: no output.

Run: `git status --short`
Expected: only intended files before the final commit.

- [ ] **Step 4: Commit**

```bash
git add HARD15_AGENT_README.md src/agent/__init__.py
git commit -m "docs: explain direct hard15 planning run"
```

- [ ] **Step 5: Final verification and handoff**

Re-run focused Hard-15/Graph tests, confirm the worktree is clean, and report the exact commands and the boundary between planning results and future SkillsBench verifier execution.

