# Agent Legacy Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the superseded toy Agent pipeline while preserving the Hard15 planning experiment and the execution-readiness audit.

**Architecture:** Extract the shared LLM response, plan schema, and DeepSeek client into `agent.llm`. Make the Hard15 planner depend on that module, then delete the old pilot/runtime path together with its CLI entry points and tests. Retain `environment_audit.py` for the future verifier-backed execution stage.

**Tech Stack:** Python 3.12, Pydantic, OpenAI-compatible DeepSeek client, pytest.

## Global Constraints

- Preserve `scripts/sync_hard15_tasks.py` and `scripts/run_hard15_experiment.py` behavior and checkpoint format.
- Do not delete `environment_audit.py`.
- Do not retain a `legacy/` source directory containing runnable toy Agent code.
- Validate with the full pytest suite and a no-API Hard15 dry-run.

---

### Task 1: Extract the shared LLM contract

**Files:**
- Create: `src/agent/llm.py`
- Modify: `src/agent/hard15_planning.py`
- Test: `tests/test_agent_llm.py`

**Interfaces:**
- Produces `LLMReply`, `PlanningClient`, `PlanStep`, `SkillPlan`, and `DeepSeekPlanningClient` from `agent.llm`.
- `hard15_planning.py` imports `PlanningClient` and `SkillPlan` from `agent.llm`.

- [ ] **Step 1: Write the failing import-contract test**

```python
from agent.llm import LLMReply, PlanStep, SkillPlan

def test_shared_llm_contract_validates_a_plan():
    plan = SkillPlan(
        selected_skill_aliases=("S01",),
        steps=(PlanStep(id="step-1", objective="do", skill_aliases=("S01",), expected_output="x"),),
        final_output="x",
    )
    assert plan.selected_skill_aliases == ("S01",)
    assert LLMReply(content="{}", prompt_tokens=0, completion_tokens=0).content == "{}"
```

- [ ] **Step 2: Run the test to verify RED**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_llm.py -q`

Expected: import failure because `agent.llm` does not exist.

- [ ] **Step 3: Create `agent.llm` with the shared models and client**

Move without behavioral changes from old `planning.py`:

```python
class LLMReply(BaseModel): ...
class PlanningClient(Protocol): ...
class PlanStep(BaseModel): ...
class SkillPlan(BaseModel): ...
class DeepSeekPlanningClient: ...
```

Keep `PlanStep.skill_aliases` non-empty in this cleanup; changing plan-validity policy is a separate experiment change.

- [ ] **Step 4: Point `hard15_planning.py` at `agent.llm`**

Replace:

```python
from agent.planning import PlanningClient, SkillPlan
```

with:

```python
from agent.llm import PlanningClient, SkillPlan
```

- [ ] **Step 5: Run focused GREEN tests**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_llm.py tests/test_agent_hard15_experiment.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/agent/llm.py src/agent/hard15_planning.py tests/test_agent_llm.py
git commit -m "refactor: extract shared agent llm contract"
```

### Task 2: Remove the replaced pilot and toy runtime path

**Files:**
- Delete: `src/agent/candidates.py`, `src/agent/hard_pilot.py`, `src/agent/hierarchy.py`, `src/agent/local_tasks.py`, `src/agent/models.py`, `src/agent/organizers.py`, `src/agent/planning.py`, `src/agent/presentation.py`, `src/agent/runtime.py`, `src/agent/selectors.py`, `src/agent/tools.py`, `src/agent/verifiers.py`
- Delete: `scripts/compare_organizers.py`, `scripts/plan_hard_pilot.py`, `scripts/prepare_hard_pilot.py`, `scripts/run_agent.py`
- Delete: old tests named `test_agent_candidates.py`, `test_agent_hard_pilot.py`, `test_agent_hierarchy.py`, `test_agent_hierarchy_runtime.py`, `test_agent_instruction_skills.py`, `test_agent_models.py`, `test_agent_organizers.py`, `test_agent_planning.py`, `test_agent_presentation.py`, `test_agent_runtime.py`, `test_agent_selectors.py`, `test_agent_tools.py`, `test_agent_verifiers.py`, `test_compare_organizers_cli.py`, `test_compare_organizers_direct_cli.py`, and `test_run_agent_cli.py`
- Modify: `src/agent/__init__.py`, `HARD15_RUN.md`
- Test: `tests/test_agent_llm.py`

**Interfaces:**
- `agent` remains an importable package.
- Current supported entry points are only `sync_hard15_tasks.py` and `run_hard15_experiment.py`.

- [ ] **Step 1: Add a failing package-import test**

```python
import agent

def test_agent_package_remains_importable_without_toy_runtime_exports():
    assert agent.__doc__
```

- [ ] **Step 2: Run it to verify RED after temporarily removing old exports**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_llm.py -q`

Expected: fail until `agent.__init__` stops importing deleted `agent.models`.

- [ ] **Step 3: Delete the complete old path and simplify package exports**

Set `src/agent/__init__.py` to a package docstring. Remove all old code, CLI
entry points, and their corresponding tests in the same change. Keep
`environment_audit.py`, all Hard15 modules, `task_catalog.py`, and
`task_packages.py`.

- [ ] **Step 4: Update documentation**

Replace the section in `HARD15_RUN.md` that describes the current toy
`VerifierRegistry` with a statement that it was intentionally removed and
that the next execution stage must integrate original SkillsBench verifiers.

- [ ] **Step 5: Run focused GREEN checks**

Run: `conda run -n agent-learn python -m pytest tests/test_agent_llm.py tests/test_agent_environment_audit.py tests/test_agent_hard15_catalog.py tests/test_agent_task_packages.py -q`

Expected: all retained focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -A
git commit -m "refactor: remove superseded toy agent path"
```

### Task 3: Verify the retained Hard15 workflow

**Files:**
- Verify only: `scripts/run_hard15_experiment.py`, `reports/agent/hard15/`

**Interfaces:**
- The dry-run still prepares 15 tasks and 45 organization prompts without API calls.

- [ ] **Step 1: Run the full test suite**

Run: `conda run -n agent-learn python -m pytest -q`

Expected: all retained tests pass.

- [ ] **Step 2: Run the real-data no-API workflow**

Run: `conda run -n agent-learn python scripts/run_hard15_experiment.py --dry-run`

Expected: `Prepared 15 tasks and 45 organization prompts; no API calls made`.

- [ ] **Step 3: Verify no deleted import remains**

Run: `Select-String -Path src\*.py,src\agent\*.py,scripts\*.py -Pattern 'agent\.(planning|hard_pilot|runtime|verifiers|models|candidates)'`

Expected: no matches.

- [ ] **Step 4: Commit verification-only documentation correction if needed**

If verification requires a documentation correction, commit it with:

```powershell
git add HARD15_RUN.md
git commit -m "docs: clarify retained agent workflow"
```
