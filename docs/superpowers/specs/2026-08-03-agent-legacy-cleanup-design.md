# Agent Legacy Cleanup Design

## Goal

Remove the superseded toy Agent path so `src/agent` represents the current
Hard15 planning experiment and only the execution-readiness utility needed for
the future SkillsBench verifier stage.

## Decision

Delete the complete old demo chain rather than moving it into a `legacy`
directory. Its one-tool runtime, first-ranked selector, and toy verifiers are
not valid execution infrastructure and keeping them risks accidental use in
research results.

Retain `environment_audit.py`: it is small, independent, and checks whether a
future full SkillsBench package contains an environment and original verifier.

Extract the shared DeepSeek client and plan schema from the old `planning.py`
into `llm.py`. The Hard15 planner will import only this shared module. This
allows deletion of `planning.py` and `hard_pilot.py` without changing the
Hard15 experiment's public behavior.

## Retained layout

`src/agent` will contain these conceptual layers:

- **Planning experiment:** `hard15_experiment.py`, `hard15_organizations.py`,
  `hard15_pilot.py`, `hard15_planning.py`, `task_catalog.py`, and
  `task_packages.py`.
- **Shared LLM contract:** `llm.py`.
- **Future execution readiness:** `environment_audit.py`.

The package `__init__.py` will no longer export models from the deleted toy
runtime.

## Removed scope

Delete the old pilot, candidate adapter, organizer, hierarchy, toy tools,
runtime, selector, renderer, verifier, and their CLI entry points and tests.
This includes `compare_organizers.py`, `run_agent.py`,
`prepare_hard_pilot.py`, and `plan_hard_pilot.py`.

## Compatibility and verification

The supported commands after cleanup are:

```powershell
python scripts/sync_hard15_tasks.py
python scripts/run_hard15_experiment.py --dry-run
python scripts/run_hard15_experiment.py
```

The cleanup must preserve the Hard15 45-call workflow and its existing
checkpoint format. Unit tests for deleted functionality are deleted with the
implementation. Remaining tests must pass, and a no-API Hard15 dry-run must
still generate all 15 tasks and 45 presentations.
