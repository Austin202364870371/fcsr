# Hierarchy Skill Organizer

This stage adds a reproducible Flat-versus-Hierarchy Agent baseline. It is an
organization experiment on top of the same FCSR retrieval candidates, not a new
retrieval method.

## What is controlled

Both methods receive the same ordered candidate Skill IDs and use the same:

- candidate cap (`--top-k`);
- final Skill budget (`--max-skills`);
- selector, tools, verifier, and Agent runtime;
- task records and expected outputs.

The only changed variable is how recalled Skills are presented to the Agent:

- **Flat:** retain the global FCSR rank and show one ranked list.
- **Hierarchy:** group by the first `category_path` segment by default, score a
  group with the sum of reciprocal member ranks, retain the top groups, and
  restore global FCSR order inside the final Skill budget.

Missing categories are placed in `uncategorized`, so candidates are not silently
dropped.

## Run locally

Activate the installed environment:

```powershell
conda activate agent-learn
$env:PYTHONPATH = "src"
```

Run the paired example:

```powershell
python -B scripts/compare_organizers.py `
  --tasks data/agent/examples/tasks.json `
  --skills data/agent/examples/skills.json `
  --output-dir reports/agent/organizer_comparison `
  --max-groups 2
```

Outputs are:

- `flat.jsonl` and `hierarchy.jsonl`: replayable per-task Agent traces;
- `summary.json`: paired success counts and organization-cost statistics.

Run all tests:

```powershell
python -B -m unittest discover -s tests -v
```

## How to interpret the current result

The two bundled smoke-test tasks both succeed under Flat and Hierarchy. This only
checks that the paired experiment is executable and auditable. It is not evidence
that Hierarchy is better: the sample is tiny, deterministic, and uses a
first-ranked selector rather than an LLM planner.

The paper-facing comparison must use a larger executable task set and report at
least end-to-end task success, verification success, tool-call count, latency,
token/context cost, and failure type. Retrieval metrics remain intermediate
diagnostics. Graph organization and improved planning should be introduced only
after this Flat/Hierarchy protocol is stable, so each ablation changes one major
factor at a time.

## Local versus server work

This implementation and smoke test run locally and do not require a GPU. A server
becomes useful when scaling FCSR inference, running an LLM-based selector/planner,
or evaluating the expanded task set repeatedly. PyTorch training tests remain
optional for this Agent stage.
