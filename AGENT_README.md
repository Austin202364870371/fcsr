# FCSR Local Skill Agent

This directory contains the first executable Agent baseline built on top of the
FCSR retrieval project. The baseline is intentionally small: it freezes the
retrieved candidate order, exposes the candidates as a flat bundle, selects one
skill, executes its bound local tool, and verifies the output deterministically.

## Boundary

The retrieval and Agent layers are separate:

```text
FCSR ranked_skill_ids
  -> candidate adapter
  -> FlatOrganizer
  -> SkillSelector
  -> ToolRegistry
  -> VerifierRegistry
  -> AgentRunResult JSONL
```

`src/agent/` does not train or run an FCSR model. It consumes an ordered list of
Skill IDs and refuses unknown Skills or Skills without a `tool_name`. This keeps
retrieval misses distinct from organization, selection, execution, and
verification failures.

## Modules

- `models.py`: validated candidate, bundle, tool, verification, and run records.
- `candidates.py`: strict adapter for FCSR `ranked_skill_ids` records.
- `organizers.py`: bounded Flat organization baseline.
- `selectors.py`: injectable selector protocol and deterministic top-ranked baseline.
- `tools.py`: fail-closed local tool registry.
- `verifiers.py`: deterministic verifier registry; no LLM-judge fallback.
- `runtime.py`: LangGraph state machine and replayable trace generation.

## Local setup

Activate the `agent-learn` environment and install the Agent dependency set:

```powershell
python -m pip install -r requirements-agent.txt
```

Run the Agent tests:

```powershell
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -p "test_agent_*.py" -v
```

Run the two offline examples:

```powershell
$env:PYTHONPATH = "src"
python -B scripts/run_agent.py `
  --tasks data/agent/examples/tasks.json `
  --skills data/agent/examples/skills.json `
  --output reports/agent/flat_example.jsonl
```

The command performs no network request and needs no GPU. Each output row
contains the selected Skill, tool result, verifier result, termination reason,
and ordered trace events.

## Input records

Each Skill record must contain `skill_id`, `name`, `description`, and
`tool_name`. `category_path` is optional in the Flat baseline but is retained
for the later Hierarchy experiment.

Each task record contains:

- `task_id` and natural-language `task`;
- ordered `ranked_skill_ids`, normally supplied by a frozen FCSR run;
- `selector_arguments` for the deterministic no-LLM baseline;
- `verifier_id` and hidden `expected` result.

The example task file includes `expected` inline only to keep the demonstration
self-contained. Benchmark tasks must keep verifier expectations outside the
Agent-visible task context.

## Server boundary

This baseline runs locally. A server is needed later for:

- regenerating FCSR candidates with GPU models;
- running local LLM selectors or planners;
- evaluating hundreds of tasks across multiple methods and repetitions.

Hierarchy, dependency Graph, LLM selection, Plan-and-Execute, local repair, and
the 240-task benchmark are intentionally separate follow-up stages. They must
reuse the same candidate, tool, verifier, run-result, and trace interfaces so
that each experiment changes one variable at a time.
