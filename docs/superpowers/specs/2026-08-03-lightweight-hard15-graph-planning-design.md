# Lightweight Hard-15 Graph Planning Experiment Design

## Goal

Connect the frozen FCSR hard-pool Top-20 rankings to a reproducible local
planning experiment that compares Flat, Hierarchy, and Graph Skill
presentations on 15 real SkillsBench tasks. This phase uses DeepSeek for
planning only. It does not execute task tools and must not report end-to-end
task success.

## Experimental boundary

- Freeze the benchmark source to SkillsBench `v1.1`, tag commit
  `b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af`.
- Keep the existing FCSR query text and final Top-20 rankings unchanged.
- Run the same 15 tasks, model, temperature, maximum Skill count, and total
  Skill-body character budget for all three organization methods.
- Never expose ground-truth Skill IDs, bundled Skills, oracle solutions, or
  verifier code to the planning model.
- Treat every generated result as a planning-stage result. End-to-end success
  begins only after a sandboxed task executor and the original SkillsBench
  verifier are connected.

## Fixed lightweight task set

The task set preserves the original four FCSR coverage strata while minimizing
the non-leaking task context downloaded from the official mirror.

| Stratum | Count | Task IDs |
| --- | ---: | --- |
| single Skill, full Top-20 coverage | 3 | `jax-computing-basics`, `dialogue-parser`, `econ-detrending-correlation` |
| single Skill, incomplete Top-20 coverage | 2 | `citation-check`, `enterprise-information-search` |
| multiple Skills, full Top-20 coverage | 5 | `flood-risk-analysis`, `syzkaller-ppdev-syzlang`, `manufacturing-fjsp-optimization`, `threejs-to-obj`, `threejs-structure-parser` |
| multiple Skills, incomplete Top-20 coverage | 5 | `setup-fuzzing-py`, `suricata-custom-exfil`, `powerlifting-coef-calc`, `xlsx-recover-data`, `parallel-tfidf-search` |

The filtered context is approximately 27 MB. The
`enterprise-information-search` package contributes about 26.6 MB but remains
because it is the only default runnable alternative, besides
`citation-check`, in the single-Skill/incomplete-coverage stratum. The other 14
tasks together are below 0.4 MB.

## Task package synchronization

Use the official `benchflow/skillsbench` Hugging Face dataset mirror and pin a
specific repository revision. Download only the 15 fixed task directories and
exclude these paths:

- `environment/skills/**`
- `oracle/**`
- `verifier/**`
- directories whose name is `groundtruth`, `ground_truth`, or `reference_answer`

The sync command writes an immutable manifest containing the task ID, source
path, source revision, expected stratum, estimated bytes, local path, and sync
status. A failed or partial sync never produces a ready pilot.

The original FCSR query remains the planner's task text so the retrieval and
Agent stages use the same input. Downloaded task files establish provenance and
make referenced task inputs locally available; private evaluator files are not
placed in the model prompt.

## Planning and execution readiness

Readiness is split explicitly:

- `planning_ready`: the task is in the frozen catalog, its synced `task.md`
  exists, and no prohibited path was downloaded.
- `execution_ready`: the complete task package, container, and original
  deterministic verifier exist.

This phase requires only `planning_ready`. The existing strict execution audit
is retained for the later Docker or cloud-sandbox phase.

## Skill metadata normalization

The current hard Skill pool usually lacks `category_path`. Normalize it with
the following precedence:

1. explicit `category_path`;
2. explicit `category`;
3. stable namespace segments derived from `skill_id`;
4. `uncategorized` only when none of the above exists.

This prevents the Hierarchy condition from collapsing all hard-pool Skills
into one meaningless group.

## Comparable organizers

### Flat

Expose the highest-ranked candidates as one ordered list within the shared
Skill-count and character budgets.

### Hierarchy

Group candidates by normalized namespace, rank groups by reciprocal-rank mass,
and expose selected groups and members within the same budgets. Preserve global
FCSR rank inside and across groups.

### Graph

Build an evidence graph over only the frozen FCSR Top-20 candidates.

- Nodes are Skill aliases.
- `explicit_reference` is a directed edge when one candidate body or
  description explicitly names another candidate by ID or unambiguous name.
- `same_namespace` is an undirected structural edge between candidates sharing
  a normalized namespace.
- Text similarity is not labeled as a dependency, and no synthetic `requires`
  edge is created.

Seed selection with the highest-ranked candidate, then add candidates that
maximize evidence-supported connectivity and namespace coverage while obeying
the shared budgets. Render nodes, typed edges, evidence, connected components,
and a deterministic reading order. Topologically order directed reference
edges when acyclic; break cycles and all remaining ties by FCSR rank. Do not
expand beyond Top-20 in the main experiment.

## Fair prompt construction

Every method receives the same task and candidate Top-20. Apply one total
Skill-body character budget per task rather than a per-Skill limit. Record the
rendered prompt size, selected aliases, omitted aliases, and organization
metadata. Use DeepSeek temperature zero and the existing JSON schema
validation. Unknown aliases or steps that use unselected aliases are errors.

## Outputs and resumability

The experiment produces:

- `data/agent/hard15/task_catalog.json`: tracked frozen 15-task catalog;
- `data/agent/hard15/task_ids.txt`: tracked human-readable task list;
- `data/agent/hard15/packages/`: ignored synchronized public task context;
- `reports/agent/hard15/task_manifest.json`: synchronization and audit record;
- `reports/agent/hard15/tasks.jsonl`: anonymous planner-visible tasks;
- `reports/agent/hard15/evaluation.jsonl`: private GT/alias mapping;
- `reports/agent/hard15/plans/{flat,hierarchy,graph}.jsonl`: validated plans;
- `reports/agent/hard15/summary.json`: paired planning metrics.

Run 45 calls (`15 tasks x 3 methods`) with one atomic checkpoint per completed
task-method pair. Reruns skip compatible completed records and reject records
created with different model, prompt budget, task catalog, or source revision.

## Metrics

Report per method and paired task:

- valid-plan rate;
- selected-GT coverage and complete-GT coverage;
- selected Skill count;
- prompt and completion tokens;
- rendered characters;
- omitted candidate count;
- group count for Hierarchy;
- node count, typed-edge count, explicit-reference count, namespace-edge count,
  and connected-component count for Graph.

Do not call these metrics task success, Pass@1, or verifier reward.

## Verifier boundary

SkillsBench verifier packages run outcome-oriented pytest assertions against
artifacts left in the task filesystem. They commonly check required file
existence and schema, domain constraints, cross-file consistency, feasibility,
and numerical or exact-answer correctness. `verifier/test.sh` records the
pytest result or partial fraction in `/logs/verifier/reward.txt`.

The current local `VerifierRegistry` only implements `exact` and
`contains_keys` for deterministic toy tasks, and the current runtime executes a
single selected tool once. It cannot establish completion of the 15 complex
SkillsBench tasks. Those components remain useful unit-test fixtures but must
not be used in Hard-15 result claims.

## Cleanup scope

Remove the generated toy and smoke artifacts currently under `reports/agent`:

- `flat_example.jsonl`
- `flat_smoke.jsonl`
- `organizer_comparison/`
- the failed all-missing contents currently under `hard15/`

Do not delete `reports/hard`, FCSR ranking files, raw datasets, reusable unit
tests, or source code merely because it supports deterministic test fixtures.
After cleanup, `reports/agent/hard15/` is reserved for the real fixed pilot.

## Minimal implementation boundary

- Add only `huggingface_hub` as a new runtime dependency for selective task
  synchronization.
- Keep task synchronization, organization, planning, and evaluation as small
  independent modules.
- Use one user-facing orchestration command for the planning experiment plus a
  separate optional synchronization command when network downloads need to be
  refreshed.
- All pure selection, filtering, graph construction, rendering, resume, and
  evaluation behavior is test-first and network-free in unit tests.

