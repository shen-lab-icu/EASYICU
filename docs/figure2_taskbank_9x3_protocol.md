# Figure 2 ICU Research Taskbank: 9 Families × 3 Tasks

Status: design protocol only. No B/C task is reportable or held out until an
independent benchmark owner seals its exact authority manifest.

## Purpose

Figure 2 evaluates whether EasyICU can complete a broad set of common ICU EHR
research workflows. The current nine Canonical9 tasks are the development set:
they expose shared-engine defects and may legitimately drive architecture
repairs. They must not also be presented as an untouched generalization test.

The submission protocol therefore assigns three distinct tasks to each of nine
method families:

| Family | Capability under test |
|---|---|
| 1 | Cohort construction and prevalence/descriptive estimation |
| 2 | Continuous-exposure association analysis |
| 3 | Ordered or dose-response analysis |
| 4 | Missingness and robustness analysis |
| 5 | Prediction, discrimination, and calibration |
| 6 | Cross-sectional phenotyping or clustering |
| 7 | Time-to-event/survival analysis |
| 8 | Causal-effect estimation under a locked estimand |
| 9 | Longitudinal trajectory phenotyping or clustering |

The two clustering families remain separate: one uses a cross-sectional feature
space, while the other uses longitudinal trajectories and therefore tests
different data, alignment, and validation contracts.

## A/B/C roles

| Role | Use | May change the shared engine? | Paper interpretation |
|---|---|---|---|
| A — development | One current Canonical9 task per family | Yes, for a general defect with a counterexample and regression test | Architecture-development coverage only |
| B — frozen validation | A scientifically distinct task per family, sealed before execution | No. If it drives an engine change, it is reclassified as development and replaced by B-prime | Frozen validation |
| C — sealed confirmation | A second distinct task per family, kept unopened while A/B work proceeds | No. An engine change after opening C invalidates the confirmation claim and requires a new held-out task | Final confirmation |

A, B, and C must change more than wording. Across each family they should vary at
least two material axes, such as database, exposure/outcome concept, cohort,
time origin/window, estimand, or missingness structure, while retaining the same
method-family capability being tested.

“Positive task” means estimable and supported by the available data, not a task
selected because it is known to yield a statistically significant or favorable
result. Outcome-direction screening is prohibited.

## Independence and sealing

Engine developers may define the family template and feasibility constraints,
but may not know the final B/C questions while changing the engine. Examples
seen during design are candidates, not held-out tasks.

Before any B task is opened, an independent benchmark owner seals:

- exact question and task ID;
- input/export authority SHA and database;
- cohort, exposure, outcome, time origin/window, and estimand;
- permitted method family and required output contracts;
- known hazards, forbidden claims, and deterministic scoring oracle;
- model, prompt pack, provider/retry budget, submission profile, and engine SHA;
- evaluator/rubric version and manifest SHA.

The repository may commit the commitment digest before execution. The concealed
B/C manifest itself remains outside the shared engine workspace until the
corresponding task is opened, then is archived for reproducibility.

All provider attempts count, including transport retries, patch attempts, full
rewrites, audits, and failed calls. A safe fail-closed result is retained as an
experimental result; it is not silently converted into success.

## Invalidation rules

1. A is always development data.
2. If B prompts a shared-engine or scorer-semantic change, B becomes development
   and a pre-sealed replacement B-prime is required. An unopened C remains valid.
3. If C prompts such a change, the confirmation claim is invalidated and a new
   sealed confirmation set is required.
4. Case-specific requirements belong in the task manifest or rubric, never in
   shared prompts, routing predicates, validators, or deterministic runners.
5. A change to scoring semantics requires an additive evaluator version. Existing
   Figure 2 v1/v2 scorers and their authority digests remain immutable.

## Stability is a separate experiment

The 9 × 3 taskbank tests task generalization. Re-running the same question tests
stochastic reproducibility and must be reported separately. At minimum, repeat
three sentinel tasks (easy, medium, and hard) three times with identical locked
coordinates and report agreement, failure-mode consistency, provider calls,
tokens, and wall time.

## Efficient execution order

1. Finish the nine A tasks and the bounded shared-engine architecture work.
2. Have the independent owner seal all B/C authorities before A development ends.
3. Freeze one engine/profile/dictionary/model/prompt/evaluator coordinate.
4. Run three B sentinels first. If the frozen engine is operationally sound, run
   the remaining B tasks without engine modification.
5. Open and run all C tasks only after B is closed.
6. Run repeated-task stability as a separate lane.

This sequence gives early feedback without consuming the final confirmation set
or turning all 27 questions into architecture-tuning cases.

## Figure and reporting contract

- The current exact 9 × 5 Figure 2 evaluator v2 remains immutable.
- A future 9 × 3 analysis is an additive v3/taskbank protocol, not a rewrite of
  v1/v2.
- The main figure should show the nine family map, the frozen B/C 9 × 5 result
  heatmap or consistency strip, and clear NA cells. The full 27 × 5 matrix belongs
  in Extended Data/source data.
- Development A and frozen B/C results are never pooled into one success rate.
- Do not average heterogeneous dimensions into a single headline score; retain
  the five dimension-level outcomes and their applicability.

## Current status (2026-07-19)

- The nine A families are defined, but all current Canonical9 typed input
  bindings remain blocked and must be re-materialized after the architecture
  freeze.
- No exact B/C task manifest has been independently sealed; therefore there is
  currently no valid claim of 18 held-out tasks.
- The previously discussed Extension3 paths are not a machine-readable sealed
  taskbank and must not be reported as completed validation.
- Architecture work remains frozen ahead of new experiments; this protocol does
  not authorize case-specific shared-engine changes.
