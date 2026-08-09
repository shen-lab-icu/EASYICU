# 2026-08-07 scientific model contract consolidation

## Scope and authority

- Task: `AGENT-SCIENTIFIC-MODEL-CONTRACT-CONSOLIDATION-20260807`
- Branch: `fix/external-review-20260724-p0-p1`
- Exact starting commit: `eadfec0d0a2cb7256af3b035f3f1f2abc1d788e0`
- Reviewed code commits:
  - `ebb83939eeb687e0e7742c7122c95b4e0ac30af2` — consolidate the scientific model contracts
  - `28f7d959577d1d0bb2f2b68be191a8ffc0d74cfc` — re-adjudicate the framework characterization golden

This batch closes the eight implementation findings and two concrete P2
findings in the review of `eadfec0`. It is an offline code-and-contract batch:
no Provider, patient data, Canonical9, M3, image build, or push was used.

## Finding-to-fix map

| Review finding | Owner contract and closure |
| --- | --- |
| Deterministic adjusted association and free-form association shared one drifting capability entry | Split the registry into `association_adjusted_v1` (deterministic) and `association_freeform_v1` (LLM-coded); selection now uses the accepted plan rather than question text alone. |
| The declared association family could disagree with the executed estimator | Added exact model tokens and exact ownership: adjusted binary outcomes use Logit and continuous outcomes use OLS. GLM and quantile requests are explicitly declined instead of being silently substituted. |
| Covariate encoding was inferred from pandas dtype/cardinality | Added frozen, extra-forbid `ModelTermSpec` and one shared model-matrix compiler. Continuous, binary, categorical, and ordinal-linear coding are explicit; levels, reference, and transforms are sealed and the compiled roster must match exactly. |
| Survival ownership matched broad aliases instead of an exact estimator/diagnostic pair | Survival ownership now requires exact Cox plus exact global Schoenfeld tokens and rejects unsupported combinations before execution. |
| Survival time units could be inferred from Planner prose | Added the dependency-neutral time-unit contract; the executor accepts only the canonical unit bound by host metadata and reconciles it with the plan and receipt. |
| PH handling was not a complete declared policy/status/paper gate | The plan and receipt bind PH alpha, diagnostic, status, and policy. The publication gate reconciles all four and fails closed on a violation or unresolved policy. |
| Semantic repair could change the scientific model design | Added the semantic-repair boundary. Method, estimator, formula, term roster, coding, reference, transform, endpoint, censoring, time unit, PH policy, and related design mutations are rejected with `scientific_design_change_requires_replan`; the original code is retained and the attempt is recorded. |
| Architecture/resource/release gates did not represent the new contract surface | Re-adjudicated the architecture and resource baselines with explicit append-only reasons, removed duplicated prompt material, added guarded emit workflows, and ran the clean no-Provider framework release command. |
| Final validation broadly demoted `step_contract` findings | Removed the broad demotion. A contract failure remains attributable and fail-closed. |
| Nested scientific Planner keys could disappear as generic validation errors | Unknown scientific-contract keys now receive a structured retry with stable code `planner_scientific_contract_unknown_key`. |

## Boundary details

The new dependency-neutral public contracts are:

- `contracts/model_terms.py`: typed term semantics and exact roster checks.
- `contracts/model_tokens.py`: exact estimator and diagnostic vocabulary.
- `contracts/time_units.py`: host-bound canonical time units.
- `execution/model_matrix.py`: the only association/survival term compiler.
- `repairs/semantic_boundary.py`: immutable scientific-design comparison and
  typed replan escalation.

The adjusted-association and survival executors consume those contracts rather
than maintaining local dtype guesses or alias sets. Capability selection,
execution receipts, result gates, readiness, and repair coordination now use
the same vocabulary.

## Characterization-golden adjudication

The first clean release run on `ebb8393` failed only
`test_char_golden_run_bundle.py`. The expected bundle was inspected rather than
blindly regenerated. Its two changes are intentional consequences of the
hardened contract:

1. a locked but unexecuted robustness specification remains an `ERROR`;
2. typed product-contract digests reflect the current public contract.

Commit `28f7d95` records that adjudication. The focused golden test and the
subsequent clean release both pass. No Figure 2 scorer or frozen scientific
authority digest was refreshed.

## Verification

The following selections overlap and must not be added together:

| Check | Result |
| --- | --- |
| New-regression replay plus adjacent parameterizations | `158 passed` |
| Broad affected research-agent selection | `687 passed` |
| Changed pipeline/resume owners | `87 passed` |
| Repair boundary selection | `112 passed` |
| Capability/governance selection | `75 passed` |
| Module-graph contract tests | `11 passed` |
| Ruff over `src/easyicu`, `tests`, and `tools` | passed |
| `compileall` | passed |
| Deptry over `src/easyicu` | no dependency issues |
| Import-linter | 7 kept, 0 broken |
| Architecture lower-is-better diff | passed |
| Resource-context diff | passed |
| Module-graph diff | passed; acyclic |
| Wheel and sdist build | passed (`/tmp/easyicu-build.81O0ev`) |
| Clean framework release at `28f7d95` | passed; 133 framework tests |

The clean release report is `/tmp/easyicu-release-28f7d95.json`, SHA-256
`bf4db6d3ba32f202d93ff32b27d647206090906662cb62fdd9f6ed313d66667e`.
It records a clean Git commit and four green gates: resource context,
architecture, module graph, and framework tests. Its static command allowlist
prohibits Provider and patient-data access.

Resource measurements remained within the adjudicated ceilings: maximum
Planner context 61,345 / 120,000 characters (56,703 without the largest
resource), maximum single resource 4,657 characters, and Coder context 2,255
characters. Provider calls were zero.

## Exact-start differential and remaining reds

A broad suite run during implementation produced `12,236 passed, 126 skipped,
159 failed`. This is not reported as a green monorepo suite. Of those failures,
108 belong to the already-frozen Figure 2 scorer/digest authority. For the 51
remaining node IDs, the exact starting commit `eadfec0` reproduced 28 failures;
23 passed at the start and therefore represented regressions from this batch.
All 23 current regressions were repaired and replayed green.

The remaining exact-start failures are historical or environment-bound (for
example Docker `/var/folders` mounts, provider DNS, host-scaffold identity,
paper readiness/replication fixtures, Arrow typing, release-archive state, and
the frozen Figure 2 authority). They were not hidden by weakening gates or
refreshing unrelated baselines. Their independent adjudication remains outside
this scientific-model-contract batch.

## Handoff

The reviewed code findings are closed and the scoped no-Provider release gate
is green. This does **not** authorize a paper claim or a formal M3 run.
Canonical9 remains 4/9 and M3 remains unrun. Before any fresh aware-only M3,
the frozen Figure 2 authority/current-SHA CI boundary must be independently
adjudicated, the exact SHA must be pushed and reviewed, and a new immutable
image and fresh ledger must be created. Historical verify ledgers must not be
resumed as evidence for this code.
