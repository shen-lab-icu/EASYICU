# E1 12/12 development execution closure

Date: 2026-08-20  
Task: `FIG2-DEV9-HELDOUT27` / E1  
Repository head after repair: `65da00f3a040d7e6d11d1da53b8c8f9ce3af4191`

## Decision

E1 has crossed the **development architecture usability** gate: a dependency-bound
terminal Planner checkpoint replay completed all 12 required analysis steps, with
all 12 step-level scientific requirements satisfied. This is enough to stop using
repeated full Provider canaries as the first diagnostic surface.

It has **not** crossed formal, publication, or top-journal readiness. The run is
correctly forced to `diagnostic_only`; paper authority remains false.

## Best development evidence

- Run:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_e1_e4d4842_checkpoint_replay_20260820/e1_sepsis3_prevalence_mortality/aware/run_20260820T091946_88edab`
- Replayed terminal Planner checkpoint:
  `progressive_planner_checkpoint_010.json`, SHA-256
  `e37eb8e85404a0e111aeef0de2f3c5058856171e11f6aee7936da073a64da310`.
- `run_status.json.gates`: `execution_complete=true`,
  `step_scientific_requirements_complete=true`, `12/12` complete, no missing or
  failed step.
- All planned analysis/report/figure steps settled through deterministic owners;
  the robustness step used the trusted deterministic fallback. The checkpoint
  replay made no new Planner calls and the 12 execution steps required no Coder
  Provider calls.
- `article_figure_strategy_complete=true`; the numeric result figure remains a
  code-backed output with registered source data.
- Development image:
  `easyicu-research-agent:e4d4842-devpatch`, digest
  `sha256:9b596a2861f1e4b7954e787c9e60c40f318ea60b523c6c8fff4cc72feb5ba4f0`.
  This is a development hotpatch image, not a reproducible formal image.
- Exact code-head image after the build-path repair:
  `easyicu-research-agent:a8fda65`, digest
  `sha256:0ecfce04652de83118fbd0956188bca811d3cf1e6a6dd962089a98fd0b5db789`.
  `tools/check_agent_runtime.py` reports `status=ready`, `network=none`, 291
  kernel files, and all required method capabilities. This image has not yet
  been used for the one fresh E1 canary.

## Architecture defects closed

| Commit | Owner-level closure |
|---|---|
| `dcd057d` | Development runtime lineage permits explicit image/kernel source transitions while keeping dependency and network authority strict. |
| `4b58b72` | Explicit revalidation preserves the original producing attempt/checkpoint; ordinary stale attempts still fail closed. |
| `663f57b` | Robustness figures use the canonical `sensitivity_forest` vocabulary. |
| `b7209c9` | Signed runtime selection can authorize the typed association model-grid owner; an unsigned Coder output cannot. |
| `1cf3ee0` | Progressive article reports consume sealed typed products, not raw cohort inputs. |
| `e4d4842` | Terminal feasibility reports remain non-analytic and ignore legacy raw-column inputs rather than mounting them. |
| `65da00f` | Planner-scoped binary labels reach adjusted and distribution publication panels without hard-coding a benchmark case. |
| `a8fda65` | Runner package installation reuses build tools from the digest-pinned layer (`--no-build-isolation`) instead of making a second unpinned PyPI request. |

The repeated E1 failures were therefore not failures of logistic regression or
plotting mathematics. They were failures of cross-layer authority, typed input
closure, resume lineage, deterministic output ownership, and final presentation
lineage. The repair strategy kept each rule in one owner and added focused
negative/contract tests instead of adding case-specific prompt text.

## Anti-homogenization finding and closure

The Planner had already declared:

- `sep3_sofa2_max=0` -> `Sepsis-3 absent`
- `sep3_sofa2_max=1` -> `Sepsis-3 present`

The publication normalizer discarded the variable scope and fell back to
`Unexposed` / `Exposed`; the adjusted forest row fell back to `Sep3 Sofa2 Max`.
`65da00f` adds one case-neutral scoped-label owner. A label is used only when a
complete pair belongs to the exact same Planner variable; incomplete or
cross-variable labels do not get inferred. The publication integration test and
SVG overlap audit now verify `Sepsis-3 present vs Sepsis-3 absent`,
`Sepsis-3 status`, and the two level labels.

This demonstrates the intended architecture: implementations and evidence
contracts are standardized, while estimands, contrasts, labels, analysis choices,
and figure narratives remain study-specific Planner authority.

## Validation

- Publication/strata owner and integration suite: `60 passed`.
- Focused runtime-lineage tests: `6 passed`.
- Result-envelope/revalidation tests: `14 passed`.
- Robustness figure vocabulary suite: `14 passed`.
- Signed association model-grid owner tests: `4 passed` plus real product-contract smoke.
- Feasibility terminal-report tests: `7 passed`.
- Progressive sealed-report compiler test: `1 passed`.
- Visual verification used a zero-Provider synthetic E1 publication bundle; PNG,
  SVG and overlap audit passed.
- Runner-image contract test: `6 passed`; exact image build succeeded, offline
  import smoke returned `easyicu 1.0.0 imports_ok`, and runtime capability check
  returned `status=ready`.

Focused suites are development evidence. A full exact-head CI has not been run at
`65da00f` and is deliberately deferred until the E1 clean canary freezes the head.

## Why the run is not paper-ready

`run_status.json` keeps `artifact_valid=false`, `evidence_complete=false`,
`numeric_verified=false`, `analysis_validated=false`, `manuscript_ready=false`,
and `paper_authorized=false`. Scientific maturity is `analysis_only` (52).

The post-step Writer encountered repeated upstream HTTP 500/EOF failures. Separate
from that transport problem, the scientific audit correctly keeps blockers for
current direct literature/novelty authority, post-baseline exposure timing,
repeated-stay dependence, exact adjustment-set confirmation, genuinely distinct
robustness axes, exact manuscript citation binding, complete manuscript sections,
and PDF render QA.

After the exact image became ready, one two-token health probe against the bound
OpenAI-compatible Provider used `max_retries=0` and a 35-second timeout. It ended
in `APITimeoutError` with no HTTP status. The fresh E1 canary was therefore not
launched; this is an external transport blocker, not a failed analysis step.

## Fastest next acceptance path

1. Run one fresh, single-image E1 canary with the verified `a8fda65` image. Accept
   only 12/12 step completion, zero
   unowned outputs, correct Planner labels, and explicit `diagnostic_only` paper
   ceiling while scientific gates remain open.
2. Freeze the head and run one full exact-head CI. Do not run full CI between small
   fixes.
3. Then close the article-grade study decisions and Writer/literature gates. Add a
   cross-case diversity benchmark that records candidate designs, selected and
   rejected analyses, and what every figure supports/cannot prove.
4. Only after the development freeze proceed to E2, Qualification12, and eventually
   the atomic Held-out27 formal batch.
