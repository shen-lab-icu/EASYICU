# Qualification12 MG04 bounded Planner canary

## Outcome

The single MG04 canary did **not** pass the literature-consumption gate. The
strict-schema Progressive Planner generated a valid outline checkpoint with
three scientifically distinct candidates, one selected design, two rejected
designs with reasons, and seven literature-design decisions. However, the
selected design cited generic method sources rather than either reviewed MG04
design-analogue card (`pmc9362765_mg04`, `pmc8116825_mg04`). No complete
`AnalysisPlan` was produced and Execute never started. This is neither a
Qualification12 result nor manuscript evidence.

## Exact coordinates

- Input literature pack HEAD: `becdaeee03eeda5a10a6c289e7f1a52e29d82131`
- Canary source HEAD: `ef429ec4c0e72be8ded22f0a7b0c647fe74a2202`
- Repaired, not-yet-Provider-reverified HEAD: `8909491ce0f41c429faf9d14d323517a61680634`
- Profile: `npj_dm_qualification12_design_canary_dev/20260825`
- Provider/model: `openai` / `gpt-5.6-luna`
- Task: MG04, exact reviewed question and two bound full-text cards
- Exact image: not built; the planner-only owner now skips execution-runtime
  preflight and cannot resume into Execute.

## Zero-Provider entry repairs

Before the first model call, the runner exposed three generic entry defects:

1. an EHRFlowBench row had no typed path for an exact-question bound literature
   bundle;
2. no additive planner-only Qualification12 canary profile existed;
3. planner-only runs incorrectly required the Docker execution image before
   planning.

These were fixed without MG04-specific logic in commits `d0f523f` and
`ef429ec`. Two launcher attempts also failed before transport because the
provider reservation floor exceeded the initial token ceiling and the declared
per-step repair capacity was underfunded. They incurred zero Provider calls and
zero cost.

## Provider calls and bounded stop

The non-strict transport path made two HTTP-200 Planner calls but both outline
responses violated the Progressive outline schema. The strict-schema path then
made four HTTP-200 Planner calls, successfully persisted outline and foundation
checkpoints, and stopped at the cumulative call budget during step
materialization.

| path | completed calls | prompt tokens | completion tokens | total tokens |
|---|---:|---:|---:|---:|
| non-strict diagnostic | 2 | 18,851 | 4,319 | 23,170 |
| strict canary continuation | 4 | 39,689 | 11,680 | 51,369 |
| cumulative | 6 | 58,540 | 15,999 | 74,539 |

At the runner's frozen conservative rates ($10/M input, $30/M output), the
cumulative estimate is `$1.065370`. Two additional attempts were denied before
transport by the hard-stop ledger and have no token or cost usage.

## Scientific audit

- candidate designs: `3` (required `2-4`)
- selected: `1`; rejected: `2`, both with explicit reasons
- selected seven-dimension decisions: `7/7`
- selected design cited reviewed MG04 cards: `false`
- seven-dimension decision keys limited to reviewed MG04 cards: `false`
- complete `analysis_plan.json`: absent
- Execute/Coder/Writer/result/figure activity: absent
- analysis authority: none
- manuscript authority: none

The structured audit is
`/Volumes/外置硬盘/easyicu_data/qualification12_planner_canary_audit_ef429ec_20260825/canary_audit.json`.
The strict checkpoint is under
`/Volumes/外置硬盘/easyicu_data/qualification12_planner_canary_ef429ec_strict_20260825/MG04/aware/run_20260825T121728_352e93/`.

## Generic owner repair after the canary

Commit `8909491` binds reviewed `LiteratureDesignEvidenceCard` objects directly
into the Progressive Planner prompt and its checkpoint authority. It also runs
`validate_selected_design_against_literature` immediately after outline parse,
before foundation or step materialization. A zero-Provider replay of the saved
bad checkpoint now stops with
`progressive_selected_design_comparator_not_bound`, before foundation.

Validation:

- literature/design/bench profile matrix: `82 passed`
- saved-failure replay: pass, expected early reason code
- focused pipeline/progressive checks: `4 passed`
- Ruff, `py_compile`, and `git diff --check`: pass

No task-, MG04-, beta-blocker-, atrial-fibrillation-, MIMIC-, or Sepsis-specific
branch was added.

## Next gate

Do not start Qualification12 execution, build an exact execution image, or open
Held-out27 yet. The next action is one new exact-HEAD strict Planner canary on
`8909491` (or its documentation-only descendant), with a fresh bounded ledger.
It passes only if the selected design cites the reviewed MG04 card(s), all seven
decisions cite only reviewed card keys, the 2-4 candidates remain genuinely
distinct, and the planner-only path stops without Execute. Because this canary
already consumed six Provider calls, no same-turn rerun was made after the
repair.

## Final checkpoint closure on 2026-08-25

The repaired canary is now closed on exact pushed HEAD
`ea744a8a1b4b3fb4080c2f04c16dd8b853c1f274`. The source checkpoint remained
`progressive_planner_checkpoint_007.json`, file SHA-256
`95e8466bdceeaab360fe330a6b0a3de7cc7893ba1e7fdab9a127513279c50664`.
The final resume completed both remaining progressive materializations on the
host, made zero Provider calls, and stopped at the governed human-review gate
in 1.2 seconds.

- final run: `/Volumes/外置硬盘/easyicu_data/qualification12_planner_canary_ea744a8_hostfinish_20260825/MG04/aware/run_20260825T133047_b853d9`
- progressive materializations: `8/8`; checkpoint sequence: `009`
- final shaped AnalysisPlan: 11 steps; no missing or failed plan steps
- candidates: 3; selected/rejected: 1/2; selected literature decisions: 7/7
- selected reviewed cards: `pmc9362765_mg04`, `pmc8116825_mg04`
- Execute/Coder/Writer/result activity: none
- exact image: not built

The deterministic association figure contract was revised from cohort/Table 1
context to the article-grade evidence hierarchy required by the association
figure strategy: absolute risk, primary adjusted estimate, robustness matrix,
and robustness ranges. Cohort accounting, Table 1, and measurement quality
remain separately planned. The synthetic renderer test exported source CSV,
PNG, SVG, PDF, TIFF, and FigureContract; this test image is not a study result.
The plan review now covers every required article role and six accepted chart
families. Focused validation ended at `223 passed`; the adjacent context,
hard-stop, and planner/bench matrices also passed in the preceding commits.

The canary is a planning-contract success but not a scientific-readiness pass.
`scientific_plan_review.json` scores 78 and remains `changes_required` with:

1. blocker `OUTCOME_DEFINITION_UNRESOLVED`;
2. `DIRECT_COMPARATOR_NOT_ESTABLISHED`;
3. `ADJUSTMENT_SET_NOT_USER_CONFIRMED`;
4. `CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED`;
5. `NOVELTY_NOT_ESTABLISHED`.

These are input/design-authority gaps, not reasons to add MG04-specific prompt
logic. Qualification12 execution, an exact execution image, and Held-out27
remain unopened until generic owner-issued endpoint, adjustment/time-role,
functional-form sensitivity, screened comparator, and novelty-positioning
contracts are present.

Across the successful billed Planner segments after the original repair, 13
completed calls reported 139,735 tokens at the frozen conservative estimate of
`$1.87409`. The final exact-head replay reported zero tokens and `$0`. Separate
timeout, HTTP 401, and pre-transport hard-stop attempts reported no usage and
are not included in the billed total.
