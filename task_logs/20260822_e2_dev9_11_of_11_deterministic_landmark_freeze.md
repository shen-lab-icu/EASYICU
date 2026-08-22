# E2 Dev9 11/11 deterministic landmark acceptance

Date: 2026-08-22

## Outcome

- Dev9 item `e2_lactate_mortality` completed all 11 required execution steps.
- Persisted completion axes: `required_step_count=11`, `completed_step_count=11`, `missing_steps=[]`, `failed_steps=[]`, `execution_complete=true`, and `execution_ok=true`.
- This is a development diagnostic only. The persisted run remains `status=diagnostic_only`, `forced_diagnostic_only=true`, `artifact_valid=false`, `scientific_requirement_complete=false`, `paper_authorized=false`, and `manuscript_ready=false`.
- No E2, sepsis, lactate, database, expected effect, or manuscript-specific rule was added to a shared prompt or validator. Repairs were assigned to generic lifecycle, runtime-authority, typed-input, robustness, and figure owners.

## Exact execution coordinates

- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_b09b1e8_e2_resume_20260822/e2_lactate_mortality/aware/run_20260822T091111_ff4bd9`
- Input package: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_cca658b_e2_input_20260822`
- Input JSONL SHA-256: `47c23dcb933f04e16d1375d046c3b69c9fc6a5408f75952753a6aeed2c1d0bd3`
- Execution completion HEAD: `eda832d79783054ff442b88c3c442d23944b5c54`
- Execution image: `sha256:7bdb0f636c325e38348a69cb462209e5c212831ab8e625063790a45b2c9a737c`
- The same Planner checkpoint was reused; resumes after the diagnostic replanner failure used replanning disabled. E2 was not restarted from outline and no full Provider canary was repeated.

## Deterministic execution closure

- The signed landmark-spline runtime authority owns the primary association and linear sensitivity execution.
- The landmark robustness owner projects the digest-bound contrast and linear-sensitivity tables into the declared robustness products without fitting another model.
- The landmark association figure owner renders the four exact source tables as PNG, SVG, PDF, and TIFF with a figure contract and source-row/table provenance.
- The generic robustness renderer produced its own exact source-data bundle and figure contract.
- The final display suite and the cohort/data-quality figures are present under the run's `steps/` directory.

## Result-boundary sanity checks

- Upper-bound contrast: OR `1.9600187955893984`, 95% CI `1.8916855238397148–2.030820467064707`.
- Linear sensitivity: OR `1.2592353780907504`, 95% CI `1.2459595657332223–1.2726526454349405`.
- Complete-case sample size: `44,095`.
- The upper-bound contrast is not a scalar summary of the nonlinear curve, and the complete-case row is not an independent replication. These values are development diagnostics, not manuscript claims.

## Provider accounting

- Successful Provider calls: `9`.
- Tokens reported by successful calls: `204,633`.
- Actual cost reported for successful calls: `$2.66061`.
- Durable hard-stop attempts: `15`.
- Conservative accounted upper bound: `1,330,833` tokens and `$29.28261`.
- Six HTTP 401 calls had unknown actual usage: two Planner, one Coder, and three Writer calls. Their reservations remain included in the conservative upper bound. The final Writer failures were settled by deterministic/system fallback; they are not hidden as successful Provider calls.

## Frozen code and CI

- Final core freeze HEAD: `ffc2335db56ee6c4114ca48dfb37e7c754cafeaf`.
- Final exact-head image: `sha256:97fd943fc85320f9fa3beec22cf4b201f20fe7cd68a2405d4a47dc819086e0e4` (`easyicu-research-agent:ffc2335`).
- Research Agent CI: [run 32568880095](https://github.com/shen-lab-icu/EASYICU/actions/runs/32568880095), success at exact `ffc2335`.
- Full CI: [run 32568880081](https://github.com/shen-lab-icu/EASYICU/actions/runs/32568880081), success at exact `ffc2335`.
- The post-execution commits only ratcheted the reviewed architecture measurement and replaced duplicate local finite-number helpers with the shared numeric-scalar contract. They did not alter the completed E2 plan, estimand, source data, or reported values; E2 was therefore not rerun. The distinction between execution HEAD and final freeze HEAD is retained explicitly.

## Decision

E2 is accepted as a completed Dev9 development diagnostic and the common contracts are frozen for progression to E3. It is not a paper-ready analysis, not expert-scored, and not part of Qualification12 or Held-out27. Expert clinical/methodological scoring remains deferred until the later formal evaluation stage, as authorized by the user.
