# E1 shared-case bias remediation

Date: 2026-08-14  
Branch: `fix/pi-workspace-review-20260809`  
Implementation commit: `0d59ad2`

## Decision

The main `ResearchAgentPipeline` did not dispatch on the E1 case identifier,
but four shared/runtime surfaces contained E1 or Sepsis/SOFA-specific bias.
The in-flight development E1 job `6a47f1d34f4b` was cancelled before analysis
and is not evidence.

## Remediation

- Replaced the real E1 denominator `94,458` in the global Writer prompt with a
  schematic, explicitly non-binding two-owner example.
- Made `research_agent.literature_concepts` the single typed owner for
  literature identities and phrases. Web Idea Mining and direct evidence
  search now consume that owner. Lactate and AKI exercise the same path in a
  non-E1 regression.
- Removed SOFA-2/Sepsis-3 priority and default rows from the legacy Web Agent
  output builder. Predictor choice is now deterministic and coverage-based,
  excluding the target `death` column.
- Corrected the legacy label for `sep3_sofa2_max` from standard `Sepsis-3` to
  `Experimental SOFA-2 Sepsis-3 phenotype`.

Clinical concept support remains available as typed metadata. No benchmark
answer, denominator, result, or case-specific routing was added to shared
prompts or execution logic.

## Verification

- 92 focused literature, PubMed, Writer, legacy-output, Web integration, and
  package-dependency tests passed.
- Ruff passed for all touched Python files.
- `git diff --check` passed.
- `tools/arch_measure.py --diff tools/arch_baselines/execution_phase.json`
  reported no lower-is-better regression; the baseline was not refreshed.
- No full exact-head CI was run because this remains E1/Web development.

## Next gate

Restart Web on an exact head containing `0d59ad2`, then run a fresh ordinary
E1 conversation. Do not reuse the cancelled job, old approval, or old run
budget. After E1, run a non-Canonical age/mortality question to verify that the
same natural-language pipeline is not case-tuned.
