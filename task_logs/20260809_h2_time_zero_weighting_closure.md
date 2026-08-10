# H2 time-zero and grace-period weighting closure

Date: 2026-08-09
Module: `benchmark实验`
Task: `FIG2-CANONICAL9-SCIENCE-PROTOCOL-V3`
Implementation commit: `598ea86`

## Review disposition

The GPT review approved E2 and H3 for human sign-off and approved H2's current fail-closed result. It identified one minor revision in H2's future target-trial coordinates: baseline adjustment must be anchored at ICU-admission time zero, while post-time-zero information during the 24-hour grace period must have an explicitly time-varying role in the adherence/censoring model. The estimation label must describe clone-censor-weight with stabilized inverse-probability-of-censoring weights, not generic stabilized IPTW.

This remains AI-assisted pre-review. It is not a clinical or methods attestation.

## Surgical change

- E2 protocol, card, packet, question, and materialization coordinates were not changed.
- H3 protocol, card, packet, question, and materialization coordinates were not changed.
- H2's current medication-capture decision remains `H2_VERIFIED_NON_USE_UNAVAILABLE`; no binary control arm, PSM, IPTW, or treatment-effect estimate is authorized.
- H2 future protocol was versioned from `20260809-v2` to `20260809-v3`.
- Baseline adjustment variables are accepted only when defined at or before ICU-admission time zero.
- Physiologic and treatment-history variables first observed after time zero may only enter a prespecified time-varying grace-period adherence or censoring model.
- The estimation method is fixed as `clone_censor_weight_with_stabilized_inverse_probability_censoring_weights`.
- The review packet now asks the human reviewer to approve these exact temporal and weighting coordinates.

H2 card content SHA-256 remains:
`5a2abb75b1404a26f01d8dd9afce409b05cc21371d9079523d5168cdc62f410f`

H2 v3 protocol content SHA-256:
`1d92f0de0fa4fd3a191e11f6baf0fca961d036ef9a17987ffafd3121777e281b`

## Verification

- Focused H2/protocol/authority/resource set: 144 passed, 0 failed.
- Canonical9 + KnowHow + resource baseline set: 551 passed, 50 skipped, 0 failed in 86.27 s.
- Ruff passed for all changed Python files.
- `git diff --check` passed.
- Negative tests reject: baseline defined relative to recorded initiation instead of time zero; generic `stabilized_iptw`; and post-time-zero information assigned to a baseline propensity model.

## Remaining gate

The three protocols are ready to be sent to real reviewers but still have `human_attestation_pending`. A clinical reviewer with ICU/critical-care expertise and a methods reviewer with clinical epidemiology, biostatistics, or causal-inference expertise must review and sign the exact card and protocol digests. Do not perform further AI-driven protocol optimization unless a human reviewer requests a concrete change.

After valid dual attestation, rebuild the final exact-SHA Docker image, typed materialization, ProductionInputAuthority, ScientificProtocolAuthority, operator declaration, ledger, and output root. Run a fresh aware-only E1 canary; if it passes, execute E1–E9 in three fixed rounds without result-driven system changes.

## Primary methods references

- TARGET Statement 2025: <https://www.bmj.com/content/390/bmj-2025-087179>; DOI `10.1136/bmj-2025-087179`.
- Target trial emulation overview: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10400102/>.
- Applied clone-censor-weight example with baseline and time-varying censoring predictors: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10947522/>.
