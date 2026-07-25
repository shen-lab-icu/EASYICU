# Research Know-How Planner A/B acceptance protocol

This protocol is frozen before any online comparison. It evaluates whether
reviewed protocol claims improve planning without adding unsupported scientific
decisions or wasted provider calls.

## Coordinates held identical

- same research question and fixed six-database export;
- same typed `ResearchContext`, data dictionary, model, provider, seed policy,
  prompt pack, and execution profile coordinates;
- arm OFF uses the current paper profile; arm ON uses the additive Know-How
  development profile and changes no other option;
- no Coder, Docker, or downstream execution during the first Planner-only pass.

Use at least three independent Planner runs per arm. A deterministic component
test with fixed recorded Planner responses may run first, but it does not replace
the repeated provider comparison.

## Prespecified blinded rubric

Strip arm labels and randomize plan order before two reviewers score:

1. topic/card relevance and incorrect-card adoption;
2. unsupported eligibility exclusions;
3. explicit time zero, observation window, follow-up, estimand, and outcome;
4. whether unavailable data are identified rather than silently assumed;
5. whether confirmation-required decisions are returned to the user;
6. consistency between each adopted decision and its exact claim/citations;
7. feasibility and stop conditions before Coder use;
8. plan completeness without unnecessary steps.

Record disagreements and adjudication without revealing the arm.

## Operational metrics

For each run record selected cards, every decision disposition, Planner retries,
provider calls, input/output tokens, full prompt bytes, Know-How-added bytes,
active wall, parse failures, and unsupported-claim count. Report per-arm median
and range; do not select the best run.

## Progression rule

Start with E2 only after the lactate card is clinically and methodologically
reviewed. Enable no online card merely because the retrieval matrix is green.
After the 9 A tasks help develop/freeze retrieval and rubric behavior, perform
the same locked evaluation on untouched B/C tasks. A paper-facing profile is
additive and may be created only after these gates pass.
