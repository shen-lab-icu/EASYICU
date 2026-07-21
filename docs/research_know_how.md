# Research Know-How: bounded, evidence-bound protocol retrieval

EasyICU has an opt-in, offline retrieval layer for source-backed ICU research
design candidates. It does not grant cohort, exclusion, time-zero, estimand, or
method authority. The Planner must record the exact claims it adopted, rejected,
left unresolved, or returned for user confirmation.

## Why this differs from a generic RAG/agent library

EasyICU borrows the useful part of systems such as Biomni: retrieve only the
resources relevant to the current task. It deliberately does not copy a broad
LLM selector or inject arbitrary full-text instructions. The host first closes
the permitted analysis family and trust boundary; deterministic topic matching
then selects cards from that set.

Missing data does not hide a relevant card. Retrieval records two separate
coordinates:

- `topic_applicable`: the protocol topic and analysis family match the task;
- `data_readiness`: `ready`, `partial`, or `not_ready`, with exact unresolved
  concepts.

Thus an AKI prediction card remains visible when urine output is unavailable,
but tells the Planner that a urine-output definition is not implementable.

## Card v2 contract

Every design, stop, and confirmation item has a stable `claim_id`, exact text,
field, evidence scope, and one or more `citation_ids`. The schema rejects a card
when any advice lacks a claim or when a claim cites an unknown source.

Trust and scientific review are separate:

- `built_in_reviewed` / `project_reviewed`: prompt-safety provenance accepted by
  default retrieval;
- `user_supplied_unreviewed`: never enters the canonical Planner prompt;
- `curated_mvp`: structured and source-linked, but not yet dual expert reviewed;
- `clinical_reviewed`: requires clinical and methods review, reviewer/date/scope,
  literature cutoff, card version, and a content digest. Editing the card makes
  the attestation invalid.

The nine bundled cards remain honestly labeled `curated_mvp`. They must not be
described as expert consensus until the review protocol is completed.

## Prompt projection and budget

Planner receives compact canonical JSON labeled as advisory data. Projection
always retains stop conditions, confirmation requirements, data readiness, and
claim-to-citation coordinates. Optional claims are included only as complete
objects; strings are never cut mid-field. If mandatory content or the total
projection exceeds its budget, the run fails closed and must reduce `top_k`.

The runtime also measures the complete initial Planner request:

- system, user, and total bytes;
- approximate input tokens;
- exact Know-How-added bytes after all other deterministic planning scaffolds;
- selected-card count and the 80,000-byte hard limit.

These metrics are registered as `planner_prompt_metrics.json`. The Know-How
projection remains capped at 8,000 characters; observed one-card projections are
about 4 KB.

## Runtime evidence and plan authority

- `know_how_retrieval.json`: query, topic match, data readiness, versions, SHA,
  citations, trust policy, and selected hits;
- `know_how_prompt.md`: exact structured Planner projection;
- `planner_prompt_metrics.json`: full request budget evidence;
- `analysis_plan.json`: claim-level `know_how_decisions`; adopted cards are
  derived from decisions whose disposition is `adopted`, so there is no second
  card-id list that can drift.

Each decision repeats the exact card version/SHA, claim ID, citation IDs,
disposition, reason code, and short rationale. The Planner cannot cite an
unretrieved claim or change its citations. Replanner and resume preserve the
decision list exactly; the decisions are part of plan scientific scope.

## Submission profiles

Historical profiles remain byte-identical and keep Know-How off. Opt-in is a
study-design change and cannot be combined with an old profile. The additive
development profile is:

```text
npj_dm_know_how_dev/20260721
```

It is not the paper default. A later paper profile may be created only after
expert review and the repeated A/B acceptance described below.

## Acceptance before online experiments

1. Canonical9 A offline retrieval matrix: correct card or honest no-card, plus
   Chinese/English aliases and adversarial negatives.
2. Clinical and methods review of every claim and citation; produce valid
   content-digest attestations.
3. Freeze an E2 blind rubric before comparison.
4. Run Know-How off/on under identical data, model, prompt pack, and provider
   coordinates, at least 2–3 runs per arm (or first use fixed recorded Planner
   responses for component testing).
5. Compare unsupported exclusions, incorrect-card adoption, time-zero/estimand
   quality, requests for confirmation, retries, calls, tokens, and wall time.
6. Freeze retrieval rules, then use sealed B/C tasks to test generalization.

The basic E2 peak-lactate question retrieves
`early_peak_lactate_association`. The separate
`lactate_trajectory_outcome` card is reserved for questions that explicitly
request clearance or longitudinal trajectories; a generic lactate mention is
not enough to select it. The evidence pre-review packet is
`docs/reviews/early_peak_lactate_association_20260721.json`; it is digest-bound
to the card but explicitly carries `authorization=false` until dual review.

The bounded comparison command is `tools/run_research_know_how_planner_ab.py`.
It runs only Planner, alternates OFF/ON within each repeat pair, writes plans
under opaque labels, and stores the arm key separately. It defaults to two
runs per arm and refuses online use of `curated_mvp` cards unless the operator
explicitly records the development-only override. A prepare-only run against
the fixed full-parent E2 context measured 65,423 bytes without Know-How and
69,932 bytes with the one selected card (+4,509 bytes), both below the 80KB
hard gate.

No new extraction is required: experiments consume the existing six-database
export at `/Volumes/外置硬盘/easyicu_data/full6_20260717`.
