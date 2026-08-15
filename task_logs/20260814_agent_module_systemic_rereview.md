# Research-agent systemic rereview after integrity closure

Date: 2026-08-14
Base HEAD: `61cfcd27e18eb3347f97550127d8876d7ba1f16f`
Scope: code/recovery/security closure plus bounded Provider verification. No
Provider batch and no patient-data read.

## Verdict

- Planner-only E1 development canary on this host: **READY AFTER the current
  closure is committed and receives a clean exact-head release**. The canary
  profile is intrinsically Planner-only and cannot resume into Execute.
- Formal E1 execution/publication run: **NO-GO**.
- The earlier clean `61cfcd2` release remains valid for that exact commit, but
  this rereview produced a new uncommitted closure and therefore requires a new
  clean exact-head release before any later launch decision.

## Systemic defects closed in this worktree

- Provider hard-stop read/modify/write transactions now reload under a
  cross-process sidecar lock; independently loaded workers cannot overwrite
  each other's call, task, token, cost, or terminal accounting.
- Review pause/resume owns the run lock before changing Provider state. Normal
  pause, checkpoint-before-pause crash, and resume-before-decision-commit crash
  converge to one exact checkpoint time anchor.
- Only typed progress control signals propagate cancellation; ordinary UI
  observer failures remain nonfatal.
- Web is fixed to `planner_canary`, requires Pi-verified credentials, ignores a
  client project root, rejects raw/unmanifested data, binds exact prepared
  manifest and declared-file bytes, and revalidates that package before
  acquisition, recovery, and review.
- Web no longer changes Provider pause/resume state outside the pipeline run
  lock. Endpoint identity is persisted as a non-secret fingerprint.
- Planner canary cannot be promoted to Execute through either the route or the
  bridge. Formal mode now has no accidental public activation path.
- Review decisions use a server-owned identity source rather than a client
  reviewer string. A fully attributable authenticated human identity remains a
  formal-run blocker below.
- Approved execution reloads the persisted approved plan, restores the complete
  approved handoff after callbacks, and revalidates review-bound evidence bytes
  after restart. Writer cannot consume a plan mutated by Execute.
- STRICT manuscript policy rejects unrelated evidence laundering, malformed
  claim/evidence tokens, metadata-prefix laundering, and common unsupported
  qualitative result phrasings, including result-bearing Markdown headings.
- Capability inventory no longer claims a fake acquisition/Pipeline test proves
  production reachability; acquisition remains experimental until a real
  bounded canary or non-fake integration traverses the boundary.

## Remaining launch blockers

1. The current worktree still needs a scoped commit, clean exact-head release,
   and an image rebuilt from that exact commit before the canary launches.
2. The data-foundation publication LOCK remains `finalized=false`; formal E1
   must use a freshly sealed native-v2 package, not merely any valid manifest.
3. Web dollar ceilings still use fixed estimated `$10/$30 per million` rates
   without a reviewed model-price upper bound. Token/call ceilings are hard;
   the dollar figure is not yet invoice-authoritative.
4. `easyicu_local_web_operator` is server-owned but not an authenticated human
   identity. Formal approval needs a host/session identity that identifies the
   reviewer rather than only the local service.
5. Formal execution is intentionally unavailable. Before adding a server-owned
   activation receipt, runtime replanning must be disabled for the exact-plan
   E1 path or converted into a durable pre-execution review; current drift
   detection prevents Writer publication but can occur after Execute starts.

## Structural debt, not a Planner-canary bypass

- `pipeline.py` remains 8,514 lines and `run_execute_phase` 4,987 lines. The
  architecture ratchet is green, but Pipeline still projects many config values
  into private state consumed through broad host protocols.
- Publication renderer ownership remains split between `pipeline.py` and
  `reporting/publication_bundles.py` for the association family.
- Runtime substantive replan advertises a review pause but is not yet the same
  durable public pause/resume lifecycle as initial Plan review.
- The Classic Agent “full Provider” control still targets the legacy summary
  engine; governed E1 uses Guided Pi. This should be relabelled or removed in a
  separate frontend-owner patch rather than wired accidentally to formal run.
- Provenance token constants and finite-number helper semantics remain
  duplicated. Existing audits prevent growth but do not yet provide one owner.

## Verification

- Provider/lifecycle/run-lock integration: `76 passed`.
- Evidence/plan/Writer/security integration: `224 passed, 1 skipped`.
- Web route/recovery integration: `123 passed`.
- Dirty-worktree framework release internal gates: resource, architecture, and
  module graph green; `137 passed`. Overall status is correctly `failed` only
  because the worktree is dirty.
- Capability inventory audit: OK.
- Repository hygiene audit: OK.
- Strict progress lint: 6 pages, 0 warnings.
- Ruff and `git diff --check`: green.
- Additive `npj_dm_e1_canary_dev/20260814` binds the current concept and SOFA-2
  dictionary bytes, disables memory/fallbacks, and emits the hashed
  `planner_only=true` Pipeline authority. Direct Pipeline approval is rejected;
  an explicit rejection may terminalize the checkpoint.
- Web package validation now uses the canonical `open_export_package()` intake
  and binds its full authority digest, including column-metadata and feature
  sidecars. Conversion manifests that the Pipeline cannot consume are rejected.
- Correctable review decisions reconcile back to the original durable Provider
  pause anchor instead of clearing it. Provider auth-header mode is now part of
  recovery identity.
- Final focused reruns: Pipeline/profile/claim/Provider owner set `113 passed`;
  Web route/recovery set `123 passed`; subsequent policy owner rerun `31 passed`.
- Colima/Docker is available. A diagnostic pre-final-patch runner image built as
  `easyicu-research-agent:1.0.0` with digest
  `sha256:a04c37358f10db014abcb9a48c96ecc8d01ea3f0a2d39db0b32283d256c173d9`;
  it must be rebuilt after the closure commit and is not the final exact-head
  image.
- Existing Pi Provider authority was reverified with a minimal Luna inference:
  HTTP 200, actual model `gpt-5.6-luna`, 312 total tokens. The separately supplied
  DeepSeek-compatible token returned HTTP 401 and was not persisted.

Diagnostic receipt:
`/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/opencode/e1-rereview-worktree-release.json`

Final pre-commit dirty-worktree receipt:
`/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/opencode/e1-final-worktree-release.json`

## First live Planner canary follow-up

- Commit `a56657b09255b30cb3b3f4f5e05eb83455f7b3f8` passed the clean
  exact-head framework release (`137 passed`) at
  `/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/opencode/e1-clean-release-a56657b.json`.
- Exact-head runner image `easyicu-research-agent:a56657b` / `:1.0.0` has
  digest `sha256:8d8fc67c0a7cc15f7a82171a3c8b0213874727542d5ac03734dfa249b1d54bdd`;
  network-none import smoke confirmed the installed canary profile is
  `planner_only=true`.
- The legacy MIMIC-IV prepared export validated as a development-only source:
  export authority `c509a5c6b2e9d5aae0f4ec6212bad5b44910adcd91ec27c19b2d3a00ead39160`,
  Web package binding `d0806e9a61945bf8b6135a34592a97d96ac7e15ca5e8091f5b69abc086e678aa`.
- Web preflight first rejected an unsupported first-stay restriction before any
  Provider call. The StudyContext was honestly narrowed to all adult ICU stays
  and descriptive proportions only; scientific configuration digest is
  `98792d7cd1885487c59073546abefe9efae561999fb6b65de2bd53b4a44c14db`.
- Live job `03a26d114d4f` reached Luna. Acquisition succeeded, then two Planner
  outputs failed structured validation; attempt 3 was blocked before transport.
  The ledger proved the canary's 24-attempt declaration could not be funded by
  its 250k-token/$10 aggregate stops because every call must conservatively
  reserve the Provider's 128k completion ceiling.
- Follow-up fix bounds the canary to its actual shape: one acquisition plus five
  Planner attempts, with six calls, 1.2M reserved tokens, and a $30 conservative
  aggregate ceiling. This changes reservation authority, not actual usage, and
  still cannot enter Execute.

## Counts-only authority and 2026-08-15 canary follow-up

- Commit `76c9bf0` introduced the typed `none_counts_only` StudyContext
  authority and `easyicu.exposure_outcome_distribution/3`. The executor and
  renderer emit point counts/proportions with null confidence, interval,
  contrast, and dependence coordinates. The clean framework release passed 137
  tests; receipt:
  `/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/opencode/e1-counts-only-release-clean.json`.
- StudyContext `e1-luna-canary-20260814-a56657b` now has
  `analysis_family=descriptive_epidemiology`, `analysis_unit=icu_stay`, and
  `variance_estimator=none_counts_only`. Its scientific configuration digest is
  `5dfee6cc8eb139003f7b063d9caba5187564321010a6cdce89a904739f4e0e7e`.
- The original flawed run `run_20260815T015859_03ee12` was durably rejected.
  Checkpoint state is `rejected`, `run_status=human_review_rejected`, one signed
  decision is recorded, and no approved executable plan exists.
- Restart rejection exposed three execution-only dependencies that must not
  gate a negative decision: Provider reconstruction, current environment/config
  equality, and a paused Provider heartbeat deadline. The rejection-only path
  now verifies checkpoint/run/request/plan/evidence/decision integrity while
  skipping those execution capabilities. It never reopens or inspects Provider
  state.
- Counts-only article contracts no longer require `baseline_context`: its only
  typed owner is Table One, which counts-only authority forbids. Measurement
  audit products also cannot claim reserved baseline, distribution, prevalence,
  outcome, risk/effect, interval/confidence, or p-value names.

### Bounded live evidence

- Job `cf07b7810ccf` stopped before Planner because the dirty shared host source
  fingerprint did not match exact image `76c9bf0`; one acquisition call, 1,577
  tokens, estimated `$0.02651`, no Execute.
- Job `88c364747b8d`, launched from a detached exact `76c9bf0` host/image pair,
  exhausted five Planner attempts across schema, dependence, and article
  contracts; six calls, 103,252 tokens, estimated `$1.29580`, no Execute.
- A local isolated image containing the article-contract projection reached a
  durable pause in job `4e46a6868aa3` / pipeline
  `run_20260815T034541_72f869`, but review found semantic laundering:
  `table:distribution_prevalence` was declared as a `measurement_process`
  audit. The pause was not approved. A rejection heartbeat bug terminalized the
  checkpoint as `failed`; Provider is terminal and no execution/result artifact
  exists.
- After closing that reserved-name loophole, job `43221088c25c` exhausted five
  further Planner attempts (article-contract, dependence-authority, and schema
  failures): six calls, 107,818 tokens, estimated `$1.40796`, no plan and no
  Execute. Live retries stopped at this boundary.
- The isolated diagnostic image is
  `easyicu-research-agent:counts-only-article-local`, local image id
  `sha256:43a82d00c7350ca03ffa49e6604f6184c06c46ce85bf6f0feaa0db8b0f1056b1`.
  It is not a release or exact committed-head image.

### Verification and current blocker

- Review/lifecycle/recovery: `92 passed`; article/plan/display: `74 passed, 2
  skipped`; Web recovery/StudyContext/planner-canary selection: `38 passed`.
  Ruff and `git diff --check` passed.
- Targeted mypy over the touched broad owners reports 37 pre-existing typing
  errors (including missing pandas stubs), so no mypy-clean claim is made.
- Formal E1 remains NO-GO. Offline follow-up now removes the redundant generic
  `distribution_prevalence` article module under counts-only authority and adds
  a compact exact nesting example that keeps `descriptive_claim` beside, not
  inside, the `/3` spec. Fixed Planner prompt size is `51,547 <= 51,600` bytes;
  prompt/catalog/cohort-owner regressions are `60 passed`, and focused
  counts-only/display regressions are `11 passed`. Before another Provider
  attempt, this uncommitted closure still needs scoped integration with the
  concurrent worktree, a clean exact-head release, and a matching image.
- During this work, independent SOFA/release-governance changes landed as
  `4f572cd` and `84d572d`. Current repository HEAD is `84d572d`; the scoped
  agent closure above is the only remaining worktree diff and is not committed.

## E1 development Planner Demo accepted

- The user explicitly delegated development-plan review to the assistant. This
  is a product-demo review only; it is not an authenticated human signature,
  formal execution activation, or publication approval.
- Final Web job `98cdb621e841` produced pipeline run
  `run_20260815T044250_9d5fc5` and stopped at its durable Planner-only review
  checkpoint. Plan SHA is
  `98077303bf041d2e1dc6c30269ea9c73d71ec6b3144b13169cdc6a4f7c53f6c4`;
  checkpoint SHA is
  `f4b7bc707b90c9cd08628d718f659745d1f65c9340e016d40ffede9200a0946d`;
  review request authority is
  `0045107dc2dd392853827c4f2227c567828b0729122e1245ca3a33abed6e4b3e`.
- Delegated review verdict: **accepted for the read-only development Planner
  Demo**. The six-step plan has exactly one typed primary distribution owner,
  schema `/3`, no model/Table One/p-value/CI/risk difference/dependence, one
  real measurement-missingness audit, and three downstream typed figures.
- A real cross-process restoration was exercised. Public plan levels remain
  privacy-preserving `__easyicu_level_N__` tokens, while the restored execution
  view resolves exposure/outcome levels to numeric `[0, 1]` and the positive
  outcome to `1` through host-only authority.
- The restore exercise exposed and closed two additional ordering defects:
  durable human-review recovery now rebuilds both Table One and generic declared
  level private bindings, and runtime capability preflight no longer asks a
  still-paused Provider task for its execution timeout.
- Final run used two Provider calls, 20,892 accounted tokens, and estimated
  `$0.30294`. Local matching image
  `easyicu-research-agent:e1-demo-local` has image id
  `sha256:404e6d1240f0202727d2dcbcf67f10d7c5b8c1cd4f72f147560f87712f1c10f0`.
- Demo artifacts are projected under wrapper `run_98cdb621e841`, including
  `agent_plan.json`, `scientific_plan_review.json`, `source_run_manifest.json`,
  `quality_gate.json`, literature/readiness projections, and evidence ledger.
  Result tables, rendered figures, and manuscript remain intentionally empty:
  the profile is Planner-only and has not Execute authority.
- Final focused verification: durable/lifecycle/opaque-level `42 passed`;
  counts-only/display `11 passed`; Web recovery/review `29 passed`; Ruff and
  `git diff --check` passed.

## E1 development execution Demo completed

- Added the server-internal, non-paper profile
  `npj_dm_e1_demo_dev/20260815`. It requires the real Provider, strict evidence,
  Docker, current concept dictionaries, no memory, and no deterministic
  Provider fallback, but sets `planner_only=false`. The public Web route remains
  fixed to `planner_canary`; only an internal `full_reviewed` runner selects the
  Demo profile. The runner image is likewise an internal factory argument and
  cannot be selected by the request body.
- The first reviewed execution exposed an exact-Plan integrity defect:
  measurement companion inputs were added at Execute start, after approval.
  Initial Plan normalization now closes those public Coder inputs before
  lifecycle sealing and review. The integrity gate correctly stopped the old
  run rather than executing a changed Plan.
- Cross-process execution then exposed a missing restored field:
  `_approved_capability_resources`. Durable restore now rebuilds it only from
  the digest-verified capability runtime. A focused regression asserts the
  restored resources exactly equal that authority.
- The counts-only distribution executor produced valid point estimates, but the
  descriptive claim compiler still required finite SE/CI fields. It now accepts
  only the exact `none_counts_only` shape: SE, CI, confidence and cluster count
  must all be null; covariance and interval method must both be
  `none_counts_only`; dependence and risk difference must remain null. It emits
  count/denominator/proportion claims without inventing uncertainty.
- Final job `699b50192c74` / pipeline `run_20260815T054050_cdba9e` executed the
  delegated-review Plan SHA
  `f44f83044556ea72b202634b1509c43b3d0dfa45abb6e023781e518059836709`.
  Checkpoint SHA before approval was
  `45ede4f1c5948711edf24294a50eafa268a5bf13e5b1edcd6872f1aa7a45b7c8`.
  All six Plan steps completed with deterministic standard executors.
- Observed development results: 94,458 adult ICU stays; phenotype absent
  60,461/94,458 (64.008342%) and present 33,997/94,458 (35.991658%). Observed
  in-hospital mortality was 4,986/60,461 (8.246638%) without the phenotype and
  4,480/33,997 (13.177633%) with it. No CI, p-value, risk difference, model, or
  causal claim was computed.
- The wrapper projects 12 aggregate tables and three embedded PNG figures:
  phenotype/mortality distribution, cohort accounting, and data quality. Manual
  image inspection found readable labels and no clipping. Provider ledger
  status is `completed`: six calls, 97,037 accounted tokens, estimated
  `$1.42043` under the development hard stop.
- Matching local diagnostic image is
  `easyicu-research-agent:e1-demo-local`, image id
  `sha256:a1ca8439ffec2dc0e6d6e2a6520a463986c1f88757cb540d454fa99c4c44eb4f`.
  Docker storage exhausted during a full rebuild, so the final exact source was
  produced as two auditable single-file layers over the prior matching image;
  runtime source-integrity validation passed. This remains a local diagnostic
  image, not a release image.
- Development execution/results/figure Demo verdict: **completed**. Publication
  verdict remains **NO-GO**. The quality gate intentionally remains blocked by
  literature, novelty, sensitivity, data-seal, formal-review and publication
  authorities. In addition, Writer currently fails because the host cohort step
  has no live step-result-envelope sidecar; therefore no manuscript completion
  is claimed.
- Focused verification after the new fixes: review/input-closure/state-drift
  `44 passed`; durable/lifecycle/opaque-level `42 passed`; counts-only executor
  and scientific claims `74 passed`; profile/runner/public-route selection
  `5 passed`; Web recovery `18 passed`. Ruff and `git diff --check` passed.

## E1 internal manuscript preview completed

- Closed the Writer host-cohort sidecar false negative without weakening the
  evidence contract. `RegisteredOutputEnvelopeConsumer` excludes the
  host-materialized cohort record only after verifying its exact registered
  host-cohort authority; a tampered cohort-table evidence id fails closed, and
  ordinary successful analysis steps still require a verified sidecar.
- Fresh job `e59d1a54feff` / pipeline `run_20260815T061842_5049c6` executed the
  six-step reviewed Plan SHA
  `8adb8bc24cef8ddc5cf08041ef4a15433ac2254a08e74c2ba27e5ff08538a65d`.
  Its pre-execution checkpoint SHA was
  `0dc0a1944fe65c6bea8180ce59bb4ee2f255bcc86c2101536a262537e09f20be`.
  The run again projected 12 aggregate tables and three embedded figures with
  the same verified counts-only results.
- The Provider ledger completed with 14 calls, 162,256 accounted tokens, and
  estimated `$2.30776`. Writer performed its generation/repair attempts and
  emitted a raw `manuscript_scaffold.md`, but STRICT binding rejected it with
  `manuscript prose lacks deterministic evidence or scientific claim
  authority`. `manuscript_draft.json` therefore remains diagnostic-only and
  truthfully states that no formal manuscript was generated.
- A separate, explicitly labelled presentation artifact was rendered from the
  verified run outputs as `internal_manuscript_preview.md`, self-contained
  `internal_manuscript_preview.html`, and a five-page
  `internal_manuscript_preview.pdf`. The preview states `NOT FOR SUBMISSION`,
  includes all three registered figures, preserves the counts-only claim
  ceiling, and does not claim publication authority.
- Preview SHA-256 values are Markdown
  `104c22171a26941e97406c0cef029f4ff0f75a69ff2c9e7ac69e6239242bbbae`,
  HTML `d90fc15c504eea2c60fc8cd1af3e9cc11d3aca8482d46bb6c2b6d44762d00113`,
  and PDF `c794ec64e8df60d01664e668ca991136ce76de755646ca5ef334c8a87c351d62`.
  Source-data comparison confirmed all displayed denominators, event counts,
  and six-decimal proportions. HTML inspection confirmed three embedded PNGs;
  PDF text/layout inspection confirmed five readable pages with no missing
  sections or truncated tables.
- Current local diagnostic image
  `easyicu-research-agent:e1-demo-local` has digest
  `sha256:573b7bb7db71abb0656121f189c206e2feeb62d5649625a25f6add48689db74e`.
  Initial Writer/sidecar regression coverage was `112 passed`. An expanded
  Writer/envelope matrix then exposed that `SE=0.13` incorrectly inherited an
  earlier OR scale and hid its valid unscaled standard-error claim. The
  mention-local scale resolver now excludes standard-error diagnostics from
  OR/HR/RR inheritance; the final expanded matrix is `195 passed`. Ruff and
  `git diff --check` passed. This is still not a clean exact-head release.
- Internal reporting-preview verdict: **completed**. Formal manuscript and
  publication verdict: **NO-GO** until deterministic sentence-level evidence
  authority, literature/novelty/sensitivity, finalized data seal,
  attributable review, formal activation, and publication review are closed.

## System-validation Demo architecture and v2 report completed

- The first internal preview was visually usable but architecturally wrong for
  the product claim: it was a manually curated phenotype-style article outside
  the run artifact allowlist, so it blurred a weak clinical manuscript with a
  research-agent systems demonstration. It is retained only as historical
  presentation output and is superseded as the primary Demo.
- Added the typed `easyicu.system-validation-report/1` owner in
  `reporting/system_validation_report.py`. Its invariant authority is
  `engineering_validation_only`; `reportable=false` and
  `publication_authorized=false` are literal schema fields. It consumes only
  bounded run projections plus digest-bound review/Provider receipts, copies a
  registered aggregate case-study table without deriving clinical numbers, and
  renders a script-free self-contained HTML dossier.
- Completed execution now projects `system_validation_report.json`,
  `system_validation_report_receipt.json`, and
  `system_validation_report.html`. An externally rendered PDF must pass a fixed
  regular-file/signature/privacy/schema boundary before
  `register_system_validation_pdf()` binds it into the receipt and evidence
  ledger. JSON, HTML, and PDF are separate from `manuscript_scaffold.*`.
- Web uses the distinct `system_validation_document` resource kind. The route
  accepts only `system_validation_report.html|pdf`, applies sandboxed
  script-free HTML CSP, and the governance projection forces
  `easyicu_system_validation_report / engineering_validation_only /
  reportable=false` even if a caller presents a signed/reportable run.
- The guided preview owner identifies the dossier as engineering evidence, not
  a clinical manuscript. The artifact renderer supplies a dedicated system
  lifecycle, metrics, and blocker view; the system report is preferred ahead
  of figures and manuscript artifacts in the completed-run resource list.
- Real run `e59d1a54feff` was reprojected through the new owner. The v2 dossier,
  **When Success Must Still Fail Closed**, makes the system result primary:
  6/6 typed steps, 125 registered evidence records, 12 aggregate tables, three
  figures, and 14 Provider calls completed, while STRICT manuscript and
  publication authority remained blocked. The phenotype output is explicitly a
  bounded case-study appendix, not the paper's novelty claim.
- Final registered v2 artifacts under the run wrapper:
  `system_validation_report.json` SHA
  `b61c4dc3779f0315345c0e6d7da7e46e86734acdb3171a9f9219721b322e3393`;
  receipt SHA
  `9069f8a58d7c2621ca7c48b2766736baa1c1c5f564e868763f2e7f666a80ce5e`;
  HTML SHA
  `cb28132cc989b79a85c3d3f6cd293acfbd1f3cd9865e055e40a44151136aff35`;
  seven-page PDF SHA
  `d04c9ff9d377b3f4e0caa62a498f3833238dc6b3736047731aab694bcbe560ea`.
- Visual QA: all three figures load; case-table values come from the bounded
  `result_tables.json` projection; desktop 1440 px and mobile 390 px have no
  horizontal overflow; print output has no clipped figures, split blocker rows,
  orphan section heading, or missing authority warning.
- Final schema/projection/document/route/replay/static/gateway matrix: `90
  passed`. Ruff and all three changed JS owner syntax checks passed. A broader
  randomized workflow file reached `138
  passed` but two unrelated runner tests stop before this code because the host
  durable recovery registry is already at `pipeline work-root capacity`; the
  existing route-contract snapshot also predates concurrent
  presentation/archive routes. No green claim is made for those external
  baseline issues.
- Guided visibility follow-up: the read-only complete-research demo now ends
  with explicit HTML and PDF links to the audited `e59d1a54feff` system
  validation dossier. The browser preview serves fixed static demo copies,
  labels them engineering-only, and does not depend on the unrelated selected
  project's transcript. The Pi Node event projector now preserves future
  `system_validation_document` resources instead of dropping them. The updated
  report/document/route/replay/static/gateway matrix is `89 passed`; live
  browser QA opened the HTML and PDF from `#guided`, rendered all three figures,
  showed both authority warnings, and found no iframe horizontal overflow.
- Revised Demo verdict: **appropriate as an engineering system-validation case
  study, not yet a top-journal systems paper**. A publishable systems paper still
  requires multi-task/multi-database benchmarks, governed-vs-ungoverned
  comparisons, authority-boundary ablations, independent expert evaluation,
  and reproducibility/time/cost analysis. Formal clinical manuscript verdict
  remains **NO-GO**.
- Death-time semantic correction: the source run was numerically correct but
  its supporting data-quality figure was reader-ambiguous. `death_time` is a
  typed `conditional_event_time`: `eligible_n=9,466`,
  `not_applicable_n=84,992`, and `value_missing_n=0`. Its 10.021% cohort share
  is therefore overall death/event applicability, not measurement coverage.
  The deterministic audit now emits explicit cohort applicability,
  available-within-applicable, missing-within-applicable, and event-prevalence
  percentages. The renderer now uses three partitions: value available,
  missing among applicable stays, and not applicable. Positive-only events are
  promoted to complete binary status only when their typed count/measured/value
  receipt reconciles; otherwise absence remains missing/conflicted.
- The finalized source run remains immutable. The dossier now carries a
  report-only corrected gallery derived from the exact audit-table binding SHA
  `c528c6a42bad7babc9b0a21dbd81e67114c2b50b32e25f4b94161878a52c2a05`.
  Updated SHA values: corrected gallery
  `c9b54b12d65e1855ee1a61fe375817447ba7a04e689400b00a8361822639e0a5`,
  report JSON `58964426e05c161db81d5c22fd2209f8135c97347a3b4f2ba5e85a327efe90dd`,
  receipt `1f512f6cb2c5c54c5dba9043402f86457f26bf7c53f46fd24a0df14ded494761`,
  HTML `5e628755fa4b7c0ff4acb75d0ebe07fe9488de6921c9f5ba2e0b2ba822309963`,
  and seven-page PDF
  `22ed6edd0350e44950aca08a326d9a034e01deface601b267460c62ff6411ba8`.
  Browser QA at 1440/390 px found no overflow and all three figures loaded.
  The final missingness/observation-semantics/report/document/route/replay/
  static/gateway matrix reached `198 passed`; receipt 3/3, report source
  bindings 11/11, corrected source binding 1/1, ledger entries 5/5, and static
  copies 2/2 matched exact bytes.
- Publication-skill audit: both `nature-figure` and `nature-writing` were
  activated in the real run. The publication figure skill explicitly records
  `generated=false`, `skipped_reason=render_failed`; all three visible figures
  were supporting deterministic step renderers, and no primary publication
  figure bundle exists. WriterAgent ran with the Nature Writing prompt, but its
  raw scaffold failed STRICT evidence binding. The internal manuscript preview
  was a later curated presentation artifact, and the systems dossier is a
  deterministic post-run projection; neither is a successfully authorized
  Writer manuscript.

## Reviewer Demo completion and presentation repair

- The visible Demo was still framed as 4/8 complete because it reused the old
  140-stay clinical-canary transcript and treated the formal manuscript gate as
  the status of the whole product demonstration. This presentation conflated
  two different outcomes: a completed systems workflow and an unauthorized
  clinical manuscript.
- Replaced the walkthrough with a reviewer-specific, bounded read-only
  projection derived from registered source run `e59d1a54feff` /
  `run_20260815T061842_5049c6`. It reuses standard Web renderers but is not the
  live artifact transport. The reviewer
  protocol is now explicit and all eight product stages are complete: protocol,
  validation scope, data contract, safe projection, exact plan, analysis,
  bounded interpretation, and reviewer dossier. The workflow and authority
  aside report `8/8`; manuscript/publication authority is a separate amber
  `WITHHELD AS DESIGNED` outcome. A redundant top-level reviewer-verdict card
  was removed so it no longer obscures or compresses the transcript.
- The reviewer transcript exposes the exact six-step plan, 94,458-stay
  denominator, 33,997 phenotype-positive stays, observed deaths
  4,986/60,461 and 4,480/33,997, applicability-aware `death_time` audit,
  125 evidence records, 12 aggregate tables, three figures, 14 Provider calls,
  and 11 source bindings. It opens the HTML dossier automatically and provides
  the registered PDF without starting a model job or mutating a project.
- Fixed a projection/privacy defect discovered during the redesign. Wide CSV
  tables previously truncated to 12 columns before checking identifiers, which
  both hid outcome events/denominators and could miss an identifier after the
  visible cap. The projection now scans the complete header first, rejects any
  identifier regardless of position, and selects review-critical aggregate
  columns before applying the cap. The dossier primary table now includes
  `outcome_events`, `outcome_denominator`, `outcome_rate_pct`, and
  `interval_method`.
- Reframed the self-contained report as **A Complete, Governed Research
  Workflow**. The document now uses `REVIEWER DEMONSTRATION COMPLETE`, explains
  that amber means an expected authority boundary rather than Demo failure, and
  keeps `engineering_validation_only / reportable=false /
  publication_authorized=false`. No STRICT or scientific gate was weakened.
- Current projection SHA-256 values: result tables
  `16c14d7f8d456eb5334df6df9fef59f028fd8d303dbf31cb198fe75e62372089`,
  corrected gallery `c9b54b12d65e1855ee1a61fe375817447ba7a04e689400b00a8361822639e0a5`,
  report JSON `3c3b879ba0ee6b9524df169f28873ec53f63f9756c3266c26d50191f6522a8df`,
  receipt `aad8bbe5c847c9a391b7e3a1b60662eb174a79f8fad92e77a9ede2babb9e404b`,
  HTML `aa6d8fb3cb6b708bbda1e1be8b921a9938957a54a0b165e827b5e26cebb7f3c0`,
  and six-page PDF
  `ce943c6bba360f9b4f3ef1d1cd0432b940a3b4f88d858edd50909bb10350ab27`.
- Browser QA at 1440x1000 and 390x844 verified automatic HTML opening, 8/8
  workflow state, three loaded figures, HTML/PDF switching, no page/report/hero
  overflow, no HTTP errors, and no console errors. Mobile intentionally shows
  the dossier first and returns to the complete transcript through the preview
  close control.
- Final focused semantic/report/document/route/replay/static/gateway/recovery
  matrix: `217 passed`; Ruff, Node syntax, strict progress lint, and diff-check
  passed. Receipt 3/3, report source bindings 11/11, corrected source 1/1,
  ledger entries 6/6, and static copies 2/2 matched exact bytes. The previously
  documented host work-root-capacity failures in unrelated runner tests remain
  external to this Demo path.

## Reviewer Demo lifecycle and real-renderer parity

- Expanded the read-only transcript from a compressed reviewer narrative into
  four native activity groups with 42 visible lifecycle steps: local preflight,
  Provider authorization, context/cohort/audit construction, literature setup,
  rejected plan draft 1/5, accepted draft 2/5, durable human-review pause,
  exact-plan resume, six execution steps, evidence/numeric/Provider ledgers,
  STRICT Writer stop, Scientific Readiness, and dossier registration. These are
  lifecycle facts and receipts, not private chain-of-thought.
- Restored 12 standard clickable artifact entries: `run_context.json`,
  `cohort_summary.json`, `quality_gate.json`, `agent_plan.json`,
  `literature_evidence.json`, `scientific_plan_review.json`,
  `scientific_readiness.json`, `manuscript_draft.json`, `figure_gallery.json`,
  `result_tables.json`, `source_run_manifest.json`, and `evidence_ledger.json`.
- Removed the Demo-only generic JSON-card rendering path. Read-only Demo
  projections now enter the same `AGENT_RENDER.artifactStructuredView` used by
  live run artifacts; literature enters the dedicated literature renderer.
  The figure gallery hydrates the three exact embedded PNGs from the registered
  self-contained dossier before entering the standard figure renderer. Raw JSON
  remains available as the secondary audit tab rather than the default view.
- Focused Guided/static/accessibility matrix: `113 passed`; Node syntax, Ruff,
  and diff-check passed. Browser QA at 1440x1000 showed table views, nine
  literature cards, and three gallery images with zero generic Demo cards. At
  390x844 all 12 standard artifacts opened through the expected renderer with
  zero preview errors, no activity clipping, no horizontal page/preview/report
  overflow, and no console errors. The auto-opened dossier retained all three
  images.

## Pre-commit reviewer closure

- Authority is now fail closed across report construction and display. A
  system-validation report cannot accept positive manuscript/publication
  authority, numeric zeroes remain visible, and engineering completion requires
  the complete approved review checkpoint rather than execution alone. The
  visible Demo consistently identifies itself as
  `bounded_reviewer_projection_from_registered_run`; it does not claim live
  artifact transport or publication evidence.
- Document serving now recomputes every registered document digest against the
  evidence ledger. PDF registration verifies the existing report/HTML receipt
  and ledger bindings, parses the PDF with PyMuPDF, and privacy-scans extracted
  text and metadata instead of random compressed bytes. The regenerated PDF is
  tagged, has a structure tree, retains all eight bounded table columns across
  six A4 pages, and is byte-identical to the static copy.
- Rejection-only recovery does not reopen the prepared package or Provider
  ledger, and a superseded setup can still terminally reject its old plan.
  Session history is filtered by Research/Workspace mode before its limit is
  applied, then prefers non-empty history across up to 100 target-mode rows.
  Deleted system-temporary work roots are pruned at registration so stale test
  roots cannot exhaust the bounded recovery index; unavailable durable roots
  outside the system temporary directory remain registered.
- Guided project rail markup is owned by `screens-guided-projects.js`, not the
  over-budget `screens-guided.js`. At <=920 px the compact home/settings/
  language/data-workspace controls remain available after closing the
  auto-opened dossier.
- Current system-document SHA-256 values: report JSON
  `3c3b879ba0ee6b9524df169f28873ec53f63f9756c3266c26d50191f6522a8df`,
  receipt `ca812644eb9fa3422f000746134cbfd48f7fc88ee7dc613fcb3c2ebc99cae4d8`,
  HTML `1758ba395bc793aaec9ec7a1b04645b6b84041d42cff863faaf3bfa8cb4ebc7f`,
  and tagged six-page PDF
  `9467664953814a2c6c946541a3b6aaf1d8b9242f1c17612f1e6f748a48c03334`.
  Source bindings are 11/11, document ledger entries 4/4, and static copies 2/2.
- Final verification: Research Agent/Pi/document/recovery matrix `524 passed`;
  Guided/static/accessibility matrix `113 passed`; post-copy Pi static rerun
  `33 passed`; Node syntax, Pi sidecar `npm run check`, Ruff, and PDF structure
  checks passed. Browser QA at 1440x1000 and 390x844 found zero horizontal
  overflow and console errors; the dossier provenance, 61 artifact receipt
  links, compact rail, and authority banner were visible in their expected
  states.
