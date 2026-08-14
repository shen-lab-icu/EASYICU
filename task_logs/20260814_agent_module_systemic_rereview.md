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
