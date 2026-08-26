# Figure 2 v3 scorer reseal audit (2026-08-23)

## Scope and baseline

- Audited code change: `ab46bb9` (`fix(agent): split survival evidence renderer`).
- Audited reseal: `156d1bc` (`Fix deterministic CI contract regressions`).
- Current main baseline: `origin/main@8115f933f54b260c392ea9cac75828294ea75d9c`.
- No Provider call, benchmark execution, stored result rewrite, score recomputation, push, merge, or main-worktree cleanup was performed.

## Decision record

The v3 manifest reseal from `17311d59...e638` to `dbe51f14...47af` is consistent with the repository's live-rubric maintenance model. Extracting the actual digest at every manifest-changing commit on the ancestry of `156d1bc` yields 15 distinct values: one initial seal plus 14 reseals including `156d1bc`, so 13 reseals preceded it. Two of those earlier transitions were package-reorganization side effects rather than commits titled as deliberate reseals; excluding them produces the narrower count of 11, but that is not the complete lineage count. The 2026-08-10 repair is not a contrary no-reseal policy: that repair removed H3-specific policy from the shared `ClusterSelectionManifest`, so the shared sealed source and digest correctly returned to their prior bytes.

`ab46bb9` made one monotonic addition to `effect_output_authorized`: the existing predicate now also recognizes the host-selected `signed_landmark_survival_suite`. That branch is fail-closed unless all of the following hold:

- exact method `signed_landmark_survival_suite`;
- `primary` planned role;
- at least one typed table output;
- a digest-shaped `scientific_runtime_contract:<sha256>` rule reference;
- exact deterministic analysis and preflight reason values;
- a mapping-valued candidate receipt whose `claimed_by` is the signed survival executor.

The signed method is produced by the Figure 2 current-case protocol only for `h1_ventilation_survival`. The other eight Canonical9 cases do not project this method. The change therefore widens live effect-output authority only for the sealed H1 deterministic survival owner; it does not make a Coder-authored or near-miss step authoritative.

This is an intentional future/runtime authorization change, not a claim that scoring behavior is universally unchanged. It can affect a future H1 run evaluated under the current v3 authority. It did not itself recompute or rewrite any stored score, and the recorded Held-out27 state remains `0/27`, `formal_ready=0/27`, and `paper_authority=false`; consequently there is no existing formal paper-facing score changed by this reseal.

The retired v2 manifest remains byte-frozen and was correctly not modified.

## Verification

- Existing H1 integration test exercises the exact positive authorization receipt.
- Added eight public-predicate fail-closed cases: wrong method, wrong role, no table output, no runtime-contract reference, wrong deterministic owner, wrong preflight reason, `claimed_by="coder"`, and a non-Mapping (`None`) step record; all must return `False`.
- Focused fail-closed result: `8 passed`.
- H1/E2/H2 current-case authority + v3 live rubric + byte-frozen v2 history: `55 passed, 7 warnings`.
- This task does not claim a new exact-head full CI run.

## Main-worktree cleanup boundary

The local main worktree was not modified. Read-only three-way comparison found that `landmark_survival_executor.py`, `screens-guided-pi.js`, and `test_pi_copilot_static.py` are stale versions already preserved by backup refs, but `static/index.html` also contains the later uncommitted `20260823-run-history-authority1` cache revision. Those four paths must not be discarded as one batch; the Monitor cache revision must first be carried onto the reconciled owner-split baseline or otherwise committed/preserved.
