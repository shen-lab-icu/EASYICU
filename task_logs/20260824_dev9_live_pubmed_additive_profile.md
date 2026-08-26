# Dev9 additive live-PubMed execution profile

Date: 2026-08-24

## Outcome

- Implementation commit: `5e61db3` on isolated branch
  `codex/dev9-quality-remediation`.
- Archived profile `npj_dm_dev9_demo_dev/20260822` is unchanged and continues
  to omit `enable_pubmed` from both public serialization and pipeline options.
- New additive profile `npj_dm_dev9_demo_dev/20260824` preserves the prior
  development-only execution, provider, dictionary, Know-How, memory, fallback,
  and Planner coordinates and pins only `enable_pubmed=true`.
- `PipelineConfig` now rejects ad-hoc live PubMed enablement when a historical
  or other registered profile does not authorize it. Profile-less exploratory
  runs retain their explicit opt-in path.
- The profile name still ends in `_dev`; it does not acquire paper-facing or
  formal authority.

## Identity evidence

Canonical sorted compact-JSON SHA-256 values:

- archived `npj_dm_dev9_demo_dev/20260822`:
  `6675fe50bd5bdbf6bbaf5ff586d65b7c90b245ea64ac6bf3b73068246e44f7b8`
  (`enable_pubmed` omitted);
- additive `npj_dm_dev9_demo_dev/20260824`:
  `03a984b85c87afcb03653369dd218e0d975a86b0dfd9cb88a653e28948caded0`
  (`enable_pubmed=true`).

`CURRENT_DEV9_AI_REVIEWED_DEMO_PROFILE_REF` now resolves to the additive
`20260824` profile; the registry retains the archived `20260822` coordinate.

## Verification

- Profile snapshot, PipelineConfig negative-boundary, and adjacent pipeline
  tests: `56 passed, 292 deselected`.
- Ruff and `git diff --check`: passed.
- Provider calls/tokens/cost: `0 / 0 / $0.00`.
- No image/full CI was run; this is not the final freeze checkpoint.

## Replay decision

The live literature shadow probe already exercised M2/M3/H2/H3 with zero
Provider calls. A full old-checkpoint pipeline resume under the new profile is
not an execution-only replay: the profile/literature authority digest is
intentionally different and should invalidate the historical checkpoint.
Therefore the next safe replay is an affected-owner/preplan replay that binds
the new profile and persists fresh literature/plan-review authority, not a
forced reuse of an old profile checkpoint.
