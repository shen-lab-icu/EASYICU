# EasyICU Release Checklist

This checklist is a gate, not evidence itself. Every checked item needs a
concrete command result, receipt, or artifact path. If a required item lacks
evidence, the release remains blocked or its scope and claims must be reduced.

## 1. Candidate identity and scope

- [ ] Record the release version, exact commit SHA, branch, and intended tag.
- [ ] Freeze a clean candidate worktree; identify any deliberately excluded
      local or concurrent changes.
- [ ] List one owner for each shipped public contract and its typed consumers.
- [ ] Confirm the PR title and description match the actual change set.
- [ ] Confirm independently reviewable workstreams were split into separate PRs.
- [ ] Record migration and rollback instructions for public API changes.

## 2. Verification

- [ ] Run direct owner, negative/fail-closed, and boundary-contract tests.
- [ ] Run the supported Python and operating-system matrix.
- [ ] Run one full exact-head CI against the frozen candidate and retain its URL
      or exported result. Focused tests do not satisfy this gate.
- [ ] Build wheel and sdist, install each in a clean environment, and verify
      package data and console entry points.
- [ ] Rebuild and smoke-test the digest-bound Docker runner when its image,
      requirements, execution contract, or runtime code changed.

## 3. Dependency and supply-chain evidence

- [ ] Generate a validated dependency snapshot or constraints set for every
      supported Python/OS target; record the resolver and source inputs.
- [ ] Execute the compatibility matrix against that snapshot before freezing it.
- [ ] Run a dependency vulnerability audit and disposition every finding.
- [ ] Confirm external workflow actions use reviewed immutable commit SHAs and
      workflow permissions are least-privilege.
- [ ] Produce and retain an SBOM for release artifacts and the runner image.
- [ ] Attach artifact provenance binding the source commit, build inputs, hashes,
      and builder identity.
- [ ] Sign release artifacts, or record why signing is unavailable and keep the
      release blocked from channels that require it.
- [ ] Confirm repository CodeQL and secret-scanning results are reviewed. Platform
      settings must be verified directly; workflow files alone are not evidence.

## 4. Clinical and database evidence

- [ ] Record each database/concept state from the shipped clinical registry.
- [ ] Do not promote `mapping_only`, algorithm golden tests, or resolver tests to
      database-specific clinical validation.
- [ ] For each claimed validated mapping, retain source-variable, unit,
      conversion, time-window, missingness, and treatment-semantics evidence.
- [ ] Compare claimed database outputs with an independent implementation and
      report agreement, disagreement classes, and unresolved cases.
- [ ] Obtain approval from an independent clinical reviewer for clinical
      definitions and an independent ICU data reviewer for database mappings.
- [ ] Confirm no patient-level data or confidential paths entered source control,
      logs, demos, issues, or release artifacts.

## 5. Research-agent and manuscript authority

- [ ] Label mock-provider, deterministic orchestration, and real-provider results
      separately; do not infer model robustness from mocks.
- [ ] Report which evaluation tiers ran: Tier 1 deterministic checks, Tier 2
      model jury, and Tier 3 clinician review.
- [ ] Limit manuscript claims to completed tiers and covered validators.
- [ ] Produce manuscript findings through the registered research-agent pipeline;
      hand-computed values may be retained only as oracle checks.
- [ ] Bind tables, figures, statistics, and prose claims to their exact evidence
      receipts and fail closed on stale or mismatched artifacts.
- [ ] Retain cost, latency, provider/model identity, retry, and failure evidence
      for any real-provider benchmark claim.

## 6. Documentation and approval

- [ ] Synchronize English and Chinese public claims with the evidence ceiling.
- [ ] Update the relevant module CURRENT file and durable dashboard row only when
      a real milestone changed status.
- [ ] Link the release task log, CI evidence, SBOM, provenance, clinical review,
      and unresolved blockers from the release record.
- [ ] Obtain required independent domain approvals; the author cannot satisfy
      their own independent-review gate.
- [ ] Recheck all external links, citations, licenses, and third-party notices.

## Release decision

- [ ] **Ready:** every required gate above has concrete evidence.
- [ ] **Blocked:** list the missing gate, owner, and next action.
- [ ] **Reduced scope:** state which feature, database, or claim was removed and
      rerun all affected gates for the reduced candidate.
