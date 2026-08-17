## Scope

**Primary owner/workstream:**

<!-- Choose one owner boundary. List required contract consumers separately. -->

- [ ] Data/concept extraction or clinical scoring
- [ ] Native Web app
- [ ] Research agent or evidence governance
- [ ] Pi workspace or security
- [ ] Benchmark/manuscript protocol
- [ ] Packaging, CI, or release engineering

**Out of scope:**

<!-- State what this PR deliberately does not change or prove. -->

If the change contains more than one independently reviewable workstream,
split it before review. A cross-layer contract change may remain together only
when one owner and its typed consumers are identified explicitly.

**Starting base HEAD (full SHA):**

- [ ] This PR contains one independently reviewable workstream.
- [ ] Work was performed in a dedicated linked worktree, or a sole-user clone
      exception is explained below.
- [ ] No unrelated staged, unstaged, or untracked path was present at commit.

## Task-scope receipt

Run `tools/verify_git_task_scope.py` immediately before committing and paste
its receipt. The guard fails closed when HEAD moved, when the task ran in the
shared primary worktree, or when any staged, unstaged, or untracked path falls
outside the declared allowlist.

```text
scope_sha256:
allowed_paths:
staged_paths:
unexpected_paths: []
```

## Contract and risk

- Owning module and public contract:
- Allowed dependencies and affected consumers:
- Stable reason/validator codes for fail-closed paths:
- Clinical, privacy, security, statistical, or compatibility risks:
- Rollback or migration note:

## Evidence class and claim ceiling

- [ ] Unit/negative contract only
- [ ] Algorithm-level golden vectors
- [ ] Database mapping only (`mapping_only`)
- [ ] Database-specific conformance evidence
- [ ] Formal experiment or publication-authorized artifact
- [ ] Documentation/governance only

Describe the strongest claim supported by this PR and the claims it does not
support. Algorithm tests and mapping presence must not be presented as
database-specific clinical validation.

## Verification

**Exact PR head (full SHA) verified:**

**Exact commands and results:**

```text
# Paste commands and concise pass/fail counts. Link long output from task_logs/.
```

- [ ] New negative/fail-closed regression reproduces the original defect
- [ ] Direct owner and boundary-contract tests pass
- [ ] Focused tests are not described as full-repository CI
- [ ] Full exact-head CI was run, or is explicitly marked not run
- [ ] Generated artifacts identify their source data, code, and commit
- [ ] Required CI is green for the exact PR head before merge

## Independent domain review

The PR author is not the independent reviewer. Name the required reviewer and
record approval before merge for every checked domain.

- [ ] Clinical definition or mapping — ICU clinician reviewer
- [ ] Database extraction or transformation — ICU data reviewer
- [ ] Statistical method or reported result — methods reviewer
- [ ] Credentials, sandboxing, filesystem, or network boundary — security reviewer
- [ ] User-facing workflow — Web/product reviewer
- [ ] No independent domain review required; rationale provided below

Reviewer(s) and rationale:

## Release and paper impact

- [ ] No release or manuscript status change
- [ ] Release candidate — complete [`docs/release_checklist.md`](../docs/release_checklist.md)
- [ ] Manuscript-facing — link the agent-produced artifact and authority receipt
- [ ] Security-sensitive — follow [`SECURITY.md`](../SECURITY.md)

Remaining blockers and follow-up owner:
