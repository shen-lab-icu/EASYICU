# Figure 2 current-source V4 engineering candidate

This is a separate engineering candidate, not a replacement signature for v3
and not a formal experiment, Provider, qualification or paper authorization.
The legacy manifest, evaluator source and all original tests remain unchanged.
The legacy production runner still uses v3 and rejects the changed source.
No V4 Provider transport or formal batch acceptance is wired here.

The three authority-coupled modules are explicit version forks because their
model Literals and function globals bind the original protocol. Their only
changes are the declared import/authority coordinates in DERIVATION.json.
Pure rubric, suite, input-binding, validity and task models stay with their
existing owner. Thresholds, the nine tasks and five dimensions are unchanged.
V4 request, receipt, task/input authority, scorecard, envelope and attempt
schemas differ from legacy identities; old artifacts cannot be reused intact.

V4 source identity binds all EasyICU Python owners, packaged authority JSON,
and the complete Canonical9 Python/config sources. This includes delegated
owners hidden behind the current schema facade. New/removed/changed owners
invalidate the candidate; the new manifest excludes only itself from its own
digest. This draft's new identity may be regenerated only for a newly reviewed
engineering candidate, never by the runtime loader or to repair an old result.

## Explicit two-tree CI contract

1. Historical v3: tools/run_figure2_historical_ci.py exports only commit
   dbbdae6a54893f0efad0466d1aac85b00e2ef6d1 (tree
   8d751c8ae88f9dd3674e963a62388124403d677d), verifies the original source
   identity, makes source read-only and runs all untouched evaluator tests.
   The subprocess receives a private HOME and an INET connection guard, not
   the user's provider environment. Its receipt clearly says historical.
2. Current source: CI explicitly loads tests.support.figure2_versioned_ci and
   --figure2-current-v4. Only four historical test modules are routed away;
   every one of their 149 parametrized contracts executes in the new V4
   namespace. This includes all 105 previously blocked safety scenarios,
   the current-source binding assertion and the other 43 contracts. The
   projection changes only namespace and exact authority coordinates.
   Assertions, parametrizations, helper control flow and original files stay
   intact. The collection guard rejects any missing/extra projected node or
   accidental legacy namespace execution and records every node mapping.
3. Every other current test remains selected normally, including other
   evaluator safety contracts and host/source/cohort controls. Additional
   direct tests assert that current source rejects the original v3 lock,
   that delegated owner mutation fails, and that V4 typed schemas reject
   legacy artifacts. Historical success is never labelled current v3 success.

Local current suite invocation:

```sh
pytest -q -m '' -p tests.support.figure2_versioned_ci --figure2-current-v4 \
  --figure2-collection-receipt=ci-receipts/current-figure2.json
```

The history command and current suite must both succeed for the versioned CI
checkpoint. An ordinary unqualified `pytest` invocation still includes the
unchanged old v3 tests and exposes the historical conflict; routing is explicit.
