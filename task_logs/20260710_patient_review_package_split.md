# Patient Review service package split

Date: 2026-07-10
Branch: `fix/easyicu-concept-bounds-enforcement`
Status: done

## Objective

Turn the 2,462-line `patient_drilldown.py` service into a package with an
explicit internal ownership seam, without changing its import path, public or
private module attributes, mutable module globals, or Patient Review behavior.

## Result

- `src/easyicu/webserver/patient_drilldown.py` became
  `src/easyicu/webserver/patient_drilldown/__init__.py`; existing imports of
  `easyicu.webserver.patient_drilldown` therefore keep the same module name.
- The eight eligibility-flow helpers moved to
  `patient_drilldown/eligibility.py` (323 lines). The package facade aliases
  them back, preserving callers that import or monkeypatch the private helper
  names.
- The remaining 2,171-line facade still owns shared mutable state and the
  tightly coupled patient timeline/review workflow. Further extraction should
  follow a tested state/renderer seam instead of mechanically moving functions.

## Compatibility contract

`tests/test_webserver_route_contracts.py` now verifies that:

- the legacy single-file path no longer exists;
- all eight facade helper attributes are identical to their new owner
  functions;
- the definitions exist only in `eligibility.py`;
- the eligibility owner does not reverse-import its package facade.

An AST comparison against the pre-split source confirmed all eight moved
function bodies are structurally equivalent.

## Verification

```text
pytest -q tests/test_webserver_route_contracts.py
17 passed

pytest -q tests/test_webserver_workspace_summary.py -k patient_review
9 passed, 111 deselected
```

`compileall`, scoped Ruff checks, facade identity checks, AST equivalence for
all eight helpers, and `git diff --check` also passed.

## Commit

- `c8185cc refactor(web): package patient review service`

## Remaining WebApp work

- Run desktop browser QA with the 94k-entity real export.
- Complete real six-database Cross-DB density/n×n verification.
- Do not continue splitting large modules solely to reduce line counts; require
  a coherent ownership boundary and focused regression contract first.
