"""The row named a file that exists only in the parent step.

The robustness replay copies the primary coefficients into its own outputs, and
the matrix row it writes carries ``coefficient_source_table`` so a reader can
trace each estimate back to the coefficient it came from.

The copy renamed the file; the row kept the upstream name.  ``figure_source_data``
resolves ``coefficient_source_table`` against the outputs of the step that owns
the row, so it opened a path that was never there and reported
``coefficient_source_unreadable`` -- on the 2026-08-01 E1 run (canary26) that
was the last finding standing between a task and its first complete pass, after
the two earlier findings on the same step had been cleared.

Measured over every recorded run: 11 matrix rows name a file their own step does
not own -- every one of them the primary row, every one naming the parent's
filename -- against 4 that name one it does.

Both spellings were already in the code: the assignment preferred the source
path's name and fell back to the name the copy actually uses.  The fallback was
right, so the branch is gone and one constant now serves the copy and the row.
"""

from __future__ import annotations

import ast
import csv
import inspect
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners import deterministic_robustness
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _PRIMARY_COEFFICIENT_COPY_NAME,
)


def test_the_row_and_the_copy_use_one_name() -> None:
    """The property that was false: two spellings of one artifact.

    Read from the module source rather than by running the replay, which needs
    a real cohort: what matters is that neither site can name the file on its
    own.
    """

    source = inspect.getsource(deterministic_robustness)
    tree = ast.parse(source)

    # The invariant is not "one constant everywhere": a variant row legitimately
    # names the variant coefficients file, which this step also writes.  What
    # every branch must satisfy is that the name is a file THIS step produces.
    owned = set(deterministic_robustness._ROBUSTNESS_PRODUCT_KINDS)
    assignments = [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "coefficient_source"
    ]
    assert assignments, "nothing assigns the row's coefficient source any more"
    for rendered in assignments:
        if rendered == "_PRIMARY_COEFFICIENT_COPY_NAME":
            continue
        literal = ast.literal_eval(rendered)
        assert Path(literal).stem in owned, (
            "the row names a coefficient file this step does not produce: " + rendered
        )

    copy_fn = inspect.getsource(
        deterministic_robustness._copy_structured_primary_contract_artifacts
    )
    assert "_PRIMARY_COEFFICIENT_COPY_NAME" in copy_fn
    assert (
        '"coefficients.csv"' not in copy_fn
    ), "the copy respells the filename instead of using the shared constant"


def test_the_row_no_longer_takes_the_upstream_filename() -> None:
    """The exact shape that produced the defect.

    ``coefficient_path`` points into the PARENT step's outputs; taking its name
    is what made the row unreadable from the step that owns it.

    Scoped to the function that builds the row.  Elsewhere in the module the
    upstream filename is a legitimate thing to read -- binding verification
    checks the parent file's own logical name -- so a module-wide ban would
    fail on correct code and say nothing about this defect.
    """

    tree = ast.parse(inspect.getsource(deterministic_robustness._matrix_model_trace))
    offenders = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Name)
        and node.value.id == "coefficient_path"
    ]
    assert not offenders, f"the upstream filename names the row again: {offenders}"


def test_the_name_is_a_bare_filename() -> None:
    """It is resolved by joining onto an outputs directory, so it must not
    carry a path of its own."""

    assert _PRIMARY_COEFFICIENT_COPY_NAME
    assert "/" not in _PRIMARY_COEFFICIENT_COPY_NAME
    assert Path(_PRIMARY_COEFFICIENT_COPY_NAME).name == _PRIMARY_COEFFICIENT_COPY_NAME


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_name_this_step_writes_is_the_one_recorded_runs_hold() -> None:
    """Real bytes: the constant must be the file the replay really produces.

    If the copy's name ever changed, this fixes the row to a filename no run
    contains -- which is the same defect pointing the other way.
    """

    owned = 0
    for path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/robustness_matrix.csv")
    ):
        if (path.parent / _PRIMARY_COEFFICIENT_COPY_NAME).is_file():
            owned += 1
    if not owned:
        pytest.skip("no recorded robustness step kept a coefficient copy")
    assert owned, "no recorded replay writes the coefficient file under this name"


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_recorded_rows_that_are_unreadable_are_the_ones_this_fixes() -> None:
    """Pin the failure this change targets, from the artifacts themselves.

    Recorded runs predate the fix, so rows naming a missing file are expected
    here; what is asserted is that every one of them is the case the fix
    covers -- a primary row whose step does hold the copy under the new name.
    If some other row shape were unreadable, this fix would not be enough and
    that must not pass silently.
    """

    unexplained = []
    for path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/robustness_matrix.csv")
    ):
        try:
            rows = list(csv.DictReader(path.open()))
        except OSError:
            continue
        for row in rows:
            named = (row.get("coefficient_source_table") or "").strip()
            if not named or (path.parent / named).is_file():
                continue
            if (path.parent / _PRIMARY_COEFFICIENT_COPY_NAME).is_file():
                continue
            unexplained.append((path.parent.parent.name, row.get("spec_id"), named))

    assert not unexplained, (
        "recorded rows name a missing coefficient file that this fix does not "
        f"explain: {unexplained[:5]}"
    )
