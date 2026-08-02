"""Counting exposure terms answers "how many", never "which one".

``_structured_model_row`` built the robustness headline by filtering the
term-level coefficient table to ``term_role == "exposure"`` and demanding
exactly one row.  A declared gradient fits one term per non-reference level, so
a four-level exposure fits three and the panel refused every one of them:

    robustness matrix requires one scalar primary exposure coefficient;
    model 'primary_death_model' emitted 3

The producer had already answered the question.  In canary44 (2026-08-02) E3's
primary model step ran as a host deterministic owner with zero provider calls
and emitted:

* ``adjusted_association_estimates.csv`` with ``is_primary_contrast=True`` on
  the ``3 vs 0`` row (OR 5.1628) -- the executor refuses unless exactly one row
  carries that mark;
* ``step_summary.json`` with ``primary_or = 5.1628``;
* ``model_contracts[0].exposure_expression = "aki_stage_max__is_1"``.

The last one is wrong: it took ``exposure_terms[0]``, the contrast that
happened to be fitted first, while the plan declared level 4 as the primary
contrast (term ``aki_stage_max__is_3``).  So the contract carried a
single-term identity that named the wrong coefficient, and the consumer never
read it anyway.

Two owners, two edits: the producer names the contrast it already marked, and
the consumer selects on that name instead of counting rows.  The refusal stays
for the genuinely unidentified cases.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners import adjusted_association_executor
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _structured_model_row,
)


# ---------------------------------------------------------------------------
# The producer names the contrast it already marked
# ---------------------------------------------------------------------------


def test_the_contract_expression_is_not_the_first_fitted_term():
    """Anchored on the source, because the defect was one index."""

    source = inspect.getsource(adjusted_association_executor)

    assert '"exposure_expression": exposure_terms[0]["term"],' not in source
    assert "_contrast_column(exposure, contrasts.primary)" in source


def test_the_single_term_case_still_uses_the_only_term():
    """A binary or continuous exposure has no contrast set to consult."""

    source = inspect.getsource(adjusted_association_executor)
    window = source[source.index('"exposure_expression": (') :][:400]

    assert 'exposure_terms[0]["term"]' in window
    assert "if contrasts is None" in window


def test_the_primary_mark_is_still_required_to_be_unique():
    """The producer's own guarantee that ``contrasts.primary`` is unambiguous.

    The fix reads ``contrasts.primary`` without re-checking it. That is only
    safe while this refusal stands.
    """

    source = inspect.getsource(adjusted_association_executor)

    assert 'if sum(1 for row in rows if row["is_primary_contrast"]) != 1:' in source


def test_the_design_matrix_name_comes_from_one_helper():
    """The term the contract names must be spelled the way the fit spells it."""

    assert (
        adjusted_association_executor._contrast_column("aki_stage_max", "3")
        == "aki_stage_max__is_3"
    )


# ---------------------------------------------------------------------------
# The consumer selects on the name instead of counting
# ---------------------------------------------------------------------------

_ORDINAL_COEFFICIENTS = pd.DataFrame(
    [
        # transcribed from canary44 e3 07_adjusted_mortality_association
        {
            "model_id": "primary_death_model",
            "term": "aki_stage_max__is_1",
            "term_role": "exposure",
            "source_variable": "aki_stage_max",
            "estimate": 2.4130078457762196,
            "ci_low": 1.4103055441577745,
            "ci_high": 4.128613751748962,
        },
        {
            "model_id": "primary_death_model",
            "term": "aki_stage_max__is_2",
            "term_role": "exposure",
            "source_variable": "aki_stage_max",
            "estimate": 1.6696323916091684,
            "ci_low": 0.9727007221497864,
            "ci_high": 2.8659095851696894,
        },
        {
            "model_id": "primary_death_model",
            "term": "aki_stage_max__is_3",
            "term_role": "exposure",
            "source_variable": "aki_stage_max",
            "estimate": 5.162785947982867,
            "ci_low": 2.4218987023567307,
            "ci_high": 11.005562998465711,
        },
        {
            "model_id": "primary_death_model",
            "term": "age",
            "term_role": "covariate",
            "source_variable": "age",
            "estimate": 1.02,
            "ci_low": 1.01,
            "ci_high": 1.03,
        },
    ]
)

_CONTRACT = {
    "model_id": "primary_death_model",
    "converged": True,
    "fit_status": "fitted",
    "exposure_source": "aki_stage_max",
    "exposure_expression": "aki_stage_max__is_3",
}


def _row(tmp_path, *, coefficients: pd.DataFrame, contract: dict):
    path = tmp_path / "coefficients.csv"
    coefficients.to_csv(path, index=False)
    return _structured_model_row(
        spec_id="primary",
        axis="primary",
        outputs_dir=tmp_path,
        contract=contract,
        evidence_id="evidence-1",
        note_prefix="",
        coefficient_path=path,
    )


def test_a_declared_gradient_reaches_the_panel(tmp_path):
    """The whole point: three contrasts, one named headline, no refusal."""

    row, _coefficients, _contract, error = _row(
        tmp_path, coefficients=_ORDINAL_COEFFICIENTS, contract=_CONTRACT
    )

    assert error is None, error
    assert row.point_estimate == pytest.approx(5.162785947982867)
    assert row.ci_low == pytest.approx(2.4218987023567307)
    assert row.ci_high == pytest.approx(11.005562998465711)


def test_the_headline_is_the_named_contrast_not_the_first(tmp_path):
    """Naming level 1 must select level 1's coefficient, not level 3's."""

    contract = {**_CONTRACT, "exposure_expression": "aki_stage_max__is_1"}
    row, _coefficients, _contract, error = _row(
        tmp_path, coefficients=_ORDINAL_COEFFICIENTS, contract=contract
    )

    assert error is None, error
    assert row.point_estimate == pytest.approx(2.4130078457762196)


def test_a_single_exposure_term_needs_no_name(tmp_path):
    """The binary/continuous case must not start depending on the contract."""

    single = _ORDINAL_COEFFICIENTS[
        _ORDINAL_COEFFICIENTS["term"].isin(["aki_stage_max__is_3", "age"])
    ]
    contract = {**_CONTRACT, "exposure_expression": ""}
    row, _coefficients, _contract, error = _row(
        tmp_path, coefficients=single, contract=contract
    )

    assert error is None, error
    assert row.point_estimate == pytest.approx(5.162785947982867)


# ---------------------------------------------------------------------------
# What must still be refused
# ---------------------------------------------------------------------------


def test_a_gradient_with_no_named_contrast_is_still_refused(tmp_path):
    """Ambiguity fails closed; the fix must not become "pick the first"."""

    contract = {**_CONTRACT, "exposure_expression": ""}
    row, _coefficients, _contract, error = _row(
        tmp_path, coefficients=_ORDINAL_COEFFICIENTS, contract=contract
    )

    assert error is not None
    assert "one scalar primary exposure coefficient" in error
    assert "names no primary contrast" in error
    assert row.converged is False


def test_a_name_matching_no_fitted_term_is_still_refused(tmp_path):
    """A contract naming a coefficient the model never fitted is a defect."""

    contract = {**_CONTRACT, "exposure_expression": "aki_stage_max__is_9"}
    row, _coefficients, _contract, error = _row(
        tmp_path, coefficients=_ORDINAL_COEFFICIENTS, contract=contract
    )

    assert error is not None
    assert "one scalar primary exposure coefficient" in error
    assert row.converged is False


def test_the_refusal_reports_what_was_fitted_not_what_survived_the_lookup(tmp_path):
    """ "emitted 0" after a failed lookup sends the reader after a healthy model."""

    contract = {**_CONTRACT, "exposure_expression": "aki_stage_max__is_9"}
    _row_, _coefficients, _contract, error = _row(
        tmp_path, coefficients=_ORDINAL_COEFFICIENTS, contract=contract
    )

    assert "emitted 3" in error
    assert "emitted 0" not in error
    assert "'aki_stage_max__is_9'" in error


# ---------------------------------------------------------------------------
# The same shape one layer down: the row's own trace field
# ---------------------------------------------------------------------------
#
# canary45 (image dev-1e1d952) proved the headline selection above works: the
# matrix row carried `exposure_expression = aki_stage_max__is_3` and the
# "requires one scalar primary exposure coefficient" refusal appeared 0 times.
# The step still failed, one field further along:
#
#     primary: fitted sensitivity estimate lacks an unambiguous
#     model-contract trace (coefficient_term)
#
# `_matrix_model_trace` records WHICH coefficient the row used, and it picked
# that with the identical "exactly one, else None" rule -- so a gradient left
# it empty and the trace check refused a row whose coefficient is in fact
# identified, by the contract sitting in the same object.


def test_the_trace_records_the_named_contrast_for_a_gradient(tmp_path):
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        PRIMARY_SPEC_ID,
        _matrix_model_trace,
    )

    path = tmp_path / "primary_coefficients.csv"
    _ORDINAL_COEFFICIENTS.to_csv(path, index=False)

    trace = _matrix_model_trace(
        spec_id=PRIMARY_SPEC_ID,
        spec=None,
        structured_source={
            "primary_contract": {**_CONTRACT, "exposure_source": "aki_stage_max"},
            "coefficient_path": str(path),
        },
        structured_replay={},
    )

    assert trace["exposure_expression"] == "aki_stage_max__is_3"
    assert trace["coefficient_term"] == "aki_stage_max__is_3"


def test_the_trace_still_reports_nothing_when_no_contrast_is_named(tmp_path):
    """Ambiguity must keep failing the trace check, not guess."""

    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        PRIMARY_SPEC_ID,
        _matrix_model_trace,
    )

    path = tmp_path / "primary_coefficients.csv"
    _ORDINAL_COEFFICIENTS.to_csv(path, index=False)

    trace = _matrix_model_trace(
        spec_id=PRIMARY_SPEC_ID,
        spec=None,
        structured_source={
            "primary_contract": {
                **_CONTRACT,
                "exposure_source": "aki_stage_max",
                "exposure_expression": "",
            },
            "coefficient_path": str(path),
        },
        structured_replay={},
    )

    assert trace["coefficient_term"] is None


def test_a_single_term_model_still_records_its_only_term(tmp_path):
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        PRIMARY_SPEC_ID,
        _matrix_model_trace,
    )

    single = _ORDINAL_COEFFICIENTS[
        _ORDINAL_COEFFICIENTS["term"].isin(["aki_stage_max__is_3", "age"])
    ]
    path = tmp_path / "primary_coefficients.csv"
    single.to_csv(path, index=False)

    trace = _matrix_model_trace(
        spec_id=PRIMARY_SPEC_ID,
        spec=None,
        structured_source={
            "primary_contract": {
                **_CONTRACT,
                "exposure_source": "aki_stage_max",
                "exposure_expression": "",
            },
            "coefficient_path": str(path),
        },
        structured_replay={},
    )

    assert trace["coefficient_term"] == "aki_stage_max__is_3"
