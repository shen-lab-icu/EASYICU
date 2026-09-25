"""A robustness variant that is the primary analysis is documented, not counted.

A complete-case replay whose rows are exactly the rows the primary fitted
reuses the primary's estimate.  It may document that identity, but it is no
second analysis: the panel, the matrix and the robustness range must not count
it as a variant.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _find_structured_primary_model_source,
    _matrix_independent_variant,
    _verified_complete_case_equivalence,
)
from easyicu.research_agent.robustness.panel import (
    RobustnessPanel,
    RobustnessPanelRow,
)
from easyicu.research_agent.schema import RobustnessSpec
from tests.support.robustness_sources import write_structured_source_authority


def _complete_case_spec(variables: list[str]) -> RobustnessSpec:
    return RobustnessSpec(
        spec_id="complete_case",
        axis="missing",
        description="Refit the primary model on complete cases.",
        missing_override={"strategy": "complete_case", "variables": variables},
    )


def _primary_source(tmp_path: Path, *, model_n: int):
    run_dir = tmp_path / "run"
    record, evidence, _script = write_structured_source_authority(
        run_dir,
        coefficient_filename="adjusted_association_coefficients.csv",
        include_primary_model_id=False,
        include_headline=False,
        model_n=model_n,
        analysis_covariates=["age"],
    )
    source = _find_structured_primary_model_source(
        records=[record], run_dir=run_dir, evidence_records=evidence
    )
    assert source is not None
    return source


def test_a_complete_case_set_equal_to_the_primary_is_not_an_independent_variant(
    tmp_path: Path,
) -> None:
    source = _primary_source(tmp_path, model_n=4)
    fitted_rows = pd.DataFrame(
        {
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 1, 0, 1],
            "age": [50.0, 60.0, 70.0, 80.0],
        }
    )

    row, _coefficients, contract, error = _verified_complete_case_equivalence(
        spec=_complete_case_spec(["exposure", "outcome", "age"]),
        source=source,
        primary_data=fitted_rows,
    )

    assert error is None and contract is not None
    # The row still carries the primary's verified estimate as documentation.
    assert row.converged is True and row.n == 4
    assert row.independent_variant is False


def test_the_matrix_counts_a_documented_identity_as_no_variant() -> None:
    spec = _complete_case_spec(["exposure", "outcome", "age"])
    identical = RobustnessPanelRow(
        spec_id="complete_case", axis="missing", n=4,
        point_estimate=1.4, ci_low=1.1, ci_high=1.8, se=0.1,
        evidence_id="table_coefficients", converged=True,
        independent_variant=False,
    )
    refit = RobustnessPanelRow(
        spec_id="complete_case", axis="missing", n=3,
        point_estimate=1.5, ci_low=1.1, ci_high=2.0, se=0.1,
        evidence_id="table_variant", converged=True,
    )

    assert _matrix_independent_variant(spec=spec, row=identical, outcome_audit={}) is False
    # A genuine refit keeps the matrix's previous, unstated value.
    assert _matrix_independent_variant(spec=spec, row=refit, outcome_audit={}) is None
    # The outcome audit still decides for outcome variants.
    assert _matrix_independent_variant(
        spec=spec, row=refit, outcome_audit={"independent_variant": False}
    ) is False
    assert _matrix_independent_variant(
        spec=spec, row=identical, outcome_audit={"independent_variant": True}
    ) is True
    # The primary row is no variant of anything.
    assert _matrix_independent_variant(spec=None, row=identical, outcome_audit={}) is None


def test_the_panel_counts_no_variant_when_only_the_primary_is_documented() -> None:
    primary = RobustnessPanelRow(
        spec_id="primary", axis="primary", n=4,
        point_estimate=1.4, ci_low=1.1, ci_high=1.8, se=0.1,
        evidence_id="table_coefficients", converged=True,
    )
    identical = RobustnessPanelRow(
        spec_id="complete_case", axis="missing", n=4,
        point_estimate=1.4, ci_low=1.1, ci_high=1.8, se=0.1,
        evidence_id="table_coefficients", converged=True,
        independent_variant=False,
    )

    panel = RobustnessPanel.from_rows([primary, identical], primary_spec_id="primary")

    assert panel.n_variants == 0


def _matrix(*variants: dict) -> pd.DataFrame:
    primary = {
        "spec_id": "primary", "axis": "primary", "effect_scale": "odds_ratio",
        "point_estimate": 1.4, "ci_low": 1.1, "ci_high": 1.8, "converged": True,
        "independent_variant": None,
    }
    return pd.DataFrame([primary, *variants])


def test_a_grid_without_an_independent_estimate_carries_no_robustness_evidence() -> None:
    from easyicu.research_agent.execution.runners.robustness_figure_executor import (
        _validated_rows,
        _variant_evidence_shown,
    )

    # A restatement keeps the primary's fit, as fitted; only its flag differs.
    identity = {
        "spec_id": "complete_case", "axis": "missing", "effect_scale": "odds_ratio",
        "point_estimate": 1.4, "ci_low": 1.1, "ci_high": 1.8, "converged": True,
        "independent_variant": False,
    }
    refit = {
        "spec_id": "first_stay", "axis": "cohort", "effect_scale": "odds_ratio",
        "point_estimate": 1.3, "ci_low": 1.0, "ci_high": 1.7, "converged": True,
        "independent_variant": None,
    }
    gap = {**refit, "spec_id": "outcome_alt", "axis": "outcome",
           "point_estimate": None, "ci_low": None, "ci_high": None, "converged": False}

    def shown(frame: pd.DataFrame) -> bool:
        rows, _scale, _gap = _validated_rows(frame)
        return _variant_evidence_shown(rows)

    assert shown(_matrix(identity)) is False
    assert shown(_matrix(gap)) is False
    assert shown(_matrix(identity, refit)) is True


def test_a_panel_declaring_no_evidence_for_its_role_does_not_cover_it() -> None:
    from easyicu.research_agent.planning.figure_strategy import (
        FigureRoleStrategy,
        _role_matches_panel,
    )

    role = FigureRoleStrategy(
        role="robustness", required=True, placement="main",
        rationale="Show sensitivity across specifications.",
        acceptable_chart_types=["specification_grid"],
        required_text_terms=[], search_terms=["robustness", "sensitivity"],
    )
    panel = {
        "panel_id": "a", "title": "Robustness of the primary estimate",
        "role": "robustness",
        "metadata": {"article_role": "robustness", "chart_type": "specification_grid"},
    }
    empty = {**panel, "metadata": {**panel["metadata"], "role_evidence_absent": True}}

    assert _role_matches_panel(role, panel) is True
    assert _role_matches_panel(role, empty) is False


def test_the_summary_range_spans_only_rows_that_vary_the_primary() -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _robustness_summary,
    )

    matrix = pd.DataFrame(
        [
            {"axis": "missing", "converged": True, "independent_variant": False,
             "ci_low": 1.1, "ci_high": 1.8},
            {"axis": "cohort", "converged": True, "independent_variant": None,
             "ci_low": 1.0, "ci_high": 2.1},
        ]
    )

    summary = _robustness_summary(matrix).set_index("axis")

    # The restatement converged and is counted as such, never as a range.
    assert summary.loc["missing", "converged_specs"] == 1
    assert summary.loc["missing", "non_independent_specs"] == 1
    assert pd.isna(summary.loc["missing", "range_low"])
    assert summary.loc["cohort", "range_low"] == 1.0
