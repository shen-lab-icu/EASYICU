"""A dose-response exposure is N contrasts, and only the plan says which is the one.

E3 asks for "the gradient of first-24h KDIGO AKI stage against mortality". That
is three contrasts against stage 0, not one number, and until 2026-07-31 the
host's association owner could only write one row -- so the paper's primary
result for every ordinal exposure fell to the Coder, and the robustness replay
downstream then blocked for want of "a completed primary estimate".

Collapsing four stages into a single term is not a rougher answer. It reports a
per-unit trend under the name of a stage comparison: a different scientific
quantity carrying the declared estimand's label. That is what these tests are
guarding.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from easyicu.research_agent.execution.output_files import bind_primary_output
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
    AdjustedAssociationError,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.schema import PlannedModelRequirement

pd = pytest.importorskip("pandas")

_LEVELS = ["0", "1", "2", "3"]


def _ordinal_cohort(n: int = 600):
    """A four-stage exposure with a real monotone gradient, plus covariates.

    Built rather than sampled so the direction is known: the risk rises with
    stage, so a test that silently fitted the wrong contrast would still have
    to produce the wrong number to pass.
    """

    import numpy as np

    rng = np.random.default_rng(20260731)
    stage = rng.integers(0, 4, size=n).astype(float)
    age = rng.normal(65, 12, size=n)
    logit = -3.0 + 0.9 * stage + 0.01 * (age - 65)
    death = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"aki_stage": stage, "death": death, "age": age})


def _run(tmp_path: Path, monkeypatch, **overrides):
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path))
    payload = dict(
        requirement_id="primary_death_by_stage",
        exposure="aki_stage",
        outcome="death",
        covariates=["age"],
        estimator_kind="logistic",
        analysis_set="complete_case",
        analysis_role="primary",
        method_family="logistic_regression",
        exposure_levels=_LEVELS,
        exposure_reference_level="0",
        primary_contrast_level="3",
        frame=_ordinal_cohort(),
        emit_step_summary=False,
    )
    payload.update(overrides)
    run_adjusted_association_from_env(**payload)
    with (tmp_path / "adjusted_association_estimates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# what the host now computes


def test_a_four_level_exposure_yields_one_row_per_contrast(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)

    assert [row["contrast"] for row in rows] == ["1 vs 0", "2 vs 0", "3 vs 0"]
    assert all(row["reference_level"] == "0" for row in rows)
    assert all(row["exposure"] == "aki_stage" for row in rows)
    # Every contrast carries its own estimate and interval; none is a placeholder.
    for row in rows:
        low, estimate, high = (
            float(row["ci_low"]),
            float(row["estimate"]),
            float(row["ci_high"]),
        )
        assert low <= estimate <= high
    # The gradient is monotone by construction, so the fit has to see it.
    estimates = [float(row["estimate"]) for row in rows]
    assert estimates[0] < estimates[-1]


def test_exactly_one_contrast_is_marked_and_it_is_the_declared_one(
    tmp_path, monkeypatch
):
    rows = _run(tmp_path, monkeypatch, primary_contrast_level="2")

    marked = [row for row in rows if row["is_primary_contrast"] == "True"]
    assert len(marked) == 1
    assert marked[0]["contrast"] == "2 vs 0"
    # ...and the mark moves with the declaration rather than with row order or
    # with which estimate happens to be largest.
    assert rows[-1]["contrast"] == "3 vs 0"
    assert float(rows[-1]["estimate"]) > float(marked[0]["estimate"])


def test_the_headline_binding_reads_the_mark_not_the_row_order(tmp_path, monkeypatch):
    """The number the manuscript quotes comes from the declared contrast.

    Before the mark existed, bind_primary_output required exactly one row and
    silently bound nothing when it saw more -- so an ordinal primary result
    produced a paper with no headline estimate at all, and the robustness
    replay downstream blocked for want of one.
    """

    rows = _run(tmp_path, monkeypatch, primary_contrast_level="1")
    declared = next(row for row in rows if row["contrast"] == "1 vs 0")

    payload = bind_primary_output(
        {
            "output_files": {
                "table:adjusted_association_estimates": (
                    "adjusted_association_estimates.csv"
                )
            }
        },
        tmp_path,
    )
    assert payload["primary_or"] == pytest.approx(float(declared["estimate"]))
    assert payload["primary_association_term"] == "aki_stage"


def test_a_binary_exposure_still_writes_exactly_one_row(tmp_path, monkeypatch):
    """The single-contrast path is untouched, and now says so in its columns."""

    frame = _ordinal_cohort()
    frame["aki_stage"] = (frame["aki_stage"] >= 2).astype(float)
    rows = _run(
        tmp_path,
        monkeypatch,
        frame=frame,
        exposure_levels=None,
        exposure_reference_level=None,
        primary_contrast_level=None,
    )
    assert len(rows) == 1
    assert rows[0]["exposure_level"] == ""
    assert rows[0]["reference_level"] == ""
    # Still the headline, so every consumer reads the mark uniformly.
    assert rows[0]["is_primary_contrast"] == "True"


def test_the_emitted_header_is_the_declared_contract(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)
    assert list(rows[0]) == list(ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS)


# --------------------------------------------------------------------------
# what it refuses, and why each refusal is not just strictness


def test_a_level_the_cohort_does_not_have_is_refused(tmp_path, monkeypatch):
    """A declared level nobody has cannot be estimated, so it is not a gradient.

    Writing the other contrasts and staying quiet about the missing one would
    show a reader three stages where the plan pre-specified four.
    """

    with pytest.raises(AdjustedAssociationError, match="no stay has"):
        _run(tmp_path, monkeypatch, exposure_levels=["0", "1", "2", "3", "4"])


def test_a_level_the_plan_never_declared_is_refused(tmp_path, monkeypatch):
    """The analysed population and the pre-specified level set must agree."""

    # The headline stays inside the declared set, so the ONLY defect is the
    # stage-3 stays the plan never mentioned. An earlier draft also moved the
    # headline out of range, and the reference/headline check fired first --
    # the test passed without ever reaching the rule it names.
    with pytest.raises(AdjustedAssociationError, match="never\\s+declared"):
        _run(
            tmp_path,
            monkeypatch,
            exposure_levels=["0", "1", "2"],
            primary_contrast_level="2",
        )


def test_a_half_declared_level_set_is_refused_rather_than_guessed(
    tmp_path, monkeypatch
):
    """Which contrast is the headline is a scientific choice, so it is required.

    This is deliberately the opposite call from the robustness replay spec,
    where a partial declaration falls through to an equally correct path. Here
    the fallback fits the four stages as one linear term, which answers a
    different question under the declared name.
    """

    with pytest.raises(AdjustedAssociationError, match="together"):
        _run(tmp_path, monkeypatch, primary_contrast_level=None)


def test_the_primary_contrast_cannot_be_the_reference(tmp_path, monkeypatch):
    with pytest.raises(AdjustedAssociationError, match="not be the reference"):
        _run(tmp_path, monkeypatch, primary_contrast_level="0")


# --------------------------------------------------------------------------
# the plan-side contract


def _requirement(**overrides) -> PlannedModelRequirement:
    payload = {
        "requirement_id": "primary",
        "outcome": "death",
        "outcome_type": "binary",
        "method_family": "logistic_regression",
        "exposure_source": "aki_stage",
        "analysis_role": "primary",
        "analysis_set": "complete_case",
        "covariates": ["age"],
        "exposure_levels": _LEVELS,
        "exposure_reference_level": "0",
        "primary_contrast_level": "3",
    }
    payload.update(overrides)
    return PlannedModelRequirement.model_validate(payload)


def test_a_complete_level_declaration_validates() -> None:
    requirement = _requirement()
    assert requirement.exposure_levels == _LEVELS
    assert requirement.primary_contrast_level == "3"


def test_a_binary_or_continuous_requirement_needs_none_of_the_three() -> None:
    requirement = _requirement(
        exposure_levels=None,
        exposure_reference_level=None,
        primary_contrast_level=None,
    )
    assert requirement.exposure_levels is None


@pytest.mark.parametrize(
    "dropped",
    ["exposure_levels", "exposure_reference_level", "primary_contrast_level"],
)
def test_any_one_of_the_three_missing_is_refused_at_plan_time(dropped: str) -> None:
    """Refused where the Planner can still fix it, not at execution."""

    with pytest.raises(ValueError, match="missing"):
        _requirement(**{dropped: None})


def test_the_reference_and_the_headline_must_be_declared_levels() -> None:
    with pytest.raises(ValueError, match="not one of the declared"):
        _requirement(exposure_reference_level="9")
    with pytest.raises(ValueError, match="not one of the declared"):
        _requirement(primary_contrast_level="9")


def test_a_level_set_with_one_level_cannot_carry_a_contrast() -> None:
    with pytest.raises(ValueError, match="at least two levels"):
        _requirement(
            exposure_levels=["0"],
            exposure_reference_level="0",
            primary_contrast_level="0",
        )


def test_a_contrast_the_fit_could_not_estimate_is_refused_not_left_blank() -> None:
    """A gradient with a hole is a different gradient, not a rougher one.

    A stage so sparse that the fit returns no interval for it must not be
    written as a blank row: a reader comparing stages would see the other
    contrasts and have no way to tell that this one was never estimated. The
    guard is unit-tested because producing the condition from data means
    engineering a near-separated fit, which would test statsmodels rather than
    this rule.
    """

    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        _contrast_rows,
        _DeclaredContrasts,
        _contrast_column,
    )
    from easyicu.research_agent.robustness.estimators import EstimatorTerm

    contrasts = _DeclaredContrasts(levels=("0", "1", "2"), reference="0", primary="2")
    fitted = EstimatorTerm(
        term=_contrast_column("aki_stage", "1"),
        source_variable=_contrast_column("aki_stage", "1"),
        estimate=2.0,
        ci_low=1.1,
        ci_high=3.6,
        se=0.3,
    )
    unusable = EstimatorTerm(
        term=_contrast_column("aki_stage", "2"),
        source_variable=_contrast_column("aki_stage", "2"),
        estimate=None,
        ci_low=None,
        ci_high=None,
        se=None,
    )

    with pytest.raises(AdjustedAssociationError, match="missing one of its levels"):
        _contrast_rows(
            (fitted, unusable),
            shared={},
            exposure="aki_stage",
            contrasts=contrasts,
            requirement_id="primary",
        )

    # The same call with both contrasts estimable is what success looks like,
    # so the refusal above is about the missing estimate and nothing else.
    repaired = EstimatorTerm(
        term=_contrast_column("aki_stage", "2"),
        source_variable=_contrast_column("aki_stage", "2"),
        estimate=4.0,
        ci_low=2.0,
        ci_high=8.0,
        se=0.4,
    )
    rows = _contrast_rows(
        (fitted, repaired),
        shared={},
        exposure="aki_stage",
        contrasts=contrasts,
        requirement_id="primary",
    )
    assert [row["contrast"] for row in rows] == ["1 vs 0", "2 vs 0"]
    assert [row["is_primary_contrast"] for row in rows] == [False, True]


def test_the_planner_prompt_offers_the_capability() -> None:
    """A field the directive never mentions is a field no plan will fill.

    The host gained the ability to fit a dose-response gradient on 2026-07-31;
    the Planner only uses it if it is told, and this is what makes the whole
    change reachable rather than dead code.
    """

    from easyicu.research_agent.agents.core import _build_planner_user_prompt
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
    )

    prompt = _build_planner_user_prompt(
        ResearchContext(
            research_question="Does severity stage grade the risk of death?",
            cohort=CohortDescriptor(
                cohort_name="synthetic",
                database="synthetic",
                n_patients=6,
                n_stays=6,
            ),
            variables=[
                ConceptDescriptor(name="stage", role="ordinal_score", dtype="float64"),
                ConceptDescriptor(name="death", role="outcome", dtype="int64"),
            ],
            primary_exposure="stage",
            target_outcome="death",
        )
    )

    assert "ORDINAL OR CATEGORICAL EXPOSURE" in prompt
    assert "`exposure_levels`" in prompt
    assert "`primary_contrast_level`" in prompt
    assert "All three together or none" in prompt


# ---------------------------------------------------------------------------
# canary13: the first run that ever reached this owner with a real plan
#
# The host's own primary-model contract refused the step it had just computed,
# with five issues. Four were one cause and the fifth was a real scientific
# defect that nothing else would have caught.
# ---------------------------------------------------------------------------


def test_a_contrast_term_reports_the_original_column_as_its_source() -> None:
    """``<exposure>__is_<level>`` is a design column; no cohort carries it.

    The primary-model contract reads ``source_variable`` as "the unique
    original authoritative cohort column" and explicitly allows ``term`` to be
    "an encoded or transformed design column". Reporting the indicator as its
    own source made every contrast unresolvable -- three issues for three
    stages, on a step whose numbers were right.
    """

    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        _coefficient_rows,
    )

    class _Term:
        def __init__(self, term: str, source: str) -> None:
            self.term = term
            self.source_variable = source
            self.estimate = 1.0
            self.ci_low = 0.5
            self.ci_high = 2.0
            self.se = 0.1

    rows = _coefficient_rows(
        [
            _Term("const", "const"),
            _Term("stage__is_1", "stage__is_1"),
            _Term("stage__is_3", "stage__is_3"),
            _Term("age", "age"),
        ],
        model_id="m",
        exposure="stage",
        adjustment=["age"],
        effect_scale="odds_ratio",
        exposure_contrast_columns=["stage__is_1", "stage__is_3"],
    )

    by_term = {row["term"]: row for row in rows}
    assert by_term["stage__is_1"]["source_variable"] == "stage"
    assert by_term["stage__is_3"]["source_variable"] == "stage"
    # The term keeps the design column: the contract asks for the pair, and
    # collapsing the term too would lose which contrast the row is.
    assert by_term["stage__is_1"]["term"] == "stage__is_1"
    assert by_term["age"]["source_variable"] == "age"
    assert by_term["const"]["term_role"] == "intercept"


def test_an_unobserved_exposure_is_not_pooled_into_the_reference(
    tmp_path, monkeypatch
) -> None:
    """Treatment coding encodes a missing value exactly like the reference.

    Every indicator is 0 for a row with no exposure, which is bit-for-bit the
    reference encoding -- so the model answers "stage 0" for a patient whose
    stage nobody recorded. canary13 had 8 such stays in 1000, and only the
    host's own denominator recomputation (992, not 1000) ever made it visible.

    This is a scientific property, not a formatting one: the assertion is on
    the analysed row count AND on the reference group's own size, because a
    test that checked only ``n`` would pass if the rows were dropped from the
    wrong group.
    """

    import numpy as np

    frame = _ordinal_cohort(n=600)
    complete = _run(tmp_path, monkeypatch, frame=frame.copy())
    assert {int(row["n"]) for row in complete} == {600}

    frame.loc[frame.index[:9], "aki_stage"] = np.nan
    rows = _run(tmp_path, monkeypatch, frame=frame)

    # Nine rows lost their exposure, so nine rows leave the analysis. Pooling
    # them into the reference would keep n at 600 and shift every contrast.
    assert {int(row["n"]) for row in rows} == {591}
    assert [row["contrast"] for row in rows] == ["1 vs 0", "2 vs 0", "3 vs 0"]

    # ...and the estimates really move, so this cannot pass by the rows having
    # been dropped somewhere harmless.
    before = {row["contrast"]: float(row["estimate"]) for row in complete}
    after = {row["contrast"]: float(row["estimate"]) for row in rows}
    assert any(before[key] != after[key] for key in before)


def test_every_contrast_indicator_is_missing_where_the_exposure_is() -> None:
    """The design matrix must not assert a level for a row that has none.

    Checking the analysed row count is not enough: complete-case drops a row on
    a NaN in ANY column, so masking one indicator and leaving the others at 0
    produces the same ``n`` while the unmasked columns still say "this row is
    not stage 2" about a row whose stage is unknown. That mutant passed the
    count test unchanged. The statement each indicator makes has to be true on
    its own, because the design is a value the fit and every later reader see.
    """

    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        _DeclaredContrasts,
        _contrast_column,
        _contrast_design,
    )

    import numpy as np

    frame = pd.DataFrame(
        {
            "aki_stage": [0.0, 1.0, np.nan, 3.0],
            "age": [60.0, 70.0, 80.0, 65.0],
        }
    )
    contrasts = _DeclaredContrasts(
        levels=("0", "1", "2", "3"), reference="0", primary="3"
    )

    design, focal = _contrast_design(
        frame, exposure="aki_stage", adjustment=["age"], contrasts=contrasts
    )

    assert focal == _contrast_column("aki_stage", "3")
    unobserved = 2
    for level in ("1", "2", "3"):
        column = design[_contrast_column("aki_stage", level)]
        assert pd.isna(column.iloc[unobserved]), level
        # ...and the observed rows keep their exact indicator, so masking has
        # not blurred the rows that do have a stage.
        assert column.iloc[0] == 0.0
    assert design[_contrast_column("aki_stage", "1")].iloc[1] == 1.0
    assert design[_contrast_column("aki_stage", "3")].iloc[3] == 1.0
    # The adjustment column is untouched: only the exposure was unobserved.
    assert not design["age"].isna().any()
