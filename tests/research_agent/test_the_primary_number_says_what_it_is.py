"""The manuscript's primary number must be able to say what it is.

The writer evidence digest is the ONLY channel through which numbers reach the
Writer, and the Writer is forbidden from stating facts that are not in it.  The
digest is built from ``WRITER_DIGEST_PREFERRED_KEYS`` (89 entries, every one a
quantity) plus ``evidence.numeric_claims()`` (numbers by construction).  An
effect SCALE is not a quantity, so it had no path at all.

MEASURED over every writer digest on disk, 2026-08-03: 17 of 17 carry no effect
scale; 15 of those 17 runs have it in their own step summary.  The cost is one
sentence in every manuscript the system has ever written -- here, e3's::

    In the canonical adjusted model, the primary stage-based estimate was
    6.47782, with a 95% confidence interval from 6.02368 to 6.96620.

6.48 of what?  The producing step's evidence says ``effect_scale=odds_ratio``,
``exposure_expression=aki_stage_max__is_3``, ``covariates=[age, sex]``.  None of
it reached the sentence, so a clinician cannot interpret the result and a
journal cannot publish it.

The chain existed and was never walked: the panel's primary row already carries
the ``evidence_id`` of the step that produced the estimate.  MEASURED over 161
recorded runs with a panel: 73 resolve end to end (all odds_ratio), 16 have no
``evidence_id`` on the primary row and are TOLD SO rather than guessed at, and
72 publish no primary row at all and are untouched.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.reporting.writer_evidence import (  # noqa: E402
    WRITER_DIGEST_PREFERRED_KEYS,
    _primary_effect_interpretation_lines,
    _render_robustness_panel_block,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")

#: The run whose manuscript reported a bare 6.47782.
_E3_RUN = (
    _CORPUS
    / "batch_20260802_luna_miiv_FULL_30f431a_verify03"
    / "e3_kdigo_gradient"
    / "aware"
    / "run_20260803T041243_66449a"
)


def test_the_preferred_key_list_cannot_carry_a_scale() -> None:
    """Why a new line was needed instead of one more allowlist entry.

    Every preferred key names a quantity.  Adding ``effect_scale`` to a list
    consumed by ``_first_present_scalar`` and by a numeric-claim registry would
    put a string where numbers are expected; the interpretation belongs beside
    the estimate it explains, not in the quantity channel.
    """

    assert "effect_scale" not in WRITER_DIGEST_PREFERRED_KEYS
    assert "primary_estimate_label" not in WRITER_DIGEST_PREFERRED_KEYS
    assert "exposure_expression" not in WRITER_DIGEST_PREFERRED_KEYS


def test_a_row_with_no_source_evidence_is_told_so_not_guessed() -> None:
    """Ambiguity fails closed: 16 recorded panels land here.

    Inventing a scale would be worse than omitting one, so the digest says the
    scale is unavailable AND instructs the Writer not to describe units.
    """

    lines = _primary_effect_interpretation_lines(
        run_dir=pathlib.Path("/nonexistent"), evidence_id=None
    )
    assert len(lines) == 1
    assert "UNAVAILABLE" in lines[0]
    assert "Do not describe the estimate's units or contrast." in lines[0]


def test_unreadable_source_evidence_is_told_so(tmp_path: pathlib.Path) -> None:
    """A named evidence id that cannot be read must not silently vanish."""

    lines = _primary_effect_interpretation_lines(
        run_dir=tmp_path, evidence_id="statistic_step_summary_missing"
    )
    assert len(lines) == 1
    assert "UNAVAILABLE" in lines[0]
    assert "statistic_step_summary_missing" in lines[0]


def test_a_step_that_declared_no_scale_is_reported_not_invented(
    tmp_path: pathlib.Path,
) -> None:
    """The producing step is the authority; silence there stays silence here."""

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "ev1__step_summary.json").write_text(
        json.dumps({"n_total": 100, "exposure": "x"}), encoding="utf-8"
    )
    lines = _primary_effect_interpretation_lines(run_dir=tmp_path, evidence_id="ev1")
    assert len(lines) == 1
    assert "declared no effect scale" in lines[0]


def test_every_declared_fact_reaches_the_writer(tmp_path: pathlib.Path) -> None:
    """Scale, contrast, outcome, adjustment set, estimator and model identity."""

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "ev1__step_summary.json").write_text(
        json.dumps(
            {
                "effect_scale": "hazard_ratio",
                "outcome": "death",
                "covariates": ["age", "sex"],
                "estimator_kind": "cox",
                "model_contracts": [
                    {
                        "exposure_expression": "vent_mode__is_invasive",
                        "outcome": "death",
                        "method_family": "cox_proportional_hazards",
                        "model_id": "primary_survival",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    lines = _primary_effect_interpretation_lines(run_dir=tmp_path, evidence_id="ev1")
    interpretation = lines[0]
    assert "effect_scale=hazard_ratio" in interpretation
    assert "exposure_contrast=vent_mode__is_invasive" in interpretation
    assert "outcome=death" in interpretation
    assert "adjusted_for=age,sex" in interpretation
    assert "estimator=cox_proportional_hazards" in interpretation
    assert "model_id=primary_survival" in interpretation
    # The instruction matters as much as the facts: without it the Writer may
    # keep the sentence it already knows how to write.
    assert any("State the scale and the contrast" in line for line in lines)


def test_a_ph_violation_withholds_the_constant_cox_headline(
    tmp_path: pathlib.Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "ev1__step_summary.json").write_text(
        json.dumps(
            {
                "effect_measure": "hazard_ratio",
                "proportional_hazards_status": ("violation_block_paper_authorization"),
                "paper_authorization_allowed": False,
                "hazard_ratio": 0.52,
            }
        ),
        encoding="utf-8",
    )
    lines = _primary_effect_interpretation_lines(run_dir=tmp_path, evidence_id="ev1")
    assert len(lines) == 1
    assert "HEADLINE EFFECT UNAUTHORIZED" in lines[0]
    assert "Do not report the Cox point estimate" in lines[0]

    (evidence / "ev2__step_summary.json").write_text(
        json.dumps(
            {
                "effect_measure": "hazard_ratio",
                "proportional_hazards_status": "not_rejected",
                "paper_authorization_allowed": False,
                "hazard_ratio": 0.52,
            }
        ),
        encoding="utf-8",
    )
    nonviolation_lines = _primary_effect_interpretation_lines(
        run_dir=tmp_path,
        evidence_id="ev2",
    )
    assert all("PH policy rejected" not in line for line in nonviolation_lines)


# --------------------------------------------------------------------------
# Against the real run whose manuscript carried the bare number
# --------------------------------------------------------------------------


def test_the_real_run_now_publishes_what_its_number_means() -> None:
    """Drives ``_render_robustness_panel_block``, the function production calls.

    Uses the sealed artifacts of the run that produced the bare-6.47782
    sentence, so this fails if the chain from panel row to producing step is
    broken anywhere along its length.
    """

    if not _E3_RUN.exists():
        pytest.skip("the e3 run that recorded this manuscript is not on disk")

    lines = _render_robustness_panel_block(run_dir=_E3_RUN)
    joined = "\n".join(lines)

    # The estimate is still published exactly as before -- this patch adds, it
    # does not restate. Precision is deliberately unchanged: the numeric binder
    # matches the Writer's literal back to the claim ledger.
    assert "point=6.47782" in joined
    assert "CI=[6.02368, 6.9662]" in joined

    interpretation = next(
        line for line in lines if line.startswith("primary interpretation:")
    )
    assert "UNAVAILABLE" not in interpretation
    assert "effect_scale=odds_ratio" in interpretation
    assert "exposure_contrast=aki_stage_max__is_3" in interpretation
    assert "adjusted_for=age,sex" in interpretation


def test_the_chain_resolves_across_the_recorded_corpus() -> None:
    """Read off the corpus, not restated from it.

    73 resolve, 16 fail closed, and nothing in between: a partially-resolved
    interpretation would mean the digest is publishing a scale it inferred.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    resolved = 0
    refused = 0
    for panel_path in _CORPUS.rglob("robustness_panel.json"):
        lines = _render_robustness_panel_block(run_dir=panel_path.parent)
        interpretation = next(
            (line for line in lines if line.startswith("primary interpretation:")),
            None,
        )
        if interpretation is None:
            continue
        if "UNAVAILABLE" in interpretation:
            refused += 1
        else:
            resolved += 1
            # Every resolved line must carry the scale; anything else means the
            # renderer emitted an interpretation it could not substantiate.
            assert "effect_scale=" in interpretation

    if not resolved and not refused:
        pytest.skip("no recorded panel publishes a primary row")
    assert resolved, "the corpus must still contain runs this fix serves"
    assert refused, "the fail-closed branch must still be exercised by the corpus"


def test_a_panel_without_a_primary_row_gains_nothing() -> None:
    """No primary estimate, no interpretation, no invented line.

    72 of the 161 recorded panels are in this state. They must be byte-identical
    to before, which is why the interpretation is appended inside the
    ``primary is not None`` branch rather than to the block.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    seen_without_primary = 0
    for panel_path in _CORPUS.rglob("robustness_panel.json"):
        lines = _render_robustness_panel_block(run_dir=panel_path.parent)
        if any(line.startswith("primary: ") for line in lines):
            continue
        seen_without_primary += 1
        assert not any(line.startswith("primary interpretation:") for line in lines), (
            panel_path
        )

    if not seen_without_primary:
        pytest.skip("every recorded panel publishes a primary row")


def test_executed_typed_robustness_supersedes_empty_legacy_panel(
    tmp_path: pathlib.Path,
) -> None:
    records = [
        {
            "step_id": "robustness_projection",
            "status": "ok",
            "step_summary_evidence_id": "statistic_robustness_summary",
            "step_summary": {
                "analysis_family": "robustness_sensitivity",
                "n_converged_variants": 3,
                "primary_effect_label": "upper boundary contrast",
                "primary_effect_scale": "odds_ratio",
                "primary_effect_is_nonlinear_curve_summary": False,
                "limitations": ["The missing-data row is not an independent refit."],
                "robustness_rows": [
                    {
                        "axis": "primary",
                        "converged": True,
                        "independent_variant": True,
                        "point_estimate": 1.96,
                        "ci_low": 1.89,
                        "ci_high": 2.03,
                        "evidence_id": "table_primary_contrast",
                    },
                    {
                        "axis": "functional_form",
                        "converged": True,
                        "independent_variant": True,
                    },
                    {
                        "axis": "missing",
                        "converged": True,
                        "independent_variant": False,
                    },
                ],
            },
        }
    ]

    lines = _render_robustness_panel_block(run_dir=tmp_path, records=records)
    joined = "\n".join(lines)

    assert "EXECUTED ROBUSTNESS AUTHORITY" in joined
    assert "n_converged=3" in joined
    assert "n_independent=2" in joined
    assert "not a summary of the entire nonlinear curve" in joined
    assert "no robustness variants converged" not in joined
