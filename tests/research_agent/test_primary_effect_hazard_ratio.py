"""Primary estimand == manuscript headline for survival (hazard-ratio) designs.

The deterministic Cox runner emits ``hazard_ratio`` (+ CIs), but the primary-
effect extractor historically keyed only on OR-family names, so a survival
estimand was dropped and a logistic OR was refit downstream — burying the real
HR (H1 fix3j: HR 1.82 buried under a near-null ~1.0). These tests lock:

1. the extractor recognises a hazard-ratio-shaped primary effect and labels it
   ``effect_measure="HR"`` (OR designs keep ``"OR"``);
2. the robustness-panel primary row carries the HR, not a refit;
3. the panel does NOT append incompatible logistic-OR variants to a hazard-ratio
   primary (which would fabricate a misleading mixed-measure robustness range).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.robustness.estimators import (
    _primary_effect_measure_from_records,
    _primary_row_from_step_records,
    fit_robustness_rows_from_records,
)
from easyicu.research_agent.robustness.primary_effect import (
    _effect_measure_from_scale,
    _extract_primary_effect_payload_from_records,
    _infer_primary_predictor_from_run_dir,
)
from easyicu.research_agent.robustness.panel import default_robustness_specs
from tests.research_agent.test_research_context_v2_authority_join import (
    _prepare_typed_run,
)


def _host_role_record(record: dict, role: str = "primary") -> dict:
    """Bind the planner role in both host-owned step-record locations."""

    return {
        **record,
        "planned_analysis_role": role,
        "analysis_request": {
            "step": {
                "step_id": record["step_id"],
                "planned_analysis_role": role,
            }
        },
    }


def _survival_records():
    return [
        _host_role_record(
            {
                "step_id": "01_survival_analysis",
                "status": "ok",
                "step_summary_evidence_id": "cox_primary",
                "step_summary": {
                    "analysis_family": "time_to_event",
                    "primary_predictor": "vent_24h_any",
                    "hazard_ratio": 1.82,
                    "hazard_ratio_ci_low": 1.74,
                    "hazard_ratio_ci_high": 1.91,
                    "hazard_ratio_p_value": 4.1e-137,
                    "n_analysis": 74454,
                    "n": 74454,
                },
            }
        )
    ]


def _logistic_frame(n: int = 800, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, 0.5, n)
    p = 1 / (1 + np.exp(-(-0.5 + np.log(1.8) * x)))
    y = rng.binomial(1, p)
    return pd.DataFrame({"x": x, "y": y})


def test_primary_predictor_loads_from_v2_research_context(tmp_path: Path) -> None:
    run_dir, _cohort_path, context, _identity, _cohort, _trajectory = (
        _prepare_typed_run(tmp_path)
    )
    (run_dir / "research_context.json").write_text(
        context.model_dump_json(),
        encoding="utf-8",
    )

    assert _infer_primary_predictor_from_run_dir(run_dir) == "lact_max"


# --- extractor -------------------------------------------------------------


def test_extractor_recognizes_hazard_ratio_as_primary_effect():
    payload = _extract_primary_effect_payload_from_records(_survival_records())
    assert payload is not None
    assert round(payload["primary_or"], 2) == 1.82  # canonical field, HR value
    assert round(payload["primary_ci_low"], 2) == 1.74
    assert round(payload["primary_ci_high"], 2) == 1.91
    assert payload["effect_measure"] == "HR"


def test_effect_measure_from_scale_maps_declared_scales():
    assert _effect_measure_from_scale("odds_ratio") == "OR"
    assert _effect_measure_from_scale("hazard_ratio") == "HR"
    assert _effect_measure_from_scale("risk_ratio") == "RR"
    assert _effect_measure_from_scale("risk_difference") == "RD"
    assert _effect_measure_from_scale("") is None
    assert _effect_measure_from_scale(None) is None
    # unrecognised scale must not be bound as if it were an OR
    assert _effect_measure_from_scale("weird_scale") is None


def test_extractor_recognizes_scale_neutral_adjusted_effect_over_probe():
    """Reproduces H2 fix3: the PS-weighted causal step writes the OR under
    ``adjusted_effect`` + ``primary_effect_scale`` (scale in a separate field). The
    extractor must pick THAT step, not a probe scalar it happened to match."""
    records = [
        _host_role_record(
            {
                "step_id": "00_probe",
                "status": "ok",
                "step_summary_evidence_id": "probe_stat",
                "step_summary": {"estimate": 28.0},
            },
            role="auxiliary",
        ),
        _host_role_record(
            {
                "step_id": "01_causal_effect_estimation",
                "status": "ok",
                "step_summary_evidence_id": "causal_stat",
                "step_summary": {
                    "primary_predictor": "early_vasopressor_any_24h",
                    "adjusted_effect": 2.7919,
                    "adjusted_effect_ci_low": 2.6467,
                    "adjusted_effect_ci_high": 2.9451,
                    "adjusted_effect_se": 0.0273,
                    "primary_effect_scale": "odds_ratio",
                    "primary_estimand": "stabilized and propensity-trimmed ATE-style",
                    "n_complete_case_primary": 74827,
                },
            }
        ),
    ]
    payload = _extract_primary_effect_payload_from_records(records)
    assert payload is not None
    assert payload["step_id"] == "01_causal_effect_estimation"
    assert round(payload["primary_or"], 2) == 2.79
    assert round(payload["primary_ci_low"], 2) == 2.65
    assert round(payload["primary_ci_high"], 2) == 2.95
    assert payload["effect_measure"] == "OR"


def test_extractor_recognizes_adjusted_effect_scale_from_iptw_runner():
    """Reproduces H2 fix8: the deterministic IPTW runner declares the scale under
    ``adjusted_effect_scale`` (not ``primary_effect_scale``). The extractor's
    scale lookup omitted that key, so a correct causal OR (3.04) was silently
    dropped and the headline bound nothing even though execution completed."""
    records = [
        _host_role_record(
            {
                "step_id": "04_causal_effect_estimation",
                "status": "ok",
                "step_summary_evidence_id": "causal_stat",
                "deterministic_standard_analysis": "causal_primary_iptw",
                "step_summary": {
                    "status": "ok",
                    "primary_exposure": "vasopressor",
                    "adjusted_effect": 3.0359,
                    "adjusted_effect_scale": "odds_ratio",
                    "adjusted_effect_ci_low": 2.8737,
                    "adjusted_effect_ci_high": 3.2072,
                    "max_smd_after_weighting": 0.0467,
                },
            }
        ),
    ]
    payload = _extract_primary_effect_payload_from_records(records)
    assert payload is not None
    assert payload["step_id"] == "04_causal_effect_estimation"
    assert round(payload["primary_or"], 2) == 3.04
    assert round(payload["primary_ci_low"], 2) == 2.87
    assert round(payload["primary_ci_high"], 2) == 3.21
    assert payload["effect_measure"] == "OR"


def test_extractor_keeps_odds_ratio_backward_compatible():
    records = [
        _host_role_record(
            {
                "step_id": "01_causal_effect_estimation",
                "status": "ok",
                "step_summary": {
                    "primary_predictor": "vaso",
                    "odds_ratio": 2.8,
                    "primary_or_ci_low": 2.6,
                    "primary_or_ci_high": 3.0,
                    "n": 5000,
                },
            }
        )
    ]
    payload = _extract_primary_effect_payload_from_records(records)
    assert payload is not None
    assert round(payload["primary_or"], 2) == 2.8
    assert payload["effect_measure"] == "OR"


def test_extractor_uses_primary_model_contract_for_predictor_and_sample_size():
    records = [
        _host_role_record(
            {
                "step_id": "05_primary_association",
                "status": "ok",
                "step_summary": {
                    "primary_or": 1.93,
                    "primary_ci_low": 1.86,
                    "primary_ci_high": 2.00,
                    "primary_model_id": "lab_source_aware_full",
                    "model_contracts": [
                        {
                            "model_id": "lab_source_aware_full",
                            "exposure_source": "lab_max",
                            "exposure_role": "primary",
                            "analysis_role": "primary",
                            "n": 94_458,
                        },
                        {
                            "model_id": "lab_complete_case",
                            "exposure_source": "lab_max",
                            "exposure_role": "primary",
                            "analysis_role": "sensitivity",
                            "n": 41_209,
                        },
                    ],
                },
            }
        )
    ]

    payload = _extract_primary_effect_payload_from_records(records)

    assert payload is not None
    assert payload["predictor"] == "lab_max"
    assert payload["sample_size"] == 94_458


def test_host_role_selects_primary_over_stronger_sensitivity_self_claims():
    primary = _host_role_record(
        {
            "step_id": "01_association",
            "status": "ok",
            "step_summary": {
                "primary_predictor": "exposure",
                "odds_ratio": 1.4,
                "primary_or_ci_low": 1.2,
                "primary_or_ci_high": 1.6,
                "n": 800,
            },
        }
    )
    sensitivity = _host_role_record(
        {
            "step_id": "02_sensitivity",
            "status": "ok",
            "step_summary": {
                "primary_predictor": "decoy",
                "primary_or": 9.9,
                "primary_ci_low": 9.0,
                "primary_ci_high": 10.8,
                "n": 10_000,
                "primary_result": {
                    "spec_id": "locked_primary",
                    "primary_or": 9.9,
                    "primary_ci_low": 9.0,
                    "primary_ci_high": 10.8,
                    "sample_size": 10_000,
                    "effect_measure": "OR",
                },
                "model_contracts": [
                    {
                        "model_id": "self_declared_primary",
                        "analysis_role": "primary",
                        "exposure_role": "primary",
                        "exposure_source": "decoy",
                        "n": 10_000,
                    }
                ],
            },
        },
        role="sensitivity",
    )

    payload = _extract_primary_effect_payload_from_records([primary, sensitivity])

    assert payload is not None
    assert payload["step_id"] == "01_association"
    assert payload["primary_or"] == 1.4


def test_primary_effect_selection_fails_closed_without_host_role_binding():
    record = {
        "step_id": "01_model",
        "status": "ok",
        "step_summary": {
            "primary_or": 1.4,
            "primary_ci_low": 1.2,
            "primary_ci_high": 1.6,
            "n": 800,
            "model_contracts": [{"analysis_role": "primary"}],
        },
    }

    assert _extract_primary_effect_payload_from_records([record]) is None


def test_primary_effect_selection_fails_closed_on_role_mismatch():
    record = _host_role_record(
        {
            "step_id": "01_model",
            "status": "ok",
            "step_summary": {"primary_or": 1.4},
        }
    )
    record["analysis_request"]["step"]["planned_analysis_role"] = "sensitivity"

    assert _extract_primary_effect_payload_from_records([record]) is None


def test_primary_effect_selection_requires_exactly_one_host_primary():
    secondary = _host_role_record(
        {
            "step_id": "02_secondary",
            "status": "ok",
            "step_summary": {"primary_or": 2.0},
        },
        role="secondary",
    )
    assert _extract_primary_effect_payload_from_records([secondary]) is None

    primary_a = _host_role_record(
        {
            "step_id": "01_primary_a",
            "status": "ok",
            "step_summary": {"primary_or": 1.2},
        }
    )
    primary_b = _host_role_record(
        {
            "step_id": "03_primary_b",
            "status": "ok",
            "step_summary": {"primary_or": 1.3},
        }
    )
    assert _extract_primary_effect_payload_from_records([primary_a, primary_b]) is None


def test_host_auxiliary_without_analysis_request_is_safely_excluded():
    probe = {
        "step_id": "00_probe",
        "status": "ok",
        "planned_analysis_role": "auxiliary",
        "step_authority_kind": "host_deterministic_probe",
        "step_summary": {
            "primary_or": 99.0,
            "primary_ci_low": 98.0,
            "primary_ci_high": 100.0,
            "n": 100_000,
        },
    }
    primary = _host_role_record(
        {
            "step_id": "01_model",
            "status": "ok",
            "step_summary": {
                "primary_or": 1.4,
                "primary_ci_low": 1.2,
                "primary_ci_high": 1.6,
                "n": 800,
            },
        }
    )

    payload = _extract_primary_effect_payload_from_records([probe, primary])

    assert payload is not None
    assert payload["step_id"] == "01_model"
    assert payload["primary_or"] == 1.4


def test_role_binding_rejects_noncanonical_literals_and_fake_host_kinds():
    primary = _host_role_record(
        {
            "step_id": "01_model",
            "status": "ok",
            "step_summary": {
                "primary_or": 1.4,
                "primary_ci_low": 1.2,
                "primary_ci_high": 1.6,
            },
        }
    )

    whitespace_role = _host_role_record(dict(primary))
    whitespace_role["planned_analysis_role"] = " PRIMARY "
    assert _extract_primary_effect_payload_from_records([whitespace_role]) is None

    fake_host = {
        "step_id": "00_fake",
        "status": "ok",
        "planned_analysis_role": "auxiliary",
        "step_authority_kind": "attacker_supplied_arbitrary_kind",
        "step_summary": {},
    }
    assert _extract_primary_effect_payload_from_records([fake_host, primary]) is None

    spoofed_probe = {
        "step_id": "00_fake",
        "status": "ok",
        "planned_analysis_role": "auxiliary",
        "step_authority_kind": "host_deterministic_probe",
        "step_summary": {},
    }
    assert (
        _extract_primary_effect_payload_from_records([spoofed_probe, primary]) is None
    )

    mismatched_step = _host_role_record(dict(primary))
    mismatched_step["analysis_request"]["step"]["step_id"] = "other_step"
    assert _extract_primary_effect_payload_from_records([mismatched_step]) is None


def test_planned_materializer_record_uses_normal_double_role_binding():
    materializer = _host_role_record(
        {
            "step_id": "01_cohort",
            "status": "ok",
            "step_authority_kind": "host_deterministic_cohort_materializer",
            "step_summary": {},
        },
        role="auxiliary",
    )
    primary = _host_role_record(
        {
            "step_id": "02_model",
            "status": "ok",
            "step_summary": {
                "primary_or": 1.4,
                "primary_ci_low": 1.2,
                "primary_ci_high": 1.6,
            },
        }
    )

    payload = _extract_primary_effect_payload_from_records([materializer, primary])

    assert payload is not None
    assert payload["step_id"] == "02_model"


def test_ordinary_auxiliary_without_embedded_role_still_fails_closed():
    auxiliary = {
        "step_id": "02_auxiliary",
        "status": "ok",
        "planned_analysis_role": "auxiliary",
        "step_summary": {},
    }
    primary = _host_role_record(
        {
            "step_id": "01_model",
            "status": "ok",
            "step_summary": {"primary_or": 1.4},
        }
    )

    assert _extract_primary_effect_payload_from_records([auxiliary, primary]) is None


# --- primary row + measure helper ------------------------------------------


def test_primary_row_carries_hazard_ratio_and_labels_hr():
    row = _primary_row_from_step_records(_survival_records())
    assert row is not None
    assert round(row.point_estimate, 2) == 1.82
    assert round(row.ci_low, 2) == 1.74
    assert round(row.ci_high, 2) == 1.91
    assert "(HR)" in row.notes


def test_primary_effect_measure_helper_reports_hr_and_or():
    assert _primary_effect_measure_from_records(_survival_records()) == "HR"
    assert _primary_effect_measure_from_records([]) is None


# --- panel guard: no mixed-measure variants for an HR primary --------------


def test_hazard_ratio_primary_skips_logistic_variants():
    df = _logistic_frame()
    records = _survival_records()
    # a second step supplies variant-fit data via the estimator adapter; the
    # primary estimand is still the HR from step 01.
    records.append(
        _host_role_record(
            {
                "step_id": "03_robustness",
                "status": "ok",
                "step_summary": {
                    "estimator_adapter": {
                        "data": df.to_dict("records"),
                        "exposure": "x",
                        "outcome": "y",
                        "estimator_kind": "logistic",
                        "missing_strategy": "complete_case",
                    }
                },
            },
            role="sensitivity",
        )
    )
    rows, warnings = fit_robustness_rows_from_records(
        specs=default_robustness_specs(),
        per_step_records=records,
        allow_implicit_cohort_refit=True,
    )
    assert len(rows) == 1, "an HR primary must not be joined by OR refit variants"
    assert rows[0].spec_id == "primary"
    assert round(rows[0].point_estimate, 2) == 1.82
    assert any("skipped logistic robustness variants" in w for w in warnings)


def test_odds_ratio_primary_still_fits_variants():
    df = _logistic_frame()
    records = [
        _host_role_record(
            {
                "step_id": "01_model",
                "status": "ok",
                "step_summary": {
                    "primary_predictor": "x",
                    "odds_ratio": 1.8,
                    "primary_or_ci_low": 1.5,
                    "primary_or_ci_high": 2.1,
                    "n": len(df),
                    "estimator_adapter": {
                        "data": df.to_dict("records"),
                        "exposure": "x",
                        "outcome": "y",
                        "estimator_kind": "logistic",
                        "missing_strategy": "complete_case",
                    },
                },
            }
        )
    ]
    rows, warnings = fit_robustness_rows_from_records(
        specs=default_robustness_specs(),
        per_step_records=records,
        allow_implicit_cohort_refit=True,
    )
    assert len(rows) > 1, "an OR primary must still get its robustness variants"
    assert not any("skipped logistic robustness variants" in w for w in warnings)
