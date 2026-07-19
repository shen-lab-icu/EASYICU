"""Pre-specified robustness panel contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_planner_emits_minimum_axes() -> None:
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    with pytest.raises(ValueError, match="at least 3 cohort"):
        AnalysisPlan(
            research_question="Does severity predict mortality?",
            steps=[
                AnalysisStep(
                    step_id="01_model",
                    intent="Fit the primary model.",
                    expected_outputs=["statistic:primary_or"],
                )
            ],
            robustness_specs=[
                {
                    "spec_id": "alt_missing_complete_case",
                    "axis": "missing",
                    "description": "Complete cases only.",
                    "cohort_override": None,
                    "missing_override": {"strategy": "complete_case"},
                    "outcome_override": None,
                }
            ],
        )


def test_default_outcome_specs_are_case_neutral_placeholders() -> None:
    from easyicu.research_agent.robustness_panel import default_robustness_specs

    outcome_specs = [
        spec for spec in default_robustness_specs() if spec.axis == "outcome"
    ]

    assert [spec.spec_id for spec in outcome_specs] == [
        "alt_outcome_author_defined_1",
        "alt_outcome_author_defined_2",
    ]
    assert [spec.outcome_override for spec in outcome_specs] == [
        {"target": "author_defined_outcome_1"},
        {"target": "author_defined_outcome_2"},
    ]
    joined = " ".join(
        " ".join([spec.spec_id, spec.description, str(spec.outcome_override)])
        for spec in outcome_specs
    ).lower()
    assert "mortality" not in joined
    assert "death" not in joined
    assert "28_day" not in joined


def test_panel_freezes_after_plan(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.robustness_panel import (
        assert_robustness_specs_locked,
        default_robustness_specs,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Does severity predict mortality?",
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit the primary model.",
                expected_outputs=["statistic:primary_or"],
            )
        ],
        robustness_specs=default_robustness_specs(),
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    changed = plan.model_copy(
        update={"robustness_specs": list(plan.robustness_specs[1:])}
    )
    with pytest.raises(Exception, match="changed after plan lock"):
        assert_robustness_specs_locked(run_dir=tmp_path, plan=changed)


def test_locked_robustness_specs_restore_after_replan_drop(ra, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        default_robustness_specs,
        robustness_specs_for_execution,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    specs = default_robustness_specs()
    plan = AnalysisPlan(
        research_question="Does severity predict mortality?",
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit the primary model.",
                expected_outputs=["statistic:primary_or"],
            )
        ],
        robustness_specs=specs,
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    replanned_without_specs = SimpleNamespace(robustness_specs=[])
    restored = robustness_specs_for_execution(
        run_dir=tmp_path,
        plan=replanned_without_specs,
    )

    assert [spec.spec_id for spec in restored] == [spec.spec_id for spec in specs]


def test_execution_rejects_replanned_specs_that_drift_from_lock(
    ra,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        default_robustness_specs,
        robustness_specs_for_execution,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    specs = default_robustness_specs()
    plan = AnalysisPlan(
        research_question="Does the exposure predict the outcome?",
        steps=[AnalysisStep(step_id="01_model", intent="Fit the primary model.")],
        robustness_specs=specs,
    )
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=ra.EvidenceStore(tmp_path),
        prompt_pack_version="test",
        llm_signature="mock",
    )

    replanned = SimpleNamespace(robustness_specs=list(reversed(specs)))
    with pytest.raises(Exception, match="changed after plan lock"):
        robustness_specs_for_execution(run_dir=tmp_path, plan=replanned)


def test_execution_rejects_tampered_lock_hash(ra, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        default_robustness_specs,
        robustness_specs_for_execution,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    specs = default_robustness_specs()
    plan = AnalysisPlan(
        research_question="Does the exposure predict the outcome?",
        steps=[AnalysisStep(step_id="01_model", intent="Fit the primary model.")],
        robustness_specs=specs,
    )
    lock_path = write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=ra.EvidenceStore(tmp_path),
        prompt_pack_version="test",
        llm_signature="mock",
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["specs"][0]["description"] = "tampered after planning"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception, match="lock hash mismatch"):
        robustness_specs_for_execution(
            run_dir=tmp_path,
            plan=SimpleNamespace(robustness_specs=[]),
        )


def test_execution_rejects_self_rehashed_lock_against_evidence_anchor(
    ra,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        RobustnessSpec,
        default_robustness_specs,
        robustness_specs_for_execution,
        robustness_specs_sha,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    specs = default_robustness_specs()
    plan = AnalysisPlan(
        research_question="Does the exposure predict the outcome?",
        steps=[AnalysisStep(step_id="01_model", intent="Fit the primary model.")],
        robustness_specs=specs,
    )
    lock_path = write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=ra.EvidenceStore(tmp_path),
        prompt_pack_version="test",
        llm_signature="mock",
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["specs"][0]["description"] = "post-lock rewrite with a fresh self-hash"
    rewritten = [RobustnessSpec.from_dict(item) for item in payload["specs"]]
    payload["spec_sha256"] = robustness_specs_sha(rewritten)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception, match="plan-time evidence anchor"):
        robustness_specs_for_execution(
            run_dir=tmp_path,
            plan=SimpleNamespace(robustness_specs=[]),
        )


def test_panel_ignores_old_success_after_newer_step_failure() -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
    )

    old_success = {
        "step_id": "01_model",
        "status": "ok",
        "step_summary_evidence_id": "stale_primary",
        "step_summary": {
            "primary_or": 1.4,
            "primary_ci_low": 1.1,
            "primary_ci_high": 1.8,
            "n_total": 100,
        },
    }
    current_failure = {
        "step_id": "01_model",
        "status": "contract_failed",
        "step_summary": {"status": "rejected"},
    }

    panel = build_robustness_panel_from_records(
        specs=[],
        per_step_records=[old_success, current_failure],
    )

    assert panel.rows[0].converged is False
    assert panel.rows[0].point_estimate is None


def test_panel_excludes_variant_outside_plan_time_lock() -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
    )

    specs = default_robustness_specs()
    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=[
            {
                "step_id": "01_model",
                "status": "ok",
                "step_summary_evidence_id": "stat_primary",
                "step_summary": {
                    "primary_or": 1.4,
                    "primary_ci_low": 1.1,
                    "primary_ci_high": 1.8,
                    "n_total": 100,
                    "robustness_rows": [
                        {
                            "spec_id": "invented_after_lock",
                            "axis": "cohort",
                            "n": 100,
                            "point_estimate": 9.9,
                            "ci_low": 9.8,
                            "ci_high": 10.0,
                            "converged": True,
                        }
                    ],
                },
            }
        ],
    )

    assert "invented_after_lock" not in {row.spec_id for row in panel.rows}
    assert panel.range_high == pytest.approx(1.8)


def test_panel_writer_rejects_nonprimary_rows_without_verified_lock(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        write_robustness_panel,
    )

    source_summary = tmp_path / "unlocked_variant_summary.json"
    source_summary.write_text(
        json.dumps(
            {
                "robustness_rows": [
                    {
                        "spec_id": "unlocked_variant",
                        "axis": "cohort",
                        "n": 90,
                        "point_estimate": 1.5,
                        "ci_low": 1.1,
                        "ci_high": 2.0,
                        "converged": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    evidence = ra.EvidenceStore(tmp_path)
    source_record = evidence.register_file(
        kind="statistic",
        description="Digest-valid but unlocked robustness variant.",
        source_path=source_summary,
        evidence_id="stat_unlocked_variant",
    )
    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "unlocked_variant",
                "cohort",
                90,
                1.5,
                1.1,
                2.0,
                0.2,
                source_record.evidence_id,
                True,
            )
        ]
    )

    with pytest.raises(Exception, match="verified plan-time lock"):
        write_robustness_panel(
            run_dir=tmp_path,
            panel=panel,
            evidence=evidence,
            prompt_pack_version="test",
        )
    assert not (tmp_path / "robustness_panel.json").exists()


@pytest.mark.parametrize("evidence_state", ["nonexistent", "stale"])
def test_panel_writer_rejects_nonexistent_or_stale_summary_evidence(
    ra,
    tmp_path: Path,
    evidence_state: str,
) -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
        write_robustness_panel,
    )

    evidence = ra.EvidenceStore(tmp_path)
    evidence_id = "missing_summary"
    if evidence_state == "stale":
        source_path = tmp_path / "source_step_summary.json"
        source_path.write_text(
            json.dumps(
                {
                    "primary_or": 1.4,
                    "primary_ci_low": 1.1,
                    "primary_ci_high": 1.8,
                    "n_total": 100,
                }
            ),
            encoding="utf-8",
        )
        record = evidence.register_file(
            kind="statistic",
            description="Primary model summary.",
            source_path=source_path,
            produced_by_step="01_model",
            evidence_id="stat_primary",
        )
        evidence_id = record.evidence_id
        (tmp_path / record.relative_path).write_text("{}", encoding="utf-8")

    panel = build_robustness_panel_from_records(
        specs=[],
        per_step_records=[
            {
                "step_id": "01_model",
                "status": "ok",
                "step_summary_evidence_id": evidence_id,
                "evidence_ids": [evidence_id],
                "step_summary": {
                    "primary_or": 1.4,
                    "primary_ci_low": 1.1,
                    "primary_ci_high": 1.8,
                    "n_total": 100,
                },
            }
        ],
    )

    with pytest.raises(Exception, match=evidence_state):
        write_robustness_panel(
            run_dir=tmp_path,
            panel=panel,
            evidence=evidence,
            prompt_pack_version="test",
        )
    assert not (tmp_path / "robustness_panel.json").exists()


def test_write_locked_robustness_specs_reuses_existing_lock_on_resume(
    ra,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        default_robustness_specs,
        write_locked_robustness_specs,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    specs = default_robustness_specs()
    plan = AnalysisPlan(
        research_question="Does severity predict mortality?",
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit the primary model.",
                expected_outputs=["statistic:primary_or"],
            )
        ],
        robustness_specs=specs,
    )
    evidence = ra.EvidenceStore(tmp_path)
    locked = write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    before = locked.read_text(encoding="utf-8")

    resume_plan_without_specs = SimpleNamespace(robustness_specs=[])
    reused = write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=resume_plan_without_specs,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    assert reused == locked
    assert locked.read_text(encoding="utf-8") == before


def test_robustness_lock_resume_rehydrates_only_legacy_timestamp_drift(
    ra,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        default_robustness_specs,
        write_locked_robustness_specs,
    )

    plan = SimpleNamespace(robustness_specs=default_robustness_specs())
    evidence = ra.EvidenceStore(tmp_path)
    locked = write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    anchored = locked.read_bytes()
    payload = json.loads(locked.read_text(encoding="utf-8"))
    payload["locked_at"] = "2099-01-01T00:00:00+00:00"
    locked.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    assert locked.read_bytes() == anchored
    repair = evidence.get("robustness_lock_resume_rehydration")
    assert repair is not None
    assert repair.metadata["llm_signature"] == "mock"


def test_plan_payload_normalizer_drops_extra_robustness_spec_keys(ra) -> None:
    from easyicu.research_agent.agents.core import _normalise_plan_payload
    from easyicu.research_agent.robustness_panel import default_robustness_specs
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    specs = []
    for spec in default_robustness_specs():
        payload = spec.to_dict()
        payload["missing_handling"] = "listwise deletion"
        specs.append(payload)

    data, dropped = _normalise_plan_payload(
        {
            "research_question": "Does severity predict mortality?",
            "steps": [
                {
                    "step_id": "01_model",
                    "intent": "Fit the primary model.",
                    "expected_outputs": ["statistic:primary_or"],
                }
            ],
            "robustness_specs": specs,
        }
    )

    assert all("missing_handling" not in spec for spec in data["robustness_specs"])
    assert len(dropped["robustness_specs"]) == len(specs)
    assert all(
        item.endswith(":missing_handling") for item in dropped["robustness_specs"]
    )
    plan = AnalysisPlan(
        research_question=data["research_question"],
        steps=[AnalysisStep(**data["steps"][0])],
        robustness_specs=data["robustness_specs"],
    )
    assert len(plan.robustness_specs) == len(specs)


def test_each_spec_produces_panel_row() -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
    )

    specs = default_robustness_specs()
    records = [
        {
            "step_id": "01_model",
            "status": "ok",
            "step_summary_evidence_id": "stat_model",
            "step_summary": {
                "primary_or": 1.4,
                "primary_ci_low": 1.1,
                "primary_ci_high": 1.8,
                "n_total": 100,
                "robustness_rows": [
                    {
                        "spec_id": spec.spec_id,
                        "axis": spec.axis,
                        "n": 90,
                        "point_estimate": 1.2 + idx / 10,
                        "ci_low": 0.9 + idx / 10,
                        "ci_high": 1.5 + idx / 10,
                        "se": 0.1,
                        "converged": True,
                        "notes": "ok",
                    }
                    for idx, spec in enumerate(specs)
                ],
            },
        }
    ]

    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=records,
        locked_at="2026-05-27T00:00:00Z",
    )

    assert len(panel.rows) == 8
    assert panel.n_variants == 7
    assert panel.rows[0].spec_id == "primary"


def test_panel_primary_row_comes_from_step_validated_primary_not_refit() -> None:
    """Regression lock for the primary-effect headline bug.

    The manuscript-facing PRIMARY panel row must be sourced from the step's
    validated primary estimate (``step_summary.primary_or`` / ``n``), NOT the
    crude ``[exposure]``-only re-fit that ``_fit_one_row`` performs for variant
    axes. In the incident, a step that reported a validated adjusted OR of 1.346
    on n=50,640 had its panel primary silently replaced by an unadjusted re-fit
    on the raw cohort (OR≈1.006, n=27,277), which the writer then headlined as
    the canonical primary effect. Variant rows must still come from the re-fit.
    """
    import numpy as np
    import pandas as pd

    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import (
        PRIMARY_SPEC_ID,
        default_robustness_specs,
    )

    # A small cohort: any re-fit on this can only ever report n≈200, never the
    # summary's n=50640, so n=50640 on the primary row proves it came from the
    # validated step summary rather than a re-fit of this frame.
    rng = np.random.default_rng(0)
    n_rows = 200
    lactate = rng.normal(3.0, 1.0, n_rows)
    death = (rng.random(n_rows) < 0.3).astype(int)
    data = pd.DataFrame({"lactate": lactate, "death": death})

    specs = default_robustness_specs()
    records = [
        {
            "step_id": "01_primary_model",
            "status": "ok",
            "step_summary_evidence_id": "stat_primary_model",
            "step_summary": {
                "primary_predictor": "lactate",
                "primary_or": 1.346,
                "primary_ci_low": 1.21,
                "primary_ci_high": 1.50,
                "n_total": 50640,
                # The coder also emits a 'primary' robustness row; the validated
                # step estimate must override it (and warn), not the reverse.
                "robustness_rows": [
                    {
                        "spec_id": PRIMARY_SPEC_ID,
                        "axis": "primary",
                        "n": 27277,
                        "point_estimate": 1.006,
                        "ci_low": 0.94,
                        "ci_high": 1.08,
                        "se": 0.03,
                        "converged": True,
                        "notes": "unadjusted re-fit",
                    }
                ],
            },
        }
    ]

    rows, warnings = fit_robustness_rows_from_records(
        specs=specs,
        per_step_records=records,
        data=data,
        exposure="lactate",
        outcome="death",
        allow_implicit_cohort_refit=True,
    )

    primary = next(r for r in rows if r.spec_id == PRIMARY_SPEC_ID)
    assert primary.point_estimate == 1.346  # validated step OR, not 1.006 re-fit
    assert primary.n == 50640  # validated step n, not 27277 / the 200-row re-fit
    assert "step_summary" in (primary.notes or "")
    assert any("overrides" in w and "primary" in w for w in warnings)

    # Variant rows are still produced by the re-fit on the supplied frame, so
    # they cannot inherit the summary's n — the fix does not disturb variants.
    variant_ns = {r.n for r in rows if r.spec_id != PRIMARY_SPEC_ID}
    assert 50640 not in variant_ns


def test_primary_row_prefers_final_repaired_effect_over_synthesis_collision() -> None:
    """The primary row must not mix a synthesis step's broad n/CI with the final
    repaired primary association contract.

    Mirrors the 2026-07-01 E2 run shape: a synthesis step carried a top-level
    primary OR but nested tables also contained a full-cohort n and categorical
    CI. The final contract-repair step carried the intended paired OR/CI/n.
    """
    import numpy as np
    import pandas as pd

    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import (
        PRIMARY_SPEC_ID,
        default_robustness_specs,
    )

    rng = np.random.default_rng(2)
    n_rows = 220
    data = pd.DataFrame(
        {
            "lact_max": rng.normal(3.0, 1.0, n_rows),
            "death": (rng.random(n_rows) < 0.3).astype(int),
        }
    )
    records = [
        {
            "step_id": "04_final_evidence_synthesis",
            "status": "ok",
            "step_summary_evidence_id": "stat_synthesis",
            "step_summary": {
                "primary_or": 1.3460230881055546,
                "primary_or_se": 0.004837765021605868,
                "outputs": {
                    "summary:cohort_attrition": {"n_total": 94458},
                    "summary:categorical_lactate_model": {
                        "ci": [1.2302420713511062, 1.4023751359118677],
                    },
                },
            },
        },
        {
            "step_id": "05_contract_repair_and_association_addendum",
            "status": "ok",
            "step_summary_evidence_id": "stat_repaired_primary",
            "step_summary": {
                "primary_or": 1.3460230881055546,
                "primary_or_ci_low": 1.3333206221140712,
                "primary_or_ci_high": 1.3588465697324286,
                "n_total": 50640,
            },
        },
    ]

    rows, _warnings = fit_robustness_rows_from_records(
        specs=default_robustness_specs(),
        per_step_records=records,
        data=data,
        exposure="lact_max",
        outcome="death",
    )

    primary = next(row for row in rows if row.spec_id == PRIMARY_SPEC_ID)
    assert primary.evidence_id == "stat_repaired_primary"
    assert primary.n == 50640
    assert primary.ci_low == 1.3333206221140712
    assert primary.ci_high == 1.3588465697324286


def test_primary_row_prefers_nested_frozen_primary_reconciliation() -> None:
    """E1 regression: a later reconciliation step can repair an earlier
    off-protocol primary model.

    The canonical primary row must prefer ``step_summary.primary_result`` when
    it explicitly names the locked/frozen primary specification, while retaining
    the off-protocol estimate as a disclosed variant.
    """
    from easyicu.research_agent.robustness_panel import (
        PRIMARY_SPEC_ID,
        build_robustness_panel_from_records,
    )

    records = [
        {
            "step_id": "04_primary_adjusted_association_model",
            "status": "ok",
            "step_summary_evidence_id": "stat_offprotocol_primary",
            "step_summary": {
                "primary_predictor": "sepsis3",
                "primary_or": 1.3582181885372382,
                "primary_ci_low": 1.2939996351138754,
                "primary_ci_high": 1.4256237773289848,
                "n_total": 88061,
            },
        },
        {
            "step_id": "05_cohort_definition_sensitivity_comparison",
            "status": "ok",
            "step_summary_evidence_id": "stat_reconciled_primary",
            "step_summary": {
                "primary_predictor": "sepsis3",
                "primary_result": {
                    "spec_id": "frozen_primary_cc",
                    "n_modeled": 71249,
                    "adjusted_or": 1.2927410203895164,
                    "ci_low": 1.2272324290331191,
                    "ci_high": 1.3617464029323074,
                },
                "offprotocol_reestimated_result": {
                    "spec_id": "adult_any_los_offprotocol_cc",
                    "n_modeled": 88061,
                    "adjusted_or": 1.3582181885372382,
                    "ci_low": 1.2939996351138754,
                    "ci_high": 1.4256237773289848,
                },
            },
        },
    ]

    panel = build_robustness_panel_from_records(specs=[], per_step_records=records)

    primary = next(row for row in panel.rows if row.spec_id == PRIMARY_SPEC_ID)
    assert primary.evidence_id == "stat_reconciled_primary"
    assert primary.n == 71249
    assert primary.point_estimate == 1.2927410203895164
    assert primary.ci_low == 1.2272324290331191
    assert primary.ci_high == 1.3617464029323074


def test_robustness_variants_adjust_for_primary_covariates(tmp_path: Path) -> None:
    """Regression lock: robustness variants must be fit on the same footing as
    the primary effect (adjusted for the primary model's covariate set), not as
    bare unadjusted single-predictor re-fits.

    The primary model's adjustment set is recovered from the run directory using
    the same covariate-recovery path the overadjustment check trusts; here it is
    declared in a step ``analysis.py`` (``covariates = ['age']``). The variant
    re-fit must then include ``age`` in the design and say so in its notes.
    """
    import numpy as np
    import pandas as pd

    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import (
        PRIMARY_SPEC_ID,
        default_robustness_specs,
    )

    # A run layout the covariate recoverer understands: steps/<id>/analysis.py
    # declaring the adjustment set, and outputs/ beside it.
    step_dir = tmp_path / "steps" / "03_primary_model"
    (step_dir / "outputs").mkdir(parents=True)
    (step_dir / "analysis.py").write_text(
        "covariates = ['age']\n" "# formula = 'death ~ lactate + age'\n",
        encoding="utf-8",
    )

    rng = np.random.default_rng(1)
    n_rows = 300
    age = rng.normal(65, 12, n_rows)
    lactate = rng.normal(3.0, 1.0, n_rows)
    death = (rng.random(n_rows) < 0.3).astype(int)
    data = pd.DataFrame({"lactate": lactate, "age": age, "death": death})

    records = [
        {
            "step_id": "03_primary_model",
            "status": "ok",
            "step_summary_evidence_id": "stat_primary_model",
            "step_summary": {
                "primary_predictor": "lactate",
                "primary_or": 1.42,
                "primary_ci_low": 1.10,
                "primary_ci_high": 1.83,
                "n_total": 300,
            },
        }
    ]

    rows, warnings = fit_robustness_rows_from_records(
        specs=default_robustness_specs(),
        per_step_records=records,
        data=data,
        exposure="lactate",
        outcome="death",
        run_dir=tmp_path,
        allow_implicit_cohort_refit=True,
    )

    assert any("adjusted for" in w and "age" in w for w in warnings)
    variant_notes = [
        r.notes or "" for r in rows if r.spec_id != PRIMARY_SPEC_ID and r.converged
    ]
    assert variant_notes, "expected at least one converged variant row"
    assert any("adjusted for age" in note for note in variant_notes)


def test_effect_summary_terms_are_not_recovered_as_primary_covariates(
    tmp_path: Path,
) -> None:
    """A per-effect summary table is not a primary adjustment-set declaration.

    The E2 synthesis table listed focal terms from multiple sensitivity models
    (``lact_max``, ``lact_measured``). Treating those terms as covariates made
    robustness variants condition on measurement status rather than the primary
    model's adjustment set.
    """
    from easyicu.research_agent.estimators import _recover_primary_covariates

    outputs = tmp_path / "steps" / "05_contract_repair" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "effect_estimates.csv").write_text(
        "model_id,term,n_complete_case,or,ci_low,ci_high\n"
        "primary_measured_only,lact_max,50640,1.34,1.33,1.36\n"
        "measurement_status,lact_measured,94458,2.72,2.59,2.86\n",
        encoding="utf-8",
    )
    records = [
        {
            "step_id": "05_contract_repair",
            "status": "ok",
            "step_summary": {
                "primary_or": 1.346,
                "primary_or_ci_low": 1.33,
                "primary_or_ci_high": 1.36,
                "n_total": 50640,
            },
        }
    ]

    covariates = _recover_primary_covariates(
        tmp_path,
        per_step_records=records,
        exposure="lact_max",
        outcome="death",
        available_columns=["lact_max", "lact_measured", "death"],
    )

    assert covariates == []


def test_non_convergence_does_not_abort() -> None:
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
    )

    specs = default_robustness_specs()
    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=[
            {
                "step_id": "01_model",
                "status": "ok",
                "step_summary": {
                    "primary_or": 1.4,
                    "primary_ci_low": 1.1,
                    "primary_ci_high": 1.8,
                    "robustness_rows": [
                        {
                            "spec_id": specs[0].spec_id,
                            "axis": specs[0].axis,
                            "n": 0,
                            "point_estimate": None,
                            "ci_low": None,
                            "ci_high": None,
                            "se": None,
                            "converged": False,
                            "notes": "n=0 after applying override",
                        }
                    ],
                },
            }
        ],
    )

    row = next(r for r in panel.rows if r.spec_id == specs[0].spec_id)
    assert row.converged is False
    assert row.n == 0


def test_panel_range_correctness() -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
    )

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True
            ),
            RobustnessPanelRow("a", "cohort", 90, 1.5, 0.8, 2.0, 0.2, "e2", True),
            RobustnessPanelRow("b", "missing", 0, None, None, None, None, "e3", False),
        ],
        locked_at="2026-05-27T00:00:00Z",
    )

    assert panel.range_low == 0.8
    assert panel.range_high == 2.0


def test_panel_numeric_digest_deduplicates_repeated_panel_values() -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        numeric_digest_for_panel,
    )

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary", "primary", 500, 1.33, 1.2, 1.47, 0.1, "e1", True
            ),
            RobustnessPanelRow(
                "alt_same", "cohort", 500, 1.33, 1.2, 1.47, 0.1, "e2", True
            ),
            RobustnessPanelRow(
                "alt_diff", "missing", 400, 0.84, 0.7, 1.1, 0.2, "e3", True
            ),
        ],
        locked_at="2026-05-27T00:00:00Z",
    )

    digest = numeric_digest_for_panel(panel)

    assert "primary_point_estimate" in digest
    assert digest["primary_point_estimate"] == 1.33
    assert "worst_cohort_point_estimate" not in digest
    assert not any(key.startswith("row_alt_same") for key in digest)
    assert list(digest.values()).count(1.33) == 1


def test_writer_digest_contains_panel_block(ra, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        default_robustness_specs,
        write_locked_robustness_specs,
        write_robustness_panel,
    )
    from easyicu.research_agent.reporting.writer_evidence import (
        _render_writer_evidence_digest_v2,
    )

    specs = default_robustness_specs()
    cohort_worst_id = specs[0].spec_id
    cohort_hidden_id = specs[1].spec_id
    source_summary = tmp_path / "panel_source_summary.json"
    source_summary.write_text(
        json.dumps(
            {
                "primary_or": 1.2,
                "primary_ci_low": 1.0,
                "primary_ci_high": 1.4,
                "n_total": 100,
                "robustness_rows": [
                    {
                        "spec_id": cohort_worst_id,
                        "axis": "cohort",
                        "n": 90,
                        "point_estimate": 1.1,
                        "ci_low": 0.7,
                        "ci_high": 1.8,
                        "converged": True,
                    },
                    {
                        "spec_id": cohort_hidden_id,
                        "axis": "cohort",
                        "n": 90,
                        "point_estimate": 1.9,
                        "ci_low": 1.5,
                        "ci_high": 2.3,
                        "converged": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = ra.EvidenceStore(tmp_path)
    source_record = evidence.register_file(
        kind="statistic",
        description="Digest-bound source summary for the robustness panel.",
        source_path=source_summary,
        produced_by_step="01_model",
        evidence_id="stat_panel_source",
    )
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=SimpleNamespace(robustness_specs=specs),
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary",
                "primary",
                100,
                1.2,
                1.0,
                1.4,
                0.1,
                source_record.evidence_id,
                True,
            ),
            RobustnessPanelRow(
                cohort_worst_id,
                "cohort",
                90,
                1.1,
                0.7,
                1.8,
                0.2,
                source_record.evidence_id,
                True,
            ),
            RobustnessPanelRow(
                cohort_hidden_id,
                "cohort",
                90,
                1.9,
                1.5,
                2.3,
                0.2,
                source_record.evidence_id,
                True,
            ),
        ],
        locked_at="2026-05-27T00:00:00Z",
    )
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )

    digest = _render_writer_evidence_digest_v2(
        [{"step_id": "01_model", "status": "ok", "step_summary": {"primary_or": 1.2}}],
        run_dir=tmp_path,
        evidence=evidence,
    )

    assert "## robustness panel" in digest
    assert "n_variants=2" in digest
    assert cohort_hidden_id not in digest
    assert "OR=1.9" not in digest


def test_writer_forbidden_terms_blocked_in_strict(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.reporting.manuscript_post import (
        enforce_writer_claim_language,
    )

    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        enforce_writer_claim_language(
            "Surprisingly, the model was stable.",
            enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
        )
    assert "surprisingly" in exc_info.value.detail["forbidden_terms"]


def test_writer_forbidden_terms_annotated_in_soft(ra) -> None:
    from easyicu.research_agent.reporting.manuscript_post import (
        enforce_writer_claim_language,
    )

    annotated, detail = enforce_writer_claim_language(
        "Surprisingly, the model was stable.",
        enforcement_mode=ra.EvidenceEnforcementMode.SOFT,
    )

    assert "<!-- LEXICON:surprisingly -->" in annotated
    assert detail["forbidden_terms"] == ["surprisingly"]


def test_manifest_records_panel_artifact(tmp_path: Path) -> None:
    from easyicu.research_agent.schema import AnalysisManifest

    manifest = AnalysisManifest(
        run_id="run",
        research_question="Question?",
        started_at="2026-05-27T00:00:00Z",
        context_path="context.json",
        robustness_panel_path="robustness_panel.json",
        robustness_panel_sha="abc",
        robustness_n_variants=7,
        robustness_range_low=0.8,
        robustness_range_high=2.1,
    )

    assert manifest.robustness_panel_sha == "abc"
    assert manifest.robustness_n_variants == 7


def test_panel_numerics_registered_in_evidence_store(ra, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        default_robustness_specs,
        write_locked_robustness_specs,
        write_robustness_panel,
    )

    specs = default_robustness_specs()
    cohort_spec_id = specs[0].spec_id
    source_summary = tmp_path / "numeric_panel_source.json"
    source_summary.write_text(
        json.dumps(
            {
                "primary_or": 1.2,
                "primary_ci_low": 1.0,
                "primary_ci_high": 1.4,
                "n_total": 100,
                "robustness_rows": [
                    {
                        "spec_id": cohort_spec_id,
                        "axis": "cohort",
                        "n": 90,
                        "point_estimate": 1.5,
                        "ci_low": 0.8,
                        "ci_high": 2.0,
                        "converged": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = ra.EvidenceStore(tmp_path)
    source_record = evidence.register_file(
        kind="statistic",
        description="Source summary for panel numeric claims.",
        source_path=source_summary,
        evidence_id="stat_panel_source",
    )
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=SimpleNamespace(robustness_specs=specs),
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary",
                "primary",
                100,
                1.2,
                1.0,
                1.4,
                0.1,
                source_record.evidence_id,
                True,
            ),
            RobustnessPanelRow(
                cohort_spec_id,
                "cohort",
                90,
                1.5,
                0.8,
                2.0,
                0.2,
                source_record.evidence_id,
                True,
            ),
        ]
    )
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )

    fields = {claim.source_field for claim in evidence.numeric_claims()}
    assert {"n_variants", "range_low", "range_high"} <= fields
    assert "primary_point_estimate" in fields
    assert "row_primary_point_estimate" not in fields


def test_panel_json_exposes_row_count_and_primary_point_estimate() -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
    )

    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary", "primary", 100, 1.2, 1.0, 1.4, 0.1, "e1", True
            ),
            RobustnessPanelRow("a", "cohort", 90, 1.5, 0.8, 2.0, 0.2, "e2", True),
        ]
    )

    payload = panel.to_dict()

    assert payload["row_count"] == 2
    assert payload["primary_point_estimate"] == 1.2


def test_primary_only_panel_does_not_register_duplicate_range_claims(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.robustness_panel import (
        RobustnessPanel,
        RobustnessPanelRow,
        write_robustness_panel,
    )

    source_summary = tmp_path / "primary_only_source.json"
    source_summary.write_text(
        json.dumps(
            {
                "primary_or": 1.2,
                "primary_ci_low": 1.0,
                "primary_ci_high": 1.4,
                "n_total": 100,
            }
        ),
        encoding="utf-8",
    )
    evidence = ra.EvidenceStore(tmp_path)
    source_record = evidence.register_file(
        kind="statistic",
        description="Source summary for a primary-only panel.",
        source_path=source_summary,
        evidence_id="stat_primary_source",
    )
    panel = RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                "primary",
                "primary",
                100,
                1.2,
                1.0,
                1.4,
                0.1,
                source_record.evidence_id,
                True,
            ),
        ]
    )
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )

    fields = {claim.source_field for claim in evidence.numeric_claims()}
    assert "primary_point_estimate" in fields
    assert "row_primary_point_estimate" not in fields
    assert "range_low" not in fields
    assert "range_high" not in fields
