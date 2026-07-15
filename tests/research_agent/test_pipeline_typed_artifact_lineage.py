from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.cohort_schema import CohortDefinition
from easyicu.research_agent.evidence import EvidenceStore, sha256_of_file
from easyicu.research_agent.pipeline_execute import (
    _plan_scientific_scope_signature,
    _plan_signature,
    _preserve_completed_step_snapshots_after_replan,
    _resolve_typed_input_evidence,
    _resolve_typed_artifact_evidence,
    _resolved_typed_input_binding,
    _write_resolved_inputs_manifest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(*, duplicate_producer: bool = False) -> AnalysisPlan:
    steps = [
        AnalysisStep(
            step_id="producer",
            intent="Produce a reusable analysis artifact.",
            expected_outputs=["artifact:analysis_dataset"],
        )
    ]
    if duplicate_producer:
        steps.append(
            AnalysisStep(
                step_id="other_producer",
                intent="Illegally claim the same typed artifact.",
                expected_outputs=["artifact:analysis_dataset"],
            )
        )
    steps.append(
        AnalysisStep(
            step_id="consumer",
            intent="Consume the reusable analysis artifact.",
            inputs=["artifact:analysis_dataset"],
        )
    )
    return AnalysisPlan(research_question="Test typed evidence lineage.", steps=steps)


def _scope_signature(plan: AnalysisPlan) -> list[str | None]:
    return list(_plan_scientific_scope_signature(plan))


def _register(
    store: EvidenceStore,
    tmp_path: Path,
    *,
    suffix: str = ".parquet",
    payload: str = "current",
    evidence_id: str = "analysis_dataset",
    kind: str = "table",
):
    source = tmp_path / "source" / f"analysis_dataset{suffix}"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(payload, encoding="utf-8")
    return store.register_file(
        kind=kind,
        description="Typed upstream analysis dataset.",
        source_path=source,
        evidence_id=evidence_id,
        produced_by_step="producer",
        on_sha_change="new_id",
    )


def _plan_for_typed_product(product: str) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Test typed evidence kind authority.",
        steps=[
            AnalysisStep(
                step_id="producer",
                intent="Produce one typed product.",
                expected_outputs=[product],
            ),
            AnalysisStep(
                step_id="consumer",
                intent="Consume the typed product.",
                inputs=[product],
            ),
        ],
    )


def test_replan_restores_completed_execution_snapshot() -> None:
    current = _plan()
    revised_producer = current.steps[0].model_copy(
        update={
            "intent": "Change outcome, exposure, and analysis window.",
            "icu_rule_refs": ["different_rule"],
        }
    )
    revised = current.model_copy(
        update={"steps": [revised_producer, *current.steps[1:]]}
    )

    preserved, findings = _preserve_completed_step_snapshots_after_replan(
        current_plan=current,
        revised_plan=revised,
        completed_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "analysis_request": {"step": current.steps[0].model_dump(mode="json")},
            }
        ],
    )

    assert preserved.steps[0] == current.steps[0]
    assert findings
    assert findings[0].detail["reason"] == "completed_step_snapshot_immutable"


def test_plan_signature_detects_plan_level_cohort_change() -> None:
    base = _plan().model_copy(update={"cohort": CohortDefinition(name="adult_primary")})
    changed = base.model_copy(
        update={"cohort": CohortDefinition(name="adult_sensitivity")}
    )

    assert _plan_signature(base) != _plan_signature(changed)
    assert _plan_signature(base) == _plan_signature(
        base.model_copy(update={"revision": base.revision + 1})
    )


def test_replan_restores_plan_scientific_scope_after_completed_step() -> None:
    current = _plan().model_copy(
        update={
            "analysis_type": "adjusted_association",
            "cohort": CohortDefinition(name="adult_primary"),
            "rationale": "Estimate the prespecified primary association.",
        }
    )
    revised = current.model_copy(
        update={
            "research_question": "Estimate an unrelated outcome.",
            "analysis_type": "prediction_model",
            "cohort": CohortDefinition(name="different_population"),
            "rationale": "Replace the original estimand.",
            "revision": current.revision + 1,
        }
    )

    preserved, findings = _preserve_completed_step_snapshots_after_replan(
        current_plan=current,
        revised_plan=revised,
        completed_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "analysis_request": {"step": current.steps[0].model_dump(mode="json")},
            }
        ],
    )

    assert preserved.research_question == current.research_question
    assert preserved.analysis_type == current.analysis_type
    assert preserved.cohort == current.cohort
    assert preserved.robustness_specs == current.robustness_specs
    assert preserved.rationale == current.rationale
    assert preserved.revision == revised.revision
    assert findings[0].detail["restored_plan_scope"] is True
    assert set(findings[0].detail["restored_plan_scope_fields"]) == {
        "research_question",
        "analysis_type",
        "cohort",
        "rationale",
    }


def _resolve(
    *,
    store: EvidenceStore,
    tmp_path: Path,
    records: list[dict],
    plan: AnalysisPlan | None = None,
):
    active_plan = plan or _plan()
    step_by_id = {step.step_id: step for step in active_plan.steps}
    snapshotted_records = []
    for raw_record in records:
        record = dict(raw_record)
        producer = step_by_id.get(str(record.get("step_id") or ""))
        if producer is not None and "analysis_request" not in record:
            record["analysis_request"] = {"step": producer.model_dump(mode="json")}
        if producer is not None and "plan_scientific_signature" not in record:
            record["plan_scientific_signature"] = _scope_signature(active_plan)
        snapshotted_records.append(record)
    return _resolve_typed_artifact_evidence(
        input_name="artifact:analysis_dataset",
        plan=active_plan,
        evidence_records=store.records(),
        per_step_records=snapshotted_records,
        run_dir=tmp_path,
    )


def test_typed_artifact_resolves_verified_current_producer_output(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)

    ref, failure = _resolve(
        store=store,
        tmp_path=tmp_path,
        records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
            }
        ],
    )

    assert failure is None
    assert ref is not None
    assert ref.evidence_id == current.evidence_id


@pytest.mark.parametrize(
    ("declared_product", "evidence_kind", "suffix"),
    [
        ("table:analysis_dataset", "table", ".csv"),
        ("dataset:analysis_dataset", "table", ".parquet"),
        ("artifact:analysis_dataset", "table", ".parquet"),
        ("artifact:analysis_dataset", "log", ".json"),
        ("model:analysis_dataset", "log", ".pkl"),
        ("manifest:analysis_dataset", "log", ".json"),
        ("figure:analysis_dataset", "figure", ".png"),
        ("log:analysis_dataset", "log", ".txt"),
    ],
)
def test_typed_input_accepts_only_closed_compatible_evidence_kinds(
    tmp_path: Path,
    declared_product: str,
    evidence_kind: str,
    suffix: str,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path, suffix=suffix, kind=evidence_kind)
    plan = _plan_for_typed_product(declared_product)

    ref, failure = _resolve_typed_input_evidence(
        input_name=declared_product,
        plan=plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "step_summary": {
                    "output_files": {
                        declared_product: f"analysis_dataset{suffix}",
                    }
                },
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
                "plan_scientific_signature": _scope_signature(plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert failure is None
    assert ref is not None
    assert ref.kind == evidence_kind


@pytest.mark.parametrize(
    ("declared_product", "evidence_kind", "suffix"),
    [
        ("table:analysis_dataset", "code", ".csv"),
        ("table:analysis_dataset", "log", ".csv"),
        ("table:analysis_dataset", "figure", ".csv"),
        ("dataset:analysis_dataset", "log", ".parquet"),
        ("model:analysis_dataset", "table", ".pkl"),
        ("artifact:analysis_dataset", "code", ".json"),
        ("artifact:analysis_dataset", "figure", ".json"),
        ("manifest:analysis_dataset", "table", ".json"),
        ("figure:analysis_dataset", "table", ".png"),
        ("log:analysis_dataset", "table", ".txt"),
    ],
)
def test_typed_input_rejects_incompatible_evidence_kind(
    tmp_path: Path,
    declared_product: str,
    evidence_kind: str,
    suffix: str,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path, suffix=suffix, kind=evidence_kind)
    plan = _plan_for_typed_product(declared_product)

    ref, failure = _resolve_typed_input_evidence(
        input_name=declared_product,
        plan=plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "step_summary": {
                    "output_files": {
                        declared_product: f"analysis_dataset{suffix}",
                    }
                },
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
                "plan_scientific_signature": _scope_signature(plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "evidence_kind_mismatch"
    assert failure["declared_kind"] == declared_product.split(":", 1)[0]
    assert failure["observed_evidence_kinds"] == [evidence_kind]


def test_resume_uses_latest_authority_not_first_write_alias(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    old = _register(store, tmp_path, payload="old")
    current = _register(store, tmp_path, payload="current")
    assert current.evidence_id.endswith("_v2")
    assert store.get("analysis_dataset").evidence_id == old.evidence_id

    ref, failure = _resolve(
        store=store,
        tmp_path=tmp_path,
        records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [old.evidence_id],
            },
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
            },
        ],
    )

    assert failure is None
    assert ref is not None
    assert ref.evidence_id == current.evidence_id


def test_old_artifact_is_rejected_after_latest_producer_failure(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    old = _register(store, tmp_path)

    ref, failure = _resolve(
        store=store,
        tmp_path=tmp_path,
        records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [old.evidence_id],
            },
            {
                "step_id": "producer",
                "status": "contract_failed",
                "evidence_ids": [],
            },
        ],
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "producer_not_successful"


def test_tampered_current_artifact_fails_closed(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)
    (tmp_path / current.relative_path).write_text("tampered", encoding="utf-8")

    ref, failure = _resolve(
        store=store,
        tmp_path=tmp_path,
        records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
            }
        ],
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "no_verified_current_artifact"


def test_ambiguous_plan_producer_fails_closed(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)

    ref, failure = _resolve(
        store=store,
        tmp_path=tmp_path,
        records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
            }
        ],
        plan=_plan(duplicate_producer=True),
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "ambiguous_producer"


def test_multiple_current_files_for_one_typed_artifact_fail_closed(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    parquet = _register(store, tmp_path, suffix=".parquet", evidence_id="dataset_pq")
    csv = _register(store, tmp_path, suffix=".csv", evidence_id="dataset_csv")

    ref, failure = _resolve(
        store=store,
        tmp_path=tmp_path,
        records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [parquet.evidence_id, csv.evidence_id],
            }
        ],
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "ambiguous_current_artifact"
    assert set(failure["evidence_ids"]) == {
        parquet.evidence_id,
        csv.evidence_id,
    }


def test_exact_typed_mapping_selects_parquet_over_named_csv_copy(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    parquet = _register(store, tmp_path, suffix=".parquet", evidence_id="dataset_pq")
    csv = _register(store, tmp_path, suffix=".csv", evidence_id="dataset_csv")

    ref, failure = _resolve(
        store=store,
        tmp_path=tmp_path,
        records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [parquet.evidence_id, csv.evidence_id],
                "step_summary": {
                    "output_files": {
                        "artifact:analysis_dataset": "analysis_dataset.parquet",
                        "artifact:analysis_dataset_csv_copy": "analysis_dataset.csv",
                    }
                },
            }
        ],
    )

    assert failure is None
    assert ref is not None
    assert ref.evidence_id == parquet.evidence_id


def test_invalid_or_multiple_exact_typed_mappings_fail_closed(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)

    for mapping, reason in (
        ("missing.parquet", "typed_mapping_not_verified"),
        (
            ["analysis_dataset.parquet", "analysis_dataset.csv"],
            "ambiguous_typed_mapping",
        ),
    ):
        ref, failure = _resolve(
            store=store,
            tmp_path=tmp_path,
            records=[
                {
                    "step_id": "producer",
                    "status": "ok",
                    "evidence_ids": [current.evidence_id],
                    "step_summary": {
                        "output_files": {"artifact:analysis_dataset": mapping}
                    },
                }
            ],
        )

        assert ref is None
        assert failure is not None
        assert failure["reason"] == reason


def test_typed_table_uses_current_resume_authority_and_writes_exact_manifest(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    source = tmp_path / "source" / "scaling_summary.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("feature,mean\nx,0\n", encoding="utf-8")
    old = store.register_file(
        kind="table",
        description="Old scaling summary.",
        source_path=source,
        evidence_id="scaling_summary",
        produced_by_step="producer",
        on_sha_change="new_id",
    )
    source.write_text("feature,mean\nx,1\n", encoding="utf-8")
    current = store.register_file(
        kind="table",
        description="Current scaling summary.",
        source_path=source,
        evidence_id="scaling_summary",
        produced_by_step="producer",
        on_sha_change="new_id",
    )
    plan = AnalysisPlan(
        research_question="Test typed table lineage.",
        steps=[
            AnalysisStep(
                step_id="producer",
                intent="Produce scaling metadata.",
                expected_outputs=["table:scaling_summary"],
            ),
            AnalysisStep(
                step_id="consumer",
                intent="Consume current scaling metadata.",
                inputs=["table:scaling_summary"],
            ),
        ],
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="table:scaling_summary",
        plan=plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [old.evidence_id],
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
                "plan_scientific_signature": _scope_signature(plan),
            },
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
                "plan_scientific_signature": _scope_signature(plan),
            },
        ],
        run_dir=tmp_path,
    )

    assert failure is None
    assert ref is not None
    assert ref.evidence_id == current.evidence_id
    binding = _resolved_typed_input_binding(
        input_name="table:scaling_summary",
        evidence_ref=ref,
        evidence_records=store.records(),
        run_dir=tmp_path,
        producer_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "step_summary": {
                    "scaling_summary": {
                        "value_column": "x",
                        "scale": "standardized",
                    }
                },
            }
        ],
    )
    assert binding is not None
    assert binding["evidence_id"] == current.evidence_id
    assert binding["sha256"] == current.sha256
    assert binding["product_contract"] == {
        "value_column": "x",
        "scale": "standardized",
    }
    assert Path(binding["absolute_path"]).read_text(encoding="utf-8").endswith("x,1\n")

    context_path = tmp_path / "research_context.json"
    context_path.write_text('{"primary_exposure":"x"}\n', encoding="utf-8")
    manifest_path = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="consumer",
        bindings={"table:scaling_summary": binding},
        context_path=context_path,
    )
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["context"]["relative_path"] == "research_context.json"
    assert payload["context"]["sha256"] == sha256_of_file(context_path)
    manifest_binding = payload["inputs"]["table:scaling_summary"]
    assert manifest_binding["evidence_id"] == current.evidence_id
    assert manifest_binding["product_contract"]["value_column"] == "x"
    assert tmp_path / manifest_binding["relative_path"] == Path(
        manifest_binding["absolute_path"]
    )


def test_resolved_inputs_manifest_rejects_context_outside_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_context.json"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contained by run_dir"):
        _write_resolved_inputs_manifest(
            run_dir=run_dir,
            step_id="consumer",
            bindings={},
            context_path=outside,
        )


def test_typed_statistic_binds_current_verified_step_summary(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    summary_path = tmp_path / "source" / "step_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('{"primary_or": 1.25}\n', encoding="utf-8")
    summary = store.register_file(
        kind="statistic",
        description="Current machine-readable step summary.",
        source_path=summary_path,
        evidence_id="producer_step_summary",
        produced_by_step="producer",
    )
    plan = AnalysisPlan(
        research_question="Test typed statistic lineage.",
        steps=[
            AnalysisStep(
                step_id="producer",
                intent="Estimate an association.",
                expected_outputs=["statistic:primary_or"],
            ),
            AnalysisStep(
                step_id="consumer",
                intent="Consume the association estimate.",
                inputs=["statistic:primary_or"],
            ),
        ],
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="statistic:primary_or",
        plan=plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [summary.evidence_id],
                "step_summary_evidence_id": summary.evidence_id,
                "step_summary": {"primary_or": 1.25},
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
                "plan_scientific_signature": _scope_signature(plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert failure is None
    assert ref is not None
    assert ref.evidence_id == summary.evidence_id


@pytest.mark.parametrize(
    ("evidence_payload", "reason"),
    [
        ("[]\n", "statistic_evidence_payload_not_mapping"),
        ('{"other_metric": 1.25}\n', "statistic_evidence_value_missing"),
        (
            '{"primary_or": 1.25, "statistics": '
            '[{"name": "primary_or", "value": 1.30}]}\n',
            "statistic_evidence_value_ambiguous",
        ),
        ('{"primary_or": 0.81}\n', "statistic_evidence_payload_mismatch"),
    ],
)
def test_typed_statistic_requires_value_bound_evidence_payload(
    tmp_path: Path,
    evidence_payload: str,
    reason: str,
) -> None:
    store = EvidenceStore(tmp_path)
    summary_path = tmp_path / "source" / "step_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(evidence_payload, encoding="utf-8")
    summary = store.register_file(
        kind="statistic",
        description="Current machine-readable step summary.",
        source_path=summary_path,
        evidence_id="producer_step_summary",
        produced_by_step="producer",
    )
    plan = AnalysisPlan(
        research_question="Test typed statistic payload authority.",
        steps=[
            AnalysisStep(
                step_id="producer",
                intent="Estimate an association.",
                expected_outputs=["statistic:primary_or"],
            ),
            AnalysisStep(
                step_id="consumer",
                intent="Consume the association estimate.",
                inputs=["statistic:primary_or"],
            ),
        ],
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="statistic:primary_or",
        plan=plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [summary.evidence_id],
                "step_summary_evidence_id": summary.evidence_id,
                "step_summary": {"primary_or": 1.25},
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
                "plan_scientific_signature": _scope_signature(plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == reason


def test_typed_input_rejects_missing_plan_scope_snapshot(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)
    plan = _plan()

    ref, failure = _resolve_typed_input_evidence(
        input_name="artifact:analysis_dataset",
        plan=plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
            }
        ],
        run_dir=tmp_path,
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "producer_plan_scope_snapshot_missing"


def test_typed_input_rejects_plan_level_cohort_mismatch(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)
    executed_plan = _plan().model_copy(
        update={"cohort": CohortDefinition(name="adult_primary")}
    )
    active_plan = executed_plan.model_copy(
        update={"cohort": CohortDefinition(name="different_population")}
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="artifact:analysis_dataset",
        plan=active_plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "analysis_request": {
                    "step": executed_plan.steps[0].model_dump(mode="json")
                },
                "plan_scientific_signature": _scope_signature(executed_plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "producer_plan_scope_snapshot_mismatch"


def test_typed_input_rejects_completed_producer_intent_rephrasing(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)
    executed_plan = _plan()
    active_producer = executed_plan.steps[0].model_copy(
        update={"intent": "Write the same reusable analysis artifact."}
    )
    active_plan = executed_plan.model_copy(
        update={"steps": [active_producer, *executed_plan.steps[1:]]}
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="artifact:analysis_dataset",
        plan=active_plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "analysis_request": {
                    "step": executed_plan.steps[0].model_dump(mode="json")
                },
                "plan_scientific_signature": _scope_signature(executed_plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "producer_plan_snapshot_mismatch"


def test_typed_input_allows_only_case_and_whitespace_normalization(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)
    executed_plan = _plan()
    active_producer = executed_plan.steps[0].model_copy(
        update={"intent": "  PRODUCE   A REUSABLE ANALYSIS ARTIFACT.  "}
    )
    active_plan = executed_plan.model_copy(
        update={"steps": [active_producer, *executed_plan.steps[1:]]}
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="artifact:analysis_dataset",
        plan=active_plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "analysis_request": {
                    "step": executed_plan.steps[0].model_dump(mode="json")
                },
                "plan_scientific_signature": _scope_signature(executed_plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert failure is None
    assert ref is not None


def test_typed_input_rejects_completed_producer_rule_mutated_by_replan(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)
    executed_plan = _plan()
    active_producer = executed_plan.steps[0].model_copy(
        update={"icu_rule_refs": ["time_zero_before_exposure"]}
    )
    active_plan = executed_plan.model_copy(
        update={"steps": [active_producer, *executed_plan.steps[1:]]}
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="artifact:analysis_dataset",
        plan=active_plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "analysis_request": {
                    "step": executed_plan.steps[0].model_dump(mode="json")
                },
                "plan_scientific_signature": _scope_signature(executed_plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "producer_plan_snapshot_mismatch"


def test_typed_input_rejects_completed_producer_science_mutated_by_replan(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    current = _register(store, tmp_path)
    executed_plan = _plan()
    active_producer = executed_plan.steps[0].model_copy(
        update={"method": "different_scientific_method"}
    )
    active_plan = executed_plan.model_copy(
        update={"steps": [active_producer, *executed_plan.steps[1:]]}
    )

    ref, failure = _resolve_typed_input_evidence(
        input_name="artifact:analysis_dataset",
        plan=active_plan,
        evidence_records=store.records(),
        per_step_records=[
            {
                "step_id": "producer",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
                "analysis_request": {
                    "step": executed_plan.steps[0].model_dump(mode="json")
                },
                "plan_scientific_signature": _scope_signature(executed_plan),
            }
        ],
        run_dir=tmp_path,
    )

    assert ref is None
    assert failure is not None
    assert failure["reason"] == "producer_plan_snapshot_mismatch"
