from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline_execute import (
    _resolve_typed_artifact_evidence,
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


def _register(
    store: EvidenceStore,
    tmp_path: Path,
    *,
    suffix: str = ".parquet",
    payload: str = "current",
    evidence_id: str = "analysis_dataset",
):
    source = tmp_path / "source" / f"analysis_dataset{suffix}"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(payload, encoding="utf-8")
    return store.register_file(
        kind="table",
        description="Typed upstream analysis dataset.",
        source_path=source,
        evidence_id=evidence_id,
        produced_by_step="producer",
        on_sha_change="new_id",
    )


def _resolve(
    *,
    store: EvidenceStore,
    tmp_path: Path,
    records: list[dict],
    plan: AnalysisPlan | None = None,
):
    return _resolve_typed_artifact_evidence(
        input_name="artifact:analysis_dataset",
        plan=plan or _plan(),
        evidence_records=store.records(),
        per_step_records=records,
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
