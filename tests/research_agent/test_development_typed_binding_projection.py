"""Development samples replace physical typed input, never paper authority."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority import typed_binding as typed_binding_module
from easyicu.research_agent.authority.plan_scope import (
    _serializable_plan_scientific_scope_signature,
)
from easyicu.research_agent.authority.typed_binding import (
    TypedBindingResolver,
    _resume_typed_input_bindings_fingerprint,
    _resolved_typed_input_binding,
    _write_resolved_inputs_manifest,
)
from easyicu.research_agent.authority.typed_input_sdk import load_typed_input
from easyicu.research_agent.execution.development_sample import (
    DEVELOPMENT_COHORT_EVIDENCE_ID,
    materialize_development_execution_sample,
    record_development_sample_authority,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRef


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _arrange_projection(tmp_path: Path):
    parent_path = tmp_path / "cohort_analysis.parquet"
    parent_frame = pd.DataFrame(
        {
            "stay_id": list(range(1, 21)),
            "exposure": [value % 2 for value in range(1, 21)],
        }
    )
    parent_frame.to_parquet(parent_path, index=False)
    store = EvidenceStore(tmp_path)
    parent_record = store.register_file(
        kind="table",
        description="Locked full analysis cohort.",
        source_path=parent_path,
        evidence_id="analysis_cohort_parent",
        produced_by_step="01_cohort",
    )
    sample = materialize_development_execution_sample(
        run_dir=tmp_path,
        target_rows=5,
        seed=17,
        declared_id_columns=("stay_id",),
        trajectory_binding=None,
    )
    findings = []
    record_development_sample_authority(
        binding=sample,
        evidence=store,
        findings=findings,
        emit_progress=lambda *_args, **_kwargs: None,
        run_id="run-test",
    )
    plan = AnalysisPlan(
        research_question="Test one development projection.",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Lock the full analysis cohort.",
                expected_outputs=["artifact:analysis_cohort"],
            ),
            AnalysisStep(
                step_id="02_model",
                intent="Use the development execution projection.",
                inputs=["artifact:analysis_cohort"],
            ),
        ],
    )
    producer_records = [
        {
            "step_id": "01_cohort",
            "status": "ok",
            "evidence_ids": [parent_record.evidence_id],
            "step_summary": {},
            "analysis_request": {
                "step": plan.steps[0].model_dump(mode="json"),
            },
            "plan_scientific_signature": (
                _serializable_plan_scientific_scope_signature(plan)
            ),
        }
    ]
    return store, parent_record, sample, plan, producer_records


def test_sample_table_is_registered_as_nonpaper_evidence(tmp_path: Path) -> None:
    store, _parent, sample, _plan, _records = _arrange_projection(tmp_path)

    record = store.get(DEVELOPMENT_COHORT_EVIDENCE_ID)

    assert record is not None
    assert record.kind == "table"
    assert record.sha256 == sample.sample_sha256
    assert record.producer == "runtime_supervisor"
    assert record.metadata["paper_authority"] is False
    assert record.metadata["parent_cohort_sha256"] == sample.parent_sha256
    assert record.metadata["projection_kind"] == (
        "ordered_subset_of_locked_analysis_cohort"
    )
    assert record.metadata["aliases_published"] is False


def test_analysis_cohort_binding_selects_sample_and_preserves_parent(
    tmp_path: Path,
) -> None:
    store, parent, sample, _plan, records = _arrange_projection(tmp_path)

    binding = _resolved_typed_input_binding(
        input_name="artifact:analysis_cohort",
        evidence_ref=EvidenceRef(evidence_id=parent.evidence_id),
        evidence_records=store.records(),
        run_dir=tmp_path,
        producer_step_records=records,
        authoritative_cohort_path=sample.cohort_path,
        development_sample=sample,
    )

    assert binding is not None
    assert binding["evidence_id"] == DEVELOPMENT_COHORT_EVIDENCE_ID
    assert binding["sha256"] == sample.sample_sha256
    assert binding["produced_by_step"] == "01_cohort"
    assert binding["product_contract"]["row_identity_column"] == "stay_id"
    assert binding["product_contract"]["row_count"] == 5
    projection = binding["execution_projection"]
    assert projection["paper_authority"] is False
    assert projection["declared_parent_input"] == {
        "evidence_id": parent.evidence_id,
        "sha256": parent.sha256,
        "produced_by_step": "01_cohort",
    }
    assert projection["locked_parent_cohort_sha256"] == sample.parent_sha256
    original_fingerprint = _resume_typed_input_bindings_fingerprint(
        {"artifact:analysis_cohort": binding}
    )
    changed_binding = {**binding, "execution_projection": dict(projection)}
    changed_binding["execution_projection"]["seed"] = 18
    assert (
        _resume_typed_input_bindings_fingerprint(
            {"artifact:analysis_cohort": changed_binding}
        )
        != original_fingerprint
    )

    manifest_path = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="02_model",
        planner_declared_inputs=["artifact:analysis_cohort"],
        bindings={"artifact:analysis_cohort": binding},
    )
    loaded = load_typed_input(
        resolved_inputs_path=manifest_path,
        expected_resolved_inputs_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        run_root=tmp_path,
        input_key="artifact:analysis_cohort",
        consumer_step_id="02_model",
        consumer_code_sha256="c" * 64,
    )
    assert loaded.payload.num_rows == 5
    assert loaded.receipt.evidence_id == DEVELOPMENT_COHORT_EVIDENCE_ID


def test_resolver_records_both_declared_parent_and_execution_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, parent, sample, plan, records = _arrange_projection(tmp_path)
    monkeypatch.setattr(
        typed_binding_module,
        "_resolve_typed_input_evidence",
        lambda **_kwargs: (EvidenceRef(evidence_id=parent.evidence_id), None),
    )
    resolver = TypedBindingResolver(
        evidence_store=store,
        per_step_records=records,
        records_lock=_Lock(),
        run_dir=tmp_path,
        authoritative_cohort_path=sample.cohort_path,
        development_sample=sample,
    )

    refs, evidence_ids, bindings = resolver.resolve_names(
        ["artifact:analysis_cohort"],
        plan=plan,
    )

    assert [ref.evidence_id for ref in refs] == [
        parent.evidence_id,
        DEVELOPMENT_COHORT_EVIDENCE_ID,
    ]
    assert evidence_ids == [parent.evidence_id, DEVELOPMENT_COHORT_EVIDENCE_ID]
    assert bindings["artifact:analysis_cohort"]["evidence_id"] == (
        DEVELOPMENT_COHORT_EVIDENCE_ID
    )


def test_changed_sample_source_cannot_reuse_registered_projection(
    tmp_path: Path,
) -> None:
    store, parent, sample, _plan, records = _arrange_projection(tmp_path)
    changed = pd.read_parquet(sample.cohort_path)
    changed.loc[0, "stay_id"] = 999999
    changed.to_parquet(sample.cohort_path, index=False)

    binding = _resolved_typed_input_binding(
        input_name="artifact:analysis_cohort",
        evidence_ref=EvidenceRef(evidence_id=parent.evidence_id),
        evidence_records=store.records(),
        run_dir=tmp_path,
        producer_step_records=records,
        authoritative_cohort_path=sample.cohort_path,
        development_sample=sample,
    )

    assert binding is None
