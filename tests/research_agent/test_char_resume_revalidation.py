"""Freeze-baseline characterization for resume evidence revalidation (G3).

These tests deliberately call the narrow authority predicates and the public
resume preparation boundary.  They describe ``main@de4af7f`` behavior without
changing validator or engine code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest


def _record_map(*records) -> dict[str, dict]:
    return {record.evidence_id: record.model_dump(mode="json") for record in records}


def _probe_fixture(ra, root: Path):
    store = ra.EvidenceStore(root)
    summary = store.register_text(
        kind="statistic",
        description="Host probe summary.",
        text='{"n": 2}',
        filename="probe_summary.json",
        evidence_id="probe_summary",
        produced_by_step="00_probe",
        producer="pipeline",
        generation_mode="deterministic_probe",
        publish_aliases=False,
    )
    table = store.register_text(
        kind="table",
        description="Host probe variable profile.",
        text="variable,non_missing_n\nexposure,2\n",
        filename="probe_variable_profile.csv",
        evidence_id="probe_table",
        produced_by_step="00_probe",
        producer="pipeline",
        generation_mode="deterministic_probe",
        publish_aliases=False,
    )
    checkpoint = {
        "step_id": "00_probe",
        "status": "ok",
        "step_authority_kind": "host_deterministic_probe",
        "probe_summary_evidence_id": summary.evidence_id,
        "probe_table_evidence_id": table.evidence_id,
        "evidence_ids": [summary.evidence_id, table.evidence_id],
    }
    return checkpoint, _record_map(summary, table), summary, table


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "authority_kind",
            "successful host probe checkpoint lacks migrated probe authority",
        ),
        (
            "missing_field",
            "successful host probe checkpoint is missing required "
            "probe_summary_evidence_id",
        ),
        (
            "not_listed",
            "successful host probe probe_summary_evidence_id probe_summary is "
            "absent from evidence_ids",
        ),
        (
            "missing_record",
            "successful host probe probe_summary_evidence_id references missing "
            "probe_summary",
        ),
        (
            "wrong_owner",
            "successful host probe probe_summary_evidence_id is not owned by "
            "step 00_probe",
        ),
        (
            "wrong_kind",
            "successful host probe probe_summary_evidence_id has wrong evidence "
            "kind; expected statistic",
        ),
        (
            "wrong_host_identity",
            "successful host probe probe_summary_evidence_id is not host-owned "
            "probe evidence",
        ),
        (
            "wrong_source_name",
            "successful host probe probe_summary_evidence_id does not name "
            "probe_summary.json",
        ),
        (
            "digest_mismatch",
            "evidence probe_summary failed path/digest verification",
        ),
    ],
)
def test_host_probe_authority_reason_matrix(
    ra,
    tmp_path: Path,
    case: str,
    expected: str,
):
    from easyicu.research_agent.run_input_capsule import (
        _host_probe_authority_error,
    )

    checkpoint, records, summary, _ = _probe_fixture(ra, tmp_path)
    checkpoint = copy.deepcopy(checkpoint)
    records = copy.deepcopy(records)
    if case == "authority_kind":
        checkpoint["step_authority_kind"] = "other"
    elif case == "missing_field":
        checkpoint.pop("probe_summary_evidence_id")
    elif case == "not_listed":
        checkpoint["evidence_ids"].remove(summary.evidence_id)
    elif case == "missing_record":
        records.pop(summary.evidence_id)
    elif case == "wrong_owner":
        records[summary.evidence_id]["produced_by_step"] = "99_other"
    elif case == "wrong_kind":
        records[summary.evidence_id]["kind"] = "table"
    elif case == "wrong_host_identity":
        records[summary.evidence_id]["producer"] = "other"
    elif case == "wrong_source_name":
        records[summary.evidence_id][
            "relative_path"
        ] = "evidence/probe_summary__wrong.json"
    elif case == "digest_mismatch":
        records[summary.evidence_id]["sha256"] = "0" * 64

    error = _host_probe_authority_error(
        record=checkpoint,
        evidence_ids=checkpoint["evidence_ids"],
        step_id="00_probe",
        run_dir=tmp_path,
        records=records,
    )

    assert error == expected


def test_host_probe_exact_authority_passes(ra, tmp_path: Path):
    from easyicu.research_agent.run_input_capsule import (
        _host_probe_authority_error,
    )

    checkpoint, records, _, _ = _probe_fixture(ra, tmp_path)
    assert (
        _host_probe_authority_error(
            record=checkpoint,
            evidence_ids=checkpoint["evidence_ids"],
            step_id="00_probe",
            run_dir=tmp_path,
            records=records,
        )
        is None
    )


def _cohort_materializer_fixture(ra, root: Path):
    cohort_path = root / "cohort_analysis.parquet"
    pd.DataFrame({"stay_id": [1, 2], "exposure": [0, 1], "outcome": [0, 1]}).to_parquet(
        cohort_path, index=False
    )
    store = ra.EvidenceStore(root)
    cohort = store.register_file(
        kind="table",
        description="Host-materialized analysis cohort.",
        source_path=cohort_path,
        evidence_id="analysis_cohort_execute_repair",
        produced_by_step="01_cohort",
        producer="cohort_repair",
        generation_mode="llm",
        metadata={"reason": "mechanical cohort materialization"},
        publish_aliases=False,
    )
    checkpoint = {
        "step_id": "01_cohort",
        "status": "ok",
        "generation_mode": "deterministic_cohort_materializer",
        "step_authority_kind": "host_deterministic_cohort_materializer",
        "cohort_table_evidence_id": cohort.evidence_id,
        "evidence_ids": [cohort.evidence_id],
        "step_summary": {
            "output_files": {"table:analysis_cohort": "cohort_analysis.parquet"},
            "n_universe": 2,
            "n_analysis_cohort": 2,
        },
    }
    return checkpoint, _record_map(cohort), cohort, cohort_path


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "authority_kind",
            "successful host cohort materializer checkpoint lacks migrated "
            "cohort authority",
        ),
        (
            "wrong_outer_id",
            "successful host cohort materializer checkpoint is missing exact "
            "cohort_table_evidence_id",
        ),
        (
            "multiple_evidence_ids",
            "successful host cohort materializer checkpoint must list only its "
            "cohort table authority",
        ),
        (
            "missing_receipt",
            "successful host cohort materializer checkpoint lacks its inline "
            "product receipt",
        ),
        (
            "wrong_product",
            "successful host cohort materializer checkpoint does not declare the "
            "analysis cohort product",
        ),
        (
            "invalid_accounting",
            "successful host cohort materializer checkpoint has invalid cohort "
            "accounting",
        ),
        (
            "missing_authority",
            "successful host cohort materializer checkpoint references missing "
            "analysis_cohort_execute_repair",
        ),
        (
            "mismatched_identity",
            "successful host cohort materializer authority has a mismatched "
            "evidence identity",
        ),
        (
            "wrong_owner",
            "successful host cohort materializer authority is not owned by step "
            "01_cohort",
        ),
        (
            "wrong_kind",
            "successful host cohort materializer authority is not a table",
        ),
        (
            "wrong_producer",
            "successful host cohort materializer authority is not the host "
            "cohort-repair product",
        ),
        (
            "wrong_source_name",
            "successful host cohort materializer authority does not name the "
            "canonical cohort product",
        ),
        (
            "script_dependency",
            "successful host cohort materializer authority has an unexpected "
            "executable dependency",
        ),
        (
            "input_dependency",
            "successful host cohort materializer authority has an unexpected "
            "executable dependency",
        ),
        (
            "missing_reason",
            "successful host cohort materializer authority lacks its "
            "materialization reason",
        ),
        (
            "evidence_sha_mismatch",
            "evidence analysis_cohort_execute_repair failed path/digest verification",
        ),
        (
            "canonical_sha_mismatch",
            "successful host cohort materializer canonical cohort differs from "
            "sealed evidence",
        ),
        (
            "row_count_mismatch",
            "successful host cohort materializer canonical cohort row count 2 "
            "does not match checkpoint 1",
        ),
    ],
)
def test_host_cohort_materializer_reason_matrix(
    ra,
    tmp_path: Path,
    case: str,
    expected: str,
):
    from easyicu.research_agent.run_input_capsule import (
        _host_cohort_materializer_authority_error,
    )

    checkpoint, records, cohort, cohort_path = _cohort_materializer_fixture(
        ra, tmp_path
    )
    checkpoint = copy.deepcopy(checkpoint)
    records = copy.deepcopy(records)
    authority = records[cohort.evidence_id]
    if case == "authority_kind":
        checkpoint["step_authority_kind"] = "other"
    elif case == "wrong_outer_id":
        checkpoint["cohort_table_evidence_id"] = "other"
    elif case == "multiple_evidence_ids":
        checkpoint["evidence_ids"].append("extra")
    elif case == "missing_receipt":
        checkpoint.pop("step_summary")
    elif case == "wrong_product":
        checkpoint["step_summary"]["output_files"] = {}
    elif case == "invalid_accounting":
        checkpoint["step_summary"]["n_analysis_cohort"] = 3
    elif case == "missing_authority":
        records.pop(cohort.evidence_id)
    elif case == "mismatched_identity":
        authority["evidence_id"] = "other"
    elif case == "wrong_owner":
        authority["produced_by_step"] = "99_other"
    elif case == "wrong_kind":
        authority["kind"] = "statistic"
    elif case == "wrong_producer":
        authority["producer"] = "other"
    elif case == "wrong_source_name":
        authority["relative_path"] = (
            "evidence/analysis_cohort_execute_repair__other.parquet"
        )
    elif case == "script_dependency":
        authority["script_evidence_id"] = "script"
    elif case == "input_dependency":
        authority["inputs"] = ["input"]
    elif case == "missing_reason":
        authority["metadata"] = {}
    elif case == "evidence_sha_mismatch":
        authority["sha256"] = "0" * 64
    elif case == "canonical_sha_mismatch":
        cohort_path.write_bytes(b"changed working copy")
    elif case == "row_count_mismatch":
        checkpoint["step_summary"]["n_analysis_cohort"] = 1

    error = _host_cohort_materializer_authority_error(
        record=checkpoint,
        evidence_ids=checkpoint["evidence_ids"],
        step_id="01_cohort",
        run_dir=tmp_path,
        records=records,
    )

    assert error == expected


def test_host_cohort_materializer_exact_authority_passes(ra, tmp_path: Path):
    from easyicu.research_agent.run_input_capsule import (
        _host_cohort_materializer_authority_error,
    )

    checkpoint, records, _, _ = _cohort_materializer_fixture(ra, tmp_path)
    assert (
        _host_cohort_materializer_authority_error(
            record=checkpoint,
            evidence_ids=checkpoint["evidence_ids"],
            step_id="01_cohort",
            run_dir=tmp_path,
            records=records,
        )
        is None
    )


def _register_executable_step(
    ra,
    *,
    root: Path,
    store,
    step_id: str,
    prefix: str,
    summary_inputs: tuple[str, ...] = (),
):
    script = store.register_text(
        kind="code",
        description="Sealed script.",
        text="value = 1\n",
        filename=f"{prefix}_analysis.py",
        evidence_id=f"{prefix}_script",
        produced_by_step=step_id,
        producer="coder",
        publish_aliases=False,
    )
    summary = store.register_text(
        kind="statistic",
        description="Sealed step summary.",
        text='{"status": "ok", "estimate": 1}',
        filename=f"{prefix}_step_summary.json",
        evidence_id=f"{prefix}_summary",
        produced_by_step=step_id,
        inputs=list(summary_inputs),
        script_evidence_id=script.evidence_id,
        producer="runner",
        publish_aliases=False,
    )
    checkpoint = {
        "step_id": step_id,
        "status": "ok",
        "evidence_ids": [script.evidence_id, summary.evidence_id],
        "script_evidence_id": script.evidence_id,
        "step_summary_evidence_id": summary.evidence_id,
    }
    return checkpoint, script, summary


def test_closed_legacy_script_chain_migrates_but_ambiguous_chain_does_not(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.run_input_capsule import (
        _migrated_legacy_step_authority,
    )

    store = ra.EvidenceStore(tmp_path)
    checkpoint, script, summary = _register_executable_step(
        ra,
        root=tmp_path,
        store=store,
        step_id="01_model",
        prefix="model",
    )
    checkpoint.pop("script_evidence_id")
    records = _record_map(script, summary)

    migrated = _migrated_legacy_step_authority(
        record=checkpoint,
        run_dir=tmp_path,
        records=records,
    )
    assert migrated is not None
    assert migrated["script_evidence_id"] == script.evidence_id
    assert migrated["resume_authority_migration_schema_version"] == (
        "easyicu.resume_step_authority_migration/1"
    )
    assert migrated["resume_authority_migrated_fields"] == ["script_evidence_id"]

    decoy = store.register_text(
        kind="code",
        description="Ambiguous same-step code.",
        text="value = 2\n",
        filename="decoy.py",
        evidence_id="model_decoy",
        produced_by_step="01_model",
        producer="coder",
        publish_aliases=False,
    )
    checkpoint["evidence_ids"].append(decoy.evidence_id)
    records[decoy.evidence_id] = decoy.model_dump(mode="json")
    assert (
        _migrated_legacy_step_authority(
            record=checkpoint,
            run_dir=tmp_path,
            records=records,
        )
        is None
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "missing_summary",
            "successful checkpoint is missing required step_summary_evidence_id",
        ),
        (
            "ambiguous_missing_script",
            "successful checkpoint is missing required script_evidence_id",
        ),
    ],
)
def test_generic_step_missing_authority_is_invalidated_when_not_safely_migratable(
    ra,
    tmp_path: Path,
    case: str,
    expected: str,
):
    from easyicu.research_agent.run_input_capsule import (
        invalidate_unverified_successful_steps,
    )

    store = ra.EvidenceStore(tmp_path)
    checkpoint, script, summary = _register_executable_step(
        ra,
        root=tmp_path,
        store=store,
        step_id="01_model",
        prefix="model",
    )
    records = _record_map(script, summary)
    if case == "missing_summary":
        checkpoint.pop("step_summary_evidence_id")
    else:
        checkpoint.pop("script_evidence_id")
        decoy = store.register_text(
            kind="code",
            description="Ambiguous same-step code.",
            text="value = 2\n",
            filename="decoy.py",
            evidence_id="model_decoy",
            produced_by_step="01_model",
            producer="coder",
            publish_aliases=False,
        )
        checkpoint["evidence_ids"].append(decoy.evidence_id)
        records[decoy.evidence_id] = decoy.model_dump(mode="json")

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=tmp_path,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {"01_model": expected}
    assert updated["per_step_records"][-1] == {
        "step_id": "01_model",
        "status": "resume_evidence_invalid",
        "resume_invalidation_reason": expected,
        "evidence_ids": [],
    }
    assert updated["findings"][-1]["validator"] == "resume_evidence_integrity"
    assert updated["findings"][-1]["severity"] == "warning"


def test_upstream_invalidation_propagates_to_exact_consumers(ra, tmp_path: Path):
    from easyicu.research_agent.run_input_capsule import (
        invalidate_unverified_successful_steps,
    )

    store = ra.EvidenceStore(tmp_path)
    upstream, upstream_script, upstream_summary = _register_executable_step(
        ra,
        root=tmp_path,
        store=store,
        step_id="01_upstream",
        prefix="upstream",
    )
    middle, middle_script, middle_summary = _register_executable_step(
        ra,
        root=tmp_path,
        store=store,
        step_id="02_middle",
        prefix="middle",
        summary_inputs=(upstream_summary.evidence_id,),
    )
    downstream, downstream_script, downstream_summary = _register_executable_step(
        ra,
        root=tmp_path,
        store=store,
        step_id="03_downstream",
        prefix="downstream",
        summary_inputs=(middle_summary.evidence_id,),
    )
    upstream.pop("step_summary_evidence_id")
    records = _record_map(
        upstream_script,
        upstream_summary,
        middle_script,
        middle_summary,
        downstream_script,
        downstream_summary,
    )

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=tmp_path,
        resume_state={
            "per_step_records": [upstream, middle, downstream],
            "findings": [],
        },
        records=records,
    )

    assert invalidated == {
        "01_upstream": (
            "successful checkpoint is missing required step_summary_evidence_id"
        ),
        "02_middle": (
            "successful checkpoint depends on invalidated step 01_upstream via "
            f"evidence {upstream_summary.evidence_id}"
        ),
        "03_downstream": (
            "successful checkpoint depends on invalidated step 02_middle via "
            f"evidence {middle_summary.evidence_id}"
        ),
    }
    latest = {record["step_id"]: record for record in updated["per_step_records"]}
    assert {
        step_id: latest[step_id]["status"]
        for step_id in ("01_upstream", "02_middle", "03_downstream")
    } == {
        "01_upstream": "resume_evidence_invalid",
        "02_middle": "resume_evidence_invalid",
        "03_downstream": "resume_evidence_invalid",
    }


def test_resume_cut_after_newly_invalidated_upstream_fails_before_receipt(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.run_input_capsule import (
        RunInputIdentityError,
        prepare_existing_resume_input,
        seal_run_input_capsule,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort = pd.DataFrame({"stay_id": [1, 2], "outcome": [0, 1]})
    cohort_path = run_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    context = ra.schema.ResearchContext(
        research_question="Characterize resume authority.",
        cohort=ra.schema.CohortDescriptor(
            cohort_name="fixture",
            database="synthetic",
            n_patients=2,
            n_stays=2,
        ),
        variables=[],
        target_outcome="outcome",
    )
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    store = ra.EvidenceStore(run_dir)
    store.register_file(
        kind="log",
        description="Frozen research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    scientific_identity = {"question": context.research_question}
    environment = {"llm_signature": "mock"}
    seal_run_input_capsule(
        run_dir=run_dir,
        evidence=store,
        scientific_identity=scientific_identity,
        initial_environment=environment,
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    upstream, upstream_script, upstream_summary = _register_executable_step(
        ra,
        root=run_dir,
        store=store,
        step_id="01_upstream",
        prefix="upstream",
    )
    downstream, _, _ = _register_executable_step(
        ra,
        root=run_dir,
        store=store,
        step_id="02_downstream",
        prefix="downstream",
        summary_inputs=(upstream_summary.evidence_id,),
    )
    upstream.pop("step_summary_evidence_id")
    plan = ra.schema.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.schema.AnalysisStep(
                step_id="01_upstream",
                intent="Produce the planned input.",
            ),
            ra.schema.AnalysisStep(
                step_id="02_downstream",
                intent="Consume the planned input.",
            ),
        ],
    )

    with pytest.raises(
        RunInputIdentityError,
        match=(
            r"Cannot start resume after invalidated upstream evidence; resume "
            r"at or before: 01_upstream"
        ),
    ):
        prepare_existing_resume_input(
            run_dir=run_dir,
            resume_state={
                "per_step_records": [upstream, downstream],
                "findings": [],
            },
            scientific_identity=scientific_identity,
            current_environment=environment,
            cohort=cohort,
            question=context.research_question,
            resume_from_step_id="02_downstream",
            enforcement_mode="soft",
            load_compatible_plan=lambda **_kwargs: (plan, None),
        )

    assert list(run_dir.glob("resume_environment_receipt_*.json")) == []
    assert upstream_script.evidence_id in {
        record.evidence_id for record in store.records()
    }
