from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.concept.loader import load_dictionary
from easyicu.concept.provenance import capture_data_provenance, metadata_digest
from easyicu.concept.sic_landmark import SIC_CASE_LANDMARK_COLUMNS, SIC_CASE_SOURCE_COLUMNS
from easyicu.config import load_src_cfg
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import time_origin_source_authority as authority


def test_raw_case_metadata_receipt_and_export_match(tmp_path, monkeypatch):
    identity = SourceIdentity(database="sic", registration_id="synthetic")
    monkeypatch.setattr(authority.sources, "registered_source_identity", lambda _: identity)
    schema = pa.schema([(field, pa.int64()) for field in SIC_CASE_SOURCE_COLUMNS])
    pq.write_table(pa.Table.from_batches([], schema=schema), tmp_path / "cases.parquet")
    receipt = authority.prepare_case_duration_metadata_plan(tmp_path, expected_identity=identity)
    assert all(len(value) == 64 for value in receipt.to_dict().values())
    config = load_src_cfg("sic")
    snapshot = capture_data_provenance(load_dictionary(include_sofa2=True),
        concepts=sorted(SIC_CASE_LANDMARK_COLUMNS), source_database="sic",
        class_prefixes=tuple(config.class_prefix), source_config=config,
        source_config_sha256=metadata_digest(config.model_dump(mode="json")))
    package = SimpleNamespace(data_provenance=SimpleNamespace(snapshots=(snapshot,)),
        concept_index={name: {} for name in SIC_CASE_LANDMARK_COLUMNS})
    authority.match_case_duration_metadata_plan(receipt, package)
    from dataclasses import replace
    with pytest.raises(ValueError, match="provenance_mismatch"):
        authority.match_case_duration_metadata_plan(replace(receipt, implementation_sha256="a" * 64), package)
    with pytest.raises(ValueError, match="source_mismatch"):
        authority.prepare_case_duration_metadata_plan(tmp_path, expected_identity=SourceIdentity(database="sic", registration_id="other"))
    pq.write_table(pa.Table.from_batches([], schema=pa.schema([("CaseID", pa.int64())])), tmp_path / "cases.parquet")
    with pytest.raises(ValueError, match="fields_unavailable"):
        authority.prepare_case_duration_metadata_plan(tmp_path, expected_identity=identity)


def test_metadata_candidate_is_visible_but_cannot_approve_execution():
    from easyicu.databases.research_approval import bind_database_capability_review
    from tests.support.signed_case_duration import signed_case_duration_case
    from tests.support.source_substitutions import _review

    context, plan, runtime, receipt, _, _, _ = signed_case_duration_case()
    metadata = authority.CaseDurationMetadataPlanReceipt(receipt.source_identity_sha256,
        *(["a" * 64] * 5))
    review = bind_database_capability_review(_review(context, plan), context=context, plan=plan,
        runtime_authority=runtime, case_duration_metadata_attestor=lambda _context, _selection: metadata)
    candidate = review.facts["database_case_duration_review"]
    assert candidate["metadata_plan_receipt"] == metadata.to_dict()
    assert candidate["metadata_status"] == "source_metadata_verified_not_executable"
    assert not candidate["execution_contract_available"]
    assert not review.approval_allowed
    bad = authority.CaseDurationMetadataPlanReceipt("f" * 64, *(["a" * 64] * 5))
    rejected = bind_database_capability_review(_review(context, plan), context=context, plan=plan,
        runtime_authority=runtime, case_duration_metadata_attestor=lambda _context, _selection: bad)
    assert rejected.facts["database_case_duration_review"]["metadata_plan_receipt"] is None
