"""The host materialises the cohort, then refuses to recognise its own work.

``adopt_existing_host_cohort_materialization`` exists so the first step of a
plan -- define the analysis cohort, report its attrition -- is executed by the
host instead of being written by the Coder.  It calls
``load_materialized_analysis_cohort_result``, which proved the parquet bytes by
requiring ``cohort_parquet_sha256`` in the ledger.

That key is written on exactly one branch of the materializer: the untyped
``analysis_cohort/1`` path.  The typed ``/2`` path publishes a content-
addressed authority sidecar instead and never writes the key, so the recovery
returned ``None`` for every cohort the host had just materialised.

Measured 2026-08-02 over ``canonical9_runs``: **164 of 164** recorded
``cohort_analysis_provenance.json`` ledgers are ``easyicu.analysis_cohort/2``
and **none** carries ``cohort_parquet_sha256`` -- this recovery had never once
succeeded on a real run.  Downstream, the first analysis step was written by
the Coder in **127 of 127** runs; it failed in 21 of them, and each failure
killed a mean of 5.1 steps -- 108 dead steps, 59% of every cascade in the
corpus.  Verified on the real fresh28 artifacts: the plan's step 01 declared
exactly ``artifact:analysis_cohort`` + ``table:cohort_flow`` (so the host's own
``_declares_host_cohort_products`` accepted it), the definition digest matched,
the typed authority verified -- and the recorded ``generation_mode`` was still
``llm``.

Every pre-existing test of this loader built its universe with a plain
``DataFrame.to_parquet``, which has no authority, so all of them exercised the
``/1`` branch.  The branch every real run takes had no test at all.

The proof substituted here is stronger than the one it replaces: the authority
pins the parquet bytes, size, row count, column list, schema digest and a
per-row identity digest, and requires the sidecar's semantic provenance to
equal this ledger.  A ledger with neither proof still fails closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.run_input import (
    RunInputIdentityError,
    _planned_host_cohort_checkpoint,
)
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    load_materialized_analysis_cohort_result,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from easyicu.research_agent.intake.materialized_metadata import (
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.schema import (
    COHORT_DEFINITION_COHORT_OUTPUT,
    COHORT_DEFINITION_FLOW_OUTPUT,
    AnalysisPlan,
    AnalysisStep,
)

from .test_materialized_column_metadata import _typed_export


def _definition() -> CohortDefinition:
    return CohortDefinition(
        name="older_adults",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admit",
                    start_offset_hours=0,
                    end_offset_hours=24,
                ),
                aggregation="first",
                op=">=",
                value=55,
            ),
        ),
    )


@pytest.fixture
def typed_run(tmp_path: Path):
    """A run directory holding a real typed (``/2``) materialization.

    Built with the host's own publisher, from a typed export -- the same path
    every recorded run takes.  Returns ``(run_dir, plan)``.
    """

    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent_path = run_dir / "cohort.parquet"
    parent = stage_materialized_cohort_authority(
        paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert parent is not None

    plan = SimpleNamespace(cohort=_definition(), steps=[])
    result = materialize_locked_analysis_cohort(
        run_dir=run_dir, plan=plan, universe_path=parent_path
    )
    assert result["status"] == "applied"
    return run_dir, plan


def _ledger(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "cohort_analysis_provenance.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# The fixture is the branch the defect lives on
# ---------------------------------------------------------------------------


def test_the_typed_branch_writes_no_parquet_digest(typed_run):
    """Otherwise every assertion below would be testing the legacy path."""

    run_dir, _plan = typed_run
    ledger = _ledger(run_dir)

    assert ledger["schema_version"] == "easyicu.analysis_cohort/2"
    assert not str(ledger.get("cohort_parquet_sha256") or "").strip()


def test_the_typed_authority_is_what_proves_these_bytes(typed_run):
    """The proof the ledger does carry."""

    run_dir, _plan = typed_run

    verified = load_verified_materialized_cohort_authority(
        run_dir / "cohort_analysis.parquet"
    )

    assert verified is not None


# ---------------------------------------------------------------------------
# The recorded failure
# ---------------------------------------------------------------------------


def test_the_host_adopts_the_cohort_it_just_materialized(typed_run):
    """The defect in one assertion: this returned None on every real run."""

    run_dir, plan = typed_run

    recovered = load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan)

    assert recovered is not None
    assert recovered["status"] == "applied"
    assert recovered["path"] == run_dir / "cohort_analysis.parquet"
    assert recovered["flow_path"] == run_dir / "cohort_analysis_flow.csv"
    assert recovered["n_cohort"] == len(
        pd.read_parquet(run_dir / "cohort_analysis.parquet")
    )


def test_the_recovered_result_reports_the_authority_it_verified(typed_run):
    """A recovered result must carry the same reference a fresh one does,
    rather than a null that a later consumer would read as "untyped"."""

    run_dir, plan = typed_run
    recovered = load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan)
    assert recovered is not None

    verified = load_verified_materialized_cohort_authority(
        run_dir / "cohort_analysis.parquet"
    )
    assert verified is not None
    assert recovered["authority_ref"] == verified.reference.to_dict()
    assert Path(recovered["authority_path"]).is_file()


# ---------------------------------------------------------------------------
# The invariants that must not move
# ---------------------------------------------------------------------------


def test_content_drift_under_the_typed_authority_still_refuses(typed_run):
    """Patience, not permission: the substitute proof must still catch bytes
    that changed without the row count changing."""

    run_dir, plan = typed_run
    cohort_path = run_dir / "cohort_analysis.parquet"
    tampered = pd.read_parquet(cohort_path)
    assert len(tampered) >= 1
    tampered.loc[tampered.index[0], "age"] = 999
    tampered.to_parquet(cohort_path, index=False)
    assert len(pd.read_parquet(cohort_path)) == _ledger(run_dir)["n_analysis_cohort"]

    assert load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan) is None


def test_a_ledger_with_neither_proof_still_refuses(typed_run):
    """Removing the authority blob leaves a ``/2`` ledger whose selector points
    at a file that is gone.

    This is the *raising* exit -- the verifier reports a broken authority
    rather than an absent one -- and the recovery handler must turn that into a
    refusal.  The absent-authority exit, where the verifier simply returns
    ``None``, is covered by the untyped test at the end of this file.
    """

    run_dir, plan = typed_run
    blobs = sorted(run_dir.glob("cohort_authority.sha256-*.json"))
    assert blobs, "the fixture published no authority blob"
    for blob in blobs:
        blob.unlink()

    assert load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan) is None


def test_flow_tampering_still_refuses_on_the_typed_branch(typed_run):
    """The attrition ledger is checked regardless of which proof was used."""

    run_dir, plan = typed_run
    flow_path = run_dir / "cohort_analysis_flow.csv"
    flow = pd.read_csv(flow_path)
    flow.loc[flow.index[-1], "n_excluded"] = 999
    flow.to_csv(flow_path, index=False)

    assert load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan) is None


def test_a_wrong_definition_still_refuses_on_the_typed_branch(typed_run):
    """Adoption is bound to the plan's own cohort, not to whatever is on disk."""

    run_dir, _plan = typed_run
    other = SimpleNamespace(
        cohort=CohortDefinition(name="everyone", inclusion=()), steps=[]
    )

    assert load_materialized_analysis_cohort_result(run_dir=run_dir, plan=other) is None


# ---------------------------------------------------------------------------
# Adopting is not enough: the checkpoint has to seal
# ---------------------------------------------------------------------------


def _sealable_plan():
    return AnalysisPlan(
        research_question="Which older adults enter the analysis cohort?",
        cohort=_definition(),
        steps=[
            AnalysisStep(
                step_id="01_define_analysis_cohort",
                intent="Define the analytic cohort and report its attrition.",
                inputs=[],
                expected_outputs=[
                    COHORT_DEFINITION_COHORT_OUTPUT,
                    COHORT_DEFINITION_FLOW_OUTPUT,
                ],
                method="cohort_definition_and_attrition",
            )
        ],
    )


def _seal(run_dir: Path, plan: AnalysisPlan):
    evidence = EvidenceStore(run_dir)
    write_locked_cohort_definition(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="test",
    )
    result = load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan)
    assert result is not None, "adoption refused, so the seal cannot be tested"
    return evidence, _planned_host_cohort_checkpoint(
        plan=plan,
        result=result,
        cohort_path=run_dir / "cohort_analysis.parquet",
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="test",
        run_dir=run_dir,
        reason="locked_plan_cohort_materialization",
        gate_stamp={},
    )


def test_the_adopted_cohort_seals_a_host_owned_checkpoint(typed_run):
    """The second half of the defect.

    Adoption produced a result the checkpoint could not seal: the receipt named
    the typed authority, the evidence record did not, and the verifier compares
    the two. Observed on the real full-cohort run of 2026-08-02 -- the only
    occurrence of this reason in the whole recorded corpus, because until the
    loader was fixed nothing ever reached the check.
    """

    run_dir, _plan = typed_run
    plan = _sealable_plan()

    _evidence, (step_id, checkpoint, error) = _seal(run_dir, plan)

    assert error is None
    assert checkpoint is not None
    assert step_id == "01_define_analysis_cohort"
    assert checkpoint["generation_mode"] == "deterministic_cohort_materializer"
    assert checkpoint["status"] == "ok"
    assert set(checkpoint["step_summary"]["output_files"]) == {
        COHORT_DEFINITION_COHORT_OUTPUT,
        COHORT_DEFINITION_FLOW_OUTPUT,
    }
    producer_flow = (
        run_dir
        / "steps"
        / "01_define_analysis_cohort"
        / "outputs"
        / "cohort_analysis_flow.csv"
    )
    assert producer_flow.read_bytes() == (
        run_dir / "cohort_analysis_flow.csv"
    ).read_bytes()


def test_writer_excludes_only_a_verified_host_cohort_checkpoint(typed_run):
    from easyicu.research_agent.audits.envelope_consumers import (
        RegisteredOutputAuthorityError,
        RegisteredOutputEnvelopeConsumer,
    )

    run_dir, _plan = typed_run
    evidence, (_step_id, checkpoint, error) = _seal(run_dir, _sealable_plan())
    assert error is None
    assert checkpoint is not None

    projected = RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
        [checkpoint], evidence_store=evidence
    )

    assert projected == []

    checkpoint["cohort_table_evidence_id"] = "missing_authority"
    with pytest.raises(RegisteredOutputAuthorityError, match="cohort authority"):
        RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
            [checkpoint], evidence_store=evidence
        )


def test_the_host_refuses_a_conflicting_cohort_flow_step_output(typed_run):
    """Resume cannot overwrite a previously sealed producer location."""

    run_dir, _plan = typed_run
    producer_flow = (
        run_dir
        / "steps"
        / "01_define_analysis_cohort"
        / "outputs"
        / "cohort_analysis_flow.csv"
    )
    producer_flow.parent.mkdir(parents=True)
    producer_flow.write_text("not,the,canonical,flow\n", encoding="utf-8")

    with pytest.raises(RunInputIdentityError, match="conflicting immutable"):
        _seal(run_dir, _sealable_plan())


def test_the_evidence_carries_the_same_authority_the_receipt_names(typed_run):
    """One value, compiled once, handed to both places that assert it."""

    run_dir, _plan = typed_run
    plan = _sealable_plan()

    evidence, (_step_id, checkpoint, error) = _seal(run_dir, plan)
    assert error is None and checkpoint is not None

    record = evidence.get("analysis_cohort_execute_repair")
    metadata = dict(record.metadata or {})
    receipt = checkpoint["step_summary"]

    assert (
        metadata["materialized_cohort_authority_ref"]
        == receipt["materialized_cohort_authority_ref"]
    )
    assert metadata["cohort_definition_sha256"] == receipt["cohort_definition_sha256"]
    assert metadata["reason"] == "locked_plan_cohort_materialization"


def test_evidence_bound_to_a_different_authority_is_still_refused(typed_run):
    """The comparison is not vacuous just because one producer fills both.

    An evidence record already registered under this id -- by an earlier
    materialization, with a different authority -- is reused rather than
    overwritten, and that is exactly what this check exists to catch.
    """

    run_dir, _plan = typed_run
    plan = _sealable_plan()
    evidence = EvidenceStore(run_dir)
    write_locked_cohort_definition(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="test",
    )
    evidence.register_file(
        kind="table",
        description="A cohort registered under this id by an earlier attempt.",
        source_path=run_dir / "cohort_analysis.parquet",
        evidence_id="analysis_cohort_execute_repair",
        produced_by_step="01_define_analysis_cohort",
        producer="cohort_repair",
        generation_mode="llm",
        prompt_pack_version="test",
        metadata={"reason": "stale", "llm_signature": "test"},
    )

    result = load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan)
    assert result is not None
    _step_id, checkpoint, error = _planned_host_cohort_checkpoint(
        plan=plan,
        result=result,
        cohort_path=run_dir / "cohort_analysis.parquet",
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="test",
        run_dir=run_dir,
        reason="locked_plan_cohort_materialization",
        gate_stamp={},
    )

    assert checkpoint is None
    assert error is not None
    assert "does not bind the typed authority reference" in error


def test_an_untyped_materialization_seals_without_typed_keys(tmp_path: Path):
    """The same verifier refuses a legacy receipt carrying PARTIAL typed
    authority, so the two keys must be absent when there is no authority to
    name -- adding them unconditionally would break the untyped path instead."""

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [10, 18, 40, 70]}).to_parquet(universe_path, index=False)
    plan = _sealable_plan()
    materialize_locked_analysis_cohort(
        run_dir=tmp_path, plan=plan, universe_path=universe_path
    )
    assert _ledger(tmp_path)["schema_version"] == "easyicu.analysis_cohort/1"

    evidence, (_step_id, checkpoint, error) = _seal(tmp_path, plan)

    assert error is None
    assert checkpoint is not None
    metadata = dict(evidence.get("analysis_cohort_execute_repair").metadata or {})
    assert "materialized_cohort_authority_ref" not in metadata
    assert "cohort_definition_sha256" not in metadata


def test_the_untyped_ledger_still_requires_its_own_digest(tmp_path: Path):
    """The ``/1`` branch is unchanged: with the digest it adopts, and a ledger
    stripped of it has no authority to fall back on, so it still refuses."""

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [10, 18, 40, 70]}).to_parquet(universe_path, index=False)
    plan = SimpleNamespace(cohort=_definition(), steps=[])
    materialize_locked_analysis_cohort(
        run_dir=tmp_path, plan=plan, universe_path=universe_path
    )

    ledger_path = tmp_path / "cohort_analysis_provenance.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["schema_version"] == "easyicu.analysis_cohort/1"
    assert ledger["cohort_parquet_sha256"]
    assert load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan)

    del ledger["cohort_parquet_sha256"]
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    assert load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan) is None
