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

from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    load_materialized_analysis_cohort_result,
    materialize_locked_analysis_cohort,
)
from easyicu.research_agent.intake.materialized_metadata import (
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
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
