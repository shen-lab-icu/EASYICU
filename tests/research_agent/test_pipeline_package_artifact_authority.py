from pathlib import Path

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline_package import _current_verified_semantic_csv


def _register_primary(
    store: EvidenceStore,
    run_dir: Path,
    *,
    odds_ratio: float,
):
    source = run_dir / "steps" / "primary" / "outputs" / "primary_association.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        f"term,odds_ratio\nexposure,{odds_ratio}\n",
        encoding="utf-8",
    )
    return store.register_file(
        kind="table",
        description="Primary association",
        source_path=source,
        evidence_id="primary_association",
        produced_by_step="primary",
        producer="coder",
        generation_mode="llm",
        on_sha_change="new_id",
    )


def test_current_semantic_csv_selects_resumed_v2_not_first_write_alias(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    old = _register_primary(store, tmp_path, odds_ratio=1.1)
    current = _register_primary(store, tmp_path, odds_ratio=2.2)
    assert old.evidence_id == "primary_association"
    assert current.evidence_id == "primary_association_v2"

    selected = _current_verified_semantic_csv(
        evidence=store,
        per_step_records=[
            {
                "step_id": "primary",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
            }
        ],
        run_dir=tmp_path,
        semantic_id="primary_association",
    )

    assert selected is not None
    record, path = selected
    assert record.evidence_id == current.evidence_id
    assert "2.2" in path.read_text(encoding="utf-8")


def test_current_semantic_csv_rejects_old_artifact_after_later_failed_rerun(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    old = _register_primary(store, tmp_path, odds_ratio=1.1)

    selected = _current_verified_semantic_csv(
        evidence=store,
        per_step_records=[
            {
                "step_id": "primary",
                "status": "ok",
                "evidence_ids": [old.evidence_id],
            },
            {
                "step_id": "primary",
                "status": "execution_failed",
                "evidence_ids": [],
            },
        ],
        run_dir=tmp_path,
        semantic_id="primary_association",
    )

    assert selected is None


def test_current_semantic_csv_rejects_digest_drift(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    current = _register_primary(store, tmp_path, odds_ratio=1.1)
    evidence_path = tmp_path / current.relative_path
    evidence_path.write_text(
        "term,odds_ratio\nexposure,9.9\n",
        encoding="utf-8",
    )

    selected = _current_verified_semantic_csv(
        evidence=store,
        per_step_records=[
            {
                "step_id": "primary",
                "status": "ok",
                "evidence_ids": [current.evidence_id],
            }
        ],
        run_dir=tmp_path,
        semantic_id="primary_association",
    )

    assert selected is None
