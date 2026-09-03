from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.manuscript_post import (
    _repair_common_writer_placeholders,
)


def _register_step(store: EvidenceStore, root: Path, step_id: str) -> None:
    path = root / f"{step_id}.json"
    path.write_text("{}", encoding="utf-8")
    store.register_file(
        kind="statistic",
        description=f"Summary for {step_id}.",
        source_path=path,
        evidence_id=f"statistic_{step_id}",
        aliases=[step_id],
        producer="test",
    )


def test_unique_step_suffix_repairs_writer_ordinal_drift(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    _register_step(store, tmp_path, "02_feature_availability_flow")
    scaffold = (
        "Age and sex were available variables "
        "{evidence:03_feature_availability_flow}."
    )

    repaired, repairs = _repair_common_writer_placeholders(
        scaffold,
        context=SimpleNamespace(research_question="Cluster sepsis phenotypes."),
        evidence=store,
    )

    assert "{evidence:02_feature_availability_flow}" in repaired
    assert "{evidence:03_feature_availability_flow}" not in repaired
    assert repairs == [
        ("03_feature_availability_flow", "02_feature_availability_flow")
    ]


def test_step_suffix_repair_stays_fail_closed_when_ambiguous(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    _register_step(store, tmp_path, "02_feature_availability_flow")
    _register_step(store, tmp_path, "04_feature_availability_flow")
    scaffold = "Variables were audited {evidence:03_feature_availability_flow}."

    repaired, repairs = _repair_common_writer_placeholders(
        scaffold,
        context=SimpleNamespace(research_question="Cluster sepsis phenotypes."),
        evidence=store,
    )

    assert repaired == scaffold
    assert repairs == []
