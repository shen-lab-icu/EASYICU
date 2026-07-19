from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.research_agent.authority.evidence_store import EvidenceRecord, EvidenceStore
from easyicu.research_agent.pipeline_report import _extract_claim_ledger_rows


def _register(
    store: EvidenceStore,
    tmp_path: Path,
    *,
    evidence_id: str,
    aliases: tuple[str, ...] = (),
    produced_by_step: str | None = None,
) -> EvidenceRecord:
    source = tmp_path / f"source_{evidence_id}.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    return store.register_file(
        kind="table",
        description=f"test evidence {evidence_id}",
        source_path=source,
        evidence_id=evidence_id,
        aliases=aliases,
        produced_by_step=produced_by_step,
        producer="test",
        generation_mode="system",
    )


def _extract(
    store: EvidenceStore,
    tmp_path: Path,
    manuscript: str,
    per_step_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    path = tmp_path / "manuscript.md"
    path.write_text(manuscript, encoding="utf-8")
    return _extract_claim_ledger_rows(
        manuscript_path=path,
        gates={},
        evidence=store,
        per_step_records=per_step_records,
    )


def test_claim_ledger_uses_exact_href_owner_not_markdown_label(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "run")
    record = _register(store, tmp_path, evidence_id="actual_table_authority")

    rows = _extract(
        store,
        tmp_path,
        f"Mortality was 7%. [friendly prose label]({record.relative_path})\n",
    )

    assert rows == [
        {
            "claim_id": "claim_001",
            "claim_text": (
                "Mortality was 7%. " f"[friendly prose label]({record.relative_path})"
            ),
            "evidence_refs": "actual_table_authority",
            "status": "bound",
            "note": "",
        }
    ]


def test_claim_ledger_resolves_published_alias_from_href_with_title(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "run")
    _register(
        store,
        tmp_path,
        evidence_id="versioned_result_7f9e",
        aliases=("stable-result",),
    )

    rows = _extract(
        store,
        tmp_path,
        'The result was stable. [display text](evidence/stable-result "sha256=x")\n',
    )

    assert rows[0]["evidence_refs"] == "versioned_result_7f9e"
    assert rows[0]["status"] == "bound"


def test_claim_ledger_preserves_semicolon_delimited_resolved_ids(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "run")
    first = _register(store, tmp_path, evidence_id="first_table")
    second = _register(store, tmp_path, evidence_id="second_table")

    rows = _extract(
        store,
        tmp_path,
        (
            f"Two results [A]({first.relative_path}) and "
            f"[B]({second.evidence_id}) agree.\n"
        ),
    )

    assert rows[0]["evidence_refs"] == "first_table;second_table"
    assert rows[0]["status"] == "bound"


def test_claim_ledger_fails_closed_when_exact_href_is_ambiguous(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "run")
    _register(
        store,
        tmp_path,
        evidence_id="first_owner",
        aliases=("colliding-name",),
    )
    _register(store, tmp_path, evidence_id="colliding-name")

    rows = _extract(
        store,
        tmp_path,
        "This cannot bind. [decorative label](evidence/colliding-name)\n",
    )

    assert rows[0]["evidence_refs"] == ""
    assert rows[0]["status"] == "missing_evidence"
    assert "ambiguous evidence href" in rows[0]["note"]


def test_claim_ledger_fails_closed_when_href_is_unresolved_even_if_label_is_id(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "run")
    _register(store, tmp_path, evidence_id="real_authority")

    rows = _extract(
        store,
        tmp_path,
        "This cannot bind. [real_authority](evidence/not_registered.csv)\n",
    )

    assert rows[0]["evidence_refs"] == ""
    assert rows[0]["status"] == "missing_evidence"
    assert "evidence/not_registered.csv" in rows[0]["note"]


def test_claim_ledger_rejects_superseded_noncurrent_evidence_path(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "run")
    stale = _register(
        store,
        tmp_path,
        evidence_id="old_attempt_table",
        aliases=("stable-table",),
        produced_by_step="01_analysis",
    )
    current = _register(
        store,
        tmp_path,
        evidence_id="current_attempt_table",
        produced_by_step="01_analysis",
    )
    step_records = [
        {
            "step_id": "01_analysis",
            "status": "ok",
            "evidence_ids": [current.evidence_id],
        }
    ]

    rows = _extract(
        store,
        tmp_path,
        f"Stale claim [old label]({stale.relative_path}).\n",
        per_step_records=step_records,
    )

    assert rows[0]["evidence_refs"] == ""
    assert rows[0]["status"] == "missing_evidence"
