from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.figure2_icu_agent_v2.protocol import load_heldout_taskbank
from benchmarks.figure2_icu_agent_v2.readiness import build_development_readiness


def _development_full6(
    root: Path,
    *,
    omit: tuple[str, str] | None = None,
) -> Path:
    root.mkdir()
    (root / "run_manifest.json").write_text(
        json.dumps({"kind": "test-development-schema"}),
        encoding="utf-8",
    )
    tasks = load_heldout_taskbank().tasks
    databases = sorted({task.database for task in tasks})
    for database in databases:
        database_root = root / database
        database_root.mkdir()
        concepts = {
            concept
            for task in tasks
            if task.database == database
            for concept in task.required_concepts
        }
        if omit is not None and omit[0] == database:
            concepts.discard(omit[1])
        table = pa.table(
            {name: pa.array([], type=pa.float64()) for name in sorted(concepts)}
        )
        pq.write_table(table, database_root / "schema_only.parquet")
    return root


def test_all_tasks_are_development_reachable_but_never_formal_ready(
    tmp_path: Path,
) -> None:
    receipt = build_development_readiness(_development_full6(tmp_path / "full6"))

    assert receipt.paper_authority is False
    assert receipt.task_count == 27
    assert receipt.contract_reachable_count == 27
    assert receipt.concepts_catalogued_count == 27
    assert receipt.development_schema_observed_count == 27
    assert receipt.development_ready_count == 27
    assert receipt.formal_ready_count == 0
    assert all(
        "FORMAL_NATIVE_V2_INPUT_NOT_AUTHORIZED" in row.blocking_reason_codes
        for row in receipt.tasks
    )


def test_missing_development_column_is_attributed_to_the_affected_task(
    tmp_path: Path,
) -> None:
    first = load_heldout_taskbank().tasks[0]
    missing = first.required_concepts[0]
    root = _development_full6(tmp_path / "full6", omit=(first.database, missing))

    receipt = build_development_readiness(root)
    row = next(item for item in receipt.tasks if item.task_id == first.task_id)

    assert row.development_full6_schema_observed is False
    assert missing in row.missing_development_schema_concepts
    assert "DEVELOPMENT_SCHEMA_CONCEPT_MISSING" in row.blocking_reason_codes
    assert row.development_ready is False
