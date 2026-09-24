"""Table 1's cohort sentence names the cohort Table 1 described.

Table 1 bound to the closed analysis cohort describes the cohort after every
exclusion and landmark, not the source database's stays; without that binding
it describes the study cohort as extracted.
"""

from __future__ import annotations

from typing import Any

import pytest


def _table_one_facts(tmp_path, summary: dict[str, Any]):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.descriptive_report_facts import (
        _compile_grouped_table_one_cohort_report_facts,
    )

    store = EvidenceStore(tmp_path)
    record = store.register_json(
        kind="statistic", description="Grouped Table 1 summary", payload=summary,
        filename="summary.json", evidence_id="baseline_summary",
        produced_by_step="baseline", generation_mode="deterministic_standard",
    )
    return _compile_grouped_table_one_cohort_report_facts(
        [{
            "step_id": "baseline", "status": "ok", "step_summary": summary,
            "step_summary_evidence_id": record.evidence_id,
        }],
        store,
    )


@pytest.mark.parametrize(
    ("cohort_input_key", "sentence"),
    [
        ("artifact:analysis_cohort", "The analysis cohort included 412 ICU stays"),
        ("cohort:analysis_set", "The analysis cohort included 412 ICU stays"),
        (None, "The study cohort included 412 ICU stays"),
    ],
)
def test_table_one_names_the_cohort_it_described(
    tmp_path, cohort_input_key: str | None, sentence: str
) -> None:
    summary = {
        "status": "ok", "analysis_family": "grouped_table_one", "cohort_n": 412,
        "variables": ["age", "sex", "lactate_tertile"],
        "output_files": {"table:table_one": "table_one.csv"},
    }
    if cohort_input_key is not None:
        summary["cohort_input_key"] = cohort_input_key

    [fact] = _table_one_facts(tmp_path, summary)

    assert fact.text == sentence
