"""The Methods and checklist use the executed MI policy, with bound numbers."""

import json
from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.model_terms import DataPreparationPlan
from easyicu.research_agent.execution.model_matrix import write_preparation_receipt
from easyicu.research_agent.reporting.reporting_checklist import build_strobe_checklist


def test_executed_imputation_has_an_exact_manuscript_method_fact(tmp_path):
    plan = DataPreparationPlan(
        missing_data={
            "strategy": "multiple_imputation",
            "m": 6,
            "seed": 71,
            "variables": [{"name": "x", "method": "pmm"}],
        }
    )
    receipt = write_preparation_receipt(
        tmp_path,
        plan=plan,
        execution={},
        missing_data={"fit_scope": "training_partition_only"},
        sensitivity={"strategy": "complete_case", "status": "fitted"},
    )
    path = tmp_path / "step_summary.json"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "executed_method_design": receipt["executed_method_design"],
            }
        )
    )
    store = EvidenceStore(tmp_path)
    store.register_file(
        kind="statistic",
        source_path=path,
        evidence_id="fit_summary",
        description="Synthetic executed MI design",
        produced_by_step="fit",
        producer="runner",
        generation_mode="deterministic_standard",
    )
    (fact,) = store.manuscript_method_facts()
    assert "6 imputations" in fact.text
    assert "seed 71" in fact.text
    assert "training partition" in fact.text
    assert "Rubin" in fact.text
    assert 6 in fact.typed_values and 71 in fact.typed_values


def test_reporting_checklist_links_the_executed_missing_data_method():
    record = SimpleNamespace(
        evidence_id="log_data_preparation_provenance_0123456789abcdef",
        relative_path="evidence/log_0123456789abcdef__data_preparation_provenance.json",
    )
    report = build_strobe_checklist(evidence_records=[record], bound_manuscript="")
    item = next(row for row in report.items if row.item_id == "12c")
    assert item.status == "addressed"
    assert "data_preparation_provenance" in item.evidence_ids
