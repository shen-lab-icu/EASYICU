"""Replication metric comparison stays pure while old imports remain valid."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from easyicu.research_agent.audits import validators
from easyicu.research_agent.replication import paper
from easyicu.research_agent.replication import metrics


def test_legacy_paper_metric_export_is_canonical_object() -> None:
    assert paper.compare_metric_values is metrics.compare_metric_values
    assert validators.compare_metric_values is metrics.compare_metric_values


def test_metric_contract_has_no_audit_or_pipeline_dependency() -> None:
    path = Path(inspect.getsourcefile(metrics))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        name.endswith(("validators", "paper", "pipeline")) for name in imports
    )


def test_metric_comparison_behavior_is_stable() -> None:
    assert metrics.compare_metric_values(
        metric="or",
        paper_value=1.5,
        paper_direction="positive",
        easyicu_value=1.4,
    ) == ("aligned", "Effect direction and magnitude were close.")
    assert metrics.compare_metric_values(
        metric="p_value",
        paper_value=0.01,
        paper_direction=None,
        easyicu_value=0.2,
    ) == ("not_aligned", "Significance state did not match.")


def test_a_paper_risk_ratio_is_compared_only_with_a_reported_risk_ratio() -> None:
    from easyicu.research_agent.audits.publication import ReplicationResultComparator
    from easyicu.research_agent.schema import PaperProfile, PaperResultLedger

    assert metrics.claim_metric_key("RR") == "primary_rr"
    assert metrics.claim_metric_key("OR") == "primary_or"
    assert paper._map_claim_metric_to_easyicu_key("rr") == "primary_rr"
    # A risk-ratio run publishes its own headline, never an odds ratio.
    collected = paper.collect_easyicu_metrics(manifest={"per_step_records": [
        {"step_summary": {"primary_rr": 1.4, "primary_ci_low": 1.1, "primary_ci_high": 1.8}},
    ]})
    assert collected["primary_rr"] == 1.4 and "primary_or" not in collected
    profile = PaperProfile.model_construct(key_claims=[
        type("Claim", (), {"claim_id": "c1", "sentence": "RR 1.5", "paper_value": "1.5",
                           "numeric_value": 1.5, "direction": "positive", "metric": "rr"})(),
    ])
    rows = ReplicationResultComparator().compare(
        paper_profile=profile,
        ledger=PaperResultLedger.model_construct(easyicu_metrics={"primary_or": 9.0, "primary_rr": 1.4}),
    )
    assert (rows[0]["easyicu_value"], rows[0]["alignment_status"]) == ("1.4", "aligned")
