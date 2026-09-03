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
