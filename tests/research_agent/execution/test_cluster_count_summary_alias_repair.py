from __future__ import annotations

import ast

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import deterministic_contract_repair


_FINDING = {
    "validator": "step_contract",
    "severity": "error",
    "message": (
        "Step 06_cluster_and_select_solution was expected to report a clustering "
        "summary, but it did not record both the selected cluster count and an "
        "agent-declared native selection/stability criterion."
    ),
    "detail": {
        "step_id": "06_cluster_and_select_solution",
        "required_keys": ["n_clusters", "cluster_count"],
    },
}


_SCRIPT = '''
import json

primary_k = choose_k(candidate_rows)
selection_manifest = {
    "criterion": "silhouette",
    "selection_rule": "maximum",
    "direction": "maximize",
    "selected_n_clusters": int(primary_k),
    "candidates": candidate_rows,
}
statistic_payload = {
    "name": "cluster_count",
    "value": int(primary_k),
}
with open(OUT_DIR / "cluster_count.json", "w", encoding="utf-8") as handle:
    json.dump(statistic_payload, handle)
step_summary = {
    "status": "ok",
    "primary_selected_n_clusters": int(primary_k),
    "cluster_selection": selection_manifest,
    "output_files": {
        "statistic:cluster_count": "cluster_count.json",
        "manifest:cluster_selection": "cluster_selection.json",
    },
}
with open(OUT_DIR / "step_summary.json", "w", encoding="utf-8") as handle:
    json.dump(step_summary, handle)
'''.lstrip()


def test_contract_repair_surfaces_proven_same_cluster_count() -> None:
    repair = deterministic_contract_repair(code=_SCRIPT, findings=[_FINDING])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "cluster_count_summary_alias_v1"
    ast.parse(repaired)
    assert '"cluster_count": int(primary_k),' in repaired
    assert repaired.count('"cluster_count": int(primary_k),') == 1
    assert "choose_k(candidate_rows)" in repaired

    metadata = repair_metadata_for(repair_id)
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert automatic_repair_allowed(repair_id)


def test_contract_repair_requires_exact_finding_and_three_way_lineage() -> None:
    wrong_finding = {
        **_FINDING,
        "detail": {"required_keys": ["cluster_selection"]},
    }
    assert (
        deterministic_contract_repair(code=_SCRIPT, findings=[wrong_finding]) is None
    )

    mismatched_selection = _SCRIPT.replace(
        '"selected_n_clusters": int(primary_k)',
        '"selected_n_clusters": int(other_k)',
    )
    assert (
        deterministic_contract_repair(
            code=mismatched_selection,
            findings=[_FINDING],
        )
        is None
    )

    mismatched_statistic = _SCRIPT.replace(
        '"value": int(primary_k)',
        '"value": int(other_k)',
    )
    assert (
        deterministic_contract_repair(
            code=mismatched_statistic,
            findings=[_FINDING],
        )
        is None
    )


def test_contract_repair_refuses_ambiguous_or_already_satisfied_summary() -> None:
    duplicated = _SCRIPT + "\n" + _SCRIPT.replace("step_summary", "other_summary")
    assert (
        deterministic_contract_repair(code=duplicated, findings=[_FINDING]) is None
    )

    repair = deterministic_contract_repair(code=_SCRIPT, findings=[_FINDING])
    assert repair is not None
    assert (
        deterministic_contract_repair(
            code=repair[1],
            findings=[_FINDING],
            previous_repair=repair[0],
        )
        is None
    )
