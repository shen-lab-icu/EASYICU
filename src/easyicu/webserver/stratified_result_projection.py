"""Evidence-bound secondary stratum results for the completed-run answer.

The plan owns the action and exact product identity; the current successful
step record owns its output filename and active evidence. The caller hashes
and parses the same identifier-free CSV snapshot. No estimate is recomputed,
no independent products are joined, and display caps cannot truncate strata.
"""

from __future__ import annotations

from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from easyicu.research_agent.authority.runtime_artifacts import (
    current_successful_step_records,
    verified_run_evidence_path,
)
from easyicu.research_agent.contracts.step_families import secondary_effect_table_products
from easyicu.research_agent.execution.runners.typed_input_binding import contained_regular_file


SCHEMA = "easyicu.web-stratified-summary/1"
_REQUEST = re.compile(r"\b(?:strat(?:a|um|if\w*)|subgroups?|interactions?|effect\s+modification)\b|分层|亚组|交互|效应修饰", re.I)
_SCALES = {"odds_ratio", "hazard_ratio", "risk_ratio", "relative_risk", "risk_difference"}
_FITTED = {"fitted", "ok", "converged", "success"}
_ALIASES = {
    "subgroup_column": ("subgroup_column", "stratum_variable", "group_variable"),
    "subgroup_value": ("subgroup_value", "stratum", "group"),
    "interaction_p_value": ("interaction_p_value", "interaction_pvalue", "p_interaction"),
}


def requested_secondary_step_ids(plan: Mapping[str, Any]) -> frozenset[str]:
    """Identify plan-owned effect actions relevant to an explicit request."""

    if not _REQUEST.search(str(plan.get("research_question") or "")):
        return frozenset()
    return frozenset(str(step.get("step_id") or "") for step in plan.get("steps") or ()
                     if isinstance(step, Mapping) and secondary_effect_table_products(step))


def requested_secondary_tables(run_dir: Path, plan: Mapping[str, Any], evidence: Any) -> dict[str, dict]:
    """Bind requested secondary products to their unique active evidence ids."""

    requested = requested_secondary_step_ids(plan)
    if not requested:
        return {}
    run_dir = run_dir.resolve()
    steps = [s for s in plan.get("steps") or () if isinstance(s, Mapping)]
    step_ids = Counter(str(s.get("step_id") or "") for s in steps)
    declarations = Counter(p for s in steps for p in s.get("expected_outputs") or ())
    records = [r for r in evidence if isinstance(r, Mapping)] if isinstance(evidence, list) else []
    ids = Counter(str(r.get("evidence_id") or "") for r in records)
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        latest = {r["step_id"]: r for r in current_successful_step_records(manifest.get("per_step_records") or [])}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}
    bindings = {}
    for step in steps:
        producer = str(step.get("step_id") or "")
        if producer not in requested or step_ids[producer] != 1:
            continue
        current = latest.get(producer, {})
        summary = current.get("step_summary")
        if not isinstance(summary, Mapping) or summary.get("effect_output_authorized") is False:
            continue
        files = summary.get("output_files")
        if not isinstance(files, Mapping):
            continue
        for product in secondary_effect_table_products(step):
            filename = files.get(product)
            if (declarations[product] != 1 or not isinstance(filename, str)
                    or Path(filename).name != filename or not filename.endswith(".csv")):
                continue
            if sum(value == filename for key, value in files.items() if str(key).startswith("table:")) != 1:
                continue
            candidates = []
            for record in records:
                eid = str(record.get("evidence_id") or "")
                metadata = record.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                if (ids[eid] != 1 or record.get("kind") != "table"
                        or record.get("produced_by_step") != producer
                        or eid not in (current.get("evidence_ids") or [])
                        or metadata.get("diagnostic_only") is True or metadata.get("aliases_published") is False
                        or Path(str(record.get("relative_path") or "")).name != f"{eid}__{filename}"):
                    continue
                try:
                    path = contained_regular_file(run_dir / str(record.get("relative_path") or ""), run_dir)
                    if path is None or path.stat().st_size > 1_500_000:
                        continue
                    path = verified_run_evidence_path(run_dir, record)
                except (OSError, ValueError):
                    continue
                if path is not None:
                    candidates.append(eid)
            if len(candidates) == 1:
                bindings[candidates[0]] = {"product": product, "step": step}
    return bindings


def stratified_summary(snapshot: str, binding: Mapping[str, Any]) -> dict | None:
    """Project complete fitted strata and their shared interaction test."""

    reader = csv.DictReader(io.StringIO(snapshot, newline=""))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        return None
    columns = {}
    for field in ("subgroup_column", "subgroup_value", "estimate", "ci_low", "ci_high",
                  "effect_scale", "interaction_p_value", "fit_status", "contrast"):
        matches = [name for name in _ALIASES.get(field, (field,)) if name in headers]
        if len(matches) != 1:
            return None
        columns[field] = matches[0]
    strata, identities, levels = [], set(), set()
    for raw in reader:
        if None in raw or any(value is None or len(value) > 500 for value in raw.values()):
            return None
        row = {key: raw[column].strip() for key, column in columns.items()}
        if (row["fit_status"].lower() not in _FITTED or row["effect_scale"] not in _SCALES
                or row["subgroup_column"] not in (binding["step"].get("inputs") or [])
                or not row["subgroup_value"] or row["subgroup_value"] in levels or not row["contrast"]):
            return None
        try:
            estimate, low, high, p = (float(row[key]) for key in ("estimate", "ci_low", "ci_high", "interaction_p_value"))
            if not all(math.isfinite(v) for v in (estimate, low, high, p)):
                return None
            if not (low <= estimate <= high and 0 <= p <= 1):
                return None
            if row["effect_scale"] != "risk_difference" and low <= 0:
                return None
            if "ci_level" in raw and float(raw["ci_level"]) not in (0.95, 95):
                return None
        except ValueError:
            return None
        identities.add((row["subgroup_column"], row["contrast"], row["effect_scale"], p))
        levels.add(row["subgroup_value"])
        strata.append({"level": row["subgroup_value"], **{key: row[key] for key in ("estimate", "ci_low", "ci_high")}})
        if len(strata) > 6 or len(identities) != 1:
            return None
    if len(strata) < 2:
        return None
    return {"schema_version": SCHEMA, "product": binding["product"],
            "subgroup_column": row["subgroup_column"], "contrast": row["contrast"],
            "effect_scale": row["effect_scale"], "interaction_p_value": row["interaction_p_value"],
            "strata": strata}
