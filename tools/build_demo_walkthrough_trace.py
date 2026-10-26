#!/usr/bin/env python3
"""Build the workflow demo's trace record from its registered source run.

The Copilot demo replays one real Research Agent run. Its traceability cards
(data dictionary, plan literature, number-to-evidence chain) read this record,
so every value they show comes from the run directory or from the concept
dictionaries at the exact bytes the run fingerprinted, never from hand-typed
copy. Only aggregate values leave the run: audit times, the plan and its
review, literature metadata, evidence ids and digests, the three-row result
table, a code excerpt, and dictionary entries. No patient-level row is read.

    python tools/build_demo_walkthrough_trace.py RUN_DIR [--out PATH]

RUN_DIR is the run's ``pipeline/run_<id>`` directory; its parent wrapper
directory holds the reviewer projection. The concept dictionaries are found
in this repository's history by the run's recorded sha256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "src/easyicu/webserver/static/assets/demo/sofa2-run-trace.json"
CONCEPT_DICT = "src/easyicu/data/concept-dict.json"
SOFA2_DICT = "src/easyicu/data/sofa2-dict.json"
SCHEMA = "easyicu.demo-walkthrough-trace/1"

# The question's clinical terms and the concepts the run bound them to
# (cohort_provenance.json: feature, outcome and inclusion concepts).
TERMS = (
    {"id": "phenotype", "concept": "sep3_sofa2"},
    {"id": "death", "concept": "death"},
    {"id": "adult", "concept": "age"},
)

# Translations of record text the cards display. Keyed by the exact source
# sentence, so a changed record fails the build instead of showing stale copy.
ZH = {
    "Apply observational reporting standards to state the cohort, denominators, variable definitions, and descriptive-only interpretation.":
        "按观察性研究报告规范写明队列、分母、变量定义，并把解读限定为描述性。",
    "Report the routinely collected database source and the reproducible extraction and variable-construction process.":
        "报告常规采集数据库的来源，以及可复现的抽取和变量构建过程。",
    "Treat phenotype classification as determined over the prespecified first 24 hours after ICU admission and keep interpretation limited to the resulting descriptive landmark population.":
        "表型按入 ICU 后预先规定的前 24 小时判定，解读只限于由此得到的描述性 landmark 人群。",
    "No landmark survival estimator is fitted; the source informs temporal alignment of the descriptive classification.":
        "没有拟合 landmark 生存估计；这篇文献只用于对齐描述性分类的时间。",
    "Flag the post-ICU-admission exposure-ascertainment period as a limitation and avoid interpreting the grouped mortality proportions as ordinary baseline exposure effects.":
        "把入 ICU 后才确定暴露的这段时间标为局限，不把分组死亡比例解读为普通的基线暴露效应。",
    "No causal or time-to-event effect estimate is attempted.":
        "不做因果或时间-事件效应估计。",
    "No dated retrieval source returned current literature for this plan.":
        "没有带日期的检索来源为这份计划返回当前文献。",
    "No retrieved study passed all population, exposure, outcome, and design-excerpt checks as a direct comparator.":
        "没有检索到的研究在人群、暴露、结局和设计上都通过核对，可作为直接对照。",
    "The study-family playbook calls for robustness evidence, but no typed, executable sensitivity authority was prespecified for this study version.":
        "这类研究需要稳健性证据，但这个研究版本没有预先规定可执行的敏感性分析。",
    "Without a direct comparator, the system cannot distinguish a genuinely novel design from a new database instantiation.":
        "没有直接对照，系统无法区分真正新颖的设计和换一个数据库重做。",
}

# Each replayed trace row spans these audit-log events: (row id, the event
# that opens it, the event that closes it). Patterns match one event each.
SPANS = (
    ("p-scope", r"^Execution runtime validated", r"^Research context built"),
    ("p-cohort", r"^Research context built", r"^Initial cohort audit passed"),
    ("p-literature", r"^Building pre-plan literature", r"^Generating plan draft 1/"),
    ("p-draft", r"^Generating plan draft 1/", r"^Plan draft 1/\d+ passed"),
    ("review-wait", r"^Analysis plan ready", r"^Skipped step already completed by pre-execution: 00_probe"),
    ("e-2", r"^Step 2/6 started", r"^Step 2/6 complete"),
    ("e-3", r"^Step 3/6 started", r"^Step 3/6 complete"),
    ("e-4", r"^Step 4/6 started", r"^Step 4/6 complete"),
    ("e-5", r"^Step 5/6 started", r"^Step 5/6 complete"),
    ("e-6", r"^Step 6/6 started", r"^Step 6/6 complete"),
    ("e-figures", r"^Auditing generated figures", r"^Drafting manuscript scaffold"),
    ("e-writer", r"^Drafting manuscript scaffold", r"^STRICT evidence enforcement blocked"),
    ("e-register", r"^STRICT evidence enforcement blocked", r"^Research-agent run complete"),
)

# The evidence a result number travels through, in reading order.
CHAIN = (
    "table_step_artifact_257971d9f784c410",
    "code_analysis_27260fbc62111090",
    "figure_step_artifact_1c86f17d397e23b3",
    "table_step_artifact_15c34169dc846f9a",
    "code_analysis_8de4337c99d89a1b",
)
CODE_EXCERPT_LINES = 12


def fail(message: str) -> None:
    raise SystemExit(f"build_demo_walkthrough_trace: {message}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def zh(text: str | None) -> str | None:
    if text is None:
        return None
    if text not in ZH:
        fail(f"no translation for record text: {text!r}")
    return ZH[text]


def dictionary_at(path: str, digest: str) -> tuple[str, dict[str, Any]]:
    """Return (commit, parsed dictionary) whose bytes hash to ``digest``."""

    commits = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--format=%H", "--all", "--", path],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    for commit in commits:
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
            capture_output=True,
        )
        if blob.returncode == 0 and sha256_bytes(blob.stdout) == digest:
            return commit, json.loads(blob.stdout)
    fail(f"no commit of {path} has sha256 {digest}")
    raise AssertionError


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def spans(audit: list[dict[str, Any]]) -> dict[str, list[str]]:
    def find(pattern: str) -> str:
        hits = [row["timestamp"] for row in audit if re.search(pattern, row["event"])]
        if len(hits) != 1:
            fail(f"audit pattern {pattern!r} matched {len(hits)} events")
        return hits[0]

    result = {}
    for row_id, opening, closing in SPANS:
        start, end = find(opening), find(closing)
        if parse_time(end) < parse_time(start):
            fail(f"{row_id} ends before it starts")
        result[row_id] = [start, end]
    return result


def concept_closure(roots: list[str], sofa2: dict, concepts: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pending = list(roots)
    while pending:
        code = pending.pop(0)
        if code in out:
            continue
        entry = sofa2.get(code) or concepts.get(code)
        if entry is None:
            out[code] = {"missing": True}
            continue
        children = entry.get("concepts") or []
        children = [children] if isinstance(children, str) else list(children)
        sources = (entry.get("sources") or {}).get("miiv") or []
        out[code] = {
            key: entry[key]
            for key in (
                "description", "class_name", "unit", "min", "max", "category",
                "definition_source", "definition_version", "clinical_status",
                "canonical_definition", "callback", "levels",
            )
            if key in entry
        }
        out[code]["dictionary"] = "sofa2" if code in sofa2 else "concept"
        if children:
            out[code]["children"] = children
        if sources:
            out[code]["miiv"] = [
                {
                    key: source[key]
                    for key in ("table", "ids", "sub_var", "val_var", "index_var", "callback", "regex")
                    if key in source
                }
                for source in sources
            ]
        pending.extend(children)
    return out


def result_rows(table: bytes) -> list[dict[str, Any]]:
    rows = []
    for row in csv.DictReader(io.StringIO(table.decode("utf-8"))):
        rows.append({
            "role": row["row_role"],
            "level": None if row["exposure_level"] == "" else int(float(row["exposure_level"])),
            "stays": int(row["n_rows"]),
            "denominator": int(row["exposure_denominator"]),
            "share_pct": float(row["exposure_pct"]),
            "deaths": int(row["outcome_events"]),
            "outcome_denominator": int(row["outcome_denominator"]),
            "rate_pct": float(row["outcome_rate_pct"]),
            "interval_method": row["interval_method"],
        })
    return rows


def build(run_dir: Path) -> dict[str, Any]:
    wrapper = run_dir.parent.parent
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    audit_bytes = (run_dir / "audit_log.jsonl").read_bytes()
    audit = [json.loads(line) for line in audit_bytes.decode("utf-8").splitlines() if line.strip()]
    finished = [row for row in audit if row["event"] == "Research-agent run complete."]
    if len(finished) != 1:
        fail("the audit log has no single completion event")

    fingerprint = read_json(run_dir / "concept_dict_fingerprint.json")
    concept_commit, concepts = dictionary_at(CONCEPT_DICT, fingerprint["concept_dict_sha"])
    sofa2_commit, sofa2 = dictionary_at(SOFA2_DICT, fingerprint["sofa2_dict_sha"])

    provenance = read_json(run_dir / "cohort_provenance.json")
    roots = [term["concept"] for term in TERMS]
    bound = set(provenance["feature_concepts"]) | set(provenance["outcome_concepts"]) | set(provenance["static_concepts"])
    if not set(roots) <= bound:
        fail(f"demo terms {roots} are not all concepts the run bound ({sorted(bound)})")

    plan = read_json(run_dir / "analysis_plan.json")
    cohort = plan["cohort"]
    inclusion = [
        {"concept": rule["concept_id"], "op": rule["op"], "value": rule["value"],
         "window_hours": [rule["time_window"]["start_offset_hours"], rule["time_window"]["end_offset_hours"]]}
        for rule in cohort["inclusion"]
    ]
    steps = []
    bindings = []
    for step in plan["steps"]:
        raw = step.get("literature_design_bindings") or []
        raw = json.loads(raw) if isinstance(raw, str) else raw
        steps.append({
            "step_id": step["step_id"], "role": step["planned_analysis_role"],
            "method": step["method"], "citations": list(step.get("literature_citation_keys") or []),
            "claim_ceiling": (step.get("descriptive_claim") or {}).get("claim_ceiling"),
            "limitations": list((step.get("descriptive_claim") or {}).get("unresolved_limitations") or []),
        })
        for binding in raw:
            bindings.append({
                "step_id": step["step_id"], "key": binding["citation_key"],
                "elements": binding["design_elements"],
                "application": binding["application"], "application_zh": zh(binding["application"]),
                "divergence": binding.get("divergence"), "divergence_zh": zh(binding.get("divergence")),
            })

    review = read_json(run_dir / "scientific_plan_review.json")
    literature = read_json(wrapper / "literature_evidence.json")
    by_id = {row["evidence_id"]: row for row in manifest["evidence"]}
    records = {}
    for evidence_id in CHAIN:
        row = by_id.get(evidence_id) or fail(f"{evidence_id} is not in the run manifest")
        data = (run_dir / row["relative_path"]).read_bytes()
        if sha256_bytes(data) != row["sha256"]:
            fail(f"{evidence_id} no longer matches its registered sha256")
        records[evidence_id] = {
            "kind": row["kind"], "step": row["produced_by_step"], "sha256": row["sha256"],
            "file": row["relative_path"].split("__", 1)[-1], "inputs": row["inputs"],
            "script": row["script_evidence_id"], "producer": row["producer"],
            "generation_mode": row["generation_mode"], "created_at": row["created_at"],
        }
    table_bytes = (run_dir / by_id[CHAIN[0]]["relative_path"]).read_bytes()
    code_bytes = (run_dir / by_id[CHAIN[1]]["relative_path"]).read_bytes()

    demo_figure = ROOT / "src/easyicu/webserver/static/assets/demo/sofa2-phenotype-mortality.png"
    if sha256_bytes(demo_figure.read_bytes()) != records[CHAIN[2]]["sha256"]:
        fail("the demo's main figure is not the run's registered figure")

    return {
        "schema_version": SCHEMA,
        "source_run_id": manifest["run_id"],
        "sources": {
            "run_manifest_sha256": sha256_bytes(manifest_bytes),
            "audit_log_sha256": sha256_bytes(audit_bytes),
            "concept_dict": {"commit": concept_commit, "sha256": fingerprint["concept_dict_sha"]},
            "sofa2_dict": {"commit": sofa2_commit, "sha256": fingerprint["sofa2_dict_sha"]},
        },
        "started_at": audit[0]["timestamp"],
        "finished_at": finished[0]["timestamp"],
        # The plan review is not an audit event; its record carries its time.
        "spans": {**spans(audit), "p-review": [spans(audit)["p-draft"][1], review["generated_at"].replace("+00:00", "Z")]},
        "dictionary": {
            "database": provenance["database"],
            "window_hours": provenance["cohort_window_hours"],
            "terms": list(TERMS),
            "concepts": concept_closure(roots, sofa2, concepts),
        },
        "cohort": {
            "name": cohort["name"], "inclusion": inclusion,
            "n_stays": provenance["n_stays_after_inclusion_exclusion"],
            "n_extracted": provenance["n_stays_extracted"],
            "cohort_sha256": provenance["cohort_sha256"],
            "export_manifest_sha256": provenance["export_authority"]["manifest_sha256"],
        },
        "plan": {
            "revision": plan["revision"], "steps": steps, "bindings": bindings,
            "review": {
                "score": review["score"], "approval_allowed": review["approval_allowed"],
                "dimension_scores": review["dimension_scores"],
                "findings": [
                    {"code": f["code"], "severity": f["severity"], "dimension": f["dimension"],
                     "message": f["message"], "message_zh": zh(f["message"])}
                    for f in review["findings"]
                ],
            },
        },
        "literature": [
            {key: row.get(key) for key in ("key", "title", "year", "venue", "pmid", "source_url")}
            for row in literature["citations"]
        ],
        "evidence": {
            "count": finished[0]["detail"]["evidence_count"],
            "chain": list(CHAIN),
            "records": records,
            "result_rows": result_rows(table_bytes),
            "code_excerpt": code_bytes.decode("utf-8").splitlines()[:CODE_EXCERPT_LINES],
        },
        "writer": {"event": next(row["event"] for row in audit if row["event"].startswith("STRICT evidence"))},
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    record = build(args.run_dir.resolve())
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
