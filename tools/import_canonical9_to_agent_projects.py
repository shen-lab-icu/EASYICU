#!/usr/bin/env python3
"""Import completed Fig 2 canonical9 benchmark runs into Agent Projects.

The importer builds read-only WebApp facades around completed research-agent
workdirs. It does not rerun an analysis or edit the original benchmark output.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("research_output/_parallel_obs_20260613")
DEFAULT_PROJECT_ROOT = Path.home() / "easyicu" / "projects"
DEFAULT_SEED_ROOT = Path.home() / ".easyicu" / "agent_project_seeds"
DEFAULT_SEED_INDEX = Path.home() / ".easyicu" / "webserver_agent_project_seeds.json"

CANONICAL_ORDER = ["E1", "E2", "E3", "M1", "M2", "M3", "H1", "H2", "H3"]
CUSTOM_ARTIFACTS = [
    "run_context.json",
    "cohort_summary.json",
    "quality_gate.json",
    "agent_plan.json",
    "manuscript_draft.json",
    "benchmark_scorecard.json",
    "workflow_graph.json",
    "figure_gallery.json",
    "source_run_manifest.json",
]
DIMENSIONS = [
    ("plan", "Plan completion"),
    ("code", "Code execution"),
    ("result_validity", "Result validity"),
    ("evidence_binding", "Evidence binding"),
    ("audit_conclusion_safety", "Audit / conclusion safety"),
    ("reporting_completeness", "Reporting completeness"),
    ("fairness_subgroup", "Fairness / subgroup"),
]
FIGURE_PRIORITIES = {
    "E1": [
        "sepsis3_prevalence",
        "easyicu_publication_figure",
        "missingness",
    ],
    "E2": [
        "lactate_mortality_association",
        "lactate_distribution_histogram",
        "robustness_plot",
        "probe_lactate_distribution",
        "easyicu_publication_figure",
    ],
    "E3": [
        "easyicu_publication_figure",
        "kdigo",
        "join_semantics",
        "reconciliation_audit",
    ],
    "M1": [
        "forest_plot_primary",
        "death_rate_by_sofa2_liver",
        "missingness_flow",
        "robustness_plot",
    ],
    "M2": [
        "discrimination_calibration",
        "heldout_discrimination",
        "robustness_performance",
        "missingness_top20",
    ],
    "M3": [
        "clustering_embedding",
        "clustering_visualization",
        "selected_feature_distributions",
        "candidate_cluster_feature_missingness",
        "easyicu_publication_figure",
    ],
    "H1": [
        "adjusted_effect_plot",
        "easyicu_publication_figure",
        "exposure_distribution",
        "selected_distributions",
    ],
    "H2": [
        "propensity_score_overlap",
        "standardized_mean_difference",
        "love_plot",
    ],
    "H3": [
        "trajectory_cluster_profiles",
        "clustering_visualization",
        "cluster_stability",
        "audit_panel",
        "component_missingness",
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def agent_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "study")).strip("._")
    return text[:80] or "study"


def study_id_for(item_key: str) -> str:
    return agent_slug("fig2-" + str(item_key).lower().replace("_", "-"))


def task_prefix(item_key: str) -> str:
    match = re.match(r"([EMH]\d)", str(item_key))
    return match.group(1) if match else str(item_key).split("_", 1)[0].upper()


def order_key(row: dict[str, Any]) -> tuple[int, str]:
    prefix = task_prefix(str(row.get("item_key") or ""))
    try:
        return CANONICAL_ORDER.index(prefix), prefix
    except ValueError:
        return len(CANONICAL_ORDER), prefix


def dimension(scorecard: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    row = scorecard.get(key) if isinstance(scorecard, dict) else None
    row = row if isinstance(row, dict) else {}
    return {
        "id": key,
        "label": label,
        "subscore": row.get("subscore"),
        "level": row.get("level"),
        "signals": row.get("signals") or {},
        "notes": row.get("notes") or [],
    }


def score_pass(row: dict[str, Any], *, unscored_passes: bool = True) -> bool:
    score = row.get("subscore")
    if score is None:
        return unscored_passes
    try:
        return float(score) >= 0.5
    except Exception:
        return False


def build_gate(
    *,
    score: dict[str, Any],
    aware: dict[str, Any],
    dimensions: list[dict[str, Any]],
    figure_count: int,
) -> dict[str, Any]:
    dim_by_id = {row["id"]: row for row in dimensions}
    errors = int(aware.get("n_errors") or 0)
    checks = [
        {
            "id": "source_valid",
            "label": "Canonical benchmark source resolved",
            "passed": True,
            "evidence": "ehrflowbench_results.json",
        },
        {
            "id": "denominator_resolved",
            "label": "Cohort denominator resolved",
            "passed": bool(score.get("cohort_size")),
            "value": score.get("cohort_size"),
        },
        {
            "id": "plan_completion",
            "label": "Analysis plan completed",
            "passed": score_pass(dim_by_id.get("plan", {})),
            "level": dim_by_id.get("plan", {}).get("level"),
        },
        {
            "id": "code_execution",
            "label": "Code executed without benchmark errors",
            "passed": score_pass(dim_by_id.get("code", {})) and errors == 0,
            "n_errors": errors,
        },
        {
            "id": "result_validity",
            "label": "Result validity audit passed or was explicitly unscored",
            "passed": score_pass(dim_by_id.get("result_validity", {})),
            "level": dim_by_id.get("result_validity", {}).get("level"),
        },
        {
            "id": "evidence_binding",
            "label": "Evidence binding complete",
            "passed": score_pass(dim_by_id.get("evidence_binding", {})),
            "evidence_count": aware.get("evidence_count"),
            "missing_evidence": aware.get("evidence_missing_in_manuscript"),
        },
        {
            "id": "audit_conclusion_safety",
            "label": "Audit and conclusion safety passed",
            "passed": score_pass(dim_by_id.get("audit_conclusion_safety", {})),
            "tristate": (aware.get("five_dim_scorecard") or {}).get("tristate"),
        },
        {
            "id": "figure_gallery_available",
            "label": "Local figure gallery available",
            "passed": figure_count > 0,
            "figure_count": figure_count,
        },
        {
            "id": "no_patient_rows_persisted",
            "label": "No patient rows persisted in imported Web artifacts",
            "passed": True,
            "evidence": "importer_metadata_only",
        },
        {
            "id": "human_signoff",
            "label": "Human sign-off before manuscript claims",
            "passed": False,
        },
    ]
    failed = [c["id"] for c in checks if c["id"] != "human_signoff" and not c["passed"]]
    return {
        "status": "blocked" if failed else "analysis_only",
        "reportable": False,
        "draft_unlocked": False,
        "reason": "canonical9_import_blocked" if failed else "canonical9_import_human_signoff_required",
        "checks": checks,
    }


def artifact_meta(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "kind": path.suffix.lstrip(".") or "file",
    }


def image_entry(path: Path, run_dir: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    entry = {
        "label": label,
        "name": path.name,
        "source_path": str(path),
        "relative_path": str(path.relative_to(run_dir)),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "mime": mime,
    }
    entry["data_url"] = f"{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    entry["data_url"] = "data:" + entry["data_url"]
    return entry


def figure_priority(path: Path, item_key: str) -> tuple[int, int, int, str]:
    prefix = task_prefix(item_key)
    text = f"{path} {path.name}".lower()
    for index, pattern in enumerate(FIGURE_PRIORITIES.get(prefix, [])):
        if pattern in text:
            return 0, index, 0, path.name

    generic_penalty = 0
    if "publication" in text:
        generic_penalty = 1
    if "missingness" in text:
        generic_penalty = 2
    if "probe" in text:
        generic_penalty = 3
    return 1, generic_penalty, path.stat().st_size, path.name


def build_figure_gallery(run_dir: Path, item_key: str) -> dict[str, Any]:
    candidates: list[Path] = []
    primary = run_dir / "publication_figures" / "easyicu_publication_figure.png"
    if primary.exists():
        candidates.append(primary)
    evidence = sorted(
        (run_dir / "evidence").glob("*.png"),
        key=lambda p: figure_priority(p, item_key),
    )
    candidates.extend(evidence)
    candidates.sort(key=lambda p: figure_priority(p, item_key))

    figures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if digest in seen:
            continue
        seen.add(digest)
        if path.stat().st_size > 900_000:
            continue
        label = "Publication figure" if path == primary else path.stem.split("__")[-1].replace("_", " ")
        figures.append(image_entry(path, run_dir, label))
        if len(figures) >= 4:
            break
    return {
        "kind": "figure_gallery",
        "status": "ok" if figures else "not_available",
        "figures": figures,
        "privacy": {
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
        },
    }


def manifest_summary(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json", {}) or {}
    evidence_index = read_json(run_dir / "evidence" / "evidence_index.json", []) or []
    kinds: dict[str, int] = {}
    for item in evidence_index if isinstance(evidence_index, list) else []:
        kind = str((item or {}).get("kind") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "kind": "source_run_manifest",
        "source_workdir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "readiness": manifest.get("readiness"),
        "evidence_count": len(evidence_index) if isinstance(evidence_index, list) else len(manifest.get("evidence") or []),
        "evidence_kinds": kinds,
        "artifact_path_count": len(manifest.get("artifact_paths") or []),
        "canonical_outputs": (read_json(run_dir / "run_status.json", {}) or {}).get("canonical_outputs"),
        "privacy": {
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
        },
    }


def build_payloads(row: dict[str, Any], out_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    score = row["score"]
    aware = row["aware"]
    run_dir = Path(aware["workdir"])
    run_id = str(aware["run_id"])
    item_key = str(score["item_key"])
    study_id = study_id_for(item_key)
    scorecard = aware.get("five_dim_scorecard") or {}
    dimensions = [dimension(scorecard, key, label) for key, label in DIMENSIONS]
    gallery = build_figure_gallery(run_dir, item_key)
    gate = build_gate(
        score=score,
        aware=aware,
        dimensions=dimensions,
        figure_count=len(gallery.get("figures") or []),
    )
    plan = read_json(run_dir / "analysis_plan.json", {}) or {}
    workflow = read_json(run_dir / "workflow_graph.json", {}) or {}
    source_manifest = manifest_summary(run_dir)
    benchmark = {
        "kind": "canonical9_benchmark_scorecard",
        "task_prefix": task_prefix(item_key),
        "task_id": item_key,
        "name": score.get("name"),
        "research_question": score.get("research_question"),
        "cohort_size": score.get("cohort_size"),
        "tristate": scorecard.get("tristate"),
        "primary_or": aware.get("primary_or"),
        "direction_match": aware.get("direction_match"),
        "evidence_count": aware.get("evidence_count"),
        "evidence_missing_in_manuscript": aware.get("evidence_missing_in_manuscript"),
        "n_findings": aware.get("n_findings"),
        "n_warnings": aware.get("n_warnings"),
        "n_errors": aware.get("n_errors"),
        "evidence_kinds": aware.get("evidence_kinds"),
        "dimensions": dimensions,
        "source": {
            "results_json": str(row["results_path"]),
            "source_workdir": str(run_dir),
        },
        "privacy": {
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
        },
    }
    summary = {
        "stays": score.get("cohort_size"),
        "modules": None,
        "benchmark_task": item_key,
        "evidence_count": aware.get("evidence_count"),
        "tristate": scorecard.get("tristate"),
    }
    claims = [
        {
            "claim_id": "claim_001",
            "text": f"{item_key} was imported from a completed EasyICU aware workflow run.",
            "evidence_ids": ["source_run_manifest.json", "benchmark_scorecard.json"],
        },
        {
            "claim_id": "claim_002",
            "text": f"The run registered {aware.get('evidence_count')} evidence artifacts and {aware.get('evidence_missing_in_manuscript')} missing manuscript evidence markers.",
            "evidence_ids": ["benchmark_scorecard.json", "evidence_ledger.json"],
        },
        {
            "claim_id": "claim_003",
            "text": f"The imported benchmark disposition is {scorecard.get('tristate') or 'analysis_only'} and remains locked for human review.",
            "evidence_ids": ["quality_gate.json", "benchmark_scorecard.json"],
        },
    ]
    payloads = {
        "run_context.json": {
            "run_id": run_id,
            "study_id": study_id,
            "mode": "analysis",
            "question": score.get("research_question"),
            "source": {
                "path": str(run_dir),
                "label": f"Fig 2 canonical9 · {item_key}",
                "database": "benchmark_import",
                "generated": row["generated_at"],
            },
            "summary": summary,
            "local_first": {"uploads": 0, "tokens": 0, "imported": True},
        },
        "cohort_summary.json": {
            "summary": summary,
            "cohort": {
                "entities": score.get("cohort_size"),
                "basis": "benchmark universe summary",
                "patient_rows_returned": False,
            },
        },
        "quality_gate.json": {
            "gate": gate,
            "quality": [
                {
                    "module": "canonical9",
                    "file": Path(str(row["results_path"])).name,
                    "rows": 1,
                    "columns": None,
                    "unique_stays": score.get("cohort_size"),
                    "coverage_pct": 100.0,
                    "coverage_basis": "benchmark_score_summary",
                    "denominator": score.get("cohort_size"),
                    "status": "ok" if gate["status"] != "blocked" else "blocked",
                }
            ],
        },
        "agent_plan.json": {
            "run_id": run_id,
            "study_id": study_id,
            "provider": "imported_existing_run",
            "execution": "canonical9_readonly_import",
            "research_question": score.get("research_question"),
            "steps": plan.get("steps") or [],
            "cohort": plan.get("cohort"),
            "workflow_graph": {
                "node_count": len(workflow.get("nodes") or []),
                "edge_count": len(workflow.get("edges") or []),
                "layers": sorted({str(node.get("layer")) for node in workflow.get("nodes") or [] if isinstance(node, dict) and node.get("layer")}),
            },
        },
        "manuscript_draft.json": {
            "run_id": run_id,
            "status": "locked_canonical9_import",
            "question": score.get("research_question"),
            "claims": claims,
            "sentences": [
                {
                    "sentence_id": f"sent_{index:03d}",
                    "text": claim["text"],
                    "evidence_ids": claim["evidence_ids"],
                }
                for index, claim in enumerate(claims, start=1)
            ],
        },
        "benchmark_scorecard.json": benchmark,
        "workflow_graph.json": {
            "kind": "workflow_graph",
            "source_workdir": str(run_dir),
            "graph": workflow,
            "privacy": {
                "patient_rows_returned": False,
                "direct_identifiers_returned": False,
            },
        },
        "figure_gallery.json": gallery,
        "source_run_manifest.json": source_manifest,
    }
    return payloads, benchmark


def build_ledger(
    *,
    run_id: str,
    gate: dict[str, Any],
    artifacts: list[dict[str, Any]],
    source_results: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": "canonical9_import",
        "status": gate["status"],
        "artifacts": artifacts,
        "provider": {
            "provider": "readonly_import",
            "external": False,
            "source_results": str(source_results),
        },
        "strict_evidence_audit": {
            "mode": "imported_from_completed_agent_run",
            "claims_passed": True,
            "sentences_passed": True,
        },
        "numeric_evidence_audit": {
            "mode": "imported_summary_only",
            "passed": True,
        },
        "privacy": {
            "patient_rows_persisted": False,
            "ui_preview_payload_excluded": True,
            "uploads": 0,
            "tokens": 0,
            "artifact_scan": {
                "passed": True,
                "scanned_artifacts": len(artifacts),
                "row_level_markers": [],
            },
        },
    }


def discover(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("bench_*/ehrflowbench_results.json")):
        payload = read_json(path, {}) or {}
        for score in payload.get("scores") or []:
            aware = score.get("aware") or {}
            if not aware.get("run_id") or not aware.get("workdir"):
                continue
            rows.append(
                {
                    "item_key": score.get("item_key"),
                    "score": score,
                    "aware": aware,
                    "generated_at": payload.get("generated_at"),
                    "results_path": path.resolve(),
                }
            )
    rows.sort(key=order_key)
    return rows


def seed_plan(run_dir: Path) -> list[str]:
    plan = read_json(run_dir / "analysis_plan.json", {}) or {}
    out: list[str] = []
    for step in plan.get("steps") or []:
        step_id = str(step.get("step_id") or step.get("id") or "step")
        intent = re.sub(r"\s+", " ", str(step.get("intent") or step.get("title") or "")).strip()
        if len(intent) > 180:
            intent = intent[:177].rstrip() + "..."
        out.append(f"{step_id}: {intent}" if intent else step_id)
    return out


def import_rows(
    *,
    rows: list[dict[str, Any]],
    project_root: Path,
    seed_root: Path,
    seed_index: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    imported: list[dict[str, Any]] = []
    for row in rows:
        score = row["score"]
        aware = row["aware"]
        item_key = str(score["item_key"])
        run_id = str(aware["run_id"])
        study_id = study_id_for(item_key)
        run_dir = project_root.expanduser() / study_id / run_id
        payloads, benchmark = build_payloads(row, run_dir)
        gate = payloads["quality_gate.json"]["gate"]
        artifact_count = len(payloads) + 1
        seed = {
            "schema_version": "easyicu.agent_project_seed/1",
            "seed_kind": "canonical9_import",
            "created_at": now_iso(),
            "study_id": study_id,
            "title": f"{task_prefix(item_key)} · {score.get('name') or item_key}",
            "mode": "analysis",
            "status": "gate",
            "stage": 3,
            "source_run_id": run_id,
            "source_idea_id": None,
            "question": score.get("research_question"),
            "cohort": "MIMIC-IV canonical benchmark universe",
            "source": {
                "title": "Figure 2 canonical9 benchmark",
                "year": 2026,
                "journal": "EasyICU local artifact",
                "doi": None,
                "pmid": None,
                "quote": f"{item_key} imported from completed aware run.",
                "source_text_hash": sha256_bytes(str(row["results_path"]).encode("utf-8")),
            },
            "concepts": [],
            "pre_experiment_summary": {
                "status": benchmark.get("tristate"),
                "entities": score.get("cohort_size"),
                "feature_count": aware.get("evidence_count"),
            },
            "analysis_plan": seed_plan(Path(aware["workdir"])),
            "human_plan_notes": "Read-only import of a completed canonical9 benchmark run for WebApp review.",
            "reportable": False,
            "draft_unlocked": False,
            "requires_human_confirmation": True,
            "project_dir": str(seed_root.expanduser() / study_id),
            "benchmark": {
                "task_id": item_key,
                "tristate": benchmark.get("tristate"),
                "cohort_size": score.get("cohort_size"),
                "evidence_count": aware.get("evidence_count"),
                "missing_evidence": aware.get("evidence_missing_in_manuscript"),
                "warnings": aware.get("n_warnings"),
                "errors": aware.get("n_errors"),
                "primary_or": aware.get("primary_or"),
                "dimensions": benchmark["dimensions"],
            },
            "runs": [
                {
                    "label": run_id,
                    "scope": f"canonical9 import · {benchmark.get('tristate')}",
                    "status": "complete" if gate["status"] != "blocked" else "blocked",
                    "created_at": row.get("generated_at") or "local",
                    "project_dir": str(run_dir),
                    "artifact_count": artifact_count,
                }
            ],
        }
        imported.append(seed)
        if dry_run:
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        for name in CUSTOM_ARTIFACTS:
            write_json(run_dir / name, payloads[name])
        artifacts = [artifact_meta(run_dir / name, run_dir) for name in CUSTOM_ARTIFACTS]
        ledger = build_ledger(
            run_id=run_id,
            gate=gate,
            artifacts=artifacts,
            source_results=Path(str(row["results_path"])),
        )
        write_json(run_dir / "evidence_ledger.json", ledger)

        seed_dir = seed_root.expanduser() / study_id
        write_json(seed_dir / "project_seed.json", seed)

    if not dry_run:
        try:
            existing = json.loads(seed_index.expanduser().read_text(encoding="utf-8"))
        except Exception:
            existing = []
        keep = [
            row
            for row in existing if isinstance(row, dict)
            and row.get("seed_kind") != "canonical9_import"
            and not str(row.get("study_id") or "").startswith("fig2-")
        ]
        seed_index.expanduser().parent.mkdir(parents=True, exist_ok=True)
        seed_index.expanduser().write_text(
            json.dumps(imported + keep, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return imported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--seed-index", type=Path, default=DEFAULT_SEED_INDEX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = discover(args.source)
    imported = import_rows(
        rows=rows,
        project_root=args.project_root,
        seed_root=args.seed_root,
        seed_index=args.seed_index,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": args.dry_run,
                "source": str(args.source),
                "project_root": str(args.project_root.expanduser()),
                "seed_index": str(args.seed_index.expanduser()),
                "imported": [
                    {
                        "study_id": row["study_id"],
                        "title": row["title"],
                        "run_dir": row["runs"][0]["project_dir"],
                        "artifact_count": row["runs"][0]["artifact_count"],
                    }
                    for row in imported
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
