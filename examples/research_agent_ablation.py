"""Ablation: agent **with** vs **without** ICU context.

Run the *same* synthetic SOFA cohort through the *same* pipeline
twice — once with the ICU-aware context (variable kinds, allowed
aggregations, pitfalls) and once with a naive context that strips all
of that out — and emit a side-by-side summary.

Run it with::

    python examples/research_agent_ablation.py

Outputs land in ``./research_output/ablation/<run_id>/`` and include:

* ``ablation_summary.json`` — machine-readable comparison.
* ``ablation_summary.md``    — Markdown comparison with figure caption.
* ``naive/manifest.json``    — naive run provenance.
* ``aware/manifest.json``    — ICU-aware run provenance.

The script is deterministic; reviewers should be able to re-run it
and obtain bit-identical numbers.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _import_repo_modules():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    # also reuse the synthetic cohort generator from the existing demo
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from research_agent_mortality_sofa import build_synthetic_cohort  # type: ignore
    from easyicu.research_agent import ResearchAgentPipeline
    from easyicu.research_agent.llm import MockLLMClient
    return build_synthetic_cohort, ResearchAgentPipeline, MockLLMClient, repo_root


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _findings_by_validator(manifest: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for f in manifest.get("findings", []):
        v = f.get("validator", "unknown")
        sev = f.get("severity", "info")
        out.setdefault(v, {"info": 0, "warning": 0, "error": 0})
        out[v][sev] = out[v].get(sev, 0) + 1
    return out


def _sofa_anomaly_flagged(run_dir: Path) -> bool:
    for ssj in run_dir.rglob("step_summary.json"):
        try:
            data = json.loads(ssj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("sofa_zero_anomaly"):
            return True
    return False


def _evidence_missing_count(run_dir: Path) -> int:
    bound = run_dir / "manuscript_scaffold_bound.md"
    if not bound.exists():
        return -1
    return bound.read_text(encoding="utf-8").count("[evidence missing:")


def _forbidden_aggregations(manifest: Dict[str, Any]) -> int:
    """Count concept_usage_auditor errors (mean-of-ordinal, etc.)."""
    n = 0
    for f in manifest.get("findings", []):
        if f.get("validator") == "concept_usage_auditor" and f.get("severity") == "error":
            n += 1
    return n


def _summarise_arm(label: str, run_dir: Path) -> Dict[str, Any]:
    manifest = _load_manifest(run_dir)
    return {
        "arm": label,
        "run_id": manifest.get("run_id"),
        "n_evidence": len(manifest.get("evidence", [])),
        "findings_by_validator": _findings_by_validator(manifest),
        "n_findings": len(manifest.get("findings", [])),
        "n_errors": sum(1 for f in manifest.get("findings", []) if f.get("severity") == "error"),
        "n_warnings": sum(1 for f in manifest.get("findings", []) if f.get("severity") == "warning"),
        "sofa_zero_anomaly_flagged": _sofa_anomaly_flagged(run_dir),
        "evidence_missing_in_manuscript": _evidence_missing_count(run_dir),
        "forbidden_aggregations_caught": _forbidden_aggregations(manifest),
    }


def _markdown_table(naive: Dict[str, Any], aware: Dict[str, Any]) -> str:
    rows = [
        ("Number of registered evidence artefacts", naive["n_evidence"], aware["n_evidence"]),
        ("Total validator findings", naive["n_findings"], aware["n_findings"]),
        ("Errors raised", naive["n_errors"], aware["n_errors"]),
        ("Warnings raised", naive["n_warnings"], aware["n_warnings"]),
        ("SOFA-zero anomaly flagged",
         "yes" if naive["sofa_zero_anomaly_flagged"] else "no",
         "yes" if aware["sofa_zero_anomaly_flagged"] else "no"),
        ("`[evidence missing]` lines in manuscript",
         naive["evidence_missing_in_manuscript"],
         aware["evidence_missing_in_manuscript"]),
        ("Forbidden-aggregation findings",
         naive["forbidden_aggregations_caught"],
         aware["forbidden_aggregations_caught"]),
    ]
    lines = [
        "| Metric | Naive (no ICU context) | ICU-aware |",
        "|---|---:|---:|",
    ]
    for name, n, a in rows:
        lines.append(f"| {name} | {n} | {a} |")
    return "\n".join(lines)


def main() -> int:
    build_synthetic_cohort, ResearchAgentPipeline, MockLLMClient, repo_root = _import_repo_modules()

    cohort = build_synthetic_cohort()
    out_root = repo_root / "research_output" / "ablation" / datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    naive_dir = out_root / "naive"
    aware_dir = out_root / "aware"
    naive_dir.mkdir()
    aware_dir.mkdir()

    common_kwargs = dict(
        question="Is admission SOFA-2 score associated with ICU mortality?",
        cohort=cohort,
        cohort_name="ablation_synthetic_cohort",
        database="synthetic",
        target_outcome="death",
        cross_database_validation=["miiv", "eicu"],
        inclusion_criteria=["First ICU admission", "Age >= 18 years", "ICU LoS >= 6 hours"],
        exclusion_criteria=["Discharged within first 6 hours"],
    )

    print("=== ABLATION ARM 1: NAIVE (no ICU context) ===")
    naive_pipeline = ResearchAgentPipeline(
        workdir=naive_dir, llm=MockLLMClient(), disable_icu_context=True,
    )
    naive_result = naive_pipeline.run(**common_kwargs)

    print("\n=== ABLATION ARM 2: ICU-AWARE ===")
    aware_pipeline = ResearchAgentPipeline(
        workdir=aware_dir, llm=MockLLMClient(), disable_icu_context=False,
    )
    aware_result = aware_pipeline.run(**common_kwargs)

    naive = _summarise_arm("naive", Path(naive_result.workdir))
    aware = _summarise_arm("aware", Path(aware_result.workdir))
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cohort": {
            "n_stays": int(len(cohort)),
            "sofa2_zero_pct": float((cohort["sofa2"] == 0).mean()),
            "death_rate": float(cohort["death"].mean()),
        },
        "naive": naive,
        "aware": aware,
        "verdict": {
            "anomaly_only_in_aware":
                aware["sofa_zero_anomaly_flagged"] and not naive["sofa_zero_anomaly_flagged"],
            "more_validator_signal_in_aware":
                aware["n_warnings"] + aware["n_errors"]
                > naive["n_warnings"] + naive["n_errors"],
            "fewer_evidence_holes_in_aware":
                aware["evidence_missing_in_manuscript"] <= naive["evidence_missing_in_manuscript"],
        },
    }
    (out_root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Ablation: with vs. without ICU context",
        "",
        f"_Generated {summary['generated_at']} on a synthetic ICU cohort with "
        f"{summary['cohort']['n_stays']} stays "
        f"({summary['cohort']['sofa2_zero_pct']:.1%} of which have sofa2==0; "
        f"overall mortality {summary['cohort']['death_rate']:.1%})._",
        "",
        _markdown_table(naive, aware),
        "",
        "## Interpretation",
        "",
        "* **Anomaly detection.** "
        + ("Only the ICU-aware arm flagged the sofa2==0 missingness anomaly."
           if summary["verdict"]["anomaly_only_in_aware"]
           else "Both arms flagged (or both missed) the sofa2==0 anomaly."),
        "* **Validator signal.** "
        + (f"The ICU-aware arm raised {aware['n_warnings']} warnings + "
           f"{aware['n_errors']} errors vs the naive arm's "
           f"{naive['n_warnings']} + {naive['n_errors']}."),
        "* **Manuscript bindability.** "
        + ("ICU-aware arm produced no `[evidence missing]` placeholders in the "
           "bound manuscript scaffold; naive arm left "
           f"{naive['evidence_missing_in_manuscript']} unresolved."
           if aware["evidence_missing_in_manuscript"] == 0
           else f"`[evidence missing]` lines: naive={naive['evidence_missing_in_manuscript']}, "
                f"aware={aware['evidence_missing_in_manuscript']}."),
        "",
        f"Naive run: `{naive_result.workdir}`",
        f"ICU-aware run: `{aware_result.workdir}`",
    ]
    (out_root / "ablation_summary.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print("=== Ablation summary ===")
    print(f"  -> {out_root / 'ablation_summary.json'}")
    print(f"  -> {out_root / 'ablation_summary.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
