"""Four-quadrant ablation: naive vs aware x mock vs real LLM.

This extends ``research_agent_ablation.py`` by adding a real-LLM arm.
It requires an OpenAI-compatible endpoint, e.g. OpenRouter:

    export OPENROUTER_API_KEY=...
    export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    python examples/research_agent_real_llm_ablation.py --provider openrouter

Outputs:
  research_output/ablation_real/<timestamp>/ablation_4q_summary.json
  research_output/ablation_real/<timestamp>/ablation_4q_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _bootstrap():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from research_agent_mortality_sofa import build_synthetic_cohort  # type: ignore
    from easyicu.research_agent import (
        ConceptUsageAuditor,
        MockLLMClient,
        OpenAIClient,
        ResearchAgentPipeline,
        build_research_context,
    )
    return (
        repo_root,
        build_synthetic_cohort,
        ConceptUsageAuditor,
        MockLLMClient,
        OpenAIClient,
        ResearchAgentPipeline,
        build_research_context,
    )


def _make_real_client(provider: str, model: str, OpenAIClient):
    from easyicu.research_agent.llm import openrouter_reasoning_extra_body

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is required for --provider openrouter")
        kwargs = dict(
            model=model,
            api_key=key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            request_timeout=180.0,
            extra_headers={
                "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                "X-Title": "EasyICU research-agent four-quadrant ablation",
            },
        )
        extra_body = openrouter_reasoning_extra_body(model)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return OpenAIClient(**kwargs)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required for --provider openai")
    return OpenAIClient(model=model, api_key=key, request_timeout=180.0)


def _summarise(run_dir: Path, *, audit_context=None, ConceptUsageAuditor=None) -> Dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bound = run_dir / "manuscript_scaffold_bound.md"
    n_steps_planned, n_steps_ok = _step_coverage(run_dir, manifest)
    return {
        "run_id": manifest.get("run_id"),
        "workdir": str(run_dir),
        "n_evidence": len(manifest.get("evidence", [])),
        "n_findings": len(manifest.get("findings", [])),
        "n_errors": sum(1 for f in manifest.get("findings", []) if f.get("severity") == "error"),
        "n_warnings": sum(1 for f in manifest.get("findings", []) if f.get("severity") == "warning"),
        "n_steps_planned": n_steps_planned,
        "n_steps_ok": n_steps_ok,
        "evidence_missing": bound.read_text(encoding="utf-8").count("[evidence missing:") if bound.exists() else -1,
        "sofa_zero_anomaly": _sofa_zero_anomaly(run_dir),
        "forbidden_aggregation_count": _forbidden_aggregation_count(
            run_dir,
            audit_context=audit_context,
            ConceptUsageAuditor=ConceptUsageAuditor,
        ),
    }


def _step_coverage(run_dir: Path, manifest: Dict[str, Any]) -> tuple[int, int]:
    plan_path = run_dir / "analysis_plan.json"
    n_planned = 0
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            n_planned = len(plan.get("steps", []) or [])
        except Exception:
            n_planned = 0

    records = manifest.get("per_step_records") or []
    if not records:
        partial = run_dir / "manifest_partial.json"
        if partial.exists():
            try:
                records = json.loads(partial.read_text(encoding="utf-8")).get(
                    "per_step_records", []
                ) or []
            except Exception:
                records = []
    n_ok = sum(1 for r in records if isinstance(r, dict) and r.get("status") == "ok")
    return n_planned, n_ok


def _forbidden_aggregation_count(run_dir: Path, *, audit_context, ConceptUsageAuditor) -> int:
    """Post-hoc full-context audit of final scripts for paper ablations.

    The naive arms intentionally run with column-only context, so their
    online validator cannot know that ``sofa2`` is a composite ordinal
    score. For the comparison table we replay the final scripts against the
    full ICU-aware context to quantify unsafe generated code directly.
    """
    if audit_context is None or ConceptUsageAuditor is None:
        return 0
    auditor = ConceptUsageAuditor()
    script_paths = sorted((run_dir / "evidence").glob("code_*__analysis.py"))
    if not script_paths:
        script_paths = sorted((run_dir / "steps").glob("*/analysis.py"))
    n = 0
    for path in script_paths:
        try:
            findings = auditor.audit(
                context=audit_context,
                script_text=path.read_text(encoding="utf-8", errors="replace"),
            )
        except Exception:
            continue
        n += sum(1 for f in findings if f.severity == "error")
    return n


def _sofa_zero_anomaly(run_dir: Path) -> bool:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            finding_msgs = " ".join(
                f.get("message", "") for f in manifest.get("findings", [])
            ).lower()
            if "non-monotonic" in finding_msgs or "score==0" in finding_msgs:
                return True
        except Exception:
            pass
    for path in run_dir.rglob("step_summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("sofa_zero_anomaly"):
            return True
    return False


def _run_arm(
    *,
    label: str,
    llm,
    disable_icu_context: bool,
    cohort,
    out_dir: Path,
    ResearchAgentPipeline,
    audit_context,
    ConceptUsageAuditor,
):
    pipeline = ResearchAgentPipeline(
        workdir=out_dir / label,
        llm=llm,
        disable_icu_context=disable_icu_context,
    )
    result = pipeline.run(
        question="Is admission SOFA-2 score associated with ICU mortality?",
        cohort=cohort.copy(),
        cohort_name=f"ablation_4q_{label}",
        database="synthetic",
        target_outcome="death",
        cross_database_validation=["miiv", "eicu"],
    )
    return _summarise(
        Path(result.workdir),
        audit_context=audit_context,
        ConceptUsageAuditor=ConceptUsageAuditor,
    )


def _reuse_arm_if_complete(
    out_root: Path,
    label: str,
    *,
    audit_context,
    ConceptUsageAuditor,
):
    arm_dir = out_root / label
    if not arm_dir.exists():
        return None
    runs = sorted(
        (p for p in arm_dir.glob("run_*") if (p / "manifest.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not runs:
        return None
    return _summarise(
        runs[0],
        audit_context=audit_context,
        ConceptUsageAuditor=ConceptUsageAuditor,
    )


def main() -> int:
    (
        repo_root,
        build_synthetic_cohort,
        ConceptUsageAuditor,
        MockLLMClient,
        OpenAIClient,
        ResearchAgentPipeline,
        build_research_context,
    ) = _bootstrap()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openrouter", "openai"], default="openrouter")
    parser.add_argument(
        "--model",
        default=os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL", "openai/gpt-oss-120b:free"),
    )
    parser.add_argument(
        "--out-root",
        default=str(
            repo_root / "research_output" / "ablation_real"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed arm runs already present under --out-root.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cohort = build_synthetic_cohort()
    audit_context = build_research_context(
        research_question="Is admission SOFA-2 score associated with ICU mortality?",
        cohort=cohort,
        cohort_name="ablation_4q_full_context_audit",
        database="synthetic",
        target_outcome="death",
        cross_database_validation=["miiv", "eicu"],
    )
    real = None

    arms = {}
    arm_specs = [
        ("mock_naive", "mock", True),
        ("mock_aware", "mock", False),
        ("real_naive", "real", True),
        ("real_aware", "real", False),
    ]
    for label, llm_kind, disable_icu_context in arm_specs:
        if args.reuse_existing:
            reused = _reuse_arm_if_complete(
                out_root,
                label,
                audit_context=audit_context,
                ConceptUsageAuditor=ConceptUsageAuditor,
            )
            if reused is not None:
                arms[label] = reused
                continue
        if llm_kind == "mock":
            llm = MockLLMClient()
        else:
            if real is None:
                real = _make_real_client(args.provider, args.model, OpenAIClient)
            llm = real
        arms[label] = _run_arm(
            label=label,
            llm=llm,
            disable_icu_context=disable_icu_context,
            cohort=cohort,
            out_dir=out_root,
            ResearchAgentPipeline=ResearchAgentPipeline,
            audit_context=audit_context,
            ConceptUsageAuditor=ConceptUsageAuditor,
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "arms": arms,
    }
    (out_root / "ablation_4q_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Four-quadrant ablation",
        "",
        "| Arm | Evidence | Steps ok/planned | Forbidden aggregations | Findings | SOFA-zero anomaly | `[evidence missing]` |",
        "|---|---:|---:|---:|---:|:-:|---:|",
    ]
    for name, s in arms.items():
        lines.append(
            f"| `{name}` | {s['n_evidence']} | "
            f"{s['n_steps_ok']}/{s['n_steps_planned']} | "
            f"{s['forbidden_aggregation_count']} | {s['n_findings']} | "
            f"{'yes' if s['sofa_zero_anomaly'] else 'no'} | {s['evidence_missing']} |"
        )
    (out_root / "ablation_4q_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {out_root / 'ablation_4q_summary.json'}")
    print(f"  -> {out_root / 'ablation_4q_summary.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
