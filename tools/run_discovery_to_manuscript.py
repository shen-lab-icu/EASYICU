#!/usr/bin/env python
"""Run the discovery-to-manuscript handoff.

This is the EasyICU analogue of AI-Scientist's ``idea.json -> experiment ->
figures -> writeup`` launcher. It starts from an idea-mining
``candidate_triage_report.json`` and writes a frozen handoff packet. With
``--run-analysis`` it also materialises a question-specific universe, launches
the aware research-agent workflow, and validates the final article package.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _bootstrap_imports() -> Path:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _bootstrap_imports()

from easyicu.research_agent.data_foundation import (  # noqa: E402
    acquire_universe_for_question,
)
from easyicu.research_agent.discovery_handoff import (  # noqa: E402
    assert_discovery_analysis_ready,
    build_handoff_from_row,
    load_discovery_ledger,
    select_discovery_row,
    write_handoff_packet,
)
from easyicu.research_agent.discovery_package import (  # noqa: E402
    validate_discovery_manuscript_package,
    write_discovery_package_assessment,
)
from easyicu.research_agent.discovery_story_figure import (  # noqa: E402
    render_discovery_story_figure,
)
from easyicu.research_agent.evidence import (  # noqa: E402
    EvidenceRecord,
    EvidenceStore,
    sha256_of_file,
)
from easyicu.research_agent.llm import OpenAIClient  # noqa: E402
from easyicu.research_agent.providers import (  # noqa: E402
    ProviderConfigurationError,
    build_provider_client,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an idea-mining handoff and optionally run analysis/writing."
    )
    parser.add_argument("--triage-report", required=True)
    parser.add_argument("--idea-index", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["agent_selected", "human_curated", "manual_scaffold"],
        default="agent_selected",
    )
    parser.add_argument("--selection-rationale", default=None)
    parser.add_argument("--research-question", default=None)
    parser.add_argument(
        "--target-outcome",
        default=None,
        help=(
            "Explicit endpoint concept. Defaults to the selected ledger row's "
            "resolved_outcome_concept; conflicting values are rejected."
        ),
    )
    parser.add_argument(
        "--human-confirm",
        action="store_true",
        help="Record explicit human approval of the selected go/recommend idea.",
    )
    parser.add_argument("--human-confirmation-note", default=None)
    parser.add_argument(
        "--outcome-concepts",
        default=None,
        help=(
            "Optional consistency assertion for deterministic outcome "
            "materialisation. It must contain exactly the frozen handoff "
            "target; the handoff remains the sole source of truth."
        ),
    )
    parser.add_argument("--database", default="miiv")
    parser.add_argument(
        "--out-root",
        default=str(
            REPO_ROOT
            / "research_output"
            / "discovery_to_manuscript"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        ),
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Materialise universe and launch the aware research-agent workflow.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Prepared EasyICU export directory required when --run-analysis is set.",
    )
    parser.add_argument(
        "--provider", choices=["openai", "openrouter"], default="openai"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL", "gpt-5.4"),
    )
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument(
        "--runner", choices=["auto", "subprocess", "docker"], default="auto"
    )
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--max-total-steps", type=int, default=None)
    parser.add_argument("--disable-replanning", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    triage_report = Path(args.triage_report).resolve()

    rows = load_discovery_ledger(triage_report)
    selected = select_discovery_row(
        rows,
        index=args.idea_index,
        require_analysis_ready=args.run_analysis,
    )
    handoff = build_handoff_from_row(
        selected,
        triage_report_path=triage_report,
        selection_mode=args.selection_mode,
        selection_rationale=args.selection_rationale,
        target_outcome=args.target_outcome,
        database=args.database,
        research_question=args.research_question,
        human_confirmed=args.human_confirm,
        human_confirmation_note=args.human_confirmation_note,
    )
    handoff_path = write_handoff_packet(handoff, out_root / "discovery_handoff.json")
    print(f"[discovery] handoff: {handoff_path}")
    print(f"[discovery] selected topic: {handoff.candidate_topic}")

    if not args.run_analysis:
        print("[discovery] --run-analysis not set; stopping after frozen handoff.")
        return 0

    if not args.export_dir:
        raise SystemExit("--export-dir is required with --run-analysis")
    try:
        assert_discovery_analysis_ready(handoff)
    except ValueError as exc:
        raise SystemExit(f"analysis gate blocked: {exc}") from exc
    llm = _build_data_foundation_llm(
        provider=args.provider,
        model=args.model,
        request_timeout=args.request_timeout,
    )
    universe_dir = out_root / "universe"
    # Deterministic outcome materialisation: a non-death target outcome only
    # appears in the universe if the data-foundation agent happens to pick its
    # concept as a feature (brittle — the LLM may select a sibling like
    # aki_stage instead of aki, leaving the target column missing and the run
    # at 0 steps). When --outcome-concepts is given we pass them as outcome
    # concepts so the materialiser emits a bare 0/1 column (+ <c>_time onset)
    # regardless of feature selection.
    outcome_concepts = _outcome_concepts_for_handoff(
        handoff_target=handoff.target_outcome,
        requested=args.outcome_concepts,
    )
    acquisition = acquire_universe_for_question(
        export_dir=Path(args.export_dir).resolve(),
        question=handoff.research_question,
        llm=llm,
        output_dir=universe_dir,
        stem="discovery_universe",
        target_outcome=handoff.target_outcome,
        outcome_concepts=outcome_concepts,
        database=handoff.database,
    )
    acquisition_path = out_root / "data_foundation_acquisition.json"
    acquisition_path.write_text(
        json.dumps(acquisition.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if acquisition.blocked or acquisition.universe_path is None:
        raise SystemExit(f"data foundation blocked: {acquisition.note}")

    # Declare the long-format trajectory in the JSONL handoff. The bench keeps
    # the cohort as a path and forwards only these explicit input paths through
    # runner_kwargs, so both CodeRunner and DockerRunner can expose it without
    # inheriting the launcher's ambient environment.
    trajectory_path = Path(acquisition.universe_path).with_name(
        f"{Path(acquisition.universe_path).stem}_trajectory.parquet"
    )
    if trajectory_path.exists():
        print(f"[discovery] trajectory: {trajectory_path}")
    else:
        trajectory_path = None

    jsonl_path = _write_ehrflowbench_row(
        out_root=out_root,
        handoff=handoff,
        cohort_path=acquisition.universe_path,
        cohort_authority_path=acquisition.cohort_authority_path,
        cohort_authority_ref=(
            acquisition.cohort_authority_ref.to_dict()
            if acquisition.cohort_authority_ref is not None
            else None
        ),
        trajectory_path=trajectory_path,
    )
    bench_root = out_root / "bench"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_research_agent_bench.py"),
        "--ehrflowbench-jsonl",
        str(jsonl_path),
        "--arms",
        "aware",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--out-root",
        str(bench_root),
        "--runner",
        args.runner,
        "--request-timeout",
        str(args.request_timeout),
    ]
    if args.llm_seed is not None:
        cmd.extend(["--llm-seed", str(args.llm_seed)])
    if args.max_total_steps is not None:
        cmd.extend(["--max-total-steps", str(args.max_total_steps)])
    if args.disable_replanning:
        cmd.append("--disable-replanning")
    if args.reuse_existing:
        cmd.append("--reuse-existing")

    print("[discovery] running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        return completed.returncode

    run_dir = _latest_aware_run_dir(bench_root)
    if run_dir is None:
        raise SystemExit(f"could not locate aware run under {bench_root}")
    evidence = EvidenceStore(run_dir)
    # Register from the already-frozen launcher packet before touching the run
    # root.  A reused run carrying the same id with different handoff content
    # therefore fails without first overwriting its plain provenance JSON.
    handoff_record = _register_file_exact(
        evidence,
        source_path=handoff_path,
        kind="log",
        description=(
            "Human-confirmed discovery-to-analysis handoff with frozen "
            "literature and endpoint provenance."
        ),
        evidence_id="discovery_handoff",
        producer="discovery_launcher",
        generation_mode="human_confirmed",
        metadata={"artifact_role": "discovery_handoff"},
    )
    run_handoff_path = write_handoff_packet(handoff, run_dir / "discovery_handoff.json")
    if sha256_of_file(run_handoff_path) != handoff_record.sha256:
        raise ValueError("run-root discovery handoff differs from registered evidence")
    _register_story_source_records(evidence=evidence, run_dir=run_dir)
    story_paths = render_discovery_story_figure(run_dir=run_dir, handoff=handoff)
    _register_story_figure_provenance(
        evidence=evidence,
        run_dir=run_dir,
        paths=story_paths,
    )
    assessment = validate_discovery_manuscript_package(run_dir=run_dir)
    assessment_path = write_discovery_package_assessment(
        assessment, run_dir / "discovery_package_assessment.json"
    )
    print(f"[discovery] package assessment: {assessment_path}")
    print(f"[discovery] package status: {assessment.status}")
    return 0 if assessment.package_ready else 3


def _normalise_endpoint(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _outcome_concepts_for_handoff(
    *,
    handoff_target: str,
    requested: Optional[str],
) -> tuple[str, ...]:
    """Return the sole licensed outcome concept for universe materialisation.

    CLI input is a consistency assertion only.  The frozen handoff remains the
    single source of truth so an AKI discovery cannot silently fall back to the
    historical mortality default.
    """

    target = str(handoff_target or "").strip()
    if not target:
        raise SystemExit("discovery handoff has no target_outcome")
    if requested is not None:
        supplied = [item.strip() for item in requested.split(",") if item.strip()]
        if len(supplied) != 1 or _normalise_endpoint(
            supplied[0]
        ) != _normalise_endpoint(target):
            raise SystemExit(
                "--outcome-concepts is a consistency assertion and must contain "
                f"exactly the frozen handoff target {target!r}"
            )
    return (target,)


def _register_file_exact(
    evidence: EvidenceStore,
    *,
    source_path: Path,
    kind: str,
    description: str,
    evidence_id: str,
    producer: str,
    generation_mode: str,
    inputs: Optional[list[str]] = None,
    script_evidence_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvidenceRecord:
    """Register an exact artifact, failing before copy on id/hash mismatch."""

    source = Path(source_path)
    digest = sha256_of_file(source)
    existing = evidence.get(evidence_id)
    if existing is not None and existing.sha256 != digest:
        raise ValueError(
            f"Evidence id collision for {evidence_id}: existing "
            f"sha256={existing.sha256[:8]} new sha256={digest[:8]}"
        )
    record = evidence.register_file(
        kind=kind,
        description=description,
        source_path=source,
        inputs=list(inputs or []),
        script_evidence_id=script_evidence_id,
        evidence_id=evidence_id,
        producer=producer,
        generation_mode=generation_mode,
        metadata=dict(metadata or {}),
        on_sha_change="raise",
    )
    if record.sha256 != digest or record.evidence_id != evidence_id:
        raise ValueError(f"strict evidence registration failed for {evidence_id}")
    return record


def _register_story_source_records(
    *,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Dict[str, EvidenceRecord]:
    """Bind the root audit files consumed by the universal story renderer."""

    specs = (
        (
            "run_status.json",
            "run_status",
            "Research-agent run status and publication readiness gates.",
        ),
        (
            "evidence_audit.json",
            "evidence_audit",
            "Research-agent evidence audit consumed by the story figure.",
        ),
        (
            "numeric_audit.json",
            "numeric_audit",
            "Research-agent numeric audit consumed by the story figure.",
        ),
    )
    records: Dict[str, EvidenceRecord] = {}
    for filename, evidence_id, description in specs:
        source_path = Path(run_dir) / filename
        if not source_path.is_file():
            raise FileNotFoundError(
                f"story figure source is missing after analysis: {source_path}"
            )
        records[evidence_id] = _register_file_exact(
            evidence,
            source_path=source_path,
            kind="log",
            description=description,
            evidence_id=evidence_id,
            producer="research_agent_pipeline",
            generation_mode="system",
            metadata={"artifact_role": "story_figure_source"},
        )
    return records


def _contract_source_evidence_ids(contract: Dict[str, Any]) -> list[str]:
    identifiers: list[str] = []
    for value in contract.get("source_data") or []:
        text = str(value or "").strip()
        if text and text not in identifiers:
            identifiers.append(text)
    for panel in contract.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        for value in panel.get("evidence_ids") or []:
            text = str(value or "").strip()
            if text and text not in identifiers:
                identifiers.append(text)
    return identifiers


def _register_story_figure_provenance(
    *,
    evidence: EvidenceStore,
    run_dir: Path,
    paths: Dict[str, Path],
) -> Dict[str, EvidenceRecord]:
    """Register code, contract, SVG and PNG with a closed evidence chain."""

    stem = "easyicu_discovery_story"
    contract_path = Path(paths.get("contract") or "")
    if not contract_path.is_file():
        raise FileNotFoundError("story figure renderer did not emit its contract")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid story figure contract: {contract_path}") from exc
    if not isinstance(contract, dict) or contract.get("figure_id") != stem:
        raise ValueError(f"story figure contract must declare figure_id={stem!r}")
    source_ids = _contract_source_evidence_ids(contract)
    if not source_ids:
        raise ValueError("story figure contract has no source evidence ids")
    missing = [
        evidence_id for evidence_id in source_ids if evidence.get(evidence_id) is None
    ]
    if missing:
        raise ValueError(
            "story figure contract references unregistered evidence: "
            + ", ".join(missing)
        )

    script_record = _register_file_exact(
        evidence,
        source_path=(
            REPO_ROOT
            / "src"
            / "easyicu"
            / "research_agent"
            / "discovery_story_figure.py"
        ),
        kind="code",
        description="Deterministic code used to render the discovery story figure.",
        evidence_id="discovery_story_figure_script",
        producer="discovery_launcher",
        generation_mode="deterministic_code",
        metadata={"artifact_role": "figure_script", "figure_id": stem},
    )
    contract_id = "discovery_story_figure_contract"
    contract_metadata = {
        "artifact_role": "figure_contract",
        "figure_id": stem,
        "source_evidence_ids": source_ids,
        "inputs": source_ids,
    }
    contract_record = _register_file_exact(
        evidence,
        source_path=contract_path,
        kind="log",
        description="Strict contract for the universal discovery story figure.",
        evidence_id=contract_id,
        producer="discovery_launcher",
        generation_mode="deterministic_contract",
        inputs=source_ids,
        metadata=contract_metadata,
    )

    figure_inputs = [contract_id, *source_ids]
    records: Dict[str, EvidenceRecord] = {
        "script": script_record,
        "contract": contract_record,
    }
    for extension in ("svg", "png", "pdf", "tiff"):
        artifact_path = Path(paths.get(extension) or "")
        if not artifact_path.is_file():
            if extension in {"svg", "png"}:
                raise FileNotFoundError(
                    f"story figure renderer did not emit a non-empty {extension} export"
                )
            continue
        if artifact_path.stat().st_size <= 0:
            raise ValueError(f"story figure {extension} export is empty")
        record = _register_file_exact(
            evidence,
            source_path=artifact_path,
            kind="figure",
            description=f"Universal discovery story figure ({extension.upper()}).",
            evidence_id=f"discovery_story_figure_{extension}",
            producer="discovery_story_figure",
            generation_mode="deterministic_matplotlib",
            inputs=figure_inputs,
            script_evidence_id=script_record.evidence_id,
            metadata={
                "artifact_role": "manuscript_figure",
                "figure_id": stem,
                "contract_evidence_id": contract_id,
                "source_evidence_ids": source_ids,
                "inputs": figure_inputs,
            },
        )
        if (
            record.script_evidence_id != script_record.evidence_id
            or record.metadata.get("contract_evidence_id") != contract_id
            or not set(source_ids).issubset(record.inputs)
        ):
            raise ValueError(
                f"story figure {extension} evidence has incomplete provenance"
            )
        records[extension] = record
    return records


def _build_data_foundation_llm(*, provider: str, model: str, request_timeout: float):
    """Use the same explicit provider contract for acquisition and benchmark."""

    try:
        return build_provider_client(
            provider=provider,
            model=model,
            request_timeout=request_timeout,
            title="EasyICU discovery-to-manuscript",
            client_cls=OpenAIClient,
        )
    except ProviderConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


def _write_ehrflowbench_row(
    *,
    out_root: Path,
    handoff,
    cohort_path: Path,
    cohort_authority_path: Optional[Path] = None,
    cohort_authority_ref: Optional[Mapping[str, object]] = None,
    trajectory_path: Optional[Path] = None,
) -> Path:
    if (cohort_authority_path is None) != (cohort_authority_ref is None):
        raise ValueError(
            "cohort authority path and reference must be handed off together"
        )
    row: Dict[str, Any] = {
        "key": f"discovery_{handoff.literature_idea_id}",
        "name": handoff.candidate_topic[:120],
        "question": handoff.research_question,
        "cohort_path": str(cohort_path.resolve()),
        "target_outcome": handoff.target_outcome,
        "primary_predictor": handoff.resolved_predictor_concept or "agent_mined_idea",
        "expected_or_direction": 0,
        "kind": "descriptive_association",
        "inclusion_criteria": list(handoff.inclusion_criteria),
    }
    if cohort_authority_path is not None and cohort_authority_ref is not None:
        row["cohort_authority_required"] = True
        row["cohort_authority_path"] = str(Path(cohort_authority_path).resolve())
        row["cohort_authority_ref"] = dict(cohort_authority_ref)
    if trajectory_path is not None:
        row["trajectory_path"] = str(Path(trajectory_path).resolve())
    path = out_root / "discovery_ehrflowbench.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _latest_aware_run_dir(bench_root: Path) -> Optional[Path]:
    candidates = sorted(
        bench_root.glob("*/aware/run_*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
