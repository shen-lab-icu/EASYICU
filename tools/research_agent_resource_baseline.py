#!/usr/bin/env python3
"""Measure the offline Canonical9 Planner/resource envelope deterministically.

This is an architecture and context-budget fixture, not a paper-input authority.
It performs no provider calls and does not read ICU patient data.  The nine fixed
contexts deliberately contain only the question, analysis family, and a small
typed concept roster.  They let each framework bundle prove that resource
selection remains deterministic while prompt/context size does not regress.

Usage::

    python tools/research_agent_resource_baseline.py
    python tools/research_agent_resource_baseline.py --emit \
        tools/arch_baselines/research_agent_resource_context.json
    python tools/research_agent_resource_baseline.py --diff \
        tools/arch_baselines/research_agent_resource_context.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.know_how import KnowHowRegistry
from easyicu.research_agent.resources import (
    CODER_RESOURCE_PROMPT_LIMIT_BYTES,
    ResourceScheduler,
    ResourceSelectionQuery,
    build_coder_resource_bundle,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

TOOL_VERSION = "2.2.0"
SCHEMA_VERSION = "easyicu.research_agent_resource_baseline/2"
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_CREATED_AT = datetime(2026, 7, 22, tzinfo=timezone.utc)

# These are offline architecture fixtures for the nine A-task families.  They
# are intentionally small and case-neutral beyond each task's public question.
# They do not replace canonical run bindings or the full6 data authority.
TASK_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "task_id": "E1",
        "question": "Estimate Sepsis-3 prevalence and mortality association.",
        "retrieval_family": "descriptive",
        "concepts": ("sepsis3", "death", "stay_id", "subject_id"),
    },
    {
        "task_id": "E2",
        "question": (
            "Estimate the association of first-24-hour peak lactate with "
            "in-hospital mortality."
        ),
        "retrieval_family": "association",
        "concepts": ("lactate", "death", "stay_id", "subject_id"),
    },
    {
        "task_id": "E3",
        "question": (
            "Estimate the KDIGO AKI stage gradient for mortality and length of stay."
        ),
        "retrieval_family": "association",
        "concepts": ("kdigo", "death", "los_icu", "stay_id", "subject_id"),
    },
    {
        "task_id": "M1",
        "question": (
            "Assess hepatobiliary SOFA and bilirubin missingness in mortality analysis."
        ),
        "retrieval_family": "association",
        "concepts": ("bilirubin", "sofa", "death", "stay_id", "subject_id"),
    },
    {
        "task_id": "M2",
        "question": "Build a first-24-hour mortality risk prediction model.",
        "retrieval_family": "prediction",
        "concepts": ("death", "age", "sex", "stay_id", "subject_id"),
    },
    {
        "task_id": "M3",
        "question": (
            "Discover candidate sepsis subphenotypes from labs and vital signs."
        ),
        "retrieval_family": "phenotyping",
        "concepts": ("sepsis3", "heart_rate", "lactate", "stay_id", "subject_id"),
    },
    {
        "task_id": "H1",
        "question": "Estimate mechanical ventilation survival and 28-day mortality.",
        "retrieval_family": "time_to_event",
        "concepts": ("mechvent", "death", "stay_id", "subject_id"),
    },
    {
        "task_id": "H2",
        "question": (
            "Compare vasopressor strategies with a confounding-aware causal analysis."
        ),
        "retrieval_family": "causal_emulation",
        "concepts": ("vaso", "death", "age", "stay_id", "subject_id"),
    },
    {
        "task_id": "H3",
        "question": "Cluster longitudinal ICU trajectories into stable subphenotypes.",
        "retrieval_family": "phenotyping",
        "concepts": ("sofa", "stay_id", "subject_id"),
    },
)

SOURCE_FILES: tuple[str, ...] = (
    "src/easyicu/research_agent/agents/core.py",
    "src/easyicu/research_agent/research_context/outbound.py",
    "src/easyicu/research_agent/know_how/registry.py",
    "src/easyicu/research_agent/planning/analysis_types.py",
    "src/easyicu/research_agent/planning/cohort_contract.py",
    "src/easyicu/research_agent/resources/schema.py",
    "src/easyicu/research_agent/resources/catalog.py",
    "src/easyicu/research_agent/resources/capability.py",
    "src/easyicu/research_agent/resources/capability_runtime.py",
    "src/easyicu/research_agent/resources/context.py",
    "src/easyicu/research_agent/resources/coder.py",
    "src/easyicu/research_agent/resources/scheduler.py",
    "src/easyicu/research_agent/learning/store.py",
    "src/easyicu/research_agent/learning/runtime.py",
    "src/easyicu/research_agent/graph.py",
    "src/easyicu/data/concept-dict.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(fixture: dict[str, Any]) -> ResearchContext:
    variables = [
        ConceptDescriptor(
            name=name,
            description=f"Offline architecture fixture concept: {name}",
            dtype=("int64" if name in {"stay_id", "subject_id"} else "float64"),
            source_concept=name,
            analysis_window=(
                "first_24h" if name in {"lactate", "age", "sex"} else None
            ),
        )
        for name in fixture["concepts"]
    ]
    return ResearchContext(
        research_question=fixture["question"],
        cohort=CohortDescriptor(
            cohort_name=f"offline_{fixture['task_id'].lower()}_fixture",
            database="miiv",
            n_patients=1_000,
            n_stays=1_000,
            id_columns=["stay_id", "subject_id"],
        ),
        variables=variables,
        created_at=FIXED_CREATED_AT,
    )


def _task_measurement(
    fixture: dict[str, Any], registry: KnowHowRegistry
) -> dict[str, Any]:
    context = _context(fixture)
    selection = ResourceScheduler.select_protocols(
        registry=registry,
        query=ResourceSelectionQuery(
            purpose="planner",
            query=context.research_question,
            analysis_family=fixture["retrieval_family"],
            database=context.cohort.database,
            available_input_roles=fixture["concepts"],
        ),
        available_concepts=fixture["concepts"],
        top_k=3,
        # This is the frozen development-context measurement, not a paper
        # profile. Paper-facing selection defaults to clinical_reviewed only.
        allowed_review_statuses=("curated_mvp", "clinical_reviewed"),
    )
    hits = selection.hits
    know_how_prompt = selection.prompt if hits else ""
    without_resources = PlannerAgent.request_metrics(context)
    with_resources = PlannerAgent.request_metrics(
        context,
        know_how_context=know_how_prompt,
    )
    coder_bundle = build_coder_resource_bundle(
        step_id=f"{fixture['task_id'].lower()}_primary_analysis",
        profile_ref="npj_dm_framework_v2_dev/20260722",
        analysis_family=fixture["retrieval_family"],
        step_role="primary",
        question=context.research_question,
        intent=context.research_question,
        method=fixture["retrieval_family"],
        planner_inputs=("cohort:analysis_cohort", *fixture["concepts"]),
        expected_outputs=("table:primary_result",),
        resolved_input_bindings={
            "cohort:analysis_cohort": {
                "evidence_id": "analysis_cohort",
                "sha256": hashlib.sha256(
                    fixture["task_id"].encode("utf-8")
                ).hexdigest(),
            }
        },
        runtime_import_names=(
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "statsmodels",
            "sklearn",
            "pyarrow",
            "lifelines",
        ),
    )
    return {
        "task_id": fixture["task_id"],
        "question": fixture["question"],
        "retrieval_family": fixture["retrieval_family"],
        "fixture_concepts": list(fixture["concepts"]),
        "selected_know_how": [
            {
                "card_id": hit.card_id,
                "version": hit.version,
                "file_sha256": hit.file_sha256,
                "score": hit.score,
                "data_readiness": hit.data_readiness,
            }
            for hit in hits
        ],
        "resource_selection_provider_calls": 0,
        "resource_catalog_sha256": selection.receipt.catalog_sha256,
        "resource_allowlist_sha256": selection.receipt.allowlist_sha256,
        "know_how_prompt_bytes": len(know_how_prompt.encode("utf-8")),
        "planner_without_resources": without_resources,
        "planner_with_resources": with_resources,
        "resource_added_bytes": (
            with_resources["total_bytes"] - without_resources["total_bytes"]
        ),
        "coder_resources": {
            "profile_ref": coder_bundle.profile_ref,
            "bundle_sha256": coder_bundle.sha256,
            "provider_calls": coder_bundle.provider_calls,
            "prompt_bytes": coder_bundle.prompt_bytes,
            "prompt_limit_bytes": CODER_RESOURCE_PROMPT_LIMIT_BYTES,
            "selected": [
                {
                    "resource_id": resource.resource_id,
                    "version": resource.version,
                    "sha256": resource.sha256,
                    "kind": resource.kind,
                }
                for receipt in coder_bundle.selections
                for resource in receipt.selected
            ],
            "selection_receipt_sha256": [
                hashlib.sha256(
                    json.dumps(
                        receipt.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for receipt in coder_bundle.selections
            ],
        },
    }


def measure() -> dict[str, Any]:
    registry = KnowHowRegistry.load()
    tasks = [_task_measurement(fixture, registry) for fixture in TASK_FIXTURES]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope": "offline_minimal_context_fixture_not_paper_input_authority",
        "provider_calls": 0,
        "patient_data_reads": 0,
        "task_order": [fixture["task_id"] for fixture in TASK_FIXTURES],
        "source_sha256": {
            "tools/research_agent_resource_baseline.py": _sha256(Path(__file__)),
            **{relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_FILES},
        },
        "summary": {
            "task_count": len(tasks),
            "selected_resource_count": sum(
                len(task["selected_know_how"]) for task in tasks
            ),
            "max_planner_without_resources_bytes": max(
                task["planner_without_resources"]["total_bytes"] for task in tasks
            ),
            "max_planner_with_resources_bytes": max(
                task["planner_with_resources"]["total_bytes"] for task in tasks
            ),
            "max_resource_added_bytes": max(
                task["resource_added_bytes"] for task in tasks
            ),
            "max_coder_resource_prompt_bytes": max(
                task["coder_resources"]["prompt_bytes"] for task in tasks
            ),
            "coder_resource_provider_calls": sum(
                task["coder_resources"]["provider_calls"] for task in tasks
            ),
        },
        "tasks": tasks,
        "online_runtime_metrics": {
            "status": "not_measured_no_provider_calls",
            "reason": (
                "Bundle 0 freezes the offline resource/context envelope only; "
                "provider latency and repair counts remain run-artifact metrics."
            ),
        },
    }


def diff(baseline: dict[str, Any], current: dict[str, Any]) -> int:
    if baseline == current:
        return 0
    baseline_text = json.dumps(baseline, ensure_ascii=False, sort_keys=True)
    current_text = json.dumps(current, ensure_ascii=False, sort_keys=True)
    if baseline_text != current_text:
        print("FAIL: research-agent resource/context baseline drifted", flush=True)
        return 1
    return 0


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--emit", type=Path)
    group.add_argument("--diff", dest="diff_path", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    current = measure()
    if args.emit:
        _write(args.emit, current)
        return 0
    if args.diff_path:
        baseline = json.loads(args.diff_path.read_text(encoding="utf-8"))
        return diff(baseline, current)
    print(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
