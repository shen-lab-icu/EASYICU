#!/usr/bin/env python3
"""Freeze exact Canonical9 scientific identities from a typed JSONL handoff.

This command performs no Provider call and launches no analysis.  It verifies
the materialized cohort/trajectory authorities, reconstructs the exact
scientific coordinates used by ``run_research_agent_bench.py``, and writes the
repository-owned Canonical9 input selector in canonical JSON form.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from benchmarks.figure2_canonical9.evaluator.input_binding_v2 import (  # noqa: E402
    CANONICAL_RUN_INPUT_BINDING_REF,
    CANONICAL_RUN_INPUT_BINDING_SCHEMA,
    CanonicalRunInputBindingManifest,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import (  # noqa: E402
    FIGURE2_TASK_IDS,
)
from easyicu.research_agent.authority.run_input import (  # noqa: E402
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2,
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3,
    build_scientific_identity,
    canonical_sha256,
)
from easyicu.research_agent.intake.materialized_metadata import (  # noqa: E402
    MaterializedCohortAuthorityRef,
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (  # noqa: E402
    MaterializedTrajectoryAuthorityRef,
    load_verified_materialized_trajectory_authority,
)

_DEFAULT_OUTPUT = (
    _REPO_ROOT / "benchmarks/figure2_canonical9/canonical_run_input_bindings_v2.json"
)
_PROFILE = {
    "ref": "npj_dm/20260718",
    "concept_dict_sha256": (
        "fccadc53622dc82fe1dc8696617e52044168b6a84a9255e97e59df9e53bc5803"
    ),
    "sofa2_dict_sha256": (
        "61f37a41083cd96df49a2e61d26c682e9d090d0a22d05ff97ba85a966b165b1c"
    ),
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_rows(path: Path) -> list[dict[str, Any]]:
    candidate = path.expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ValueError("Canonical9 JSONL must be an absolute, non-symlink path")
    rows: list[dict[str, Any]] = []
    for line in candidate.resolve(strict=True).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Canonical9 JSONL rows must be objects")
        rows.append(row)
    task_ids = tuple(str(row.get("key") or row.get("id") or "") for row in rows)
    if task_ids != tuple(FIGURE2_TASK_IDS):
        raise ValueError("JSONL must contain the exact ordered Canonical9")
    return rows


def _concept_descriptions(row: Mapping[str, Any]) -> dict[str, str] | None:
    operational = str(row.get("operational_exposure") or "").strip()
    display_name = str(row.get("primary_predictor") or "").strip()
    question = str(row.get("question") or "")
    normalized_question = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    normalized_display = re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_")
    if (
        operational
        and display_name
        and normalized_display
        and re.search(
            rf"(?:^|_){re.escape(normalized_display)}(?:_|$)",
            normalized_question,
        )
    ):
        return {operational: display_name}
    return None


def _string_list(row: Mapping[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_manifest(jsonl_path: Path) -> CanonicalRunInputBindingManifest:
    tasks: list[dict[str, Any]] = []
    for row in _strict_rows(jsonl_path):
        task_id = str(row["key"])
        question = str(row["question"])
        cohort_path = Path(str(row["cohort_path"])).expanduser().resolve(strict=True)
        cohort_ref = MaterializedCohortAuthorityRef.from_dict(
            row["cohort_authority_ref"]
        )
        cohort = load_verified_materialized_cohort_authority(
            cohort_path,
            expected_authority=cohort_ref,
        )
        if cohort is None:
            raise ValueError(f"{task_id}: typed cohort authority is missing")

        trajectory_path: Path | None = None
        trajectory_ref: MaterializedTrajectoryAuthorityRef | None = None
        if row.get("trajectory_path") is not None:
            trajectory_path = (
                Path(str(row["trajectory_path"])).expanduser().resolve(strict=True)
            )
            trajectory_ref = MaterializedTrajectoryAuthorityRef.from_dict(
                row["trajectory_authority_ref"]
            )
            trajectory = load_verified_materialized_trajectory_authority(
                trajectory_path,
                expected_authority=trajectory_ref,
                expected_universe_authority=cohort_ref,
            )
            if trajectory is None:
                raise ValueError(f"{task_id}: typed trajectory authority is missing")

        operational_exposure = (
            str(row.get("operational_exposure")).strip()
            if row.get("operational_exposure") is not None
            else None
        )
        target_outcome = str(row.get("target_outcome") or "").strip()
        scientific_identity = build_scientific_identity(
            cohort=cohort_path,
            question=question,
            cohort_name=f"bench_{task_id}",
            database=str(row.get("database") or "bench").strip() or "bench",
            target_outcome=target_outcome or None,
            primary_exposure=operational_exposure or None,
            cross_database_validation=None,
            inclusion_criteria=_string_list(row, "inclusion_criteria"),
            exclusion_criteria=None,
            id_columns=_string_list(row, "id_columns") or None,
            time_columns=None,
            outcome_columns=None,
            time_windows=None,
            concept_descriptions=_concept_descriptions(row),
            user_preferences=None,
            notes=(str(row.get("notes") or "").strip() or None),
            skill_key=None,
            experiment_spec=None,
            source_files=None,
            disable_icu_context=False,
            materialized_cohort_authority_ref=cohort_ref.to_dict(),
            trajectory_path=trajectory_path,
            materialized_trajectory_authority_ref=(
                trajectory_ref.to_dict() if trajectory_ref is not None else None
            ),
            capability_workflow=None,
        )
        tasks.append(
            {
                "task_id": task_id,
                "state": "ready",
                "research_question_sha256": hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest(),
                "database": scientific_identity["database"],
                "operational_exposure": scientific_identity["primary_exposure"],
                "target_outcome": scientific_identity["target_outcome"],
                "expected_run_input_capsule_schema_version": (
                    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3
                    if trajectory_ref is not None
                    else RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2
                ),
                "scientific_identity_sha256": canonical_sha256(scientific_identity),
                "source_materialized_cohort_authority_ref": cohort_ref.to_dict(),
                "source_materialized_trajectory_authority_ref": (
                    trajectory_ref.to_dict() if trajectory_ref is not None else None
                ),
            }
        )
    return CanonicalRunInputBindingManifest.model_validate(
        {
            "schema_version": CANONICAL_RUN_INPUT_BINDING_SCHEMA,
            "manifest_ref": CANONICAL_RUN_INPUT_BINDING_REF,
            "submission_profile": _PROFILE,
            "tasks": tuple(tasks),
        },
        strict=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ehrflowbench-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify output bytes instead of writing them.",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.ehrflowbench_jsonl)
    payload = _canonical_bytes(manifest.model_dump(mode="json"))
    output = args.output.expanduser().resolve()
    if args.check:
        if output.read_bytes() != payload:
            raise SystemExit("Canonical9 run-input binding is stale")
    else:
        output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "tasks": len(manifest.tasks),
                "state": "ready",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
