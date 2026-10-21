"""Verify and reuse the input of a grouped preparation, without a new acquisition.

Launch rehearses the actual grouping restore in a temporary directory. Offers
memoize that verdict only after rechecking its content-bound inputs. The capsule
supplies scientific coordinates only; neither its
old plan nor any approval is replayed. All cohort checks remain with intake.
"""

from __future__ import annotations

import functools
import hashlib
import json
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from easyicu.research_agent.acquisition.catalog import CoverageReport
from easyicu.research_agent.acquisition.foundation import AcquisitionResult, ConceptSelection
from easyicu.research_agent.authority.run_input import load_verified_run_input_capsule
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthority,
    MaterializedCohortAuthorityRef,
    load_verified_materialized_cohort_authority,
    materialized_provenance_path,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    ResumeExposureGroupings,
    bound_context_on,
    restore_exposure_grouping_phase,
)
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import TimeWindow

_SCIENTIFIC_ARGUMENTS = (
    "question", "cohort_name", "database", "target_outcome", "primary_exposure",
    "cross_database_validation", "inclusion_criteria", "exclusion_criteria",
    "id_columns", "time_columns", "outcome_columns", "concept_descriptions",
    "user_preferences", "notes",
)
_REHEARSAL_CONFIG = (
    "bound_target_trial", "bound_baseline_requirements",
    "bound_population_requirements", "bound_analysis_inputs",
)
_OFFER_CONFIG = (*_REHEARSAL_CONFIG, "required_primary_cohort_selection_mode", "bound_plan_revision_contract")
_OFFER_MEMO_LIMIT = 128
_OFFER_VERDICTS: OrderedDict[str, bool] = OrderedDict()
_OFFER_LOCK = RLock()


@dataclass(frozen=True)
class PreparedGroupedContinuation:
    cohort_path: Path
    authority_ref: MaterializedCohortAuthorityRef
    columns: tuple[str, ...]
    scientific_identity: Mapping[str, Any]
    verification_sha256: str = field(default="", compare=False)

    def run_arguments(self) -> dict[str, Any]:
        """A bounded allowlist of scientific inputs, never execution options."""
        identity = self.scientific_identity
        return {
            **{key: identity[key] for key in _SCIENTIFIC_ARGUMENTS if key in identity},
            "endpoint": EndpointSpec.model_validate(identity["endpoint"]) if identity.get("endpoint") else None,
            "time_windows": [TimeWindow.model_validate(item) for item in identity.get("time_windows") or ()],
        }

    def acquisition(self) -> AcquisitionResult:
        """Declare the verified original source; the pipeline stages its own copy."""
        columns = list(self.columns)
        coverage = CoverageReport(requested=columns, resolved={c: c for c in columns}, available=columns, missing=[])
        return AcquisitionResult(
            universe_path=self.cohort_path,
            provenance_path=materialized_provenance_path(self.cohort_path),
            selection=ConceptSelection(selected_concepts=columns, coverage=coverage, selection_authority="host_exact"),
            materialized_concepts=columns, coverage=coverage,
            cohort_authority_path=self.cohort_path.parent / self.authority_ref.file,
            cohort_authority_ref=self.authority_ref,
            materialized_columns=self.columns,
            analysis_columns={c: c for c in columns},
            endpoint=self.run_arguments()["endpoint"],
            note="Reused the verified preparation input for Planner continuation; new plan review is required.",
        )


def verify_prepared_grouped_continuation(
    *, run_dir: Path, source: ResumeExposureGroupings,
    pipeline_config: Mapping[str, Any], package_binding: Mapping[str, Any] | None,
    scope_identity: tuple[str, ...] = (), offer_only: bool = False,
    previously_verified: PreparedGroupedContinuation | None = None,
) -> PreparedGroupedContinuation | None:
    """Run the launch's restore offline, retaining every source/row/context seal.

    Metadata-only groupings still use the catalog restore. Prepared groupings
    have no trajectory (the grouping phase enforces that invariant). Reusing
    their original input therefore cannot introduce a new trajectory or lose
    the source's verified patient identity, as a fresh acquisition could.
    """
    if source.materialized_authority_raw is None:
        return None
    capsule_path = run_dir / "run_input_capsule.json"
    if capsule_path.is_symlink() or capsule_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("the preparation input capsule is not a bounded regular file")
    capsule_raw = capsule_path.read_bytes()
    identity = json.loads(capsule_raw)["scientific_identity"]
    verified = load_verified_run_input_capsule(run_dir=run_dir, scientific_identity=identity)
    if verified.capsule.trajectory_relative_path is not None:
        raise ValueError("the grouped preparation unexpectedly contains a trajectory")
    reference = MaterializedCohortAuthorityRef.from_dict(identity["materialized_cohort_authority_ref"])
    grouped_authority = MaterializedCohortAuthority.from_dict(json.loads(source.materialized_authority_raw))
    if grouped_authority.parent_authority_sha256 != reference.sha256:
        raise ValueError("the grouped preparation belongs to another input")
    if package_binding is not None and (
        package_binding.get("export_authority_sha256") != grouped_authority.source_export_authority_sha256
    ):
        raise ValueError("the grouped preparation belongs to another prepared package")
    # The source snapshot names the input, never an arbitrary client path.
    authority_path = run_dir / reference.file
    if authority_path.is_symlink() or authority_path.stat().st_size != reference.size:
        raise ValueError("the preparation source authority changed")
    raw = authority_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("the preparation source authority changed")
    original = MaterializedCohortAuthority.from_dict(json.loads(raw))
    wrapper = run_dir.parents[1]
    # A second continuation can point to the original sibling's input, just
    # as a prepared plan revision does. Only this study's owned wrappers are
    # eligible, and the exact source authority is verified again below.
    wrappers = [wrapper, *sorted(p for p in wrapper.parent.iterdir() if p != wrapper and p.name.startswith("run_"))]
    cohort_path = None
    for candidate in wrappers[:200]:
        input_dir = candidate / "pipeline_input"
        if candidate.is_symlink() or input_dir.is_symlink():
            continue
        path = input_dir / original.cohort_file
        if (input_dir / reference.file).is_file():
            trajectory = path.with_name(f"{path.stem}_trajectory.parquet")
            if trajectory.exists() or trajectory.is_symlink():
                raise ValueError("the preparation input acquired a trajectory after it was sealed")
            current = load_verified_materialized_cohort_authority(path, expected_authority=reference)
            if current is None:
                raise ValueError("the preparation source lost its typed authority")
            cohort_path = path
            break
    if cohort_path is None:
        raise ValueError("the verified preparation input is unavailable")
    verification_sha256 = _rehearsal_identity({
        "scope": scope_identity,
        "capsule": hashlib.sha256(capsule_raw).hexdigest(),
        "grouping_record": hashlib.sha256(source.raw).hexdigest(),
        "grouped_authority": hashlib.sha256(source.materialized_authority_raw).hexdigest(),
        "source_cohort_authority": reference.sha256,
        "reviewed_configuration": {key: pipeline_config[key] for key in _OFFER_CONFIG if key in pipeline_config},
        "prepared_package": (package_binding or {}).get("export_authority_sha256"),
    })
    result = PreparedGroupedContinuation(cohort_path, reference, original.cohort_columns, identity, verification_sha256)
    if previously_verified is not None:
        # The runner checks the freshly verified inputs against this launch's
        # receipt. This is never a lookup in the offer memo.
        if verification_sha256 != previously_verified.verification_sha256 or result != previously_verified:
            raise ValueError("the preparation changed after launch validation")
    elif offer_only:
        # Keep both verdicts and serialize misses: concurrent page projections
        # must not each stage the same cohort. Store no paths or patient rows.
        with _OFFER_LOCK:
            if verification_sha256 not in _OFFER_VERDICTS:
                try:
                    _rehearse_grouped_continuation(result, run_dir, pipeline_config)
                except Exception:
                    _remember_offer(verification_sha256, False)
                    raise
                _remember_offer(verification_sha256, True)
            _OFFER_VERDICTS.move_to_end(verification_sha256)
            if not _OFFER_VERDICTS[verification_sha256]:
                raise ValueError("the saved preparation restore rehearsal failed")
    else:
        _rehearse_grouped_continuation(result, run_dir, pipeline_config)
    return result


def _rehearsal_identity(inputs: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _remember_offer(key: str, verdict: bool) -> None:
    _OFFER_VERDICTS[key] = verdict
    while len(_OFFER_VERDICTS) > _OFFER_MEMO_LIMIT:
        _OFFER_VERDICTS.popitem(last=False)


def _rehearse_grouped_continuation(
    result: PreparedGroupedContinuation, run_dir: Path, pipeline_config: Mapping[str, Any],
) -> None:
    # This is the same builder, reviewed requirements and grouping restore the
    # pipeline uses, on the same materialization, with no Provider or approval.
    with tempfile.TemporaryDirectory(prefix="easyicu-grouped-resume-") as directory:
        temporary_root = Path(directory).resolve()
        target = temporary_root / "cohort.parquet"
        stage_materialized_cohort_authority(
            result.cohort_path, target, expected_source_authority=result.authority_ref,
            producer_implementation_sha256="0" * 64,
        )
        kwargs = result.run_arguments()
        kwargs["research_question"] = kwargs.pop("question")
        config = PipelineConfig(workdir=directory, **{
            key: pipeline_config[key] for key in _REHEARSAL_CONFIG if key in pipeline_config
        })
        rebuild = functools.partial(bound_context_on, build_research_context, kwargs, config)
        restore_exposure_grouping_phase(
            source_run_dir=run_dir, context=rebuild(target), cohort_path=target,
            run_dir=temporary_root, rebuild_context=rebuild,
        )
