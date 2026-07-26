"""Host-owned Action/Software/Data selection for one Coder step.

The selector is deliberately deterministic and has no provider dependency.
Selected resources are written once, bound to the active submission profile,
and rendered as one bounded :class:`HostCoderAuthority` attachment.  An honest
zero-match is valid and does not widen the Coder's scientific authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Literal, Mapping, MutableMapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..authority.coder_authority import HostCoderAuthority
from ..authority.typed_binding import (
    _coder_authority_with_typed_parent_schema_receipts,
)
from ..contracts.method_packages import (
    BASELINE_PACKAGES,
    CURATED_METHOD_PACKAGES,
    OPTIONAL_BASELINE_PACKAGES,
)
from ..learning.runtime import ReviewedMemoryRuntime
from ..planning.analysis_types import infer_analysis_type
from ..research_context.prompt_scope import scoped_coder_context
from ..research_context.typed import materialized_input_prompt_attachment
from ..schema import AnalysisStep, ResearchContext
from .catalog import ResourceCatalog
from .scheduler import ResourceScheduler
from .schema import (
    ResourceDescriptor,
    ResourceSelectionPolicy,
    ResourceSelectionQuery,
    ResourceSelectionReceipt,
)

CODER_RESOURCE_BUNDLE_SCHEMA = "easyicu.coder_resource_bundle/1"
CODER_RESOURCE_PROMPT_LIMIT_BYTES = 8_000


class CoderResourceIntegrityError(RuntimeError):
    """A persisted Coder-resource authority is missing or inconsistent."""


class CoderResourceBundle(BaseModel):
    """Profile- and step-bound proof of deterministic Coder resource selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.coder_resource_bundle/1"] = (
        CODER_RESOURCE_BUNDLE_SCHEMA
    )
    step_id: str = Field(min_length=1, max_length=160)
    profile_ref: str = Field(min_length=3, max_length=200)
    analysis_family: str = Field(min_length=1, max_length=120)
    selections: tuple[ResourceSelectionReceipt, ...]
    prompt_projection: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_bytes: int = Field(ge=0, le=CODER_RESOURCE_PROMPT_LIMIT_BYTES)
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _exact_projection_digest(self) -> "CoderResourceBundle":
        payload = self.prompt_projection.encode("utf-8")
        if len(payload) != self.prompt_bytes:
            raise ValueError("Coder resource prompt byte count changed")
        if hashlib.sha256(payload).hexdigest() != self.prompt_sha256:
            raise ValueError("Coder resource prompt digest changed")
        kinds = [receipt.policy.allowed_kinds for receipt in self.selections]
        if kinds != [("action",), ("software",), ("data",)]:
            raise ValueError("Coder resource receipts must be action/software/data")
        if any(receipt.query.purpose != "coder" for receipt in self.selections):
            raise ValueError("Coder resource bundle contains a non-Coder query")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.model_dump(mode="json"))
        ).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _descriptor(
    *,
    resource_id: str,
    kind: Literal["action", "software", "data"],
    projection: Mapping[str, object],
    analysis_families: Sequence[str] = (),
    required_input_roles: Sequence[str] = (),
    produced_output_roles: Sequence[str] = (),
    permissions: Sequence[str],
    search_terms: Sequence[str],
) -> ResourceDescriptor:
    prompt_projection = _canonical_bytes(projection).decode("utf-8")
    return ResourceDescriptor(
        resource_id=resource_id,
        version="1.0.0",
        sha256=hashlib.sha256(prompt_projection.encode("utf-8")).hexdigest(),
        kind=kind,
        analysis_families=tuple(analysis_families),
        required_input_roles=tuple(required_input_roles),
        produced_output_roles=tuple(produced_output_roles),
        permissions=tuple(permissions),
        review_status="validated",
        search_terms=tuple(dict.fromkeys(str(item) for item in search_terms if item)),
        prompt_projection=prompt_projection,
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_.:-]+", "-", value.lower()).strip("-.:_")
    if not slug:
        slug = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return slug[:96]


def _binding_sha256(binding: Mapping[str, object]) -> str:
    recorded = str(binding.get("sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", recorded):
        return recorded
    return hashlib.sha256(_canonical_bytes(dict(binding))).hexdigest()


def _action_resources(
    *,
    available_input_roles: Sequence[str],
    expected_outputs: Sequence[str],
    has_table_one_spec: bool,
) -> list[ResourceDescriptor]:
    resources: list[ResourceDescriptor] = []
    typed_roles = tuple(role for role in available_input_roles if ":" in role)
    if typed_roles:
        resources.append(
            _descriptor(
                resource_id="action:typed-input-sdk",
                kind="action",
                projection={
                    "action": "load_typed_input",
                    "authority": "host_owned",
                    "contract": "consume only resolved typed inputs and emit receipts",
                },
                permissions=("coder_context",),
                search_terms=(
                    "typed input",
                    "resolved input",
                    "artifact",
                    "evidence",
                    *typed_roles,
                ),
            )
        )
    if has_table_one_spec and "table:table_one" in expected_outputs:
        resources.append(
            _descriptor(
                resource_id="action:table-one",
                kind="action",
                projection={
                    "action": "execute_table_one_spec",
                    "authority": "planner_owned_spec",
                    "contract": "grouping, summaries, and tests come from table_one_spec",
                },
                produced_output_roles=("table:table_one",),
                permissions=("coder_context",),
                search_terms=("table one", "table_one_spec", "table:table_one"),
            )
        )
    if any(str(output).lower().startswith("figure:") for output in expected_outputs):
        resources.append(
            _descriptor(
                resource_id="action:publication-figure-export",
                kind="action",
                projection={
                    "action": "save_publication_figure",
                    "authority": "declared_figure_contract",
                    "contract": "render only the Planner-declared figure products",
                },
                permissions=("coder_context",),
                search_terms=("publication figure", "figure", *expected_outputs),
            )
        )
    return resources


_BASELINE_FAMILIES: dict[str, tuple[str, ...]] = {
    "pandas": (),
    "numpy": (),
    "scipy": ("association", "causal_emulation", "survival", "time_to_event"),
    "statsmodels": ("association", "causal_emulation", "survival", "time_to_event"),
    "sklearn": (
        "prediction",
        "prediction_model",
        "phenotyping",
        "trajectory_clustering",
    ),
    "matplotlib": (),
    "pyarrow": (),
    "seaborn": (),
}


def _software_resources(
    runtime_import_names: Iterable[str],
) -> list[ResourceDescriptor]:
    snapshot = frozenset(str(name) for name in runtime_import_names)
    resources: list[ResourceDescriptor] = []
    for import_name in (*BASELINE_PACKAGES, *OPTIONAL_BASELINE_PACKAGES):
        if import_name not in snapshot:
            continue
        resources.append(
            _descriptor(
                resource_id=f"software:{_slug(import_name)}",
                kind="software",
                projection={
                    "import_name": import_name,
                    "availability": "verified_in_runner_snapshot",
                    "runtime_install_allowed": False,
                },
                analysis_families=_BASELINE_FAMILIES.get(import_name, ()),
                permissions=("coder_context", "sandbox_import"),
                search_terms=(import_name,),
            )
        )
    for package in CURATED_METHOD_PACKAGES:
        if package.import_name not in snapshot:
            continue
        resources.append(
            _descriptor(
                resource_id=f"software:{_slug(package.import_name)}",
                kind="software",
                projection={
                    "import_name": package.import_name,
                    "capability": package.capability,
                    "fallback": package.fallback,
                    "availability": "verified_in_runner_snapshot",
                    "runtime_install_allowed": False,
                },
                analysis_families=package.families,
                permissions=("coder_context", "sandbox_import"),
                search_terms=(
                    package.import_name,
                    package.pip_name,
                    package.capability,
                    *package.families,
                ),
            )
        )
    return resources


def _data_resources(
    resolved_input_bindings: Mapping[str, Mapping[str, object]],
) -> list[ResourceDescriptor]:
    resources: list[ResourceDescriptor] = []
    for input_key in sorted(resolved_input_bindings):
        binding = resolved_input_bindings[input_key]
        evidence_id = str(binding.get("evidence_id") or "")
        resources.append(
            ResourceDescriptor(
                resource_id=f"data:{_slug(input_key)}",
                version="1.0.0",
                sha256=_binding_sha256(binding),
                kind="data",
                permissions=("coder_context", "data_read"),
                review_status="validated",
                search_terms=tuple(
                    dict.fromkeys(
                        item
                        for item in (input_key, evidence_id, "typed input", "data")
                        if item
                    )
                ),
                prompt_projection=_canonical_bytes(
                    {
                        "input_key": input_key,
                        "evidence_id": evidence_id,
                        "sha256": _binding_sha256(binding),
                        "access": "EASYICU_RESOLVED_INPUTS_JSON",
                    }
                ).decode("utf-8"),
            )
        )
    return resources


def build_coder_resource_bundle(
    *,
    step_id: str,
    profile_ref: str,
    analysis_family: str,
    step_role: str,
    question: str,
    intent: str,
    method: str | None,
    planner_inputs: Sequence[str],
    expected_outputs: Sequence[str],
    resolved_input_bindings: Mapping[str, Mapping[str, object]],
    runtime_import_names: Iterable[str],
    has_table_one_spec: bool = False,
    approved_software_resources: Sequence[ResourceDescriptor] = (),
) -> CoderResourceBundle:
    """Select reviewed resources for one exact Coder step without an LLM."""

    available_roles = tuple(
        dict.fromkeys([*planner_inputs, *resolved_input_bindings.keys()])
    )
    query = ResourceSelectionQuery(
        purpose="coder",
        query=" ".join(
            str(value or "")
            for value in (
                question,
                intent,
                method,
                " ".join(planner_inputs),
                " ".join(expected_outputs),
            )
        ).strip(),
        analysis_family=analysis_family,
        step_role=step_role,
        available_input_roles=available_roles,
    )
    catalog = ResourceCatalog(
        (
            *_action_resources(
                available_input_roles=available_roles,
                expected_outputs=expected_outputs,
                has_table_one_spec=has_table_one_spec,
            ),
            *_software_resources(runtime_import_names),
            *approved_software_resources,
            *_data_resources(resolved_input_bindings),
        )
    )
    selections = []
    policies = {
        "action": ResourceSelectionPolicy(
            allowed_kinds=("action",),
            allowed_review_statuses=("validated",),
            allowed_permissions=("coder_context",),
            max_actions=8,
        ),
        "software": ResourceSelectionPolicy(
            allowed_kinds=("software",),
            allowed_review_statuses=("validated", "approved"),
            allowed_permissions=("coder_context", "sandbox_import"),
            max_software=3,
        ),
        "data": ResourceSelectionPolicy(
            allowed_kinds=("data",),
            allowed_review_statuses=("validated",),
            allowed_permissions=("coder_context", "data_read"),
            max_data=16,
        ),
    }
    prompt_sections: list[str] = []
    for kind in ("action", "software", "data"):
        selection = ResourceScheduler.select_resources(
            catalog=catalog,
            query=query,
            policy=policies[kind],
            kind=kind,
        )
        selections.append(selection.receipt)
        if selection.prompt:
            prompt_sections.append(
                _canonical_bytes(
                    {
                        "kind": kind,
                        "resources": [
                            json.loads(resource.prompt_projection)
                            for resource in selection.resources
                        ],
                    }
                ).decode("utf-8")
            )
    prompt = "\n".join(prompt_sections)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > CODER_RESOURCE_PROMPT_LIMIT_BYTES:
        raise CoderResourceIntegrityError(
            "Coder resource projection exceeds the fixed byte budget: "
            f"{prompt_bytes}>{CODER_RESOURCE_PROMPT_LIMIT_BYTES}"
        )
    return CoderResourceBundle(
        step_id=step_id,
        profile_ref=profile_ref,
        analysis_family=analysis_family,
        selections=tuple(selections),
        prompt_projection=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_bytes=prompt_bytes,
    )


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CoderResourceIntegrityError(
                f"Coder resource receipt changed at {path}"
            )
        return
    fd, temp_name = tempfile.mkstemp(prefix=".coder-resource-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise CoderResourceIntegrityError(
                    f"Coder resource receipt raced with different bytes at {path}"
                ) from None
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def persist_coder_resource_bundle(
    *, run_dir: Path, bundle: CoderResourceBundle
) -> tuple[Path, str]:
    """Persist an exact bundle and return its path plus Coder attachment."""

    safe_step = _slug(bundle.step_id)
    path = Path(run_dir) / "resource_selections" / "coder" / f"{safe_step}.json"
    payload = _canonical_bytes(bundle.model_dump(mode="json")) + b"\n"
    _write_once(path, payload)
    relative_path = path.relative_to(run_dir).as_posix()
    attachment = _canonical_bytes(
        {
            "schema_version": "easyicu.coder_resource_prompt/1",
            "profile_ref": bundle.profile_ref,
            "step_id": bundle.step_id,
            "bundle_sha256": bundle.sha256,
            "receipt_path": relative_path,
            "provider_calls": 0,
            "selected_context": bundle.prompt_projection,
        }
    ).decode("utf-8")
    return path, attachment


def attach_coder_resources(
    *,
    authority: HostCoderAuthority,
    run_dir: Path,
    bundle: CoderResourceBundle,
) -> tuple[HostCoderAuthority, Path]:
    """Persist and bind one resource bundle into Coder/capsule authority."""

    path, attachment = persist_coder_resource_bundle(run_dir=run_dir, bundle=bundle)
    return authority.append(attachment), path


def bind_materialized_coder_authority(
    context: ResearchContext,
    step: AnalysisStep,
    authority: HostCoderAuthority,
) -> tuple[ResearchContext, HostCoderAuthority]:
    """Bind one step-scoped typed-materialization attachment to Coder calls."""

    scoped_context = scoped_coder_context(context, step)
    attachment = materialized_input_prompt_attachment(scoped_context)
    if not attachment:
        return context, authority
    return scoped_context, authority.append(attachment)


def bind_primary_cohort_role(
    *,
    authority: HostCoderAuthority,
    locked_cohort_payload: str | None,
    materialized_execution_payload: str | None = None,
) -> HostCoderAuthority:
    """Bind the unique universe-to-analysis-cohort producer role."""

    if locked_cohort_payload is None:
        return authority
    attachment = (
        "CURRENT STEP INPUT ROLE (host-owned execution contract): this is the "
        "plan's unique primary analysis_cohort + attrition producer, so "
        "COHORT_PARQUET is the raw study universe for this step only; it is not "
        "already filtered to the analysis cohort. Apply "
        "exactly the Planner-locked cohort definition, report truthful "
        "universe-to-final attrition, and emit an analysis_cohort whose ordered "
        "row identity matches the locked host cohort. Downstream steps receive "
        "the filtered cohort. Planner-locked cohort definition JSON: "
        f"{locked_cohort_payload}."
    )
    if materialized_execution_payload is not None:
        attachment += (
            " HOST-VERIFIED COHORT EXECUTION RECEIPT (binding): the host "
            "deterministically resolved the Planner-owned predicates against "
            "the sealed raw universe. Use every `resolved_column` and operation "
            "in order, and assert the recorded before/excluded/remaining counts. "
            "Before applying a predicate, enforce any host-proven closed domain "
            "in the matching ResearchContext variable descriptor; in particular, "
            "an observed binary column must fail closed unless every non-missing "
            "value is exactly in {0, 1}. A threshold check alone is not a domain "
            "check. "
            "The counts and digests are integrity checks, not permission to "
            "select rows by position, truncate, sample, or copy an arbitrary "
            "same-sized frame. Receipt JSON: "
            f"{materialized_execution_payload}."
        )
    return authority.append(attachment)


def bind_execution_cohort_runtime(
    *,
    authority: HostCoderAuthority,
) -> HostCoderAuthority:
    """Explain the host-owned current-cohort cardinality coordinate."""

    return authority.append(
        "CURRENT EXECUTION COHORT (host-owned runtime contract): "
        "COHORT_PARQUET is the exact cohort selected for this step. The "
        "ResearchContext cohort cardinality can describe the earlier run input "
        "before eligibility or a typed cohort override, so never hard-code it "
        "as the current row count. Derive ordinary denominators from the loaded "
        "DataFrame. If an explicit row-count integrity assertion is needed, "
        "compare len(the loaded COHORT_PARQUET frame) with "
        'int(os.environ["EASYICU_COHORT_ROWS"]); the runner owns that value. '
        "The prompt's outbound-safe variable view uses "
        "observed_shape.opaque_levels, but the digest-verified local "
        "ResearchContext JSON uses observed_domain.levels. Read the latter only "
        "at local execution time when closed categorical helpers need the real "
        "binding; never copy private literals into generated source."
    )


def attach_step_coder_input_authority(
    *,
    enabled: bool,
    authority: HostCoderAuthority,
    run_dir: Path,
    profile_ref: str,
    context: ResearchContext,
    step: AnalysisStep,
    resolved_input_bindings: Mapping[str, Mapping[str, object]],
    runtime_import_names: Iterable[str],
    step_record: MutableMapping[str, object],
    reviewed_memory_runtime: ReviewedMemoryRuntime | None = None,
    approved_software_resources: Sequence[ResourceDescriptor] = (),
) -> HostCoderAuthority:
    """Bind typed-input receipts and optional selected resources for one step."""

    authority = bind_execution_cohort_runtime(authority=authority)
    authority = _coder_authority_with_typed_parent_schema_receipts(
        authority=authority,
        bindings=resolved_input_bindings,
    )
    analysis_family = infer_analysis_type(context).key
    if enabled:
        bundle = build_coder_resource_bundle(
            step_id=step.step_id,
            profile_ref=profile_ref,
            analysis_family=analysis_family,
            step_role=step.planned_analysis_role,
            question=context.research_question,
            intent=step.intent,
            method=step.method,
            planner_inputs=step.inputs,
            expected_outputs=step.expected_outputs,
            resolved_input_bindings=resolved_input_bindings,
            runtime_import_names=runtime_import_names,
            has_table_one_spec=step.table_one_spec is not None,
            approved_software_resources=approved_software_resources,
        )
        authority, path = attach_coder_resources(
            authority=authority, run_dir=run_dir, bundle=bundle
        )
        step_record.update(
            {
                "coder_resource_selection_path": path.relative_to(run_dir).as_posix(),
                "coder_resource_selection_sha256": bundle.sha256,
                "coder_resource_prompt_bytes": bundle.prompt_bytes,
                "coder_resource_provider_calls": 0,
            }
        )
    if reviewed_memory_runtime is not None:
        memory_result = reviewed_memory_runtime.attach(
            authority=authority,
            run_dir=run_dir,
            profile_ref=profile_ref,
            step_id=step.step_id,
            analysis_family=analysis_family,
            step_role=step.planned_analysis_role,
            question=context.research_question,
            method=step.method,
        )
        if memory_result is not None:
            authority, memory_bundle, memory_path = memory_result
            step_record.update(
                {
                    "reviewed_memory_selection_path": memory_path.relative_to(
                        run_dir
                    ).as_posix(),
                    "reviewed_memory_selection_sha256": memory_bundle.sha256,
                    "reviewed_memory_prompt_bytes": memory_bundle.prompt_bytes,
                    "reviewed_memory_provider_calls": 0,
                }
            )
    return authority


__all__ = [
    "CODER_RESOURCE_BUNDLE_SCHEMA",
    "CODER_RESOURCE_PROMPT_LIMIT_BYTES",
    "CoderResourceBundle",
    "CoderResourceIntegrityError",
    "attach_coder_resources",
    "attach_step_coder_input_authority",
    "bind_execution_cohort_runtime",
    "bind_materialized_coder_authority",
    "bind_primary_cohort_role",
    "build_coder_resource_bundle",
    "persist_coder_resource_bundle",
]
