"""Strict typed-input extensions for :class:`ResearchContext`.

Version 1 remains the literal archived contract in :mod:`schema`.  Version 2
added host-verified physical and lineage facts.  Version 3 additionally binds
the descriptor's materialization-window role to those facts.  Archived V2
payloads remain readable under their original contract; new typed contexts are
written as V3.  None of these facts assigns the study cohort, exposure,
outcome, method, or estimand; those remain Planner and Coder decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from easyicu.concept.metadata_sidecar import (
    ColumnMetadataBinding,
    SidecarRef,
    TimeCoordinate,
    binding_payload_sha256,
)
from easyicu.concept.metadata_projection import (
    NumericBounds,
    is_range_preserving_projection,
)

from ..intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    MaterializedMetadataError,
    VerifiedMaterializedCohortAuthority,
)
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    TrajectoryConceptBinding,
    VerifiedMaterializedTrajectoryAuthority,
)
from ..contracts.cohort_receipt import COHORT_RECEIPT_COLUMN_FIELDS
from ..icu_rules import ICU_RULES
from .implementation_identity import metadata_implementation_identity
from ..schema import ConceptDescriptor, ResearchContext

RESEARCH_CONTEXT_V2_SCHEMA_VERSION = "easyicu.research_context/2"
RESEARCH_CONTEXT_V3_SCHEMA_VERSION = "easyicu.research_context/3"
MATERIALIZED_INPUT_PROMPT_SCHEMA_VERSION = "easyicu.materialized_input_prompt_facts/2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_PROMPT_COLUMN_BINDINGS = 48
_MAX_PROMPT_TRAJECTORY_CONCEPTS = 48
_MAX_PROMPT_LINEAGES_PER_COLUMN = 1
_MAX_MATERIALIZED_PROMPT_BYTES = 4 * 1024
_MATERIALIZED_PROMPT_HEADING = (
    "MATERIALIZED INPUT FACTS (host-verified; physical/lineage authority "
    "only; not a scientific-design instruction):\n"
)


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def binding_preserves_analysis_range(binding: ColumnMetadataBinding) -> bool:
    """Whether one physical binding preserves single-measurement ranges.

    Numeric sums are deliberately excluded: a valid per-measurement range is
    not a valid range for a cumulative total.  This mirrors the typed metadata
    projector without reinterpreting a column name or analysis role.
    """

    return is_range_preserving_projection(
        binding.metadata.role,
        binding.metadata.aggregation,
    )


def effective_analysis_plausibility_range(
    binding: ColumnMetadataBinding,
) -> Optional[Dict[str, Optional[float]]]:
    """Return the sealed sidecar range or source-concept ICU fallback.

    The physical column name is not scientific authority.  When the sidecar
    does not declare an analysis range, the independently versioned ICU rules
    are therefore queried with the sealed ``source_concept``.  Non-range-
    preserving projections never inherit the single-measurement fallback.
    """

    if not binding_preserves_analysis_range(binding):
        return None
    metadata = binding.metadata
    if metadata.analysis_plausibility_range is not None:
        return metadata.analysis_plausibility_range.to_dict()
    hint = ICU_RULES.classify_variable(metadata.source_concept, "", None)
    if hint.valid_range is None:
        return None
    return {
        "minimum": float(hint.valid_range[0]),
        "maximum": float(hint.valid_range[1]),
    }


class CanonicalColumnBinding(BaseModel):
    """One canonical column binding plus its independently checkable digest."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    binding: Dict[str, Any]
    binding_sha256: str
    analysis_plausibility_range: Optional[Dict[str, Optional[float]]] = None

    @model_validator(mode="after")
    def _validate_canonical_binding(self) -> "CanonicalColumnBinding":
        parsed = ColumnMetadataBinding.from_dict(self.binding)
        if parsed.to_dict() != self.binding:
            raise ValueError("column binding is not canonical")
        expected = binding_payload_sha256({parsed.metadata.column_name: parsed})
        if self.binding_sha256 != expected:
            raise ValueError("column binding digest mismatch")
        if self.analysis_plausibility_range is not None:
            NumericBounds.from_dict(self.analysis_plausibility_range)
            if not binding_preserves_analysis_range(parsed):
                raise ValueError(
                    "analysis plausibility range requires a range-preserving binding"
                )
        return self


def canonical_column_binding(
    column: str,
    binding: ColumnMetadataBinding,
) -> CanonicalColumnBinding:
    """Seal one physical binding and its effective analysis range."""

    return CanonicalColumnBinding(
        binding=binding.to_dict(),
        binding_sha256=binding_payload_sha256({column: binding}),
        analysis_plausibility_range=effective_analysis_plausibility_range(binding),
    )


def descriptor_physical_updates(
    binding: CanonicalColumnBinding,
) -> Dict[str, Any]:
    """Reconstruct legacy descriptor fields owned by typed input authority.

    ``ConceptDescriptor`` predates the typed materialized-input contract and
    therefore repeats a small set of physical facts.  V2 keeps those fields for
    API compatibility, but they must be a deterministic view of the sealed
    binding rather than a second, independently mutable authority.  Scientific
    fields such as analysis role, aggregation choice, ordinal interpretation,
    exposure/outcome, covariates, and estimand are intentionally absent.
    """

    parsed = ColumnMetadataBinding.from_dict(binding.binding)
    metadata = parsed.metadata
    value_like = metadata.role.value in {"value", "numeric_aggregate"}
    time_like = metadata.role.value in {
        "first_observation_time",
        "last_observation_time",
        "event_time",
    }
    plausible = binding.analysis_plausibility_range
    valid_range = None
    if (
        plausible is not None
        and plausible.get("minimum") is not None
        and plausible.get("maximum") is not None
    ):
        valid_range = [plausible["minimum"], plausible["maximum"]]
    lineage = metadata.source_lineage
    updates: Dict[str, Any] = {
        "unit": (
            metadata.time_unit
            if time_like
            else metadata.canonical_unit if value_like else None
        ),
        "valid_range": valid_range,
        "source_concept": metadata.source_concept,
        "derived_from_concepts": list(metadata.derived_from_concepts),
        "source_tables": sorted(
            {entry.table for entry in lineage if entry.table is not None}
        ),
        "item_ids": sorted({item for entry in lineage for item in entry.item_ids_json}),
        "unit_normalization": parsed.representation_transform,
        "source_databases": list(metadata.available_databases),
    }
    if metadata.description:
        updates["description"] = metadata.description
    if parsed.derivation_window is not None:
        window = parsed.derivation_window
        updates["analysis_window"] = (
            f"{window.origin}[{window.start_hours:g},{window.end_hours:g}]h"
        )
        updates["analysis_window_role"] = "outer_observation_window"
    if metadata.time_origin is not None and metadata.time_unit is not None:
        updates["temporal_resolution"] = (
            f"relative to {metadata.time_origin} in {metadata.time_unit}"
        )
    return updates


_V2_DESCRIPTOR_PHYSICAL_FIELDS = frozenset(
    {
        "unit",
        "valid_range",
        "source_concept",
        "derived_from_concepts",
        "source_tables",
        "item_ids",
        "unit_normalization",
        "source_databases",
        "description",
        "analysis_window",
        "temporal_resolution",
    }
)


def _v2_descriptor_physical_updates(
    binding: CanonicalColumnBinding,
) -> Dict[str, Any]:
    """Return the descriptor projection frozen by the archived V2 contract.

    ``analysis_window_role`` was added to the closure validator after V2
    artifacts had already been sealed.  Filtering through an explicit frozen
    field set keeps future descriptor additions from silently narrowing V2
    again.
    """

    return {
        key: value
        for key, value in descriptor_physical_updates(binding).items()
        if key in _V2_DESCRIPTOR_PHYSICAL_FIELDS
    }


class MaterializedCohortContext(BaseModel):
    """Verified staged cohort facts exposed to the research context."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    authority_ref: Dict[str, Any]
    cohort_file: str
    cohort_sha256: str
    cohort_size: int = Field(ge=0)
    cohort_rows: int = Field(ge=0)
    cohort_columns: List[str]
    cohort_schema_sha256: str
    identity_column: str
    row_identity_sha256: str
    column_metadata_ref: Dict[str, Any]
    file_metadata_payload_sha256: str
    source_export_authority_sha256: str
    source_database: str
    time_coordinates: List[Dict[str, str]]
    projection_scope: Literal["full", "scoped"] = "full"
    column_bindings: Dict[str, CanonicalColumnBinding]
    column_binding_payload_sha256: str
    producer_implementation_sha256: str
    metadata_projection_sha256: str
    metadata_sidecar_sha256: str
    icu_rules_sha256: str
    metadata_implementation_bundle_sha256: str

    @model_validator(mode="after")
    def _validate_coordinates(self) -> "MaterializedCohortContext":
        reference = MaterializedCohortAuthorityRef.from_dict(self.authority_ref)
        sidecar_ref = SidecarRef.from_dict(self.column_metadata_ref)
        for label, digest in (
            ("cohort_sha256", self.cohort_sha256),
            ("cohort_schema_sha256", self.cohort_schema_sha256),
            ("row_identity_sha256", self.row_identity_sha256),
            ("file_metadata_payload_sha256", self.file_metadata_payload_sha256),
            ("source_export_authority_sha256", self.source_export_authority_sha256),
            ("column_binding_payload_sha256", self.column_binding_payload_sha256),
            ("producer_implementation_sha256", self.producer_implementation_sha256),
            ("metadata_projection_sha256", self.metadata_projection_sha256),
            ("metadata_sidecar_sha256", self.metadata_sidecar_sha256),
            ("icu_rules_sha256", self.icu_rules_sha256),
            (
                "metadata_implementation_bundle_sha256",
                self.metadata_implementation_bundle_sha256,
            ),
        ):
            _require_sha256(digest, label=label)
        if reference.size <= 0:
            raise ValueError("cohort authority reference must not be empty")
        if self.projection_scope == "full" and sidecar_ref.record_count != len(
            self.column_bindings
        ):
            raise ValueError("cohort metadata record count mismatch")
        if self.identity_column not in self.cohort_columns:
            raise ValueError("cohort identity column is absent")
        time_coordinates = tuple(
            TimeCoordinate.from_dict(item) for item in self.time_coordinates
        )
        time_columns = {item.column for item in time_coordinates}
        expected_columns = set(self.cohort_columns) - {
            self.identity_column,
            *time_columns,
        }
        binding_columns = set(self.column_bindings)
        if (
            self.projection_scope == "full" and binding_columns != expected_columns
        ) or (
            self.projection_scope == "scoped"
            and not binding_columns.issubset(expected_columns)
        ):
            raise ValueError("cohort metadata bindings do not cover columns")
        canonical_bindings: dict[str, ColumnMetadataBinding] = {}
        for column, binding in self.column_bindings.items():
            metadata = binding.binding.get("metadata")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("column_name") != column
            ):
                raise ValueError("cohort binding key does not match column metadata")
            canonical_bindings[column] = ColumnMetadataBinding.from_dict(
                binding.binding
            )
        if (
            binding_payload_sha256(canonical_bindings)
            != self.column_binding_payload_sha256
        ):
            raise ValueError("cohort binding-set digest mismatch")
        return self


class MaterializedTrajectoryContext(BaseModel):
    """Verified canonical long-trajectory facts, when explicitly staged."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    authority_ref: Dict[str, Any]
    trajectory_file: str
    trajectory_sha256: str
    trajectory_size: int = Field(ge=0)
    trajectory_rows: int = Field(ge=0)
    trajectory_columns: List[str]
    trajectory_schema_sha256: str
    identity_column: str
    time_column: str
    time_origin: str
    time_unit: str
    concept_column: str
    numeric_value_column: str
    text_value_column: str
    ordered_row_key_sha256: str
    stay_identity_set_sha256: str
    trajectory_stays: int = Field(ge=0)
    bound_universe_authority_ref: Dict[str, Any]
    bound_universe_row_identity_sha256: str
    source_export_authority_sha256: str
    requested_concepts: List[str]
    materialized_concepts: List[str]
    available_unobserved_concepts: List[str]
    unavailable_concepts: List[str]
    window: Optional[Dict[str, Any]]
    projection_scope: Literal["full", "scoped"] = "full"
    concept_bindings: Dict[str, Dict[str, Any]]
    concept_analysis_plausibility_ranges: Dict[
        str, Optional[Dict[str, Optional[float]]]
    ]
    concept_binding_payload_sha256: str
    producer_implementation_sha256: str

    @model_validator(mode="after")
    def _validate_coordinates(self) -> "MaterializedTrajectoryContext":
        MaterializedTrajectoryAuthorityRef.from_dict(self.authority_ref)
        MaterializedCohortAuthorityRef.from_dict(self.bound_universe_authority_ref)
        for label, digest in (
            ("trajectory_sha256", self.trajectory_sha256),
            ("trajectory_schema_sha256", self.trajectory_schema_sha256),
            ("ordered_row_key_sha256", self.ordered_row_key_sha256),
            ("stay_identity_set_sha256", self.stay_identity_set_sha256),
            (
                "bound_universe_row_identity_sha256",
                self.bound_universe_row_identity_sha256,
            ),
            ("source_export_authority_sha256", self.source_export_authority_sha256),
            ("producer_implementation_sha256", self.producer_implementation_sha256),
            ("concept_binding_payload_sha256", self.concept_binding_payload_sha256),
        ):
            _require_sha256(digest, label=label)
        requested = set(self.requested_concepts)
        materialized = set(self.materialized_concepts)
        unobserved = set(self.available_unobserved_concepts)
        unavailable = set(self.unavailable_concepts)
        if (
            len(requested) != len(self.requested_concepts)
            or len(materialized) != len(self.materialized_concepts)
            or len(unobserved) != len(self.available_unobserved_concepts)
            or len(unavailable) != len(self.unavailable_concepts)
            or materialized & unobserved
            or materialized & unavailable
            or unobserved & unavailable
            or materialized | unobserved | unavailable != requested
        ):
            raise ValueError("trajectory availability states do not close")
        expected_bound = {
            concept
            for concept in self.requested_concepts
            if concept in materialized or concept in unobserved
        }
        if set(self.concept_bindings) != expected_bound:
            raise ValueError("trajectory bindings do not match available concepts")
        if set(self.concept_analysis_plausibility_ranges) != expected_bound:
            raise ValueError(
                "trajectory plausibility ranges do not match available concepts"
            )
        for concept, payload in self.concept_bindings.items():
            parsed = TrajectoryConceptBinding.from_dict(payload)
            if parsed.concept_id != concept or parsed.to_dict() != payload:
                raise ValueError("trajectory concept binding is not canonical")
            plausible = self.concept_analysis_plausibility_ranges[concept]
            if plausible is not None:
                NumericBounds.from_dict(plausible)
                if not binding_preserves_analysis_range(parsed.binding):
                    raise ValueError(
                        "trajectory plausibility range requires a range-preserving binding"
                    )
            explicit = parsed.binding.metadata.analysis_plausibility_range
            if explicit is not None and plausible != explicit.to_dict():
                raise ValueError(
                    "trajectory explicit plausibility range does not match its binding"
                )
        expected_digest = hashlib.sha256(
            json.dumps(
                self.concept_bindings,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if expected_digest != self.concept_binding_payload_sha256:
            raise ValueError("trajectory binding-set digest mismatch")
        return self


class MaterializedResearchInputs(BaseModel):
    """Host-owned physical input facts; never study-design assignments."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cohort: MaterializedCohortContext
    trajectory: Optional[MaterializedTrajectoryContext] = None

    @model_validator(mode="after")
    def _validate_shared_universe(self) -> "MaterializedResearchInputs":
        trajectory = self.trajectory
        if trajectory is not None and (
            trajectory.bound_universe_authority_ref != self.cohort.authority_ref
            or trajectory.bound_universe_row_identity_sha256
            != self.cohort.row_identity_sha256
            or trajectory.source_export_authority_sha256
            != self.cohort.source_export_authority_sha256
        ):
            raise ValueError("trajectory is not bound to the context cohort")
        return self


class ResearchContextV2(ResearchContext):
    """Archived typed context with the original V2 closure semantics."""

    # The inherited V1 fields retain their archived coercion/JSON behaviour
    # (notably ISO datetime parsing). Strictness is applied to every new V2
    # authority model above, where numeric/string coercion would be unsafe.
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[RESEARCH_CONTEXT_V2_SCHEMA_VERSION] = (
        RESEARCH_CONTEXT_V2_SCHEMA_VERSION
    )
    materialized_inputs: MaterializedResearchInputs

    @model_validator(mode="after")
    def _validate_context_authority_closure(self) -> "ResearchContextV2":
        cohort = self.materialized_inputs.cohort
        if self.cohort.database != cohort.source_database:
            raise ValueError("context database does not match typed cohort source")
        if cohort.identity_column not in self.cohort.id_columns:
            raise ValueError("typed cohort identity is absent from context ids")
        time_columns = {
            str(item.get("column") or "") for item in cohort.time_coordinates
        }
        if not time_columns.issubset(set(self.cohort.time_columns)):
            raise ValueError("typed time coordinates are absent from context")
        if self.cohort.n_stays != cohort.cohort_rows:
            raise ValueError("context stay count does not match typed cohort rows")
        excluded_columns = {cohort.identity_column, *time_columns}
        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("typed context variable names must be unique")
        outside_columns = set(variable_names) - set(cohort.cohort_columns)
        if outside_columns:
            raise ValueError(
                "typed context variables are absent from the cohort: "
                + ", ".join(sorted(outside_columns))
            )
        if cohort.projection_scope == "full" and set(variable_names) != set(
            cohort.cohort_columns
        ):
            raise ValueError("full typed context does not cover the cohort columns")
        selected_variables = {
            variable.name: variable
            for variable in self.variables
            if variable.name not in excluded_columns
        }
        missing = set(selected_variables) - set(cohort.column_bindings)
        if missing:
            raise ValueError(
                "selected context variables lack typed cohort bindings: "
                + ", ".join(sorted(missing))
            )
        for column, variable in selected_variables.items():
            binding = cohort.column_bindings[column]
            legacy_updates = _v2_descriptor_physical_updates(binding)
            for field_name, expected in legacy_updates.items():
                if getattr(variable, field_name) != expected:
                    raise ValueError(
                        "context descriptor physical field does not match typed "
                        f"column binding: {column}.{field_name}"
                    )
        if self.endpoint is not None:
            # The receipt for the endpoint declaration. A declaration is only
            # worth trusting if the columns it names exist in the cohort that
            # was actually verified -- otherwise it is prose with a type
            # annotation, and a downstream consumer resolving the columns would
            # be back to guessing. Fail closed here, at declaration time,
            # rather than at the step that first tries to read the column.
            absent = [
                column
                for column in self.endpoint.declared_columns()
                if column not in set(cohort.cohort_columns)
            ]
            if absent:
                raise ValueError(
                    "endpoint declaration names columns absent from the typed "
                    "cohort: " + ", ".join(sorted(absent))
                )
        return self


class ResearchContextV3(ResearchContextV2):
    """Current typed context with a bound materialization-window role."""

    schema_version: Literal[RESEARCH_CONTEXT_V3_SCHEMA_VERSION] = (
        RESEARCH_CONTEXT_V3_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_window_role_closure(self) -> "ResearchContextV3":
        cohort = self.materialized_inputs.cohort
        time_columns = {
            str(item.get("column") or "") for item in cohort.time_coordinates
        }
        excluded_columns = {cohort.identity_column, *time_columns}
        selected_variables = {
            variable.name: variable
            for variable in self.variables
            if variable.name not in excluded_columns
        }
        for column, variable in selected_variables.items():
            binding = cohort.column_bindings[column]
            expected = descriptor_physical_updates(binding).get(
                "analysis_window_role"
            )
            if expected is not None and variable.analysis_window_role != expected:
                raise ValueError(
                    "context descriptor physical field does not match typed "
                    f"column binding: {column}.analysis_window_role"
                )
        return self


ResearchContextAuthority = Union[
    ResearchContext,
    ResearchContextV2,
    ResearchContextV3,
]


def _revalidate_typed_research_context(
    context: ResearchContextV2,
) -> ResearchContextV2:
    """Revalidate nested mutable payloads without changing schema versions."""

    return type(context).model_validate(context.model_dump(mode="python"))


def _without_empty_values(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None and value != [] and value != {} and value != ""
    }


def _column_prompt_fact(
    column: str,
    binding: CanonicalColumnBinding,
) -> Dict[str, Any]:
    """Return a bounded fact-only view of one verified physical binding."""

    parsed = ColumnMetadataBinding.from_dict(binding.binding)
    metadata = parsed.metadata
    lineages = []
    for lineage in metadata.source_lineage[:_MAX_PROMPT_LINEAGES_PER_COLUMN]:
        lineages.append(
            _without_empty_values(
                {
                    "database": lineage.database,
                    "table": lineage.table,
                    "value_variable": lineage.value_variable,
                    "time_variable": lineage.time_variable,
                    "source_class_name": lineage.source_class_name,
                    "target": lineage.target,
                }
            )
        )
    return _without_empty_values(
        {
            "column": column,
            "source_concept": metadata.source_concept,
            "physical_role": metadata.role.value,
            "aggregation": metadata.aggregation,
            "canonical_unit": metadata.canonical_unit,
            "extraction_bounds": (
                metadata.extraction_bounds.to_dict()
                if metadata.extraction_bounds is not None
                else None
            ),
            # This is the effective context-time plausibility range. It may
            # originate from host ICU rules when the extraction sidecar has no
            # analysis range; it is deliberately distinct from extraction
            # bounds.
            "analysis_plausibility_range": binding.analysis_plausibility_range,
            "allowed_values": (
                list(metadata.allowed_values)
                if metadata.allowed_values is not None
                else None
            ),
            "time_origin": metadata.time_origin,
            "time_unit": metadata.time_unit,
            "source_database_actual": metadata.source_database,
            "availability_basis": metadata.availability_basis,
            "source_lineage": lineages,
            "source_lineage_omitted_count": max(
                0,
                len(metadata.source_lineage) - len(lineages),
            ),
            "derivation_window": (
                parsed.derivation_window.to_dict()
                if parsed.derivation_window is not None
                else None
            ),
            "representation_transform": parsed.representation_transform,
        }
    )


def materialized_input_prompt_projection(
    context: ResearchContextAuthority,
) -> Optional[Dict[str, Any]]:
    """Build the bounded host-fact projection shared by Planner and Coder.

    The projection exposes only physical representation, availability, and
    lineage coordinates. It cannot assign the study cohort, exposure,
    outcome, method, covariate set, or estimand. Full authority remains in the
    sealed ResearchContext; explicit counts make every transport omission
    visible rather than silently pretending the prompt contains the full
    sidecar.
    """

    if not isinstance(context, ResearchContextV2):
        return None
    # Typed contexts are frozen at the model surface, while nested
    # compatibility payloads remain ordinary Python containers. Revalidate the
    # canonical dump before every authority-bearing prompt render so an
    # in-memory nested mutation cannot bypass binding digests or closure
    # validators.
    context = _revalidate_typed_research_context(context)
    cohort = context.materialized_inputs.cohort
    exact_variable_order = [
        variable.name
        for variable in context.variables
        if variable.name in cohort.column_bindings
    ]
    protected_science_names = {
        str(value).strip().casefold()
        for value in (context.primary_exposure, context.target_outcome)
        if str(value or "").strip()
    }

    def column_priority(column: str) -> tuple[int, int, str]:
        binding = cohort.column_bindings[column]
        metadata = binding.binding.get("metadata")
        source_concept = (
            str(metadata.get("source_concept") or "").strip().casefold()
            if isinstance(metadata, Mapping)
            else ""
        )
        normalized = column.casefold()
        protected = int(
            normalized in protected_science_names
            or source_concept in protected_science_names
        )
        try:
            variable_index = exact_variable_order.index(column)
        except ValueError:
            variable_index = len(exact_variable_order)
        return (-protected, variable_index, column)

    column_names = sorted(cohort.column_bindings, key=column_priority)
    selected_columns = column_names[:_MAX_PROMPT_COLUMN_BINDINGS]
    protected_source_concepts = set(protected_science_names)
    for column in column_names:
        binding = cohort.column_bindings[column]
        metadata = binding.binding.get("metadata")
        source_concept = (
            str(metadata.get("source_concept") or "").strip().casefold()
            if isinstance(metadata, Mapping)
            else ""
        )
        if column.casefold() in protected_science_names or (
            source_concept in protected_science_names
        ):
            protected_source_concepts.add(source_concept)
    cohort_facts: Dict[str, Any] = {
        "projection_scope": cohort.projection_scope,
        "authority_sha256": cohort.authority_ref["sha256"],
        "cohort_sha256": cohort.cohort_sha256,
        "cohort_rows": cohort.cohort_rows,
        "identity_column": cohort.identity_column,
        "row_identity_sha256": cohort.row_identity_sha256,
        "source_database_actual": cohort.source_database,
        "time_coordinates": list(cohort.time_coordinates),
        "column_bindings": [
            _column_prompt_fact(column, cohort.column_bindings[column])
            for column in selected_columns
        ],
        "column_binding_total_count": len(column_names),
        "column_binding_omitted_count": max(
            0,
            len(column_names) - len(selected_columns),
        ),
        "column_binding_payload_sha256": cohort.column_binding_payload_sha256,
    }

    trajectory_facts: Optional[Dict[str, Any]] = None
    trajectory = context.materialized_inputs.trajectory
    if trajectory is not None:
        requested = sorted(
            trajectory.requested_concepts,
            key=lambda concept: (
                -int(concept.casefold() in protected_source_concepts),
                trajectory.requested_concepts.index(concept),
            ),
        )
        selected_concepts = requested[:_MAX_PROMPT_TRAJECTORY_CONCEPTS]
        materialized = set(trajectory.materialized_concepts)
        unobserved = set(trajectory.available_unobserved_concepts)
        concept_facts: List[Dict[str, Any]] = []
        for concept in selected_concepts:
            if concept in materialized:
                status = "materialized"
            elif concept in unobserved:
                status = "available_unobserved"
            else:
                status = "unavailable"
            fact: Dict[str, Any] = {"concept": concept, "status": status}
            raw_binding = trajectory.concept_bindings.get(concept)
            if raw_binding is not None:
                parsed = TrajectoryConceptBinding.from_dict(raw_binding)
                binding_digest = binding_payload_sha256(
                    {parsed.source.column: parsed.binding}
                )
                fact.update(
                    {
                        "source": {
                            "file": parsed.source.file,
                            "column": parsed.source.column,
                        },
                        "physical_binding": _column_prompt_fact(
                            parsed.source.column,
                            CanonicalColumnBinding(
                                binding=parsed.binding.to_dict(),
                                binding_sha256=binding_digest,
                                analysis_plausibility_range=trajectory.concept_analysis_plausibility_ranges[
                                    concept
                                ],
                            ),
                        ),
                    }
                )
            concept_facts.append(fact)
        trajectory_facts = _without_empty_values(
            {
                "projection_scope": trajectory.projection_scope,
                "authority_sha256": trajectory.authority_ref["sha256"],
                "trajectory_sha256": trajectory.trajectory_sha256,
                "trajectory_rows": trajectory.trajectory_rows,
                "identity_column": trajectory.identity_column,
                "time_column": trajectory.time_column,
                "time_origin": trajectory.time_origin,
                "time_unit": trajectory.time_unit,
                "concept_column": trajectory.concept_column,
                "numeric_value_column": trajectory.numeric_value_column,
                "text_value_column": trajectory.text_value_column,
                "window": trajectory.window,
                "concepts": concept_facts,
                "concept_total_count": len(requested),
                "concept_omitted_count": max(
                    0,
                    len(requested) - len(selected_concepts),
                ),
                "concept_binding_payload_sha256": (
                    trajectory.concept_binding_payload_sha256
                ),
            }
        )

    facts: Dict[str, Any] = {
        "schema_version": MATERIALIZED_INPUT_PROMPT_SCHEMA_VERSION,
        "authority_scope": "physical_representation_availability_and_lineage_only",
        "scientific_ownership": (
            "Planner/Coder retain cohort, exposure, outcome, method, covariates, "
            "and estimand decisions"
        ),
        "cohort": cohort_facts,
        "metadata_implementation": {
            "metadata_implementation_bundle_sha256": (
                cohort.metadata_implementation_bundle_sha256
            ),
        },
    }
    if trajectory_facts is not None:
        facts["trajectory"] = trajectory_facts

    def with_projection_digest(value: Dict[str, Any]) -> Dict[str, Any]:
        canonical_without_digest = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return {
            **value,
            "projection_sha256": hashlib.sha256(
                canonical_without_digest.encode("utf-8")
            ).hexdigest(),
        }

    def rendered_size(value: Dict[str, Any]) -> int:
        return len(
            (
                _MATERIALIZED_PROMPT_HEADING
                + json.dumps(
                    with_projection_digest(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ).encode("utf-8")
        )

    # Prompt transport is a bounded view, never the authority store. Remove
    # least-priority companion facts first while keeping exact primary
    # exposure/outcome facts whenever they are physically bound. Every
    # omission remains explicit and points back to the binding-set digest.
    while rendered_size(facts) > _MAX_MATERIALIZED_PROMPT_BYTES:
        removable: Optional[int] = None
        rows = cohort_facts["column_bindings"]
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            if (
                str(row.get("column") or "").casefold() not in protected_science_names
                and str(row.get("source_concept") or "").casefold()
                not in protected_science_names
            ):
                removable = index
                break
        if removable is not None:
            rows.pop(removable)
            cohort_facts["column_binding_omitted_count"] += 1
            continue
        if trajectory_facts is not None:
            concepts = trajectory_facts.get("concepts") or []
            trajectory_removable = next(
                (
                    index
                    for index in range(len(concepts) - 1, -1, -1)
                    if str(concepts[index].get("concept") or "").casefold()
                    not in protected_source_concepts
                ),
                None,
            )
            if trajectory_removable is not None:
                concepts.pop(trajectory_removable)
                trajectory_facts["concept_omitted_count"] += 1
                continue
        break
    result = with_projection_digest(facts)
    if rendered_size(facts) > _MAX_MATERIALIZED_PROMPT_BYTES:
        raise ValueError(
            "materialized input prompt facts exceed the 4 KiB transport limit"
        )
    return result


def _closed_observed_levels(
    domain: Optional[Mapping[str, Any]],
) -> Optional[List[Any]]:
    """Return a small exact observed domain, or ``None`` when it is unsafe."""

    if domain is None:
        return None
    observed_levels = domain.get("levels")
    n_unique = domain.get("n_unique")
    if not (
        isinstance(observed_levels, list)
        and 1 <= len(observed_levels) <= 8
        and isinstance(n_unique, int)
        and not isinstance(n_unique, bool)
        and n_unique == len(observed_levels)
    ):
        return None
    try:
        encoded_levels = [
            (
                type(value).__name__,
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            for value in observed_levels
            if value is not None and isinstance(value, (bool, int, float, str))
        ]
    except (TypeError, ValueError):
        return None
    if len(encoded_levels) != len(observed_levels) or len(set(encoded_levels)) != len(
        encoded_levels
    ):
        return None
    return list(observed_levels)


def _legacy_raw_input_contract(variable: ConceptDescriptor) -> Dict[str, Any]:
    """Project a V1 descriptor into an executable, step-sealed contract.

    V1 contexts do not carry a materialized sidecar, but their descriptors are
    still host-generated from the exact in-memory cohort and are sealed in the
    run input.  Persisting this bounded projection keeps legacy/offline runs
    executable without letting generated code rediscover policy from the whole
    ResearchContext.
    """

    fact: Dict[str, Any] = {
        "column": variable.name,
        "dtype": variable.dtype,
    }
    if variable.unit:
        fact["unit"] = variable.unit
    if variable.valid_range is not None:
        bounds = list(variable.valid_range)
        if len(bounds) != 2 or any(
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(bound))
            for bound in bounds
        ):
            raise ValueError(
                f"legacy descriptor {variable.name!r} has an invalid valid_range"
            )
        minimum, maximum = bounds
        if float(minimum) > float(maximum):
            raise ValueError(
                f"legacy descriptor {variable.name!r} has a reversed valid_range"
            )
        fact["analysis_plausibility_range"] = {
            "minimum": minimum,
            "maximum": maximum,
        }
        fact["plausibility_policy"] = {
            "range_policy": "flag_only",
            "out_of_range_action": "retain_and_flag",
        }
    _apply_domain(fact, variable)
    return fact


def _apply_domain(fact: Dict[str, Any], variable: ConceptDescriptor) -> None:
    """Publish a DECLARED domain when one exists, an observed one otherwise.

    ``ordinal_levels`` is a declaration: ``icu_rules`` fixes SOFA components at
    0-4 and KDIGO stage at 0-3 by construction, and 687 of 4,318 recorded
    context variables already carry it.  This layer ignored it and re-derived a
    level set from whatever the cohort happened to contain -- so the host held
    the codebook and published a guess.

    The difference is not cosmetic.  An observed set cannot contain a level with
    zero cases, which is exactly what a stage-stratified table has to show; the
    2026-07-30 note that introduced the observed fallback names "not displaying
    the zero-count one" as part of the defect it was fixing, and deriving the
    levels from data structurally cannot deliver it.  Measured on the recorded
    runs: 327 contracts published an ordinal-score domain taken from
    observation, and every one of those concepts has a declaration available.

    The concept dictionary is the second declaration.  A ``fct_cncpt`` states
    its own closed factor -- ``adm`` is ``['med','surg','other']`` -- and that
    is a codebook, not a sample.  Reading it also makes a mapping defect
    visible instead of blessing it: MIMIC-IV's service table contains ``EYE``,
    the dictionary's own ``apply_map`` for ``adm`` has no entry for it, and the
    raw code passed straight through.  All 201 recorded contexts observed
    ``['EYE','med','other','surg']`` on a concept that declares three levels,
    and publishing the observation labelled the leak legal.  Whether ``EYE``
    belongs in ``surg`` or ``other`` is a clinical mapping decision and is NOT
    made here; this only stops the host asserting it is already legal.

    The observed fallback stays for concepts with neither declaration -- 702
    binary 0/1 flags on the recorded runs -- and keeps saying so in
    ``allowed_values_basis``.
    """

    observed = _closed_observed_levels(variable.observed_domain)
    declared, basis = _declared_domain(variable)
    if declared is not None:
        fact["allowed_values"] = list(declared)
        fact["allowed_values_basis"] = basis
        if observed is not None:
            # Keep the observation visible rather than replacing it: a declared
            # level this cohort never saw, and an observed value the codebook
            # does not declare, are both real reportable facts, and a consumer
            # can only see either by comparing the two.
            fact["observed_values"] = observed
        return
    if observed is not None:
        fact["allowed_values"] = observed
        fact["allowed_values_basis"] = "sealed_research_context_observed_domain"


_CONCEPT_LEVELS_CACHE: Dict[str, Optional[List[Any]]] = {}


def _dictionary_declared_levels(source_concept: Optional[str]) -> Optional[List[Any]]:
    """Return the concept dictionary's own closed factor levels, if it has one."""

    if not source_concept:
        return None
    if source_concept in _CONCEPT_LEVELS_CACHE:
        return _CONCEPT_LEVELS_CACHE[source_concept]
    levels: Optional[List[Any]] = None
    try:  # local import to avoid import-time cost / cycles, as icu_rules does
        from ...concept.loader import load_dictionary

        definition = load_dictionary().get(source_concept)
        raw = getattr(definition, "levels", None)
        if isinstance(raw, (list, tuple)) and raw:
            levels = list(raw)
    except Exception:
        levels = None
    _CONCEPT_LEVELS_CACHE[source_concept] = levels
    return levels


#: Representation transforms whose column still holds the SOURCE CONCEPT'S OWN
#: VALUES, and is therefore described by that concept's declared levels.
#:
#: Everything else is a different quantity derived from those values -- a count
#: of them, a flag that any were seen, the time of the first one, a numeric
#: summary -- and the concept's level set says nothing about its domain.
#:
#: MEASURED over every recorded resolved-input contract, which is where this
#: enumeration comes from rather than from judgement: 13 distinct
#: (physical_role, representation_transform) pairs exist, and exactly one --
#: ``stay_level_unique_value`` -- carries the value itself. The rest are
#: ``window_nonnull_count`` (a count), ``window_measurement_status`` /
#: ``window_presence_max`` / ``whole_stay_any_truthy`` (0-1 flags),
#: ``window_first_time`` / ``window_last_time`` / ``first_truthy_event_time``
#: (timestamps) and ``window_numeric_first|max|mean|min`` (numeric summaries).
_VALUE_PRESERVING_TRANSFORMS = frozenset({"stay_level_unique_value"})


def _transform_preserves_concept_values(transform: Any) -> bool:
    """Whether a column under this transform still holds the concept's values.

    An unknown transform answers False. A new derived representation must say
    it preserves the value domain before it inherits one -- the failure this
    guards is a contract nothing can satisfy, so ambiguity fails closed.
    """

    return str(transform or "") in _VALUE_PRESERVING_TRANSFORMS


def _declared_domain(
    variable: ConceptDescriptor,
) -> tuple[Optional[List[Any]], Optional[str]]:
    """Return the highest-authority declared domain for one descriptor.

    Order is authority, not convenience: the host's own ICU rules fix a score's
    range by construction, so they outrank the dictionary; the dictionary
    outranks anything derived from this cohort's rows.
    """

    ordinal = getattr(variable, "ordinal_levels", None)
    if ordinal:
        return list(ordinal), "declared_ordinal_levels"
    levels = _dictionary_declared_levels(getattr(variable, "source_concept", None))
    if levels:
        return levels, "declared_concept_dictionary_levels"
    return None, None


def resolved_raw_input_contracts(
    context: ResearchContextAuthority,
    planner_declared_inputs: Sequence[str],
) -> Dict[str, Any]:
    """Return exact executable metadata for Planner-declared raw columns.

    The outbound prompt projection is explanatory transport, while generated
    code executes against ``EASYICU_RESOLVED_INPUTS_JSON``.  Put the same
    host-verified physical/domain facts in that manifest so code never has to
    rediscover a range policy or closed domain by scanning the broader
    ResearchContext.  Typed ``kind:name`` products remain under the manifest's
    existing ``inputs`` authority and are intentionally excluded here.

    A repeated name is normalized away rather than rejected.  Every contract
    here is a pure function of the name -- both branches below resolve it from
    ``name`` alone and store it in a name-keyed dict -- so a second occurrence
    produces a byte-identical entry and cannot make the manifest ambiguous.
    ``raw_contract_inputs_for_step`` already applies exactly this rule to the
    cohort predicate columns it appends (``if resolved_column not in names``),
    so rejecting the Planner's own repeat made one call chain treat the same
    fact as harmless in one line and fatal four lines later.
    """

    raw_names = list(
        dict.fromkeys(
            str(value).strip()
            for value in planner_declared_inputs
            if isinstance(value, str) and str(value).strip() and ":" not in str(value)
        )
    )
    variables = {variable.name: variable for variable in context.variables}
    contracts: Dict[str, Any] = {}
    if not isinstance(context, ResearchContextV2):
        for name in raw_names:
            variable = variables.get(name)
            if variable is None:
                raise ValueError(
                    f"Planner-declared raw input {name!r} lacks a context descriptor"
                )
            contracts[name] = _legacy_raw_input_contract(variable)
        payload: Dict[str, Any] = {
            "schema_version": "easyicu.resolved_raw_input_contracts/1",
            "authority_scope": ("host_generated_sealed_research_context_constraints"),
            "scientific_ownership": (
                "Planner retains cohort, exposure, outcome, method, covariates, "
                "and estimand decisions"
            ),
            "contracts": contracts,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        payload["contracts_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    context = _revalidate_typed_research_context(context)
    cohort = context.materialized_inputs.cohort
    variables = {variable.name: variable for variable in context.variables}
    for name in raw_names:
        binding = cohort.column_bindings.get(name)
        variable = variables.get(name)
        if binding is None and (
            name != cohort.identity_column
            or name not in cohort.cohort_columns
            or variable is None
            or str(variable.role.value) != "id"
        ):
            raise ValueError(
                f"Planner-declared raw input {name!r} lacks a typed cohort binding"
            )
        if binding is None:
            # Identity is deliberately excluded from concept-column bindings:
            # it has row-identity authority, not clinical concept metadata.
            # Still expose its exact materialized-cohort contract so a Planner
            # may use it for row identity or deduplication without pretending
            # it is a physiological variable or inventing a plausibility range.
            fact = {
                "column": name,
                "dtype": variable.dtype,
                "physical_role": "identity",
                "representation_transform": "row_identity",
                "source_database_actual": cohort.source_database,
                "authority_kind": "materialized_cohort_identity",
                "row_identity_sha256": cohort.row_identity_sha256,
            }
            contracts[name] = fact
            continue
        fact = {
            **_column_prompt_fact(name, binding),
            "binding_sha256": binding.binding_sha256,
        }
        domain = (
            variable.observed_domain
            if variable is not None and isinstance(variable.observed_domain, Mapping)
            else None
        )
        observed_levels = _closed_observed_levels(domain)
        # Same order of authority as ``_apply_domain``: a declaration outranks
        # an observation, and outranks it here too -- this is the manifest the
        # sandbox actually executes against, so a guess winning on this path
        # would undo the fix on the other one.
        declared_levels, declared_basis = (
            _declared_domain(variable) if variable is not None else (None, None)
        )
        if declared_basis == "declared_concept_dictionary_levels" and not (
            _transform_preserves_concept_values(fact.get("representation_transform"))
        ):
            # A CONCEPT'S LEVELS DESCRIBE ITS VALUES, NOT A COUNT OF THEM.
            #
            # This fallback borrows the source concept's declared levels when
            # nothing else fixed a domain. For the concept's own column that is
            # right; for a companion derived from it, it publishes a contract
            # the column cannot satisfy. h1 (2026-08-03) died on exactly that:
            # ``mech_vent_n`` is ``physical_role=count`` /
            # ``window_nonnull_count`` / int64, holding 20-25 observations per
            # stay, and it was handed ``['invasive', 'noninvasive']`` --
            # 92,398 of 92,398 rows outside their own declared domain, and the
            # generated code raised, correctly, on what it had been told.
            #
            # The sibling ``mech_vent_measured`` escaped only by accident: its
            # metadata already declared ``[0, 1]``, so this fallback never ran
            # for it. The count had no metadata domain, so it took the
            # concept's.
            #
            # Only the ORDINAL branch is left alone: those levels come from the
            # host's own ICU rules, and a max over ordinal stages is still a
            # stage. The dictionary branch is the one that borrows a
            # categorical value domain, so it is the one gated.
            declared_levels, declared_basis = None, None
        if "allowed_values" not in fact and declared_levels:
            fact["allowed_values"] = list(declared_levels)
            fact["allowed_values_basis"] = declared_basis
            if observed_levels is not None:
                fact["observed_values"] = observed_levels
        elif "allowed_values" not in fact and observed_levels is not None:
            fact["allowed_values"] = observed_levels
            fact["allowed_values_basis"] = "sealed_research_context_observed_domain"
        if binding.analysis_plausibility_range is not None:
            fact["plausibility_policy"] = {
                "range_policy": "flag_only",
                "out_of_range_action": "retain_and_flag",
            }
        contracts[name] = fact
    payload: Dict[str, Any] = {
        "schema_version": "easyicu.resolved_raw_input_contracts/1",
        "authority_scope": (
            "host_verified_physical_representation_and_domain_constraints"
        ),
        "scientific_ownership": (
            "Planner retains cohort, exposure, outcome, method, covariates, "
            "and estimand decisions"
        ),
        "contracts": contracts,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["contracts_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def raw_contract_inputs_for_step(
    *,
    planner_declared_inputs: Sequence[str],
    primary_cohort_execution_receipt: Optional[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Add only host-resolved cohort predicates to exact Planner inputs.

    A predicate coordinate is whichever column the host's mask actually read.
    For a predicate the host narrowed to an event-time window that is two
    columns, not one, so the event-time column is authorized on the same
    footing as ``resolved_column``: the Coder is asked to reproduce that
    predicate's counts, and it cannot do so from a column it has no contract
    for. Rows without the field are unrefined and add nothing.
    """

    names = [str(value) for value in planner_declared_inputs]
    if primary_cohort_execution_receipt is None:
        return tuple(names)
    flow = primary_cohort_execution_receipt.get("ordered_predicate_flow")
    if not isinstance(flow, list):
        raise MaterializedMetadataError(
            "primary cohort execution receipt lacks ordered predicate flow"
        )
    for row in flow:
        if not isinstance(row, Mapping):
            raise MaterializedMetadataError(
                "primary cohort execution receipt contains an invalid predicate"
            )
        for field, reason in COHORT_RECEIPT_COLUMN_FIELDS:
            column = row.get(field)
            if column is None:
                continue
            if not (isinstance(column, str) and column.strip() and ":" not in column):
                raise MaterializedMetadataError(
                    f"primary cohort execution receipt has an invalid {reason}"
                )
            if column not in names:
                names.append(column)
    return tuple(names)


def resolved_raw_input_contracts_for_step(
    *,
    coder_base_context: ResearchContextAuthority,
    coder_context: ResearchContextAuthority,
    planner_declared_inputs: Sequence[str],
    primary_cohort_execution_receipt: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve exact step receipt columns without widening the Coder prompt."""

    contract_inputs = raw_contract_inputs_for_step(
        planner_declared_inputs=planner_declared_inputs,
        primary_cohort_execution_receipt=primary_cohort_execution_receipt,
    )
    authority_context = (
        coder_base_context
        if primary_cohort_execution_receipt is not None
        else coder_context
    )
    return resolved_raw_input_contracts(authority_context, contract_inputs)


def materialized_input_prompt_attachment(
    context: ResearchContextAuthority,
) -> str:
    """Render the canonical bounded projection as one host-owned attachment."""

    payload = materialized_input_prompt_projection(context)
    if payload is None:
        return ""
    rendered = _MATERIALIZED_PROMPT_HEADING + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(rendered.encode("utf-8")) > _MAX_MATERIALIZED_PROMPT_BYTES:
        raise ValueError("materialized input prompt attachment exceeds 4 KiB")
    return rendered


def materialized_research_inputs_from_authority(
    *,
    cohort: VerifiedMaterializedCohortAuthority,
    trajectory: Optional[VerifiedMaterializedTrajectoryAuthority] = None,
) -> MaterializedResearchInputs:
    """Project already-verified host authorities into compact context facts."""

    matching_bindings = tuple(
        item
        for item in cohort.sidecar.files
        if item.relative_path == cohort.authority.cohort_file
    )
    if len(matching_bindings) != 1:
        raise ValueError("cohort authority has no unique sidecar file binding")
    file_binding = matching_bindings[0]
    implementation = metadata_implementation_identity()

    cohort_context = MaterializedCohortContext(
        authority_ref=cohort.reference.to_dict(),
        cohort_file=cohort.authority.cohort_file,
        cohort_sha256=cohort.authority.cohort_sha256,
        cohort_size=cohort.authority.cohort_size,
        cohort_rows=cohort.authority.cohort_rows,
        cohort_columns=list(cohort.authority.cohort_columns),
        cohort_schema_sha256=cohort.authority.cohort_schema_sha256,
        identity_column=cohort.authority.identity_column,
        row_identity_sha256=cohort.authority.row_identity_sha256,
        column_metadata_ref=cohort.authority.column_metadata.to_dict(),
        file_metadata_payload_sha256=(cohort.authority.file_metadata_payload_sha256),
        source_export_authority_sha256=(
            cohort.authority.source_export_authority_sha256
        ),
        source_database=cohort.sidecar.source_database,
        time_coordinates=[
            coordinate.to_dict() for coordinate in file_binding.time_coordinates
        ],
        projection_scope="full",
        column_bindings={
            column: canonical_column_binding(column, binding)
            for column, binding in file_binding.columns.items()
        },
        column_binding_payload_sha256=binding_payload_sha256(file_binding.columns),
        producer_implementation_sha256=(
            cohort.authority.producer_implementation_sha256
        ),
        metadata_projection_sha256=implementation["metadata_projection_sha256"],
        metadata_sidecar_sha256=implementation["metadata_sidecar_sha256"],
        icu_rules_sha256=implementation["icu_rules_sha256"],
        metadata_implementation_bundle_sha256=(
            implementation["metadata_implementation_bundle_sha256"]
        ),
    )
    trajectory_context = None
    if trajectory is not None:
        authority = trajectory.authority
        trajectory_context = MaterializedTrajectoryContext(
            authority_ref=trajectory.reference.to_dict(),
            trajectory_file=authority.trajectory_file,
            trajectory_sha256=authority.trajectory_sha256,
            trajectory_size=authority.trajectory_size,
            trajectory_rows=authority.trajectory_rows,
            trajectory_columns=list(authority.trajectory_columns),
            trajectory_schema_sha256=authority.trajectory_schema_sha256,
            identity_column=authority.identity_column,
            time_column=authority.time_column,
            time_origin=authority.time_origin,
            time_unit=authority.time_unit,
            concept_column=authority.concept_column,
            numeric_value_column=authority.numeric_value_column,
            text_value_column=authority.text_value_column,
            ordered_row_key_sha256=authority.ordered_row_key_sha256,
            stay_identity_set_sha256=authority.stay_identity_set_sha256,
            trajectory_stays=authority.trajectory_stays,
            bound_universe_authority_ref=(authority.bound_universe_authority.to_dict()),
            bound_universe_row_identity_sha256=(
                authority.bound_universe_row_identity_sha256
            ),
            source_export_authority_sha256=(authority.source_export_authority_sha256),
            requested_concepts=list(authority.requested_concepts),
            materialized_concepts=list(authority.materialized_concepts),
            available_unobserved_concepts=list(authority.available_unobserved_concepts),
            unavailable_concepts=list(authority.unavailable_concepts),
            window=authority.window.to_dict() if authority.window else None,
            projection_scope="full",
            concept_bindings={
                item.concept_id: item.to_dict() for item in authority.concept_bindings
            },
            concept_analysis_plausibility_ranges={
                item.concept_id: effective_analysis_plausibility_range(item.binding)
                for item in authority.concept_bindings
            },
            concept_binding_payload_sha256=hashlib.sha256(
                json.dumps(
                    {
                        item.concept_id: item.to_dict()
                        for item in authority.concept_bindings
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            producer_implementation_sha256=(authority.producer_implementation_sha256),
        )
    return MaterializedResearchInputs(
        cohort=cohort_context,
        trajectory=trajectory_context,
    )


def project_research_context_variables(
    context: ResearchContextAuthority,
    selected_variables: List[ConceptDescriptor],
    *,
    additional_concept_ids: tuple[str, ...] = (),
    include_source_concept_siblings: bool = True,
) -> ResearchContextAuthority:
    """Atomically scope variables and their typed physical authority.

    Full authority references and artifact digests stay unchanged.  Only the
    prompt-facing binding projection is reduced, labelled ``scoped``, and
    independently digested.  This prevents a small variable list from being
    paired with unrelated full-cohort metadata.
    """

    if isinstance(context, ResearchContextV2):
        context = _revalidate_typed_research_context(context)
    full_by_name = {item.name: item for item in context.variables}
    names = [item.name for item in selected_variables]
    if len(names) != len(set(names)):
        raise ValueError("selected ResearchContext variables must be unique")
    if any(
        name not in full_by_name or full_by_name[name] != item
        for name, item in zip(names, selected_variables)
    ):
        raise ValueError("selected ResearchContext variable is not authoritative")
    if not isinstance(context, ResearchContextV2):
        return context.model_copy(update={"variables": list(selected_variables)})

    selected_names = {name.lower() for name in names}
    selected_source_concepts = {
        str(item.source_concept).strip().lower()
        for item in selected_variables
        if item.source_concept
    }
    cohort = context.materialized_inputs.cohort
    cohort_bindings = {
        column: binding
        for column, binding in cohort.column_bindings.items()
        if column.lower() in selected_names
        or (
            include_source_concept_siblings
            and str((binding.binding.get("metadata") or {}).get("source_concept"))
            .strip()
            .lower()
            in selected_source_concepts
        )
    }
    canonical_cohort_bindings = {
        column: ColumnMetadataBinding.from_dict(binding.binding)
        for column, binding in cohort_bindings.items()
    }
    scoped_cohort = MaterializedCohortContext.model_validate(
        {
            **cohort.model_dump(mode="python"),
            "projection_scope": "scoped",
            "column_bindings": cohort_bindings,
            "column_binding_payload_sha256": binding_payload_sha256(
                canonical_cohort_bindings
            ),
        }
    )

    trajectory = context.materialized_inputs.trajectory
    scoped_trajectory = None
    if trajectory is not None:
        explicitly_selected = {
            str(value).strip().lower()
            for value in additional_concept_ids
            if str(value).strip()
        }
        selected_concepts = (
            selected_names | selected_source_concepts | explicitly_selected
        )
        requested = [
            concept
            for concept in trajectory.requested_concepts
            if concept.lower() in selected_concepts
        ]
        requested_set = set(requested)
        materialized = [
            item for item in trajectory.materialized_concepts if item in requested_set
        ]
        unobserved = [
            item
            for item in trajectory.available_unobserved_concepts
            if item in requested_set
        ]
        unavailable = [
            item for item in trajectory.unavailable_concepts if item in requested_set
        ]
        trajectory_bindings = {
            concept: payload
            for concept, payload in trajectory.concept_bindings.items()
            if concept in requested_set
        }
        trajectory_ranges = {
            concept: payload
            for concept, payload in trajectory.concept_analysis_plausibility_ranges.items()
            if concept in requested_set
        }
        scoped_trajectory = MaterializedTrajectoryContext.model_validate(
            {
                **trajectory.model_dump(mode="python"),
                "projection_scope": "scoped",
                "requested_concepts": requested,
                "materialized_concepts": materialized,
                "available_unobserved_concepts": unobserved,
                "unavailable_concepts": unavailable,
                "concept_bindings": trajectory_bindings,
                "concept_analysis_plausibility_ranges": trajectory_ranges,
                "concept_binding_payload_sha256": hashlib.sha256(
                    json.dumps(
                        trajectory_bindings,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    scoped_inputs = MaterializedResearchInputs.model_validate(
        {
            "cohort": scoped_cohort,
            "trajectory": scoped_trajectory,
        }
    )
    return type(context).model_validate(
        {
            **context.model_dump(mode="python"),
            "variables": list(selected_variables),
            "materialized_inputs": scoped_inputs,
        }
    )


def migrate_research_context_v2(
    payload: Union[Mapping[str, Any], ResearchContextV2],
) -> ResearchContextV3:
    """Explicitly upgrade an archived V2 context to the current V3 contract.

    The upgrade is deterministic from the sealed column binding.  A missing
    window role is populated, while an explicit conflicting role is rejected
    rather than overwritten.  Parsing alone never mutates an archived V2
    document or changes its schema identity.
    """

    if isinstance(payload, ResearchContextV3):
        return ResearchContextV3.model_validate(payload.model_dump(mode="python"))
    context = (
        ResearchContextV2.model_validate(payload)
        if isinstance(payload, Mapping)
        else ResearchContextV2.model_validate(payload.model_dump(mode="python"))
    )
    migrated = context.model_dump(mode="python")
    cohort = context.materialized_inputs.cohort
    time_columns = {
        str(item.get("column") or "") for item in cohort.time_coordinates
    }
    excluded_columns = {cohort.identity_column, *time_columns}
    for variable in migrated["variables"]:
        column = str(variable.get("name") or "")
        if column in excluded_columns:
            continue
        binding = cohort.column_bindings[column]
        expected = descriptor_physical_updates(binding).get(
            "analysis_window_role"
        )
        if expected is None:
            continue
        actual = variable.get("analysis_window_role")
        if actual not in {None, expected}:
            raise ValueError(
                "archived V2 context descriptor physical field conflicts with "
                f"typed column binding: {column}.analysis_window_role"
            )
        variable["analysis_window_role"] = expected
    migrated["schema_version"] = RESEARCH_CONTEXT_V3_SCHEMA_VERSION
    return ResearchContextV3.model_validate(migrated)


def parse_research_context(payload: Mapping[str, Any]) -> ResearchContextAuthority:
    """Parse only an explicitly supported ResearchContext version."""

    if not isinstance(payload, Mapping):
        raise ValueError("research context payload must be an object")
    version = payload.get("schema_version")
    if version == "easyicu.research_context/1":
        return ResearchContext.model_validate(payload)
    if version == RESEARCH_CONTEXT_V2_SCHEMA_VERSION:
        return ResearchContextV2.model_validate(payload)
    if version == RESEARCH_CONTEXT_V3_SCHEMA_VERSION:
        return ResearchContextV3.model_validate(payload)
    raise ValueError(f"unsupported research context schema: {version!r}")


def _reject_duplicate_json_pairs(
    pairs: List[tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate research context JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite research context JSON constant: {value}")


def parse_research_context_json(
    raw: Union[str, bytes, bytearray],
) -> ResearchContextAuthority:
    """Decode JSON and dispatch to the strict versioned model."""

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("research context JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("research context JSON must contain an object")
    return parse_research_context(payload)


__all__ = [
    "CanonicalColumnBinding",
    "MaterializedCohortContext",
    "MaterializedResearchInputs",
    "MaterializedTrajectoryContext",
    "MATERIALIZED_INPUT_PROMPT_SCHEMA_VERSION",
    "RESEARCH_CONTEXT_V2_SCHEMA_VERSION",
    "RESEARCH_CONTEXT_V3_SCHEMA_VERSION",
    "ResearchContextAuthority",
    "ResearchContextV2",
    "ResearchContextV3",
    "binding_preserves_analysis_range",
    "canonical_column_binding",
    "descriptor_physical_updates",
    "effective_analysis_plausibility_range",
    "materialized_input_prompt_attachment",
    "materialized_input_prompt_projection",
    "materialized_research_inputs_from_authority",
    "migrate_research_context_v2",
    "parse_research_context",
    "parse_research_context_json",
    "project_research_context_variables",
    "raw_contract_inputs_for_step",
    "resolved_raw_input_contracts",
    "resolved_raw_input_contracts_for_step",
]
