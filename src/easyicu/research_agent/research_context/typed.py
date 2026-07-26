"""Strict typed-input extension for :class:`ResearchContext`.

Version 1 remains the literal archived contract in :mod:`schema`.  Version 2
adds only host-verified physical and lineage facts.  These facts never assign
the study cohort, exposure, outcome, method, or estimand; those remain Planner
and Coder decisions.
"""

from __future__ import annotations

import hashlib
import json
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

from ..intake.materialized_metadata import MaterializedCohortAuthorityRef
from ..intake.materialized_metadata import VerifiedMaterializedCohortAuthority
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    TrajectoryConceptBinding,
    VerifiedMaterializedTrajectoryAuthority,
)
from ..icu_rules import ICU_RULES
from .implementation_identity import metadata_implementation_identity
from ..schema import ConceptDescriptor, ResearchContext

RESEARCH_CONTEXT_V2_SCHEMA_VERSION = "easyicu.research_context/2"
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
        "unit": metadata.canonical_unit if value_like else None,
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
    if metadata.time_origin is not None and metadata.time_unit is not None:
        updates["temporal_resolution"] = (
            f"relative to {metadata.time_origin} in {metadata.time_unit}"
        )
    return updates


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
    """ResearchContext plus exact host-verified materialized-input facts."""

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
            for field_name, expected in descriptor_physical_updates(binding).items():
                if getattr(variable, field_name) != expected:
                    raise ValueError(
                        "context descriptor physical field does not match typed "
                        f"column binding: {column}.{field_name}"
                    )
        return self


ResearchContextAuthority = Union[ResearchContext, ResearchContextV2]


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
    # V2 is frozen at the model surface, while nested compatibility payloads
    # remain ordinary Python containers. Revalidate the canonical dump before
    # every authority-bearing prompt render so an in-memory nested mutation
    # cannot bypass binding digests or closure validators.
    context = ResearchContextV2.model_validate(context.model_dump(mode="python"))
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


def resolved_raw_input_contracts(
    context: ResearchContextAuthority,
    planner_declared_inputs: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Return exact executable metadata for Planner-declared raw columns.

    The outbound prompt projection is explanatory transport, while generated
    code executes against ``EASYICU_RESOLVED_INPUTS_JSON``.  Put the same
    host-verified physical/domain facts in that manifest so code never has to
    rediscover a range policy or closed domain by scanning the broader
    ResearchContext.  Typed ``kind:name`` products remain under the manifest's
    existing ``inputs`` authority and are intentionally excluded here.
    """

    if not isinstance(context, ResearchContextV2):
        return None
    context = ResearchContextV2.model_validate(context.model_dump(mode="python"))
    raw_names = [
        str(value).strip()
        for value in planner_declared_inputs
        if isinstance(value, str) and str(value).strip() and ":" not in str(value)
    ]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("Planner-declared raw inputs must be unique")
    cohort = context.materialized_inputs.cohort
    variables = {variable.name: variable for variable in context.variables}
    contracts: Dict[str, Any] = {}
    for name in raw_names:
        binding = cohort.column_bindings.get(name)
        if binding is None:
            raise ValueError(
                f"Planner-declared raw input {name!r} lacks a typed cohort binding"
            )
        fact = {
            **_column_prompt_fact(name, binding),
            "binding_sha256": binding.binding_sha256,
        }
        variable = variables.get(name)
        domain = (
            variable.observed_domain
            if variable is not None
            and isinstance(variable.observed_domain, Mapping)
            else None
        )
        observed_levels = domain.get("levels") if domain is not None else None
        n_unique = domain.get("n_unique") if domain is not None else None
        if (
            "allowed_values" not in fact
            and isinstance(observed_levels, list)
            and 1 <= len(observed_levels) <= 8
            and isinstance(n_unique, int)
            and not isinstance(n_unique, bool)
            and n_unique == len(observed_levels)
        ):
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
                    if value is not None
                    and isinstance(value, (bool, int, float, str))
                ]
            except (TypeError, ValueError):
                encoded_levels = []
            if (
                len(encoded_levels) == len(observed_levels)
                and len(set(encoded_levels)) == len(encoded_levels)
            ):
                fact["allowed_values"] = list(observed_levels)
                fact["allowed_values_basis"] = (
                    "sealed_research_context_observed_domain"
                )
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
        context = ResearchContextV2.model_validate(context.model_dump(mode="python"))
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
    return ResearchContextV2.model_validate(
        {
            **context.model_dump(mode="python"),
            "variables": list(selected_variables),
            "materialized_inputs": scoped_inputs,
        }
    )


def parse_research_context(payload: Mapping[str, Any]) -> ResearchContextAuthority:
    """Parse only an explicitly supported ResearchContext version."""

    if not isinstance(payload, Mapping):
        raise ValueError("research context payload must be an object")
    version = payload.get("schema_version")
    if version == "easyicu.research_context/1":
        return ResearchContext.model_validate(payload)
    if version == RESEARCH_CONTEXT_V2_SCHEMA_VERSION:
        return ResearchContextV2.model_validate(payload)
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
    "ResearchContextAuthority",
    "ResearchContextV2",
    "binding_preserves_analysis_range",
    "canonical_column_binding",
    "descriptor_physical_updates",
    "effective_analysis_plausibility_range",
    "materialized_input_prompt_attachment",
    "materialized_input_prompt_projection",
    "materialized_research_inputs_from_authority",
    "parse_research_context",
    "parse_research_context_json",
    "project_research_context_variables",
    "resolved_raw_input_contracts",
]
