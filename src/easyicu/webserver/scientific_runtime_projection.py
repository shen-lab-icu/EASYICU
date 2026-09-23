"""Compile fully typed Web study decisions into immutable runtime authority.

Owner
-----
This module owns the Web-to-Research-Agent projection for closed landmark
associations, including categorical exposures and their prespecified model
grids, and routes the time-varying, RMST and landmark-survival designs to
their sibling adapters. StudyContext owns the user's scientific choices;
current-case runtime authority owns deterministic execution. The public contract joins those two
typed boundaries only after every required coordinate is explicit. For a
categorical landmark study whose adjustment roster is ``planner_selectable``,
the projection seals the executable column domain and the runtime owner seals
the Planner-selected roster from the reviewed plan at bind time.

Allowed dependencies are the dependency-neutral sensitivity contract, the
current-case authority builder, a parquet *schema* reader, and canonical
hashing.  It never reads patient rows, selects a variable, or infers a landmark
from prose.  Failures use ``WebScientificRuntimeProjectionError`` so the Web
runner can attribute the blocker to this owner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.concept.selection_policy import (
    OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.icu_rules import VariableKind, classify_variable
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)


class WebScientificRuntimeProjectionError(ValueError):
    """A user-reviewed design could not be bound to its deterministic owner."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)


@dataclass(frozen=True)
class WebScientificRuntimeProjection:
    authority: dict[str, Any]
    projection_sha256: str
    analysis_only_execution: bool = False


#: The concept owner declares which KDIGO stage bindings coalesce an
#: unobserved component to zero; this owner decides what that means for an
#: exposure.  Keeping one list would have let the two drift, and the refusal
#: was already a no-op once the renal contract renamed its columns.
_COLLAPSING_KDIGO_STAGE_BINDINGS = frozenset(
    OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS
)
#: A materialized exposure carries the cohort aggregation in its name; the
#: binding underneath is what this owner judges.
_WINDOW_AGGREGATION_SUFFIXES = ("_max", "_min", "_mean", "_first", "_last")


def _kdigo_binding_stem(name: str) -> str:
    for suffix in _WINDOW_AGGREGATION_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def kdigo_observability_authority_missing(*names: str | None) -> bool:
    """Whether a KDIGO binding can collapse missing evidence into stage 0.

    The public-reference KDIGO stage follows its upstream semantics: a component
    with no usable evidence contributes zero, so ``stage 0`` means "no positive
    evidence was found".  Using it as a 0/1/2/3 exposure puts never-assessed
    patients in the reference group, which this owner refuses.  The judgement
    is about that semantics, not about one release's column names: the retired
    phenotype, the reference port and the source-native profiles are all
    refused, in concept form and in materialized ``<binding>_<aggregation>``
    form.

    Only an exposure that is observability-preserving *by construction* is
    authorized -- a host-materialized strict stage, whose unknown rows stay
    unknown.  Another column existing in the same artifact never authorizes
    this one: receipts make the strict exposure derivable, they do not make a
    collapsed exposure safe.  Use
    :func:`kdigo_strict_derivation_available` to tell a caller which of the two
    situations they are in.
    """

    normalized = {
        str(name or "").strip().lower()
        for name in names
        if str(name or "").strip()
    }
    if any(
        "aki_stage_strict" in name or "kdigo_strict" in name
        for name in normalized
    ):
        return False
    return any(
        _kdigo_binding_stem(name) in _COLLAPSING_KDIGO_STAGE_BINDINGS
        for name in normalized
    )


def required_host_derivations(*names: str | None) -> tuple[str, ...]:
    """Declared cross-concept derivations the named plan variables require.

    A host derivation is never inferred from the data: a column whose value
    depends on several concepts at once is produced only when the reviewed
    design names it.  This resolves those names to the declared derivation ids
    that acquisition must run, and returns nothing for an ordinary concept.
    """

    from easyicu.research_agent.contracts.host_derivations import (
        host_derivation_producing,
    )

    resolved: list[str] = []
    for name in names:
        declared = host_derivation_producing(str(name or "").strip())
        if declared is None:
            continue
        derivation_id = declared[0].derivation_id
        if derivation_id not in resolved:
            resolved.append(derivation_id)
    return tuple(resolved)


def host_derived_design_sources(column: str | None) -> tuple[str, ...] | None:
    """Source concepts of a declared host derivation a plan may name, if any.

    A derived design variable (the strict KDIGO stage) has no single exported
    source; it is available exactly when every concept its declaration reads
    is.  Receipts a derivation publishes for audit are not design variables,
    so they resolve like any other column instead.
    """

    from easyicu.research_agent.contracts.host_derivations import (
        host_derivation_producing,
    )

    declared = host_derivation_producing(str(column or "").strip())
    if declared is None or not declared[1].selectable:
        return None
    return declared[0].source_concepts


def kdigo_strict_derivation_available(columns: Sequence[str]) -> bool:
    """Whether a set of columns carries the receipts a strict KDIGO stage needs.

    This is diagnostic, not authorization: it answers whether the remedy is
    "materialize the strict exposure from receipts this export already has" or
    "re-extract the renal module so it carries those receipts at all".
    """

    from easyicu.scores.aki_strict import strict_kdigo_receipt_columns

    return strict_kdigo_receipt_columns(tuple(columns)) is not None


def export_kdigo_strict_derivation_available(source_path: object) -> bool:
    """Whether the *export* behind a study can support a strict KDIGO stage.

    The receipts must be read from the export, not from the materialized wide
    cohort: the cohort summarizer reduces a categorical evidence status to a
    numeric presence indicator, which can no longer distinguish an observed
    negative from an absent measurement.  An unreadable export answers ``False``
    so the caller keeps its fail-closed default.
    """

    from easyicu.research_agent.intake.export_package import (
        ExportPackageError,
        is_export_package,
        open_export_package,
    )

    root = Path(str(source_path or "")).expanduser()
    try:
        if not root.is_dir() or not is_export_package(root):
            return False
        with open_export_package(root) as package:
            columns = {
                column
                for physical_file in package.files
                for column in physical_file.columns
            }
    except (ExportPackageError, OSError, ValueError):
        return False
    return kdigo_strict_derivation_available(tuple(columns))


def compile_web_scientific_runtime_projection(**coordinates: Any) -> WebScientificRuntimeProjection | None:
    """Route only explicit typed specifications to their execution owner."""
    from .time_varying_runtime_projection import compile_time_varying_runtime_projection

    projection = compile_time_varying_runtime_projection(**coordinates)
    if projection is not None:
        return projection
    from .rmst_runtime_projection import compile_rmst_runtime_projection

    projection = compile_rmst_runtime_projection(**coordinates)
    if projection is not None:
        return projection
    from .landmark_survival_runtime_projection import (
        compile_landmark_survival_runtime_projection,
    )

    # A declared survival family with a landmark binds the sealed survival
    # suite; the association adapters below never re-model a survival design.
    projection = compile_landmark_survival_runtime_projection(**coordinates)
    if projection is not None:
        return projection
    landmark_coordinates = dict(coordinates)
    landmark_coordinates.pop("literature_citation_keys", None)
    landmark_coordinates.pop("direct_comparator_literature_keys", None)
    landmark = one_sensitivity_spec(
        landmark_coordinates["sensitivity_specs"], strategy="landmark"
    )
    if landmark is None:
        return None
    exposure_kind, _levels = primary_exposure_kind(
        universe_path=landmark_coordinates["universe_path"],
        primary_exposure=landmark_coordinates.get("primary_exposure"),
        primary_exposure_source=landmark_coordinates.get("primary_exposure_source"),
    )
    if exposure_kind in {
        VariableKind.ORDINAL,
        VariableKind.CATEGORICAL,
        VariableKind.BINARY,
    }:
        exposure_names = {
            landmark_coordinates.get("primary_exposure"),
            landmark_coordinates.get("primary_exposure_source"),
        }
        exposure_splines = [
            spec for spec in landmark_coordinates["sensitivity_specs"]
            if spec.strategy == "restricted_cubic_spline"
            and bool(set(spec.execution_variables) & exposure_names)
        ]
        if exposure_splines:
            raise WebScientificRuntimeProjectionError(
                "web_landmark_exposure_model_incompatible",
                "A categorical or ordinal landmark exposure cannot use the continuous spline runtime.",
                details={
                    "exposure_kind": exposure_kind.value,
                    "spec_ids": [spec.spec_id for spec in exposure_splines],
                },
            )
        return compile_landmark_categorical_runtime_projection(
            **landmark_coordinates
        )
    return compile_landmark_spline_runtime_projection(**landmark_coordinates)


def signed_projection(
    authority: Mapping[str, Any], *, scientific_configuration_sha256: str
) -> WebScientificRuntimeProjection:
    """Join one compiled authority to the study configuration it came from."""

    projection_body = {
        "schema_version": "easyicu.web_scientific_runtime_projection/1",
        "study_scientific_configuration_sha256": scientific_configuration_sha256,
        "deterministic_execution_contract": dict(authority),
    }
    return WebScientificRuntimeProjection(
        authority=dict(authority),
        projection_sha256=hashlib.sha256(_canonical_bytes(projection_body)).hexdigest(),
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def one_sensitivity_spec(
    specs: Sequence[PrespecifiedSensitivitySpec], *, strategy: str
) -> PrespecifiedSensitivitySpec | None:
    matching = [spec for spec in specs if spec.strategy == strategy]
    if not matching:
        return None
    if len(matching) != 1:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_projection_ambiguous",
            f"The Web study declares more than one {strategy} sensitivity.",
            details={"strategy": strategy, "spec_ids": [spec.spec_id for spec in matching]},
        )
    return matching[0]


def primary_exposure_kind(
    *,
    universe_path: Path,
    primary_exposure: str | None,
    primary_exposure_source: str | None,
) -> tuple[VariableKind, tuple[str, ...]]:
    """Classify a physical exposure without reading patient rows."""

    dtype = ""
    if primary_exposure:
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(universe_path)
            if primary_exposure in schema.names:
                dtype = str(schema.field(primary_exposure).type)
        except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
            raise WebScientificRuntimeProjectionError(
                "web_scientific_runtime_schema_unavailable",
                "The materialized universe schema could not be read for exposure routing.",
                details={"artifact": Path(universe_path).name, "reason": str(exc)[:500]},
            ) from exc
    hint = classify_variable(
        str(primary_exposure_source or primary_exposure or ""),
        dtype,
    )
    return hint.kind, tuple(str(value) for value in (hint.ordinal_levels or ()))


def categorical_adjustments(
    universe_path: Path, *, covariates: Sequence[str]
) -> tuple[str, ...]:
    """Classify only physically typed string/dictionary/boolean columns."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.read_schema(universe_path)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    categorical: list[str] = []
    missing = [column for column in covariates if column not in schema.names]
    if missing:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_columns_missing",
            "The exact adjustment set is absent from the materialized universe.",
            details={"missing_columns": missing},
        )
    for column in covariates:
        column_type = schema.field(column).type
        if (
            pa.types.is_string(column_type)
            or pa.types.is_large_string(column_type)
            or pa.types.is_dictionary(column_type)
            or pa.types.is_boolean(column_type)
        ):
            categorical.append(column)
            continue
        if not (
            pa.types.is_integer(column_type)
            or pa.types.is_floating(column_type)
            or pa.types.is_decimal(column_type)
        ):
            raise WebScientificRuntimeProjectionError(
                "web_scientific_runtime_covariate_encoding_unsupported",
                "An exact adjustment column has no deterministic supported encoding.",
                details={"column": column, "parquet_type": str(column_type)},
            )
    return tuple(categorical)


def _plan_bound_adjustment_domain(
    universe_path: Path, *, design_columns: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Seal every physically executable adjustment column and its encoding.

    A Planner-selectable roster is chosen during planning, so the projection
    cannot name it. It can still close the domain from the materialized
    universe *schema*: every column with a deterministic supported encoding
    (numeric → continuous; string/dictionary/boolean → categorical), minus the
    signed design coordinates. Columns with any other physical type are simply
    not admissible; the runtime owner rejects a roster outside this domain.
    Scientific admissibility (pre-time-zero availability) stays with the
    planning authority, which is stricter than this physical domain.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.read_schema(universe_path)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    excluded = set(design_columns)
    admissible: list[str] = []
    categorical: list[str] = []
    for field in schema:
        if field.name in excluded:
            continue
        column_type = field.type
        if (
            pa.types.is_string(column_type)
            or pa.types.is_large_string(column_type)
            or pa.types.is_dictionary(column_type)
            or pa.types.is_boolean(column_type)
        ):
            admissible.append(field.name)
            categorical.append(field.name)
        elif (
            pa.types.is_integer(column_type)
            or pa.types.is_floating(column_type)
            or pa.types.is_decimal(column_type)
        ):
            admissible.append(field.name)
    return tuple(admissible), tuple(categorical)


def operational_covariates(
    universe_path: Path,
    *,
    declared_covariates: Sequence[str],
    operationalizations: Mapping[str, str],
) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(universe_path).names)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    resolved = tuple(
        str(operationalizations.get(name) or name) for name in declared_covariates
    )
    missing = [
        declared
        for declared, operational in zip(declared_covariates, resolved)
        if operational not in schema_names
    ]
    if missing:
        raise WebScientificRuntimeProjectionError(
            "web_covariate_operationalization_required",
            "An exact covariate requires a user-reviewed materialized-column binding.",
            details={
                "missing_operationalizations": missing,
                "field": "covariate_operationalizations",
            },
        )
    if len(resolved) != len(set(resolved)):
        raise WebScientificRuntimeProjectionError(
            "web_covariate_operationalization_ambiguous",
            "Two exact covariates resolve to the same materialized column.",
            details={"resolved_covariates": list(resolved)},
        )
    return resolved


def _categorical_model_grid(
    *,
    sensitivity_specs: Sequence[PrespecifiedSensitivitySpec],
    primary_exposure: str,
    exposure_kind: VariableKind,
    exposure_levels: tuple[str, ...],
    universe_path: Path,
    covariates: tuple[str, ...],
    categorical_covariates: tuple[str, ...],
    declared_covariates: Sequence[str],
    scientific_configuration_sha256: str,
    plan_bound_roster: bool = False,
) -> dict[str, Any] | None:
    """Bind only typed alternate exposures and covariate forms to one grid."""

    specs = [
        spec for spec in sensitivity_specs
        if spec.strategy in {"alternate_exposure", "restricted_cubic_spline"}
    ]
    if not specs:
        return None
    if plan_bound_roster:
        # A covariate functional-form variant names an exact covariate. With a
        # Planner-selected roster the family template derives its
        # functional-form checks from the selected roster instead, so the
        # StudyContext spec has no executable owner here.
        unbound_forms = [
            spec.spec_id for spec in specs if spec.strategy == "restricted_cubic_spline"
        ]
        if unbound_forms:
            raise WebScientificRuntimeProjectionError(
                "web_model_grid_functional_form_requires_exact_roster",
                "A covariate functional-form sensitivity names an exact covariate; "
                "with a Planner-selected roster, remove the spec or set "
                "covariate_selection=exact.",
                details={"spec_ids": unbound_forms, "field": "sensitivity_specs"},
            )
    variants: list[dict[str, Any]] = [
        {"analysis_id": "reference", "metadata": {"axis": "primary", "source_spec_id": "primary"}}
    ]
    used_ids = {"reference"}
    for spec in specs:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", spec.spec_id) or spec.spec_id in used_ids:
            raise WebScientificRuntimeProjectionError(
                "web_model_grid_spec_id_invalid",
                "A model-grid sensitivity needs a unique stable analysis id.",
                details={"spec_id": spec.spec_id},
            )
        used_ids.add(spec.spec_id)
        if len(spec.execution_variables) != 1:
            raise WebScientificRuntimeProjectionError(
                "web_model_grid_source_ambiguous",
                "A model-grid variant requires one exact materialized source column.",
                details={"spec_id": spec.spec_id},
            )
        source = spec.execution_variables[0]
        variant: dict[str, Any] = {
            "analysis_id": spec.spec_id,
            "metadata": {"axis": spec.axis, "source_spec_id": spec.spec_id},
        }
        if spec.strategy == "alternate_exposure":
            kind, levels = primary_exposure_kind(
                universe_path=universe_path,
                primary_exposure=source,
                primary_exposure_source=source,
            )
            if source == primary_exposure or kind != exposure_kind or levels != exposure_levels:
                raise WebScientificRuntimeProjectionError(
                    "web_model_grid_exposure_definition_incompatible",
                    "The alternate exposure must have the primary model's closed level set.",
                    details={"spec_id": spec.spec_id, "source": source},
                )
            variant["exposure_column"] = source
        else:
            if source not in covariates and source in declared_covariates:
                source = covariates[list(declared_covariates).index(source)]
            if source not in covariates or source in categorical_covariates:
                raise WebScientificRuntimeProjectionError(
                    "web_model_grid_functional_form_source_invalid",
                    "The nonlinear source must be a primary continuous covariate.",
                    details={"spec_id": spec.spec_id, "source": source},
                )
            variant["nonlinear_terms"] = [{
                "source_column": source,
                "basis": "natural_cubic_spline",
                "degrees_of_freedom": 3,
                "center_before_basis": True,
            }]
        variants.append(variant)
    authority = build_current_case_scientific_runtime_authority({
        "schema_version": "easyicu.association_model_grid_runtime_authority/1",
        "authority_kind": "association_model_grid",
        "protocol_content_sha256": scientific_configuration_sha256,
        "plan_method": "verified_association_model_grid",
        "plan_intent": "Compare prespecified definitions and covariate forms on the signed landmark cohort.",
        "cohort_product": "artifact:analysis_cohort",
        "parent_product": "table:adjusted_association_estimates",
        "output_product": "table:association_sensitivity_grid",
        "reference_variant_id": "reference",
        "metadata_columns": ["axis", "source_spec_id"],
        "output_aliases": {},
        "variants": variants,
    })
    return authority.model_dump(mode="json")


def compile_landmark_categorical_runtime_projection(
    *,
    study: Mapping[str, Any],
    sensitivity_specs: Sequence[PrespecifiedSensitivitySpec],
    primary_exposure: str | None,
    primary_exposure_source: str | None,
    target_outcome: str | None,
    declared_covariates: Sequence[str],
    covariate_operationalizations: Mapping[str, str],
    target_is_event_status: bool,
    universe_path: Path,
    scientific_configuration_sha256: str,
    dependence: PlannedDependenceRequirement | None = None,
) -> WebScientificRuntimeProjection | None:
    """Compile an ordered/categorical exposure through the verified logit adapter."""

    landmark = one_sensitivity_spec(sensitivity_specs, strategy="landmark")
    if landmark is None:
        return None
    # ``exact`` seals the user-reviewed roster now; ``planner_selectable``
    # defers the roster to the reviewed plan and seals the executable domain
    # instead, so an agent-planned adjustment set still reaches the signed
    # deterministic executor without a second run.
    covariate_selection = str(study.get("covariate_selection") or "planner_selectable")
    if covariate_selection not in {"exact", "planner_selectable"}:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_categorical_authority_incomplete",
            "The landmark categorical design lacks executable typed coordinates.",
            details={"missing_fields": ["covariate_selection"]},
        )
    plan_bound_roster = covariate_selection != "exact"
    missing_fields: list[str] = []
    if not primary_exposure:
        missing_fields.append("primary_exposure")
    if not primary_exposure_source:
        missing_fields.append("primary_exposure_source")
    if not target_outcome:
        missing_fields.append("target_outcome")
    if not target_is_event_status:
        missing_fields.append("binary_event_status_outcome")
    if not plan_bound_roster and not declared_covariates:
        missing_fields.append("exact_covariates")
    if landmark.event_time_variable is None:
        missing_fields.append("landmark.event_time_variable")
    if landmark.observation_duration_variable is None:
        missing_fields.append("landmark.observation_duration_variable")
    if landmark.observation_duration_unit is None:
        missing_fields.append("landmark.observation_duration_unit")
    if not landmark.require_alive_at_landmark:
        missing_fields.append("landmark.require_alive_at_landmark")
    if not landmark.exclude_negative_event_times:
        missing_fields.append("landmark.exclude_negative_event_times")
    if missing_fields:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_categorical_authority_incomplete",
            "The landmark categorical design lacks executable typed coordinates.",
            details={"missing_fields": missing_fields},
        )
    if float(landmark.landmark_hours or 0.0) != 24.0:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_categorical_landmark_unsupported",
            "The verified categorical association adapter currently supports a 24-hour landmark.",
            details={"landmark_hours": landmark.landmark_hours},
        )
    if kdigo_observability_authority_missing(
        primary_exposure_source, primary_exposure
    ):
        source = study.get("data_source") if isinstance(study, Mapping) else None
        derivable = export_kdigo_strict_derivation_available(
            (source or {}).get("path") if isinstance(source, Mapping) else None
        )
        raise WebScientificRuntimeProjectionError(
            "web_kdigo_observability_authority_missing",
            "KDIGO execution requires an exposure that keeps incomplete observation evidence unknown.",
            details={
                "primary_exposure": primary_exposure,
                "primary_exposure_source": primary_exposure_source,
                "required_binding": "aki_stage_strict",
                "strict_derivation_available": derivable,
                "remedy": (
                    "select aki_stage_strict as the exposure: this source "
                    "carries the KDIGO evidence receipts, so the host "
                    "materializes the observability-preserving stage"
                    if derivable
                    else "re-extract the renal module so it carries the "
                    "per-component KDIGO evidence receipts"
                ),
            },
        )

    exposure_kind, levels = primary_exposure_kind(
        universe_path=universe_path,
        primary_exposure=primary_exposure,
        primary_exposure_source=primary_exposure_source,
    )
    if exposure_kind not in {
        VariableKind.ORDINAL,
        VariableKind.CATEGORICAL,
        VariableKind.BINARY,
    }:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_categorical_exposure_incompatible",
            "The selected exposure is not typed as categorical, ordinal, or binary.",
            details={"exposure_kind": exposure_kind.value},
        )
    if not levels:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_categorical_levels_unavailable",
            "The categorical landmark runtime requires a source-owned closed level set.",
            details={
                "primary_exposure": primary_exposure,
                "primary_exposure_source": primary_exposure_source,
            },
        )

    if plan_bound_roster:
        covariates: tuple[str, ...] = ()
        categorical: tuple[str, ...] = ()
    else:
        covariates = operational_covariates(
            universe_path,
            declared_covariates=tuple(declared_covariates),
            operationalizations=covariate_operationalizations,
        )
        categorical = categorical_adjustments(universe_path, covariates=covariates)
    grid = _categorical_model_grid(
        sensitivity_specs=sensitivity_specs,
        primary_exposure=str(primary_exposure),
        exposure_kind=exposure_kind,
        exposure_levels=levels,
        universe_path=universe_path,
        covariates=covariates,
        categorical_covariates=categorical,
        declared_covariates=() if plan_bound_roster else declared_covariates,
        scientific_configuration_sha256=scientific_configuration_sha256,
        plan_bound_roster=plan_bound_roster,
    )
    required_columns = {
        str(primary_exposure),
        str(target_outcome),
        str(landmark.event_time_variable),
        str(landmark.observation_duration_variable),
        *map(str, covariates),
        *((dependence.group_source,) if dependence is not None else ()),
    }
    if grid is not None:
        required_columns.update(
            variant["exposure_column"]
            for variant in grid["variants"]
            if variant.get("exposure_column")
        )
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(universe_path).names)
    except Exception as exc:  # pragma: no cover - classified above
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    absent = sorted(required_columns - schema_names)
    if absent:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_columns_missing",
            "The landmark categorical runtime inputs are absent from the materialized universe.",
            details={"missing_columns": absent},
        )

    if plan_bound_roster:
        schema_version = "easyicu.landmark_categorical_association_runtime_authority/3"
    elif grid is not None:
        schema_version = "easyicu.landmark_categorical_association_runtime_authority/2"
    else:
        schema_version = "easyicu.landmark_categorical_association_runtime_authority/1"
    authority_body = {
        "schema_version": schema_version,
        "authority_kind": "landmark_categorical_association",
        "protocol_content_sha256": scientific_configuration_sha256,
        "cohort_method": "signed_landmark_analysis_cohort",
        "primary_method": "signed_landmark_categorical_association",
        "plan_intent": (
            "Estimate the adjusted categorical association among patients alive "
            "and observed at the prespecified 24-hour landmark."
        ),
        "landmark_spec_id": landmark.spec_id,
        "cohort_product": "artifact:analysis_cohort",
        "cohort_flow_product": "table:cohort_flow",
        "primary_product": "table:adjusted_association_estimates",
        "exposure_column": primary_exposure,
        "exposure_kind": exposure_kind.value,
        "exposure_levels": list(levels),
        "exposure_reference_level": levels[0],
        "primary_contrast_level": levels[-1],
        "outcome_column": target_outcome,
        "event_time_column": landmark.event_time_variable,
        "observation_duration_column": landmark.observation_duration_variable,
        "observation_duration_unit": landmark.observation_duration_unit,
        "landmark_hours": 24,
        "exclude_negative_event_times": True,
        "require_alive_at_landmark": True,
        "required_adjustment_columns": list(covariates),
        "categorical_adjustment_columns": list(categorical),
        "dependence": (
            dependence.model_dump(mode="json")
            if dependence is not None
            else None
        ),
        "interpretation": "descriptive_prognostic_association_not_causal",
    }
    if grid is not None:
        authority_body["association_model_grid"] = grid
    if plan_bound_roster:
        # v3 bodies sign both optional coordinates explicitly (``None`` when
        # absent) so the digest matches the model's canonical dump without a
        # legacy field-drop rule.
        authority_body["association_model_grid"] = grid
        design_columns = [
            str(primary_exposure),
            str(target_outcome),
            str(landmark.event_time_variable),
            str(landmark.observation_duration_variable),
            *((dependence.group_source,) if dependence is not None else ()),
            *(
                variant["exposure_column"]
                for variant in (grid["variants"] if grid is not None else ())
                if variant.get("exposure_column")
            ),
        ]
        admissible, admissible_categorical = _plan_bound_adjustment_domain(
            universe_path, design_columns=design_columns
        )
        authority_body["plan_bound_adjustment_roster"] = {
            "authority": "plan_primary_model",
            "admissible_columns": list(admissible),
            "admissible_categorical_columns": list(admissible_categorical),
            "sealed": False,
        }
    authority = build_current_case_scientific_runtime_authority(
        authority_body
    ).model_dump(mode="json")
    return signed_projection(
        authority, scientific_configuration_sha256=scientific_configuration_sha256
    )


def compile_landmark_spline_runtime_projection(
    *,
    study: Mapping[str, Any],
    sensitivity_specs: Sequence[PrespecifiedSensitivitySpec],
    primary_exposure: str | None,
    primary_exposure_source: str | None,
    target_outcome: str | None,
    declared_covariates: Sequence[str],
    covariate_operationalizations: Mapping[str, str],
    target_is_event_status: bool,
    universe_path: Path,
    scientific_configuration_sha256: str,
    dependence: PlannedDependenceRequirement | None = None,
) -> WebScientificRuntimeProjection | None:
    """Return a signed runtime projection only for one fully closed design.

    Absence of either a landmark or RCS request means this owner is not
    applicable.  Once both are present, every remaining field is required and
    failures are explicit rather than silently falling back to agent-coded
    analysis.
    """

    landmark = one_sensitivity_spec(sensitivity_specs, strategy="landmark")
    spline = one_sensitivity_spec(sensitivity_specs, strategy="restricted_cubic_spline")
    if landmark is None or spline is None:
        return None

    covariate_selection = str(study.get("covariate_selection") or "planner_selectable")
    if covariate_selection not in {"exact", "planner_selectable"}:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_authority_incomplete",
            "The landmark spline design is user-selected but lacks executable coordinates.",
            details={"missing_fields": ["covariate_selection"]},
        )
    plan_bound_roster = covariate_selection != "exact"
    missing_fields: list[str] = []
    if not primary_exposure:
        missing_fields.append("primary_exposure")
    if not primary_exposure_source:
        missing_fields.append("primary_exposure_source")
    if not target_outcome:
        missing_fields.append("target_outcome")
    if not target_is_event_status:
        missing_fields.append("binary_event_status_outcome")
    if not plan_bound_roster and not declared_covariates:
        missing_fields.append("exact_covariates")
    if landmark.event_time_variable is None:
        missing_fields.append("landmark.event_time_variable")
    if landmark.observation_duration_variable is None:
        missing_fields.append("landmark.observation_duration_variable")
    if landmark.observation_duration_unit is None:
        missing_fields.append("landmark.observation_duration_unit")
    if not landmark.require_alive_at_landmark:
        missing_fields.append("landmark.require_alive_at_landmark")
    if not landmark.exclude_negative_event_times:
        missing_fields.append("landmark.exclude_negative_event_times")
    if missing_fields:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_authority_incomplete",
            "The landmark spline design is user-selected but lacks executable coordinates.",
            details={"missing_fields": missing_fields},
        )

    if float(landmark.landmark_hours or 0.0) != 24.0:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_landmark_unsupported",
            "The verified Web landmark spline adapter currently supports a 24-hour landmark.",
            details={"landmark_hours": landmark.landmark_hours},
        )
    spline_sources = set(spline.execution_variables)
    if primary_exposure_source not in spline_sources and primary_exposure not in spline_sources:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_exposure_binding_mismatch",
            "The RCS sensitivity is not bound to the configured primary exposure.",
            details={
                "primary_exposure": primary_exposure,
                "primary_exposure_source": primary_exposure_source,
                "spline_execution_variables": sorted(spline_sources),
            },
        )

    if plan_bound_roster:
        covariates: tuple[str, ...] = ()
        categorical: tuple[str, ...] = ()
    else:
        covariates = operational_covariates(
            universe_path,
            declared_covariates=tuple(declared_covariates),
            operationalizations=covariate_operationalizations,
        )
        categorical = categorical_adjustments(
            universe_path, covariates=covariates
        )
    required_columns = {
        str(primary_exposure),
        str(target_outcome),
        str(landmark.event_time_variable),
        str(landmark.observation_duration_variable),
        *map(str, covariates),
        *((dependence.group_source,) if dependence is not None else ()),
    }
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(universe_path).names)
    except Exception as exc:  # pragma: no cover - classified by helper above
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    absent = sorted(required_columns - schema_names)
    if absent:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_columns_missing",
            "The landmark spline runtime inputs are absent from the materialized universe.",
            details={"missing_columns": absent},
        )

    if plan_bound_roster:
        spline_schema_version = "easyicu.landmark_spline_runtime_authority/5"
    elif dependence is not None:
        spline_schema_version = "easyicu.landmark_spline_runtime_authority/4"
    else:
        spline_schema_version = "easyicu.landmark_spline_runtime_authority/2"
    plan_bound_body: dict[str, Any] = {}
    if plan_bound_roster:
        admissible, admissible_categorical = _plan_bound_adjustment_domain(
            universe_path,
            design_columns=[
                str(primary_exposure),
                str(target_outcome),
                str(landmark.event_time_variable),
                str(landmark.observation_duration_variable),
                *((dependence.group_source,) if dependence is not None else ()),
            ],
        )
        plan_bound_body["plan_bound_adjustment_roster"] = {
            "authority": "plan_primary_model",
            "admissible_columns": list(admissible),
            "admissible_categorical_columns": list(admissible_categorical),
            "sealed": False,
        }
    authority = build_current_case_scientific_runtime_authority(
        {
            "schema_version": spline_schema_version,
            "authority_kind": "landmark_spline_association",
            "protocol_content_sha256": scientific_configuration_sha256,
            "plan_method": "signed_landmark_restricted_cubic_spline",
            "plan_intent": (
                "Execute the user-reviewed 24-hour landmark restricted-cubic-"
                "spline association and its prespecified linear sensitivity."
            ),
            "plan_outputs": [
                "table:landmark_rcs_curve",
                "table:landmark_rcs_contrasts",
                "table:landmark_linear_sensitivity",
                "table:landmark_adjusted_absolute_risk",
                "table:landmark_population_flow",
                "table:landmark_variable_opportunity_sensitivity",
                "log:landmark_scientific_runtime_receipt",
            ],
            "exposure_column": primary_exposure,
            "outcome_column": target_outcome,
            "outcome_time_column": landmark.event_time_variable,
            "observation_duration_column": landmark.observation_duration_variable,
            "observation_duration_unit": landmark.observation_duration_unit,
            "landmark_hours": 24,
            "required_adjustment_columns": list(covariates),
            "categorical_adjustment_columns": list(categorical),
            "alternative_exposure_columns": [],
            "dependence": (
                dependence.model_dump(mode="json")
                if dependence is not None
                else None
            ),
            "adjusted_absolute_risk_product": (
                "table:landmark_adjusted_absolute_risk"
            ),
            "population_flow_product": "table:landmark_population_flow",
            "variable_opportunity_sensitivity_product": (
                "table:landmark_variable_opportunity_sensitivity"
            ),
            "spline_knot_quantiles": [0.10, 0.50, 0.90],
            "spline_reference": "median_in_primary_population",
            "curve_quantile_range": [0.10, 0.90],
            "curve_points": 41,
            "linear_sensitivity_per_unit": 1.0,
            "interpretation": "descriptive_prognostic_association_not_causal",
            **plan_bound_body,
        }
    ).model_dump(mode="json")
    return signed_projection(
        authority, scientific_configuration_sha256=scientific_configuration_sha256
    )


__all__ = [
    "WebScientificRuntimeProjection",
    "WebScientificRuntimeProjectionError",
    "categorical_adjustments",
    "compile_landmark_categorical_runtime_projection",
    "compile_landmark_spline_runtime_projection",
    "compile_web_scientific_runtime_projection",
    "kdigo_observability_authority_missing",
    "export_kdigo_strict_derivation_available",
    "kdigo_strict_derivation_available",
    "required_host_derivations",
    "one_sensitivity_spec",
    "operational_covariates",
    "primary_exposure_kind",
    "signed_projection",
]
