"""Private prepared-source handles, pinned to ec2 registration and exact bytes."""

from dataclasses import dataclass
from pathlib import Path

from easyicu.research_agent.canonical_json import canonical_sha256, sha256_file
from easyicu.research_agent.contracts.dependence import (
    PatientGroupResolutionError, PlannedDependenceRequirement, resolve_patient_groups,
)
from easyicu.research_agent.planning.dependence_authority import context_patient_group_authority
from easyicu.research_agent.providers.client_trust import (
    ProviderConfigurationError, require_credentialed_data_clearance, source_requires_clearance,
)
from easyicu.research_agent.research_context.export_selection import (
    SourceIdentity, source_identity_provenance,
)
from easyicu.research_agent.schema import ResearchContext

from .contracts import QuickStop


@dataclass(frozen=True)
class PreparedSource:
    """Host-only handle; never deserialize a browser-supplied path or identity.

    The source preparation owner supplies the materialized context and its
    verified unit/patient columns. Binding makes no extraction and reads no rows.
    """

    path: Path
    context: ResearchContext
    identity: SourceIdentity
    unit_column: str
    patient_column: str | None
    analysis_unit: str
    data_sha256: str
    binding: str


def binding_digest(source: PreparedSource) -> str:
    return canonical_sha256({
        "source": source.identity.to_dict(), "data_sha256": source.data_sha256,
        "context": source.context.model_dump(mode="json"),
        "unit_column": source.unit_column, "patient_column": source.patient_column,
        "analysis_unit": source.analysis_unit,
    })


def bind_prepared_source(path: Path, *, context: ResearchContext, identity: SourceIdentity,
                         unit_column: str, patient_column: str | None = None,
                         analysis_unit: str = "icu_stay") -> PreparedSource:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.suffix != ".parquet":
        raise QuickStop("quick_prepared_source_required")
    if (SourceIdentity.from_record(identity.to_dict()) != identity
            or identity.database != context.cohort.database
            or source_identity_provenance(context.user_preferences).get("source_identity") != identity.to_dict()
            or analysis_unit not in {"icu_stay", "hospitalization", "patient"}):
        raise QuickStop("quick_source_identity_mismatch")
    declared = set(context.cohort.id_columns)
    if unit_column not in declared or patient_column is not None and patient_column not in declared:
        raise QuickStop("quick_unit_identity_missing")
    _patient_requirement(context, unit_column, patient_column)
    # Detach mutable context children from the preparation caller.
    source = PreparedSource(path.resolve(), context.model_copy(deep=True), identity,
                            unit_column, patient_column, analysis_unit, sha256_file(path), "")
    from dataclasses import replace

    return replace(source, binding=binding_digest(source))


def _patient_requirement(context, unit_column, patient_column):
    """How a row's patient is read from ``patient_column``.

    A column the context's owner-issued grouping names is read by that
    grouping's derivation (a verified ``p<patient>:s<stay>`` row key, for
    one); any other declared column holds the patient itself. A stay key
    never counts as its own patient.
    """
    if patient_column is None:
        return None
    issued = context_patient_group_authority(context)
    if issued is not None and issued.group_source == patient_column:
        return issued
    if patient_column == unit_column:
        raise QuickStop("quick_unit_identity_missing")
    return PlannedDependenceRequirement(group_source=patient_column, group_derivation="identity")


def patient_groups(source: PreparedSource, frame) -> dict:
    """Each prepared unit's patient, read through the shared dependence contract."""
    requirement = _patient_requirement(source.context, source.unit_column, source.patient_column)
    if requirement is None:
        raise QuickStop("quick_independent_units_unverified")
    try:
        groups = resolve_patient_groups(frame[requirement.group_source].tolist(), requirement=requirement).groups
    except PatientGroupResolutionError as exc:
        raise QuickStop("quick_unit_identity_invalid") from exc
    return dict(zip(frame[source.unit_column].tolist(), groups, strict=True))


def require_source(source: PreparedSource, provider: object) -> None:
    """Repeat clearance at both approval preparation and execution."""
    try:
        if provider is None:
            if source_requires_clearance(source.identity.database):
                raise QuickStop("quick_qualifying_provider_required")
        else:
            require_credentialed_data_clearance(provider, source.identity.database)
    except ProviderConfigurationError as exc:
        raise QuickStop("quick_qualifying_provider_required") from exc
    if source.binding != binding_digest(source):
        raise QuickStop("quick_source_changed")
    if source.path.is_symlink() or not source.path.is_file() or sha256_file(source.path) != source.data_sha256:
        raise QuickStop("quick_source_changed")
