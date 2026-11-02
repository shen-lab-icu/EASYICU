"""Prove that the selected export and approved first-stay rule produced the cohort."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from easyicu.research_agent.acquisition.first_icu_stay import (
    FirstIcuStayBinding,
    FirstIcuStayPerHospitalizationBinding,
    load_verified_first_icu_stay,
    load_verified_first_icu_stay_per_hospitalization,
)
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    _local_authority_reference,
    load_verified_materialized_cohort_authority,
    sealed_first_icu_stay_per_hospitalization_restriction,
    sealed_first_icu_stay_restriction,
)
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import primary_cohort, source_identity_authority, sources


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _restriction_coordinates(verified: Any, path: Path, field: str) -> Any:
    authority = verified.authority
    seen = set()
    while authority is not None:
        restriction = authority.producer_parameters.get(field + "_restriction")
        if restriction is not None:
            return restriction.get("authority_coordinates") if isinstance(restriction, Mapping) else None
        parent = authority.parent_authority_sha256
        if parent is None or parent in seen:
            return None
        seen.add(parent)
        _, authority = _local_authority_reference(path.parent, authority_sha256=parent)
    return None


def verified_first_stay_applied(
    scientific: Any, acquisition: Any, *, export_path: str, source_identity: Any,
) -> bool:
    """Require independent launch, source, restriction and acquisition identities."""
    if acquisition is None or scientific.metadata_only_planning or acquisition.blocked:
        return False
    cohort = scientific.study.get("cohort")
    per_hospital = primary_cohort.first_icu_stay_per_hospitalization_only(cohort)
    field = "first_icu_stay_per_hospitalization" if per_hospital else "first_icu_stay"
    if not per_hospital and not primary_cohort.first_icu_stay_only(cohort):
        return False
    binding = scientific.foundation_profile.get(field)
    binding_type = FirstIcuStayPerHospitalizationBinding if per_hospital else FirstIcuStayBinding
    if not isinstance(binding, binding_type) or not acquisition.universe_path:
        return False
    reference = getattr(acquisition, "cohort_authority_ref", None)
    authority_path = getattr(acquisition, "cohort_authority_path", None)
    if not isinstance(reference, MaterializedCohortAuthorityRef) or authority_path is None:
        return False
    path = Path(acquisition.universe_path)
    if Path(authority_path) != path.parent / reference.file:
        return False
    identity = SourceIdentity.from_record(source_identity)
    if identity is None or identity.to_dict() != source_identity:
        return False
    try:
        selected = Path(export_path).expanduser().resolve(strict=True)
        if sources.stored_source_path_for_identity(identity) != str(selected):
            return False
        coordinates = binding.authority_coordinates
        expected_schema = f"easyicu.{field}_runtime_authority/1"
        if (coordinates.get("schema_version") != expected_schema
                or coordinates.get("database") != source_identity_authority._database(identity.database)
                or coordinates.get("coordinate_sha256") != binding.coordinate_sha256
                or not isinstance(coordinates.get("authority_ref"), str)
                or not coordinates["authority_ref"]):
            return False
        verified = load_verified_materialized_cohort_authority(path, expected_authority=reference)
        if verified is None:
            return False
        with open_export_package(selected) as package:
            if (package.authority_sha256 != verified.authority.source_export_authority_sha256
                    or package.manifest_sha256 != coordinates.get("export_manifest_sha256")
                    or package.manifest_path.name != coordinates.get("export_manifest_file")):
                return False
        load_coordinate = (load_verified_first_icu_stay_per_hospitalization
                           if per_hospital else load_verified_first_icu_stay)
        load_coordinate(binding.coordinate_path, expected_sha256=binding.coordinate_sha256)
        read = sealed_first_icu_stay_per_hospitalization_restriction if per_hospital else sealed_first_icu_stay_restriction
        restriction = read(verified, cohort_path=path)
        return (restriction is not None
                and restriction.get("coordinate_sha256") == binding.coordinate_sha256
                and json.dumps(_plain(_restriction_coordinates(verified, path, field)), sort_keys=True)
                == json.dumps(_plain(coordinates), sort_keys=True))
    except (OSError, ValueError, TypeError):
        return False
