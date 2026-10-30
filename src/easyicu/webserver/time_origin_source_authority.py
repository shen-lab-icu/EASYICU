"""Bind a registered export to content-verified source facts for review.

This is not a plan approval and it grants no materialization authority. It
supports a registration whose selected directory is itself the export; raw
source to subsequently produced export requires a separate producer receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
from easyicu.webserver import sources


@dataclass(frozen=True)
class RegisteredExportTimeOrigin:
    source_identity_sha256: str
    export_authority_sha256: str
    manifest_sha256: str
    selected_origin: str

    def __post_init__(self) -> None:
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in (
            self.source_identity_sha256, self.export_authority_sha256, self.manifest_sha256,
        )) or self.selected_origin != SIC_CARE_UNIT_ORIGIN:
            raise ValueError("time_origin_export_receipt_invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_identity_sha256": self.source_identity_sha256,
            "export_authority_sha256": self.export_authority_sha256,
            "manifest_sha256": self.manifest_sha256,
            "selected_origin": self.selected_origin,
        }


@dataclass(frozen=True)
class CaseDurationSourceProvenance:
    """Metadata-only derivation evidence pinned inside a verified export."""

    snapshot_sha256: str
    implementation_sha256: str
    source_config_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"snapshot_sha256": self.snapshot_sha256,
                "implementation_sha256": self.implementation_sha256,
                "source_config_sha256": self.source_config_sha256}


@dataclass(frozen=True)
class CaseDurationMetadataPlanReceipt:
    """Read-only source schema and resolver facts; not extracted data."""

    source_identity_sha256: str
    dictionary_sha256: str
    implementation_sha256: str
    source_config_sha256: str
    case_concepts_sha256: str
    case_schema_sha256: str

    def __post_init__(self) -> None:
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in self.to_dict().values()):
            raise ValueError("case_metadata_plan_receipt_invalid")

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in (
            "source_identity_sha256", "dictionary_sha256", "implementation_sha256",
            "source_config_sha256", "case_concepts_sha256", "case_schema_sha256",
        )}


def prepare_case_duration_metadata_plan(raw_path: str | Path, *, expected_identity: SourceIdentity) -> CaseDurationMetadataPlanReceipt:
    """Bind a registered raw SIC schema and exact producer metadata, without rows."""
    import pyarrow.parquet as pq
    from easyicu.concept.loader import load_dictionary
    from easyicu.concept.provenance import capture_data_provenance, metadata_digest
    from easyicu.concept.sic_landmark import SIC_CASE_LANDMARK_COLUMNS, SIC_CASE_SOURCE_COLUMNS
    from easyicu.config import load_src_cfg

    root = Path(raw_path).expanduser()
    cases = root / "cases.parquet"
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir()
            or cases.is_symlink() or not cases.is_file()):
        raise ValueError("case_metadata_source_unavailable")
    identity = sources.registered_source_identity(str(root))
    if identity is None or identity != expected_identity or identity.database != "sic":
        raise ValueError("case_metadata_source_mismatch")
    schema = pq.read_schema(cases)
    if not set(SIC_CASE_SOURCE_COLUMNS) <= set(schema.names):
        raise ValueError("case_metadata_fields_unavailable")
    config = load_src_cfg("sic")
    snapshot = capture_data_provenance(
        load_dictionary(include_sofa2=True), concepts=sorted(SIC_CASE_LANDMARK_COLUMNS),
        source_database="sic", class_prefixes=tuple(config.class_prefix),
        source_config=config, source_config_sha256=metadata_digest(config.model_dump(mode="json")),
    )
    selected = [item.model_dump(mode="json") for item in snapshot.concepts
                if item.concept_id in SIC_CASE_LANDMARK_COLUMNS]
    if not is_case_duration_provenance_snapshot(snapshot):
        raise ValueError("case_duration_source_provenance_unavailable")
    if sources.registered_source_identity(str(root)) != identity:
        raise ValueError("case_metadata_registration_changed")
    schema_fields = [(name, str(schema.field(name).type)) for name in sorted(SIC_CASE_SOURCE_COLUMNS)]
    return CaseDurationMetadataPlanReceipt(
        source_identity_sha256=canonical_sha256(identity.to_dict()), dictionary_sha256=snapshot.dictionary_sha256,
        implementation_sha256=snapshot.implementation_sha256, source_config_sha256=snapshot.source_config_sha256,
        case_concepts_sha256=canonical_sha256(sorted(selected, key=lambda item: item["concept_id"])),
        case_schema_sha256=canonical_sha256(schema_fields),
    )


def match_case_duration_metadata_plan(receipt: CaseDurationMetadataPlanReceipt, package) -> None:
    """Compare the post-extraction export's resolver to the pre-review facts."""
    from easyicu.concept.sic_landmark import SIC_CASE_LANDMARK_COLUMNS

    proof = case_duration_source_provenance(package)
    snapshot = next(item for item in package.data_provenance.snapshots if item.snapshot_sha256 == proof.snapshot_sha256)
    selected = [item.model_dump(mode="json") for item in snapshot.concepts
                if item.concept_id in SIC_CASE_LANDMARK_COLUMNS]
    if (snapshot.dictionary_sha256 != receipt.dictionary_sha256
            or proof.implementation_sha256 != receipt.implementation_sha256
            or proof.source_config_sha256 != receipt.source_config_sha256
            or canonical_sha256(sorted(selected, key=lambda item: item["concept_id"])) != receipt.case_concepts_sha256):
        raise ValueError("case_metadata_export_provenance_mismatch")


def case_duration_source_provenance(package) -> CaseDurationSourceProvenance:
    """Find one digest-verified snapshot declaring every case-risk input.

    This proves the export's recorded resolver inputs, not an independent
    replay of patient values nor the original raw-source registration.
    """
    from easyicu.concept.sic_landmark import SIC_CASE_LANDMARK_COLUMNS

    if not set(SIC_CASE_LANDMARK_COLUMNS) <= set(package.concept_index):
        raise ValueError("case_duration_source_provenance_unavailable")
    provenance = package.data_provenance
    matches = []
    for snapshot in provenance.snapshots if provenance is not None else ():
        if is_case_duration_provenance_snapshot(snapshot):
            matches.append(snapshot)
    if len(matches) != 1:
        raise ValueError("case_duration_source_provenance_unavailable")
    snapshot = matches[0]
    return CaseDurationSourceProvenance(snapshot.snapshot_sha256, snapshot.implementation_sha256,
                                        snapshot.source_config_sha256)


def is_case_duration_provenance_snapshot(snapshot) -> bool:
    from easyicu.concept.sic_landmark import SIC_CASE_LANDMARK_COLUMNS, SIC_CASE_SOURCE_COLUMNS

    if (snapshot.source_database != "sic" or not snapshot.source_config_sha256
            or not snapshot.implementation_sha256):
        return False
    concepts = {item.concept_id: item for item in snapshot.concepts}
    if not set(SIC_CASE_LANDMARK_COLUMNS) <= concepts.keys():
        return False
    for concept in SIC_CASE_LANDMARK_COLUMNS:
        recorded = concepts[concept]
        fields = {source.value_variable for source in recorded.sources
                  if source.table == "cases" and source.callback == "sic_case_landmark_fields"}
        if recorded.status != "recorded" or fields != set(SIC_CASE_SOURCE_COLUMNS):
            return False
    return True


def attest_registered_export_time_origin(
    export_path: str | Path, *, expected_identity: SourceIdentity,
) -> RegisteredExportTimeOrigin:
    """Snapshot one exact registration and its same-directory export package.

    The verified package is reopened by the consumer before materialization;
    a receipt cannot authorize executing a research plan. Source registration
    and package facts are checked again after opening to catch replacement.
    """
    path = Path(export_path).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("time_origin_export_source_invalid")
    before = sources.registered_source_identity(str(path))
    if before is None or before != expected_identity or before.database != "sic":
        raise ValueError("time_origin_export_registration_mismatch")
    with open_export_package(path) as package:
        if (package.database != "sic" or package.root.resolve() != path.resolve()
                or package.declared_time_origin != SIC_CARE_UNIT_ORIGIN):
            raise ValueError("time_origin_export_database_mismatch")
        result = RegisteredExportTimeOrigin(
            source_identity_sha256=canonical_sha256(before.to_dict()),
            export_authority_sha256=package.authority_sha256,
            manifest_sha256=package.manifest_sha256,
            selected_origin=SIC_CARE_UNIT_ORIGIN,
        )
        if sources.registered_source_identity(str(path)) != before:
            raise ValueError("time_origin_export_registration_changed")
        return result


def attest_selection_for_review(context, selection) -> RegisteredExportTimeOrigin:
    """Resolve the frozen registration uniquely and bind its verified export.

    Only the host registry provides the path. An unregistered source or raw
    directory without a verified export cannot produce an attestation.
    """
    from easyicu.databases.research_plan_requirements import _constraints

    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if identity is None or canonical_sha256(identity.to_dict()) != selection.source_identity_sha256:
        raise ValueError("time_origin_export_registration_mismatch")
    rows = sources.load_registry().get("sources") or []
    paths = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("ok") or not isinstance(row.get("path"), str):
            continue
        current = SourceIdentity.from_record({**row, "registration_id": row.get("id")})
        if current == identity:
            paths.append(row["path"])
    if len(paths) != 1:
        raise ValueError("time_origin_export_registration_not_unique")
    return attest_registered_export_time_origin(paths[0], expected_identity=identity)


def attest_case_duration_for_review(context, selection):
    """Reopen the selected export and bind its verified case derivation snapshot."""
    from easyicu.databases.research_plan_requirements import _constraints

    axis = attest_selection_for_review(context, selection)
    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    paths = []
    for row in sources.load_registry().get("sources") or []:
        if isinstance(row, dict) and row.get("ok") and isinstance(row.get("path"), str):
            if SourceIdentity.from_record({**row, "registration_id": row.get("id")}) == identity:
                paths.append(row["path"])
    if len(paths) != 1:
        raise ValueError("case_duration_source_registration_not_unique")
    with open_export_package(paths[0]) as package:
        if (package.authority_sha256 != axis.export_authority_sha256
                or package.manifest_sha256 != axis.manifest_sha256
                or package.declared_time_origin != axis.selected_origin):
            raise ValueError("case_duration_source_export_changed")
        return axis, case_duration_source_provenance(package)


def propose_registered_export_time_origin(context):
    """Offer a source-verified candidate; do not select or approve it."""
    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.databases.research_requirements import TimeOriginSelection, review_only_time_origin
    from easyicu.databases.research_substitutions import registered_source_profile

    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    profile = registered_source_profile(context.cohort.database)
    proposal = review_only_time_origin(profile) if profile is not None else None
    if identity is None or identity.database != context.cohort.database or proposal is None:
        raise ValueError("time_origin_export_registration_mismatch")
    selection = TimeOriginSelection(
        requested_origin=proposal.requested_origin,
        selected_origin=proposal.proposed_origin,
        source_field=proposal.source_field,
        source_identity_sha256=canonical_sha256(identity.to_dict()),
    )
    return selection, attest_selection_for_review(context, selection)


def review_time_origin_attestor(context, selection):
    """One host entry for a proposed or explicitly recorded candidate."""
    if selection is None:
        return propose_registered_export_time_origin(context)
    return selection, attest_selection_for_review(context, selection)
