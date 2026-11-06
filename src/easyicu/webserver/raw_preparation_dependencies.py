"""Compile a producer's proposed input declaration without reading data rows.

Declarations are candidates, not proof that arbitrary code consumes only these
files. The trusted producer and its runtime still have to prove that closure.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest

_SHA = r"^[0-9a-f]{64}$"


def reviewed_plan_cohort_concepts(*, context, plan) -> tuple[str, ...]:
    """List typed cohort-predicate concepts using explicit variable bindings."""
    variables = {}
    for item in context.variables:
        name, concept = str(item.name), str(item.source_concept or item.name)
        if name in variables and variables[name] != concept:
            raise ValueError("dependency_variable_source_ambiguous")
        variables[name] = concept
    cohort = plan.cohort
    if cohort is None:
        return ()
    return tuple(dict.fromkeys(variables.get(str(predicate.concept_id), str(predicate.concept_id))
                               for predicate in (*cohort.inclusion, *cohort.exclusion)))


def reviewed_plan_provenance_concepts(*, context, plan) -> tuple[str, ...]:
    """List exact declared variable and typed cohort-predicate concepts.

    A cohort predicate can depend on a concept absent from the analysis
    variable list. Include it in metadata discovery; do not infer additional
    population rules, identity mappings, or producer read closure.
    """
    names = [str(item.source_concept or item.name) for item in context.variables]
    names.extend(reviewed_plan_cohort_concepts(context=context, plan=plan))
    return tuple(dict.fromkeys(names))


def reviewed_population_provenance_concepts(*, projection: dict | None,
                                            plan_sha256: str) -> tuple[str, ...]:
    """Name concepts in a host-verified population, including unapplied ones.

    ``projection`` must come from ``reviewed_population_spec`` for the same
    paused plan. These are metadata-discovery candidates, not proof that the
    criterion is executable, a complete input declaration, or a data grant.
    """
    from easyicu.research_agent.planning.population_compile import population_spec_concepts
    from easyicu.research_agent.planning.population_spec import PopulationSpec

    if projection is None:
        return ()
    if (not isinstance(projection, dict)
            or projection.get("schema_version") != "easyicu.reviewed_population_spec_projection/1"
            or projection.get("plan_sha256") != plan_sha256
            or not isinstance(projection.get("criteria"), list)):
        raise ValueError("dependency_population_projection_mismatch")
    try:
        spec = PopulationSpec.model_validate(
            {"criteria": [item["criterion"] for item in projection["criteria"]]}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dependency_population_projection_invalid") from exc
    return population_spec_concepts(spec)


class SourceTableRequirement(BaseModel):
    """Explicit dependencies provided by a database adapter, not guessed by AI."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    table: str = Field(min_length=1, max_length=160)
    partitions: tuple[str, ...] = ()
    columns: tuple[str, ...] = Field(min_length=1)
    format: Literal["csv", "parquet"]
    # False means the adapter has not determined all dynamically selected parts.
    partition_roster_declared_complete: bool = False
    window: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _validate_names(self):
        if len(set(self.partitions)) != len(self.partitions) or len(set(self.columns)) != len(self.columns):
            raise ValueError("dependency_duplicate_name")
        for name in self.partitions:
            path = PurePosixPath(name)
            if (not name or "\\" in name or "\x00" in name or path.is_absolute()
                    or path.as_posix() != name or any(part in (".", "..") for part in path.parts)):
                raise ValueError("dependency_invalid_partition")
        if any(not column or len(column) > 160 for column in self.columns):
            raise ValueError("dependency_invalid_column")
        return self


class ProducerDependencyDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    adapter: str = Field(min_length=1, max_length=160)
    source_identity_sha256: str = Field(pattern=_SHA)
    plan_sha256: str = Field(pattern=_SHA)
    context_sha256: str | None = Field(default=None, pattern=_SHA)
    configuration_sha256: str = Field(pattern=_SHA)
    implementation_sha256: str = Field(pattern=_SHA)
    requirements: tuple[SourceTableRequirement, ...] = Field(min_length=1)
    unresolved_dependencies: tuple[str, ...] = ()


class CandidateStepInput(BaseModel):
    """One plan-declared input and its context binding, if explicitly present."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    step_id: str = Field(min_length=1, max_length=160)
    input_name: str = Field(min_length=1, max_length=200)
    source_concept: str | None = Field(default=None, max_length=160)
    context_window: str | None = Field(default=None, max_length=160)


class CandidateWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=160)
    anchor: str = Field(min_length=1, max_length=80)
    start_hours: float
    end_hours: float


class CandidatePopulationNeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    criterion_id: str = Field(min_length=1, max_length=10)
    kind: str = Field(min_length=1, max_length=80)
    role: Literal["include", "exclude"]
    concepts: tuple[str, ...] = Field(max_length=10)
    window: CandidateWindow | None = None
    diagnosis_system: str | None = Field(default=None, max_length=16)
    diagnosis_codes: tuple[str, ...] = Field(default=(), max_length=64)
    measurement_summary: str | None = Field(default=None, max_length=16)
    operator: str | None = Field(default=None, max_length=4)
    value: float | None = None
    unit: str | None = Field(default=None, max_length=32)
    missing_policy: str | None = Field(default=None, max_length=16)
    min_years: float | None = None
    max_years: float | None = None
    min_hours: float | None = None
    max_hours: float | None = None
    hours: float | None = None


class CandidatePreparationPreview(BaseModel):
    """Read-only evidence, never an extraction or analysis grant."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["easyicu.candidate-preparation-preview/1"] = "easyicu.candidate-preparation-preview/1"
    plan_sha256: str = Field(pattern=_SHA)
    scientific_review_sha256: str = Field(pattern=_SHA)
    context_sha256: str = Field(pattern=_SHA)
    step_inputs: tuple[CandidateStepInput, ...] = Field(max_length=500)
    population_needs: tuple[CandidatePopulationNeed, ...] = Field(max_length=8)
    context_windows: tuple[CandidateWindow, ...] = Field(max_length=50)
    blockers: tuple[str, ...]
    complete: Literal[False] = False
    preparation_authorized: Literal[False] = False
    analysis_authorized: Literal[False] = False


def declared_native_export_inventory(*, export_path: Path, expected_manifest_sha256: str,
                                     expected_database: str) -> dict:
    """Describe manifest-declared files without opening them or granting preparation.

    The future native-package reader snapshots every listed file. This inventory
    is only a candidate: it does not hash or validate file contents, establish
    decoded columns or callbacks, or authorize any data access.
    """
    from easyicu.research_agent.intake.export_package import _FORMAT_BY_SUFFIX, _manifest_file_entries

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("candidate_export_manifest_duplicate_key")
            result[key] = value
        return result

    from easyicu.export_publication import require_no_in_progress_marker

    require_no_in_progress_marker(Path(export_path))  # an interrupted export
    with AnchoredDirectory.open(Path(export_path)) as directory:
        raw = directory.read_bytes("_manifest.json", max_bytes=16 * 1024 * 1024)
        directory.assert_still_selected()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("candidate_export_manifest_changed")
    manifest = json.loads(raw, object_pairs_hook=reject_duplicates)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "easyicu_native_export_v2"
            or manifest.get("database") != expected_database):
        raise ValueError("candidate_export_manifest_unavailable")
    entries = _manifest_file_entries(manifest, manifest_kind="native")
    if not 0 < len(entries) <= 1000:
        raise ValueError("candidate_export_roster_invalid")
    files = []
    for entry in entries:
        path = entry.get("file")
        module = entry.get("module") or entry.get("group")
        relative = PurePosixPath(path) if isinstance(path, str) else PurePosixPath("")
        if (not isinstance(path, str) or not path or "\\" in path or "\x00" in path
                or relative.is_absolute() or relative.as_posix() != path
                or any(part in {".", ".."} for part in relative.parts)
                or ".derivation-context" in relative.parts or relative.suffix.lower() not in _FORMAT_BY_SUFFIX
                or not isinstance(module, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", module)):
            raise ValueError("candidate_export_roster_invalid")
        files.append({"file": path, "module": module})
    if len({item["file"] for item in files}) != len(files):
        raise ValueError("candidate_export_roster_invalid")
    metadata_files = ["_manifest.json"]
    features = manifest.get("feature_definitions")
    if isinstance(features, Mapping) and features.get("included") is True:
        feature_files = features.get("files")
        if not isinstance(feature_files, list) or any(not isinstance(item, Mapping) for item in feature_files):
            raise ValueError("candidate_export_roster_invalid")
        metadata_files.extend(item.get("file") for item in feature_files)
    column_metadata = manifest.get("column_metadata")
    if column_metadata is not None:
        if not isinstance(column_metadata, Mapping):
            raise ValueError("candidate_export_roster_invalid")
        metadata_files.append(column_metadata.get("file"))
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", name)
           or name in {".", ".."} for name in metadata_files) or len(set(metadata_files)) != len(metadata_files):
        raise ValueError("candidate_export_roster_invalid")
    return {"schema_version": "easyicu.declared-export-read-inventory/1",
            "manifest_sha256": expected_manifest_sha256, "database": expected_database, "files": files,
            "metadata_files": metadata_files,
            "all_manifest_files_will_be_snapshotted": True, "file_contents_verified": False,
            "decoded_column_scope_verified": False, "preparation_authorized": False, "analysis_authorized": False}


def _declared_typed_export_columns(*, export_path: Path, inventory: Mapping) -> dict:
    """Verify manifest and sidecar metadata bytes, without opening data files."""
    from easyicu.concept.metadata_sidecar import ColumnMetadataSidecar, EXPORT_PHYSICAL_SCOPE, SidecarRef

    with AnchoredDirectory.open(export_path) as directory:
        raw_manifest = directory.read_bytes("_manifest.json", max_bytes=16 * 1024 * 1024)
        if hashlib.sha256(raw_manifest).hexdigest() != inventory["manifest_sha256"]:
            raise ValueError("candidate_export_manifest_changed")
        manifest = json.loads(raw_manifest)
        reference = SidecarRef.from_dict(manifest["column_metadata"])
        if reference.file != f"column_metadata.sha256-{reference.sha256}.json":
            raise ValueError("candidate_column_metadata_invalid")
        raw_sidecar = directory.read_bytes(reference.file, max_bytes=64 * 1024 * 1024)
        directory.assert_still_selected()
    if len(raw_sidecar) != reference.size or hashlib.sha256(raw_sidecar).hexdigest() != reference.sha256:
        raise ValueError("candidate_column_metadata_changed")
    sidecar = ColumnMetadataSidecar.from_dict(json.loads(raw_sidecar))
    if (sidecar.record_count != reference.record_count or sidecar.scope != EXPORT_PHYSICAL_SCOPE
            or sidecar.source_database != inventory["database"]):
        raise ValueError("candidate_column_metadata_invalid")
    expected = {(item["file"], item["module"]) for item in inventory["files"]}
    if {(item.relative_path, item.module) for item in sidecar.files} != expected:
        raise ValueError("candidate_column_metadata_inventory_changed")
    if any(PurePosixPath(item.relative_path).suffix.lower() != ".parquet" for item in sidecar.files):
        raise ValueError("candidate_non_parquet_export_unverified")
    files = []
    for item in sidecar.files:
        columns = [{"column": name, "source_concept": binding.metadata.source_concept,
                    "role": binding.metadata.role.value} for name, binding in item.columns.items()]
        files.append({"file": item.relative_path, "module": item.module,
                      "identity_column": item.identity_column,
                      "time_columns": sorted(coordinate.column for coordinate in item.time_coordinates),
                      "declared_columns": columns})
    return {"sidecar_sha256": reference.sha256, "files": files}


def native_planning_metadata(*, export_path: Path, database: str | None = None):
    """Validate native manifest/sidecar for planning without opening data members."""
    import os
    from easyicu.concept.metadata_sidecar import ColumnMetadataSidecar, SidecarRef
    from easyicu.concept.selection_policy import concept_selection_policy
    from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept, _methodology_tag
    from easyicu.research_agent.concept_availability import normalize_database_name
    from easyicu.research_agent.intake.export_package import _WITHOUT_VALUES
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
    from easyicu.webserver.dataio import ExportCohortError

    from easyicu.export_publication import (
        UNPUBLISHED_CODE,
        NativeExportUnpublishedError,
        require_no_in_progress_marker,
    )

    root = Path(export_path).expanduser()
    try:
        # Refused unconditionally: with or without a manifest beside it, a
        # marked folder is an interrupted export, never a plannable source.
        require_no_in_progress_marker(root)
    except NativeExportUnpublishedError as exc:
        raise ExportCohortError(
            UNPUBLISHED_CODE, {"reason": exc.reason, "message": str(exc)}
        ) from exc
    if not os.path.lexists(root / "_manifest.json"):
        return None
    try:
        with AnchoredDirectory.open(root) as directory:
            raw = directory.read_bytes("_manifest.json", max_bytes=16 * 1024 * 1024)
            directory.assert_still_selected()
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            from easyicu.research_agent.intake.export_package import ExportPackageError
            from easyicu.webserver.dataio import _research_pipeline_intake_diagnostic

            intake_error = ExportPackageError(
                "invalid export manifest JSON", code="export_manifest_json_invalid"
            )
            raise ExportCohortError(
                "research_pipeline_manifest_invalid", _research_pipeline_intake_diagnostic(intake_error)
            ) from exc
        name = manifest.get("database") if isinstance(manifest, dict) else None
        if (not isinstance(name, str) or not name
                or (database and normalize_database_name(name) != normalize_database_name(database))):
            raise ValueError("native metadata database mismatch")
        inventory = declared_native_export_inventory(export_path=root,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(), expected_database=name)
        declared = _declared_typed_export_columns(export_path=root, inventory=inventory)
        files = declared.get("files")
        # One ICU-stay identity across files. The exporter keeps ``stay_id`` when
        # present, else the source's own ICU-stay key (eICU: patientunitstayid);
        # see concept.export_metadata._export_identity_and_time_coordinates.
        from easyicu.config import load_src_cfg
        icu_identifier = load_src_cfg(name.lower()).id_configs.get("icustay")
        stay_identities = {"stay_id", *((icu_identifier.id,) if icu_identifier is not None else ())}
        identities = {item.get("identity_column") for item in files or ()}
        if not files or len(identities) != 1 or not identities <= stay_identities:
            raise ValueError("native metadata identity mismatch")
        for item in files:
            member = root
            for part in PurePosixPath(item["file"]).parts:
                member = member / part
                if member.is_symlink():
                    raise ValueError("native metadata member unavailable")
            if not member.is_file():
                raise ValueError("native metadata member unavailable")
            member.resolve(strict=True).relative_to(root.resolve(strict=True))
        reference = SidecarRef.from_dict(manifest["column_metadata"])
        with AnchoredDirectory.open(root) as directory:
            sidecar_raw = directory.read_bytes(reference.file, max_bytes=64 * 1024 * 1024,
                                               expected_sha256=reference.sha256)
            directory.assert_still_selected()
        sidecar = ColumnMetadataSidecar.from_dict(json.loads(sidecar_raw))
        statuses: dict[str, set[str]] = {}
        for entry in manifest.get("files", []):
            if isinstance(entry, Mapping):
                recorded = entry.get("concept_status")
                for concept, status in (recorded.items() if isinstance(recorded, Mapping) else ()):
                    value = status.get("availability") if isinstance(status, Mapping) else None
                    statuses.setdefault(str(concept), set()).add(str(value))
        absent = {concept for concept, values in statuses.items() if values <= _WITHOUT_VALUES}
        by_concept = {}
        for item in sidecar.files:
            for name_, binding in item.columns.items():
                meta = binding.metadata
                concept = meta.source_concept
                if isinstance(concept, str) and concept and meta.role.value in {"value", "event_status"}:
                    by_concept.setdefault(concept, []).append((item, name_, meta))
        concepts = []
        for concept, values in sorted(by_concept.items()):
            if concept in absent or len(values) != 1:
                continue
            item, column, meta = values[0]
            policy = concept_selection_policy(concept)
            category = meta.category or ""
            concepts.append(CatalogConcept(concept_id=concept, description=meta.description or "",
                category=category, file_name=item.relative_path, resolved_column=column,
                typed_metadata=True, column_role=meta.role.value, methodology=_methodology_tag(concept, category),
                selection_mode=policy.selection_mode if policy else "ordinary",
                selection_note=policy.rationale if policy else "",
                canonical_alternative=(policy.canonical_alternative or "") if policy else ""))
        return {"database": name, "inventory": inventory, "files": files,
                "catalog": AvailableCatalog(source="verified_native_column_metadata", concepts=concepts)}
    except ExportCohortError:
        raise
    except (AuthorityFilesystemError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ResearchPipelineRunError("research_pipeline_planning_native_metadata_invalid",
            "The native export column metadata cannot be verified without reading patient rows.") from exc


def _candidate_plan_input_dependencies(*, plan, context, population_spec, structural_columns=()) -> dict:
    """Compile declared dependencies only; this cannot authorize any file reads."""
    from easyicu.research_agent.contracts.declared_product import typed_product
    from easyicu.research_agent.contracts.host_derivations import host_derivation_producing
    from easyicu.research_agent.planning.plan_graph import _typed_plan_dependency_graph
    from easyicu.research_agent.planning.population_compile import population_spec_concepts

    steps = list(plan.steps)
    graph, findings = _typed_plan_dependency_graph(steps)
    blockers = []
    if len(graph) != len(steps) or findings:
        blockers.append("candidate_typed_plan_graph_invalid")
    unresolved_graph = {key: set(value) for key, value in graph.items()}
    while unresolved_graph:
        ready = {key for key, value in unresolved_graph.items() if not value}
        if not ready:
            blockers.append("candidate_typed_plan_graph_cycle")
            break
        unresolved_graph = {key: value - ready for key, value in unresolved_graph.items() if key not in ready}
    variables = {item.name: item for item in context.variables}
    structural_columns = frozenset(structural_columns)
    if len(variables) != len(context.variables):
        raise ValueError("candidate_preparation_variable_ambiguous")
    required = set(population_spec_concepts(population_spec))
    products = []
    for step in steps:
        for name in step.inputs:
            product = typed_product(name)
            if product is not None:
                products.append({"step_id": step.step_id, "product": name})
            elif name in structural_columns:
                continue
            elif name in variables and variables[name].source_concept:
                required.add(variables[name].source_concept)
            else:
                blockers.append("candidate_step_input_binding_unverified")
    # Context variables can be needed by typed method contracts even if the
    # plan omits an explicit input. Include them conservatively.
    for variable in context.variables:
        if variable.name in structural_columns:
            continue
        if variable.source_concept:
            required.add(variable.source_concept)
        else:
            blockers.append("candidate_context_variable_binding_unverified")
    required.update(reviewed_plan_cohort_concepts(context=context, plan=plan))
    derivations = {}
    for concept in list(required):
        derivation = host_derivation_producing(concept)
        if derivation is not None:
            required.discard(concept)
            required.update(derivation[0].source_concepts)
            derivations[derivation[0].derivation_id] = sorted(derivation[0].source_concepts)
    if any(item.kind == "diagnosis_codes" for item in population_spec.criteria):
        blockers.append("candidate_diagnosis_source_binding_unverified")
    return {"required_source_concepts": sorted(required),
            "typed_step_dependencies": {key: sorted(value) for key, value in sorted(graph.items())},
            "typed_product_inputs": sorted(products, key=lambda item: (item["step_id"], item["product"])),
            "host_derivation_sources": dict(sorted(derivations.items())),
            "blockers": sorted(set(blockers))}


def first_observed_candidate_preparation_preview(*, run_dir: Path, pending, checkpoint) -> dict:
    """Recheck the host-reviewed scope before projecting known plan coordinates.

    The result names evidence already in the review, not the producer's full
    dependency closure. Source identity verification opens and hashes the raw
    table but does not decode or project its patient rows.
    """
    import hashlib
    from easyicu.research_agent.planning.population_compile import population_spec_concepts
    from easyicu.research_agent.planning.population_spec import PopulationSpec
    from easyicu.research_agent.research_context.typed import parse_research_context_json
    from easyicu.research_agent.schema import AnalysisPlan
    from easyicu.webserver.first_observed_preparation import _SCOPE_KEY, proposed_first_observed_scope
    from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
    from easyicu.webserver.raw_preparation_candidate import _plan_authority

    authority = _plan_authority(pending)
    if checkpoint.request.payload.get("plan_review_authority") != authority.model_dump(mode="json"):
        raise ValueError("candidate_preparation_plan_changed")
    scope = checkpoint.request.payload.get(_SCOPE_KEY)
    if not isinstance(scope, Mapping) or proposed_first_observed_scope(run_dir=run_dir, request=checkpoint.request) != dict(scope):
        raise ValueError("candidate_preparation_scope_changed")
    with AnchoredDirectory.open(run_dir) as directory:
        raw = directory.read_bytes("research_context.json", max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    if hashlib.sha256(raw).hexdigest() != scope.get("context_sha256"):
        raise ValueError("candidate_preparation_context_changed")
    context = parse_research_context_json(raw.decode("utf-8"))
    plan = AnalysisPlan.model_validate(authority.model_dump(mode="json")["plan_payload"])
    population = reviewed_population_spec(run_dir=run_dir, pending=pending)
    if not isinstance(population, dict) or population.get("plan_sha256") != authority.plan_sha256:
        raise ValueError("candidate_preparation_population_changed")
    spec = PopulationSpec.model_validate({"criteria": [item["criterion"] for item in population["criteria"]]})
    variables = {variable.name: variable for variable in context.variables}
    if len(variables) != len(context.variables):
        raise ValueError("candidate_preparation_variable_ambiguous")
    step_inputs = []
    for step in plan.steps:
        for name in step.inputs:
            variable = variables.get(name)
            step_inputs.append(CandidateStepInput(step_id=step.step_id, input_name=name,
                source_concept=variable.source_concept if variable else None,
                context_window=variable.analysis_window if variable else None))
    windows = tuple(CandidateWindow(name=item.name, anchor=item.anchor, start_hours=item.start_hours,
                                    end_hours=item.end_hours) for item in context.time_windows)
    population_needs = []
    for criterion in spec.criteria:
        span = getattr(criterion, "window", None)
        window = (CandidateWindow(name=criterion.id, anchor="icu_admission", start_hours=span.start_hours,
                                  end_hours=span.end_hours) if span is not None else None)
        one = PopulationSpec.model_validate({"criteria": [criterion.model_dump(mode="json")]})
        population_needs.append(CandidatePopulationNeed(criterion_id=criterion.id, kind=criterion.kind,
            role=criterion.role, concepts=population_spec_concepts(one), window=window,
            diagnosis_system=getattr(criterion, "system", None),
            diagnosis_codes=tuple(getattr(criterion, "codes", ())),
            measurement_summary=getattr(criterion, "summary", None),
            operator=getattr(criterion, "op", None), value=getattr(criterion, "value", None),
            unit=getattr(criterion, "unit", None), missing_policy=getattr(criterion, "missing_policy", None),
            min_years=getattr(criterion, "min_years", None), max_years=getattr(criterion, "max_years", None),
            min_hours=getattr(criterion, "min_hours", None), max_hours=getattr(criterion, "max_hours", None),
            hours=getattr(criterion, "hours", None)))
    blockers = ["candidate_dependency_closure_unverified", "candidate_source_read_closure_unverified",
                "candidate_extraction_window_unverified", "candidate_coordinate_export_binding_unavailable"]
    if any(item.source_concept is None for item in step_inputs):
        blockers.append("candidate_step_input_binding_unverified")
    result = CandidatePreparationPreview(plan_sha256=authority.plan_sha256,
        scientific_review_sha256=scope["scientific_review_sha256"], context_sha256=scope["context_sha256"],
        step_inputs=tuple(step_inputs), population_needs=tuple(population_needs),
        context_windows=windows, blockers=tuple(blockers)).model_dump(mode="json")
    result["declared_plan_dependencies"] = _candidate_plan_input_dependencies(
        plan=plan, context=context, population_spec=spec)
    return result


def declaration_from_provenance(*, plan: PlanReviewAuthority, context_sha256: str,
                                snapshot, source_config, source_root: Path,
                                expected_identity, window: str) -> ProducerDependencyDeclaration:
    """Compile a real resolver snapshot to a conservative review candidate.

    The caller resolves and verifies the scientific-review context and its
    window. This helper binds those values; it does not approve them. Callbacks,
    dataset partition discovery, cohort filters and unknown concepts remain
    blockers until an owner supplies complete machine-verifiable dependencies.
    """
    from easyicu.concept.provenance import DataProvenanceSnapshot, metadata_digest
    from easyicu.config import DataSourceConfig
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver.sources import registered_source_identity

    if (not isinstance(snapshot, DataProvenanceSnapshot)
            or not isinstance(source_config, DataSourceConfig)
            or not isinstance(expected_identity, SourceIdentity)
            or not isinstance(context_sha256, str) or len(context_sha256) != 64
            or not window or len(window) > 500):
        raise ValueError("dependency_provenance_arguments_invalid")
    # Revalidate typed objects to catch mutated nested state.
    snapshot = DataProvenanceSnapshot.model_validate(snapshot.model_dump(mode="json"))
    root = Path(source_root)
    if (not root.is_absolute() or ".." in root.parts
            or registered_source_identity(str(root)) != expected_identity
            or expected_identity.database != snapshot.source_database
            or source_config.name != snapshot.source_database
            or metadata_digest(source_config.model_dump(mode="json")) != snapshot.source_config_sha256):
        raise ValueError("dependency_source_provenance_mismatch")
    unknown: list[str] = []
    if snapshot.cohort_filters:
        unknown.append("dependency_cohort_filters_unknown")
    tables: dict[str, set[str]] = {}
    for concept in snapshot.concepts:
        if concept.status != "recorded":
            unknown.append("dependency_concept_unrecorded")
        if concept.derivation_callback:
            unknown.append("dependency_derivation_callback_unknown")
        for source in concept.sources:
            if source.callback:
                unknown.append("dependency_source_callback_unknown")
            if not source.table or source.table not in source_config.tables:
                unknown.append("dependency_source_table_unknown")
                continue
            columns = tables.setdefault(source.table, set())
            table = source_config.tables[source.table]
            columns.update(filter(None, (table.defaults.id_var, source.selector_variable,
                source.value_variable, source.unit_variable, source.time_variable, source.duration_variable)))
    requirements = []
    for table_name, columns in sorted(tables.items()):
        table = source_config.tables[table_name]
        paths = tuple(str(item.get("path") or item.get("file") or "") for item in table.files)
        formats = {Path(path).suffix.lower() for path in paths}
        dynamic = table.dataset is not None or not paths or not all(paths) or len(formats) != 1
        fmt = {".csv": "csv", ".parquet": "parquet"}.get(next(iter(formats), ""))
        if dynamic or not fmt:
            unknown.append("dependency_partition_roster_unknown")
            continue
        if not columns:
            unknown.append("dependency_columns_unknown")
            continue
        requirements.append(SourceTableRequirement(table=table_name, partitions=paths,
            columns=tuple(sorted(columns)), format=fmt,
            partition_roster_declared_complete=True, window=window))
    if not requirements:
        raise ValueError("dependency_source_tables_unknown")
    # Recheck registration after metadata projection; it is not authorization.
    if (registered_source_identity(str(root)) != expected_identity
            or metadata_digest(source_config.model_dump(mode="json")) != snapshot.source_config_sha256):
        raise ValueError("dependency_source_provenance_mismatch")
    return ProducerDependencyDeclaration(adapter=f"resolver:{snapshot.source_database}",
        source_identity_sha256=canonical_sha256(expected_identity.to_dict()), plan_sha256=plan.plan_sha256,
        context_sha256=context_sha256, configuration_sha256=snapshot.source_config_sha256,
        implementation_sha256=snapshot.implementation_sha256, requirements=tuple(requirements),
        unresolved_dependencies=tuple(dict.fromkeys(unknown)))


def _schema(root: Path, relative: str, file_format: str) -> tuple[tuple[str, str], ...]:
    path = PurePosixPath(relative)
    with AnchoredDirectory.open(root.joinpath(*path.parts[:-1])) as directory:
        with directory.open_regular(path.name) as stream:
            if file_format == "parquet":
                import pyarrow.parquet as pq
                arrow_schema = pq.read_schema(stream)
                columns = tuple((field.name, str(field.type)) for field in arrow_schema)
            else:
                header = stream.readline(1024 * 1024 + 1)
                if len(header) > 1024 * 1024 or not header:
                    raise ValueError("dependency_csv_header_invalid")
                columns = tuple((name, "untyped") for name in next(csv.reader(io.StringIO(header.decode("utf-8-sig")))))
        directory.assert_still_selected()
    names = [name for name, _ in columns]
    if not columns or len(set(names)) != len(names):
        raise ValueError("dependency_schema_invalid")
    return columns


def compile_producer_dependencies(*, source_root: Path,
                                  declaration: ProducerDependencyDeclaration) -> dict:
    """Check declared partitions' schemas, never open rows or widen a roster."""
    root = Path(source_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("dependency_source_invalid")
    blockers: list[str] = list(declaration.unresolved_dependencies)
    files: list[dict] = []
    seen = set()
    for item in declaration.requirements:
        if not item.partition_roster_declared_complete or not item.partitions:
            blockers.append("dependency_partition_roster_unknown")
        for name in item.partitions:
            if name in seen:
                blockers.append("dependency_partition_assigned_twice")
                continue
            seen.add(name)
            try:
                schema = _schema(root, name, item.format)
                if not set(item.columns).issubset(name for name, _ in schema):
                    blockers.append("dependency_columns_missing")
                    continue
                files.append({"table": item.table, "path": name, "format": item.format,
                              "columns": list(item.columns), "window": item.window,
                              "schema_sha256": canonical_sha256(list(schema))})
            except (AuthorityFilesystemError, OSError, ValueError, UnicodeError, csv.Error):
                blockers.append("dependency_schema_unavailable")
    body = {"schema_version": "easyicu.producer-dependencies/1",
            "declaration": declaration.model_dump(mode="json"), "files": files,
            "blocker_codes": list(dict.fromkeys(blockers)),
            # Candidate checks only: no authority or completeness is inferred.
            "source_registration_verified": False, "actual_consumption_verified": False,
            "producer_read_closure_verified": False}
    return {**body, "declaration_sha256": canonical_sha256(body)}


def pending_preparation_from_dependencies(*, plan: PlanReviewAuthority, dependencies: dict) -> RawPreparationCheckpoint:
    """Construct a pending proposal; this never records a human decision."""
    digest = dependencies.get("declaration_sha256")
    body = {k: v for k, v in dependencies.items() if k != "declaration_sha256"}
    if digest != canonical_sha256(body) or body.get("blocker_codes"):
        raise ValueError("dependency_declaration_not_reviewable")
    declaration = ProducerDependencyDeclaration.model_validate(body.get("declaration"))
    if declaration.plan_sha256 != plan.plan_sha256:
        raise ValueError("dependency_plan_mismatch")
    scope = {"purpose": "prepare_only", "adapter_id": declaration.adapter, "plan_sha256": plan.plan_sha256,
             "dependencies": dependencies, "analysis_authorized": False}
    payload = {"plan_review_authority": plan.model_dump(mode="json"), "raw_preparation_scope": scope}
    details = "; ".join(
        f"{item['table']}: {item['path']} ({', '.join(item['columns'])}); {item['window']}"
        for item in body["files"]
    )
    if len(details) > 700:
        details = details[:697] + "..."
    summary = ("Proposed preparation inputs: " + details
               + ". Actual producer reads and source registration remain unverified; no analysis is authorized.")
    request = HumanReviewRequest.create(kind="scientific_stop", summary=summary, payload=payload,
                                        authority_sha256=canonical_sha256(payload))
    return RawPreparationCheckpoint.pending(plan_sha256=plan.plan_sha256, request=request)


def candidate_from_scientific_review(*, plan_result, plan_authority: PlanReviewAuthority,
                                     run_dir: Path, source_root: Path, snapshot, source_config) -> dict:
    """Compile only an exact, evidence-verified plan/context into a proposal.

    Returns explicit blockers rather than a pending approval if dependencies are
    unknown. This does not persist a checkpoint or authorize patient data reads.
    """
    import json
    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    context = plan_result.context
    if (canonical_sha256(plan_result.plan.model_dump(mode="json")) != plan_authority.plan_sha256
            or canonical_sha256(plan_authority.model_dump(mode="json")["plan_payload"]) != plan_authority.plan_sha256):
        raise ValueError("dependency_plan_mismatch")
    record = plan_result.evidence.get("scientific_plan_review")
    path = verified_run_evidence_path(run_dir, record) if record is not None else None
    if path is None or plan_authority.evidence_sha256.get("scientific_plan_review") != record.sha256:
        raise ValueError("dependency_scientific_review_missing")
    review = PlanScientificReview.model_validate_json(path.read_bytes())
    context_sha = canonical_sha256(context.model_dump(mode="json"))
    if review.plan_sha256 != plan_authority.plan_sha256 or review.context_sha256 != context_sha:
        raise ValueError("dependency_scientific_review_mismatch")
    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if identity is None:
        raise ValueError("dependency_source_identity_missing")
    window_facts = {"time_windows": [item.model_dump(mode="json") for item in context.time_windows],
                    "variable_windows": {item.name: item.analysis_window for item in context.variables
                                         if item.analysis_window}}
    window = json.dumps(window_facts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if not window_facts["time_windows"] and not window_facts["variable_windows"]:
        raise ValueError("dependency_reviewed_window_missing")
    if len(window) > 500:
        raise ValueError("dependency_reviewed_window_too_large")
    declaration = declaration_from_provenance(plan=plan_authority, context_sha256=context_sha,
        snapshot=snapshot, source_config=source_config, source_root=source_root,
        expected_identity=identity, window=window)
    # Concept provenance is not a complete description of cohort/patient-id
    # selection. Until a producer owner supplies those dependencies as well,
    # even a plan with no explicit filters cannot yield a complete roster.
    declaration = declaration.model_copy(update={"unresolved_dependencies": (
        *declaration.unresolved_dependencies, "dependency_cohort_inputs_unknown")})
    recorded = {item.concept_id for item in snapshot.concepts if item.status == "recorded"}
    requested = {str(item.source_concept or item.name) for item in context.variables}
    if not requested.issubset(recorded):
        declaration = declaration.model_copy(update={"unresolved_dependencies": (
            *declaration.unresolved_dependencies, "dependency_context_concepts_unrecorded")})
    if not set(reviewed_plan_cohort_concepts(context=context, plan=plan_result.plan)).issubset(recorded):
        declaration = declaration.model_copy(update={"unresolved_dependencies": (
            *declaration.unresolved_dependencies, "dependency_cohort_concepts_unrecorded")})
    dependencies = compile_producer_dependencies(source_root=source_root, declaration=declaration)
    if dependencies["blocker_codes"]:
        return {"status": "blocked", "dependencies": dependencies, "pending": None}
    pending = pending_preparation_from_dependencies(plan=plan_authority, dependencies=dependencies)
    return {"status": "pending", "dependencies": dependencies, "pending": pending}
