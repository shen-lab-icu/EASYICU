"""Compile a producer's proposed input declaration without reading data rows.

Declarations are candidates, not proof that arbitrary code consumes only these
files. The trusted producer and its runtime still have to prove that closure.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path, PurePosixPath
from typing import Literal

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
