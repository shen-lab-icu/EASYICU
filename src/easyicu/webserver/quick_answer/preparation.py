"""Prepare the conversation's registered export through existing data owners.

Only the host supplies the export path. The typed reader chooses no source,
provider, code, values or approval. Qualification stays in planning.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import assess_study
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.planning.dependence_authority import context_patient_group_authority
from easyicu.research_agent.planning.population_compile import population_spec_concepts
from easyicu.research_agent.planning.population_spec import FirstIcuStay
from easyicu.research_agent.providers.client_trust import require_credentialed_data_clearance
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver import dataio, source_identity_authority, sources

from .contracts import QuickProposal, QuickReading, QuickStop, StudyHandoff
from .planning import _effective, _requirements, bind_concept_names, prepare_quick_answer, qualification_stop
from .source import PreparedSource, bind_prepared_source

MAX_PREPARATION_BYTES = 256 * 1024 * 1024


def _preparation_budget(receipt, export_path):
    import pyarrow.parquet as pq

    size = 0
    for file in receipt["binding"]["files"]:
        path = Path(export_path) / file["relative_path"]
        if path.suffix != ".parquet":
            raise QuickStop("quick_prepared_source_required")
        with pq.ParquetFile(path) as parquet:
            metadata = parquet.metadata
        size += sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        if metadata.num_rows > 2_000_000 or size > MAX_PREPARATION_BYTES:
            raise QuickStop("quick_prepared_source_too_large")


def _verified_identity(export_path, database, population):
    """The source's verified patient grouping and, when asked, first-stay coordinate.

    Both come from the owners a full study binds; a source without one (or
    whose authority fails its checks) is unverified, never guessed.
    """
    authority = source_identity_authority
    try:
        grouping = authority.resolve_study_patient_grouping(export_path=export_path, database=database)
    except authority.PatientGroupingAuthorityError:
        grouping = None
    first_stay = None
    if any(isinstance(item, FirstIcuStay) for item in population.criteria):
        try:
            first_stay = authority.resolve_study_first_icu_stay(export_path=export_path, database=database)
        except authority.PatientGroupingAuthorityError:
            first_stay = None
    return grouping, first_stay


def _first_stays_orderable(export_path, database) -> bool:
    """Whether the bound raw source can order each patient's ICU stays.

    A repeated-patient stop offers "ask about each patient's first ICU stay"
    only where that criterion can be prepared (``resolve_study_first_icu_stay``).
    """
    from easyicu.webserver import raw_source_authority

    try:
        binding = raw_source_authority.resolve_raw_source_identity_binding(
            export_path=export_path, database=database,
        )
    except raw_source_authority.RawSourceAuthorityError:
        return False
    return bool(binding is not None and getattr(binding, "first_icu_stay_available", False))


def prepare_registered_question(
    reading: QuickReading, *, question: str, export_path: str,
    database: str, output_dir: Path, provider: object,
) -> tuple[QuickProposal, PreparedSource | None]:
    """Materialize only a qualified reading, without a model or extraction run."""
    identity = sources.registered_source_identity(export_path)
    if identity is None or identity.database != database:
        raise QuickStop("quick_source_identity_mismatch")
    reading = bind_concept_names(reading)
    if question != reading.question:
        reading = reading.model_copy(update={
            "question": question, "ambiguities": (*reading.ambiguities, "question_mismatch"),
        })
    handoff = StudyHandoff(reading=reading, source=identity.to_dict())
    reason = qualification_stop(reading)
    if reason:
        return QuickProposal(status="full_study", reason=reason, expand=handoff), None
    # The same provider owner as study setup must clear the source before any
    # materialization. Public-demo provenance is scoped by the caller.
    from easyicu.research_agent.providers.client_trust import ProviderConfigurationError, source_requires_clearance

    try:
        if provider is None and source_requires_clearance(database):
            raise QuickStop("quick_qualifying_provider_required")
        require_credentialed_data_clearance(provider, database)
        decision = assess_study(research_capability_profile(database), _requirements(reading))
        if decision.status == "refusal":
            return QuickProposal(status="stopped", reason="quick_capability_refused", expand=handoff,
                                 capability=decision), None
        effective = _effective(reading, decision)
        receipt = dataio.validate_research_pipeline_source(export_path, database=database)
        # Bound local preparation before a wide table can be allocated. Larger
        # registered packages retain the normal study path.
        _preparation_budget(receipt, export_path)
        concepts = tuple(dict.fromkeys(filter(None, (
            effective.measure.concept, *effective.grouping, *population_spec_concepts(effective.population),
        ))))
        # The existing materializer establishes its stay universe from the
        # static demographic module, preserving stays with missing measures.
        static = tuple(dict.fromkeys(("age", *(c for c in concepts if c != effective.measure.concept or effective.measure.summary == "value"))))
        window = effective.measure.window
        grouping, first_stay = _verified_identity(export_path, database, effective.population)
        acquired = acquire_universe_for_question(
            export_dir=export_path, question=reading.question, llm=None, output_dir=output_dir,
            database=database, target_outcome=None, outcome_concepts=(), require_outcome=False,
            required_feature_concepts=tuple(c for c in concepts if c not in static),
            static_concepts=static,
            concept_selection_authority="host_exact", emit_trajectory=False,
            cohort_window=(window.start_hours, window.end_hours) if window else (0.0, 24.0),
            patient_grouping=grouping, first_icu_stay=first_stay,
        )
        if acquired.blocked or acquired.universe_path is None:
            raise QuickStop("quick_prepared_source_required")
        # A grouped universe keys each stay as p<patient>:s<stay>.
        unit = grouping.output_identity_column if grouping is not None else "stay_id"
        context = build_research_context(
            research_question=reading.question, cohort=acquired.universe_path,
            cohort_name="quick_answer", database=database, id_columns=[unit],
            outcome_columns=[], user_preferences={"data_constraints": json.dumps({
                "source_identity": identity.to_dict(),
                "first_icu_stays_orderable": grouping is not None and _first_stays_orderable(export_path, database),
            })},
        )
        issued = context_patient_group_authority(context)
        source = bind_prepared_source(
            acquired.universe_path, context=context, identity=identity, unit_column=unit,
            patient_column=issued.group_source if issued is not None else None,
        )
        return prepare_quick_answer(reading, source=source, provider=provider, question=question), source
    except ProviderConfigurationError:
        reason = "quick_qualifying_provider_required"
    except QuickStop as exc:
        reason = exc.code
    except (OSError, ValueError, RuntimeError, dataio.ExportCohortError):
        reason = "quick_prepared_source_required"
    return QuickProposal(status="stopped", reason=reason, expand=handoff), None
