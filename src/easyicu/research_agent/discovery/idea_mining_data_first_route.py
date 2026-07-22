"""Standard Idea Mining route for data-first candidate generation.

The leaf generator in :mod:`idea_mining_data_first` finds concept pairs that
are genuinely resolvable across the harmonized ICU databases.  This module
adapts those pairs into the existing Idea Mining pipeline; it does *not* create
an alternative ledger or a shortcut around novelty, feasibility, or human
confirmation.

The route is deliberately provider-free for candidate generation.  The only
network dependency is the caller-supplied prior-art search client.  Every
candidate is backed by a frozen data-profile excerpt containing the prepared
cohort SHA, then passes through the normal concept mapping, real-data
feasibility probe, PubMed assessment, registry, discovery ledger, and handoff
contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..concept_availability import (
    default_public_databases,
    hypothesis_cross_database_feasibility,
)
from ..literature import CitationRecord
from ..providers.protocol import LLMMessage
from ..schema import ConceptDescriptor
from .idea_mining import freeze_source_snapshot, run_idea_mining_dry_run
from .idea_mining_data_first import DataFirstCandidate, generate_data_first_candidates
from .idea_mining_schema import (
    IdeaMiningDryRunResult,
    LiteratureIdeaCandidate,
    OutcomeDeterminability,
    SourceMaterial,
)

DATA_FIRST_ROUTE_SCHEMA_VERSION = "easyicu.data_first_discovery_route/2"
DATA_FIRST_SHORTLIST_SCHEMA_VERSION = "easyicu.data_first_review_shortlist/1"

_MIN_EXTERNAL_VALIDATION_COMPLETENESS = 0.90
_MIN_MEASUREMENT_AUDIT_COMPLETENESS = 0.20
_MAX_MEASUREMENT_AUDIT_COMPLETENESS = 0.70
_MAX_EXACT_HITS_FOR_EXTERNAL_VALIDATION_REVIEW = 25


class _ProviderCallForbidden:
    """Sentinel LLM proving the deterministic route never calls a provider."""

    name = "data-first-provider-forbidden"
    __easyicu_mock_client__ = True

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ) -> str:
        del messages, max_tokens, temperature, seed
        raise RuntimeError("data-first discovery must not call an LLM provider")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_evidence_text(
    candidate: DataFirstCandidate,
    *,
    data_sha256: str,
) -> str:
    databases = ", ".join(candidate.harmonized_databases)
    return (
        "EasyICU data-first profile: predictor "
        f"'{candidate.predictor}' and outcome '{candidate.outcome}' are "
        f"dictionary-resolvable with full availability in "
        f"{candidate.harmonized_db_count}/{candidate.total_databases} harmonized "
        f"public ICU databases ({databases}). The prepared cohort used for "
        f"joint-feasibility probing has SHA-256 {data_sha256}. This is a "
        "measurability signal, not a novelty claim."
    )


def _source_materials_and_ideas(
    candidates: Sequence[DataFirstCandidate],
    *,
    data_path: Path,
    data_sha256: str,
    literature_terms: Mapping[str, str],
    literature_aliases: Mapping[str, Sequence[str]],
    analysis_family: str,
) -> tuple[list[SourceMaterial], list[LiteratureIdeaCandidate]]:
    materials: list[SourceMaterial] = []
    evidence_by_key: dict[str, str] = {}
    for candidate in candidates:
        key_digest = hashlib.sha256(
            (
                f"{candidate.predictor}\0{candidate.outcome}\0{data_sha256}\0"
                + "\0".join(candidate.harmonized_databases)
            ).encode("utf-8")
        ).hexdigest()[:16]
        citation_key = f"easyicu_data_profile_{key_digest}"
        evidence_text = _candidate_evidence_text(
            candidate,
            data_sha256=data_sha256,
        )
        evidence_by_key[citation_key] = evidence_text
        materials.append(
            SourceMaterial(
                citation=CitationRecord(
                    key=citation_key,
                    title=(
                        "EasyICU harmonized data profile: "
                        f"{candidate.predictor} -> {candidate.outcome}"
                    ),
                    year="2026",
                    venue="EasyICU host-owned data profile",
                    relevance=(
                        "Frozen cross-database measurability evidence; not a "
                        "literature novelty source."
                    ),
                ),
                source_adapter_level="user_supplied_excerpt",
                locator=str(data_path),
                source_text=evidence_text,
                discovery_route="data_first",
                source_text_role="data_profile_evidence",
            )
        )

    snapshot = freeze_source_snapshot(materials)
    ideas: list[LiteratureIdeaCandidate] = []
    for candidate, material in zip(candidates, materials):
        evidence_text = evidence_by_key[material.citation.key]
        predictor_phrase = str(
            literature_terms.get(candidate.predictor) or candidate.predictor
        ).strip()
        outcome_phrase = str(
            literature_terms.get(candidate.outcome) or candidate.outcome
        ).strip()
        ideas.append(
            LiteratureIdeaCandidate(
                source_snapshot_id=snapshot.source_snapshot_id,
                citation_key=material.citation.key,
                source_adapter_level=material.source_adapter_level,
                population="adult ICU stays in harmonized public ICU databases",
                exposure_or_predictor=predictor_phrase,
                outcome=outcome_phrase,
                rationale=(
                    candidate.differentiator_note
                    + " The association itself remains unproven until prior-art "
                    "and clinical review are complete."
                ),
                source_quote=evidence_text,
                analysis_family=analysis_family,
                exposure_core_concept=candidate.predictor,
                outcome_core_concept=candidate.outcome,
                exposure_literature_aliases=list(
                    literature_aliases.get(candidate.predictor, ())
                ),
                outcome_literature_aliases=list(
                    literature_aliases.get(candidate.outcome, ())
                ),
            )
        )
    return materials, ideas


def _review_shortlist(result: IdeaMiningDryRunResult) -> list[dict[str, Any]]:
    """Select bounded, differentiated human-review routes from standard rows.

    The shortlist is deliberately not a second novelty or acceptance gate.  It
    only decides where scarce human review effort should go after every idea has
    passed the standard mapping, real-data, PubMed, registry, and discovery
    ledger.  Route selection is semantic and name-neutral:

    * high-completeness pairs with few exact same-topic hits are candidates for
      cross-database external validation;
    * measurable but substantially incomplete pairs are candidates for a
      cross-database measurement/missingness audit, not an association claim;
      saturation of association papers does not answer that different audit.

    Both routes remain ``requires_human_confirmation`` and explicitly require a
    temporal protocol before any outcome analysis.
    """

    external_validation: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    measurement_audit: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for record in result.discovery_records:
        if record.executable_candidate_id is None or record.go_no_go == "db-cannot-do":
            continue
        feasibility = record.database_feasibility
        numerator = feasibility.get("n_joint_complete")
        denominator = feasibility.get("denominator_n")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            continue
        if denominator <= 0 or numerator < 0 or numerator > denominator:
            continue
        completeness = numerator / denominator
        exact_records = [
            query
            for query in record.prior_art.query_records
            if query.query_type == "exact"
        ]
        if not exact_records or not all(query.search_ok for query in exact_records):
            continue
        exact_hits = sum(query.hit_count for query in exact_records)
        base = {
            "literature_idea_id": record.literature_idea_id,
            "executable_candidate_id": record.executable_candidate_id,
            "candidate_topic": record.candidate_topic,
            "go_no_go": record.go_no_go,
            "exact_same_topic_hit_count": exact_hits,
            "joint_complete_n": numerator,
            "denominator_n": denominator,
            "joint_completeness": round(completeness, 6),
            "requires_human_confirmation": True,
            "paper_authorized": False,
        }
        cross_db_prior_art = int(
            record.prior_art.evidence_map_counts.get(
                "same_topic_cross_database_or_external", 0
            )
        )
        if (
            completeness >= _MIN_EXTERNAL_VALIDATION_COMPLETENESS
            and cross_db_prior_art == 0
            and exact_hits <= _MAX_EXACT_HITS_FOR_EXTERNAL_VALIDATION_REVIEW
        ):
            payload = {
                **base,
                "review_route": "cross_database_external_validation",
                "selection_reason": (
                    "high prepared-cohort completeness, <=25 exact same-topic "
                    "PubMed hits, and no retrieved title classified as comparable "
                    "cross-database/external-validation work"
                ),
                "required_next_action": (
                    "human-review the exact hits and licensed/non-PubMed literature; "
                    "then define a pre-outcome predictor ascertainment window and "
                    "database-specific transportability estimand"
                ),
            }
            external_validation.append(
                ((exact_hits, -completeness, record.candidate_topic), payload)
            )
        elif (
            _MIN_MEASUREMENT_AUDIT_COMPLETENESS
            <= completeness
            < _MAX_MEASUREMENT_AUDIT_COMPLETENESS
        ):
            payload = {
                **base,
                "review_route": "cross_database_measurement_bias_audit",
                "selection_reason": (
                    "dictionary-resolvable across harmonized databases but only "
                    "20-70% jointly observed in the prepared cohort; association "
                    "literature saturation does not answer the separate "
                    "measurement/source-status audit question"
                ),
                "required_next_action": (
                    "profile source-status and missingness in the existing full6 "
                    "exports before defining any outcome association; do not treat "
                    "unmeasured values as clinical absence"
                ),
            }
            measurement_audit.append(
                ((exact_hits, completeness, record.candidate_topic), payload)
            )

    external_validation.sort(key=lambda item: item[0])
    measurement_audit.sort(key=lambda item: item[0])
    selected: list[dict[str, Any]] = []
    if external_validation:
        selected.append(external_validation[0][1])
    if measurement_audit:
        selected.append(measurement_audit[0][1])
    return selected


def run_data_first_idea_mining_dry_run(
    *,
    predictor_concepts: Sequence[str],
    outcome_concepts: Sequence[str],
    available_concepts: Sequence[ConceptDescriptor | str],
    output_dir: str | Path,
    data_path: str | Path,
    prior_art_search_client: Any,
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    outcome_determinability: Optional[
        Mapping[str, OutcomeDeterminability | Mapping[str, Any] | str]
    ] = None,
    databases: Optional[Sequence[str]] = None,
    database: str = "miiv",
    feasibility_probe: Optional[Any] = None,
    min_harmonized_dbs: int = 4,
    top_k: int = 25,
    prior_art_top_n: int = 20,
    source_item_index: Optional[Any] = None,
    cross_database_feasibility_fn: Any = hypothesis_cross_database_feasibility,
    literature_terms: Optional[Mapping[str, str]] = None,
    literature_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    analysis_family: str = "cross_database_replication",
) -> IdeaMiningDryRunResult:
    """Run deterministic data-first generation through the standard funnel.

    No pair is promoted because it is merely measurable.  Cross-database
    availability creates the candidate set; the existing pair-level real-data
    probe and prior-art gate determine whether each row is executable and
    differentiated.  All rows remain proposed and require human confirmation.
    """

    prepared_data = Path(data_path).resolve()
    if not prepared_data.is_file():
        raise FileNotFoundError(f"prepared data path not found: {prepared_data}")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_sha256 = _sha256_file(prepared_data)

    effective_databases = (
        list(databases) if databases is not None else default_public_databases()
    )
    candidates = generate_data_first_candidates(
        predictor_concepts=predictor_concepts,
        outcome_concepts=outcome_concepts,
        databases=effective_databases,
        feasibility_fn=cross_database_feasibility_fn,
        min_harmonized_dbs=min_harmonized_dbs,
        limit=max(len(predictor_concepts) * max(len(outcome_concepts), 1), top_k),
    )
    materials, ideas = _source_materials_and_ideas(
        candidates,
        data_path=prepared_data,
        data_sha256=data_sha256,
        literature_terms=literature_terms or {},
        literature_aliases=literature_aliases or {},
        analysis_family=analysis_family,
    )

    result = run_idea_mining_dry_run(
        materials=materials,
        precomputed_literature_ideas=ideas,
        llm=_ProviderCallForbidden(),
        available_concepts=available_concepts,
        concept_aliases=concept_aliases,
        outcome_determinability=outcome_determinability,
        output_dir=out_dir,
        database=database,
        data_path=prepared_data,
        analytic_unit="stay",
        feasibility_probe=feasibility_probe,
        top_k=top_k,
        prior_art_search_client=prior_art_search_client,
        prior_art_top_n=prior_art_top_n,
        source_item_index=source_item_index,
        cross_db_targets=effective_databases,
    )

    shortlist = _review_shortlist(result)
    shortlist_path = out_dir / "data_first_review_shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "schema_version": DATA_FIRST_SHORTLIST_SCHEMA_VERSION,
                "selection_scope": (
                    "bounded human-review prioritization; not a novelty, go, or "
                    "paper-authorization decision"
                ),
                "candidates": shortlist,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    route_manifest = {
        "schema_version": DATA_FIRST_ROUTE_SCHEMA_VERSION,
        "prepared_data_path": str(prepared_data),
        "prepared_data_sha256": data_sha256,
        "min_harmonized_dbs": min_harmonized_dbs,
        "harmonized_databases_considered": effective_databases,
        "predictor_concepts_considered": list(predictor_concepts),
        "outcome_concepts_considered": list(outcome_concepts),
        "analysis_family": analysis_family,
        "literature_terms": dict(literature_terms or {}),
        "literature_aliases": {
            key: list(values) for key, values in (literature_aliases or {}).items()
        },
        "data_first_candidates": [candidate.__dict__ for candidate in candidates],
        "candidate_triage_report": result.triage_report_path,
        "review_shortlist": str(shortlist_path),
        "review_shortlist_count": len(shortlist),
    }
    (out_dir / "data_first_route_manifest.json").write_text(
        json.dumps(route_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


__all__ = [
    "DATA_FIRST_ROUTE_SCHEMA_VERSION",
    "DATA_FIRST_SHORTLIST_SCHEMA_VERSION",
    "run_data_first_idea_mining_dry_run",
]
