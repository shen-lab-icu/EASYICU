"""Read-only deterministic preflight for Track 3 literature hard dependency.

The gate checks that a plan binds retrieval evidence for the three hard
decisions (population / outcome-window / method applicability):

* which library was searched (``searches[].source``),
* which record was hit (``hits[].citation_key`` + returning query),
* which sentence supports which decision
  (``supports[].supporting_statement`` for one decision + citation key),
* and that the plan itself cites the same retrieved key through a compatible
  design element or seven-dimension decision.

Two-tier honesty (review finding): the error tier above verifies retrieval
lineage plus citation binding only. Whether the recorded sentence's content
actually comes from the literature is a separate question, answered per
decision by reviewed full-text design evidence
(``contract.source_backed``): decisions covered without source backing get
a ``warning`` finding (never a block), so reviewers see exactly which
claims lack source verification.

Read-only: returns :class:`ValidationFinding` items only. Performs no Provider
call, no network I/O, no filesystem access, and no authority or evidence
mutation. Offline/mock callers reuse the existing opt-in exemption only for
transport (stubbed PubMed client + mock LLM choice need no new opt-in); the
verdict itself stays strict so cutting retrieval still blocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from easyicu.ai_optin import is_offline_llm_choice

from ..planning.literature_retrieval_contract import (
    LITERATURE_RETRIEVAL_DECISIONS,
    LiteratureRetrievalContract,
    decision_for_design_dimension,
    decision_for_design_element,
    is_bibliographic_source,
)
from ..schema import ValidationFinding

VALIDATOR = "literature_retrieval_gate"


def _plan_bound_keys(plan: Any) -> set[str]:
    bound: set[str] = set()
    for step in list(getattr(plan, "steps", None) or []):
        for key in list(getattr(step, "literature_citation_keys", None) or []):
            text = str(key or "").strip()
            if text:
                bound.add(text)
    selection = getattr(plan, "design_selection", None)
    candidates = list(getattr(selection, "candidates", None) or []) if selection is not None else []
    for candidate in candidates:
        if str(getattr(candidate, "disposition", "") or "") != "selected":
            continue
        for key in list(getattr(candidate, "literature_citation_keys", None) or []):
            text = str(key or "").strip()
            if text:
                bound.add(text)
    return bound


def _plan_decision_statements(plan: Any) -> Dict[str, List[Dict[str, str]]]:
    """Project plan bindings onto hard decisions as decision -> [{key, statement}]."""

    coverage: Dict[str, List[Dict[str, str]]] = {
        decision: [] for decision in LITERATURE_RETRIEVAL_DECISIONS
    }
    for step in list(getattr(plan, "steps", None) or []):
        for binding in list(getattr(binding_holder(step), "literature_design_bindings", None) or []):
            key = str(getattr(binding, "citation_key", "") or "").strip()
            statement = str(getattr(binding, "application", "") or "").strip()
            if not key:
                continue
            for element in list(getattr(binding, "design_elements", None) or []):
                decision = decision_for_design_element(str(element or ""))
                if decision is not None:
                    coverage[decision].append({"key": key, "statement": statement})
    selection = getattr(plan, "design_selection", None)
    candidates = list(getattr(selection, "candidates", None) or []) if selection is not None else []
    for candidate in candidates:
        if str(getattr(candidate, "disposition", "") or "") != "selected":
            continue
        for item in list(getattr(candidate, "literature_design_decisions", None) or []):
            decision = decision_for_design_dimension(
                str(getattr(item, "dimension", "") or "")
            )
            if decision is None:
                continue
            rationale = str(getattr(item, "rationale", "") or "").strip()
            for key in list(getattr(item, "citation_keys", None) or []):
                text = str(key or "").strip()
                if text:
                    coverage[decision].append({"key": text, "statement": rationale})
    return coverage


def binding_holder(step: Any) -> Any:
    """Return the object carrying literature bindings (identity for steps)."""
    return step


def literature_retrieval_findings(
    *,
    plan: Any,
    contract: LiteratureRetrievalContract,
    llm_choice: str | None = None,
) -> List[ValidationFinding]:
    """Return blocking findings when retrieval evidence is missing.

    Pure function of its inputs. ``llm_choice`` only records whether the
    caller ran under the existing offline/mock exemption; it never relaxes
    the verdict.
    """

    offline_transport = bool(
        is_offline_llm_choice(llm_choice) if llm_choice is not None else False
    )
    base_detail: Dict[str, Any] = {
        "provider_called": False,
        "authority_mutated": False,
        "offline_mock_transport": offline_transport,
        "search_conducted": bool(contract.search_conducted),
        "sources_returning": list(contract.sources_returning),
    }
    findings: List[ValidationFinding] = []

    if not contract.search_conducted:
        findings.append(
            ValidationFinding(
                validator=VALIDATOR,
                severity="error",
                message=(
                    "Plan finalization requires conducted bibliographic retrieval "
                    "for population, outcome-window, and method-applicability "
                    "decisions; this contract records no conducted search."
                ),
                detail={
                    **base_detail,
                    "kind": "literature_search_not_conducted",
                    "missing_decisions": list(LITERATURE_RETRIEVAL_DECISIONS),
                },
            )
        )
        return findings
    if not contract.searches:
        findings.append(
            ValidationFinding(
                validator=VALIDATOR,
                severity="error",
                message=(
                    "Plan finalization requires recorded retrieval queries "
                    "(source + exact query); none are present."
                ),
                detail={
                    **base_detail,
                    "kind": "literature_search_queries_missing",
                    "missing_decisions": list(LITERATURE_RETRIEVAL_DECISIONS),
                },
            )
        )
        return findings
    if not contract.hits:
        findings.append(
            ValidationFinding(
                validator=VALIDATOR,
                severity="error",
                message=(
                    "Bibliographic retrieval returned no bound hits; each hard "
                    "decision needs at least one retrieved record."
                ),
                detail={
                    **base_detail,
                    "kind": "literature_hits_missing",
                    "missing_decisions": list(LITERATURE_RETRIEVAL_DECISIONS),
                },
            )
        )
        return findings
    bibliographic_hits = [
        hit for hit in contract.hits if is_bibliographic_source(hit.source)
    ]
    if not bibliographic_hits:
        findings.append(
            ValidationFinding(
                validator=VALIDATOR,
                severity="error",
                message=(
                    "Retrieved hits come from no bibliographic source "
                    "(need pubmed, tavily, or bound_search lineage)."
                ),
                detail={
                    **base_detail,
                    "kind": "literature_source_not_bibliographic",
                    "hit_sources": sorted({hit.source for hit in contract.hits}),
                    "missing_decisions": list(LITERATURE_RETRIEVAL_DECISIONS),
                },
            )
        )
        return findings

    hit_keys = {hit.citation_key for hit in bibliographic_hits}
    supports_by_decision: Dict[str, List[Any]] = {
        decision: [] for decision in LITERATURE_RETRIEVAL_DECISIONS
    }
    for support in contract.supports:
        if support.citation_key not in hit_keys:
            continue
        supports_by_decision[support.decision].append(support)

    plan_keys = _plan_bound_keys(plan)
    plan_coverage = _plan_decision_statements(plan)

    unbound = sorted(
        {
            support.citation_key
            for supports in supports_by_decision.values()
            for support in supports
            if support.citation_key not in plan_keys
        }
    )
    if unbound:
        findings.append(
            ValidationFinding(
                validator=VALIDATOR,
                severity="error",
                message=(
                    "Retrieval supports records the plan never binds; cite each "
                    "supporting record in plan literature_citation_keys: "
                    + ", ".join(unbound)
                    + "."
                ),
                detail={
                    **base_detail,
                    "kind": "literature_support_not_bound_by_plan",
                    "unbound_keys": unbound,
                    "plan_bound_keys": sorted(plan_keys),
                },
            )
        )

    backed_by_key = getattr(contract, "source_backed", None) or {}
    for decision in LITERATURE_RETRIEVAL_DECISIONS:
        plan_keys_for_decision = {
            item["key"]
            for item in plan_coverage.get(decision, [])
            if item["key"] in hit_keys
        }
        supported_keys = {
            support.citation_key for support in supports_by_decision.get(decision, [])
        }
        covering = sorted(plan_keys_for_decision & supported_keys)
        if not covering:
            findings.append(
                ValidationFinding(
                    validator=VALIDATOR,
                    severity="error",
                    message=(
                        f"Hard decision {decision!r} has no retrieval-backed plan "
                        "binding: need one retrieved record, one recorded query, "
                        "one supporting sentence, and a matching plan citation."
                    ),
                    detail={
                        **base_detail,
                        "kind": "literature_decision_without_retrieval",
                        "missing_decision": decision,
                        "contract_supported_keys": sorted(supported_keys),
                        "plan_keys_for_decision": sorted(plan_keys_for_decision),
                        "retrieved_hit_keys": sorted(hit_keys),
                    },
                )
            )
            continue
        # Review finding: a recorded supporting sentence is not proof its
        # content came from the literature. What this gate verifies is
        # retrieval lineage plus citation binding; source backing (reviewed
        # full-text design evidence for the cited record and decision) is
        # reported here as a warning so reviewers see the exact gap.
        backed = [
            key
            for key in covering
            if decision in list(backed_by_key.get(key) or [])
        ]
        if not backed:
            findings.append(
                ValidationFinding(
                    validator=VALIDATOR,
                    severity="warning",
                    message=(
                        f"Hard decision {decision!r} is citation-bound but no "
                        "cited record carries reviewed full-text design "
                        "evidence for it; the supporting sentence is "
                        "recorded, not source-verified. Add a design "
                        "evidence card to clear this."
                    ),
                    detail={
                        **base_detail,
                        "kind": "literature_support_not_source_backed",
                        "missing_decision": decision,
                        "covering_keys": covering,
                    },
                )
            )
    return findings


def literature_retrieval_gate_blocks(
    findings: Sequence[ValidationFinding],
) -> bool:
    """Return True when findings contain a blocking error from this gate."""
    return any(
        finding.validator == VALIDATOR and finding.severity == "error"
        for finding in findings
    )


__all__ = [
    "VALIDATOR",
    "literature_retrieval_findings",
    "literature_retrieval_gate_blocks",
]
