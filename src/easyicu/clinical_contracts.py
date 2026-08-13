"""Typed governance owner for high-risk clinical definitions.

Concept dictionaries describe extraction. This module separately states which
clinical definition/version a derived concept claims, which independent test
vector protects it, and how far cross-database conformance has actually been
demonstrated. A mapping-only database status must never be read as clinical
algorithm validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from .concept.schema import ConceptDictionary


_CONTRACT_PATH = Path(__file__).resolve().parent / "data" / "clinical-contracts.json"
_DATABASES = ("mimic", "miiv", "eicu", "aumc", "hirid", "sic")
_CONFORMANCE_LEVELS = {"not_assessed", "mapping_only", "algorithm_golden"}
_CLINICAL_STATUS_RANK = {
    "experimental": 0,
    "source_bound_golden": 1,
    "automated_conformance": 2,
    "validated_definition": 3,
}
_SOFA2_COMPONENT_CONTRACTS = {
    "sofa2_resp": "sofa2_resp_2025",
    "sofa2_coag": "sofa2_coag_2025",
    "sofa2_liver": "sofa2_liver_2025",
    "sofa2_cardio": "sofa2_cardio_2025",
    "sofa2_cns": "sofa2_cns_2025",
    "sofa2_renal": "sofa2_renal_2025",
}


@dataclass(frozen=True)
class ClinicalConceptContract:
    contract_id: str
    concepts: tuple[str, ...]
    definition: str
    version: str
    source_id: str
    source_table: Optional[str]
    definition_time_anchor: Optional[str]
    status: str
    canonical_definition: bool
    requires_explicit_opt_in: bool
    spec_golden_vectors: tuple[str, ...]
    runtime_golden_vectors: tuple[str, ...]
    test_vector_version: str
    validation_status: str
    validated_against: tuple[str, ...]
    reviewer: str
    last_reviewed_at: str
    reference_implementation: Optional[str]
    production_executor: Optional[str]
    production_callback: Optional[str]
    runtime_inputs: Mapping[str, str]
    reference_commit: Optional[str]
    depends_on_contracts: tuple[str, ...]
    ascertainment_limitations: tuple[str, ...]
    database_conformance: Mapping[str, str]

    @property
    def golden_vector(self) -> str:
        """Deprecated singular-vector compatibility surface."""

        vectors = self.runtime_golden_vectors or self.spec_golden_vectors
        return vectors[0] if vectors else ""

    @classmethod
    def from_mapping(cls, contract_id: str, payload: Mapping[str, Any]) -> "ClinicalConceptContract":
        def vector_paths(key: str) -> tuple[str, ...]:
            raw = payload.get(key)
            if raw is None and payload.get("golden_vector"):
                raw = payload["golden_vector"]
            if isinstance(raw, str):
                return (raw,)
            return tuple(str(item) for item in (raw or ()))

        return cls(
            contract_id=contract_id,
            concepts=tuple(str(item) for item in payload.get("concepts", ())),
            definition=str(payload.get("definition") or ""),
            version=str(payload.get("version") or ""),
            source_id=str(payload.get("source_id") or ""),
            source_table=(str(payload["source_table"]) if payload.get("source_table") else None),
            definition_time_anchor=(
                str(payload["definition_time_anchor"])
                if payload.get("definition_time_anchor")
                else None
            ),
            status=str(payload.get("status") or ""),
            canonical_definition=bool(payload.get("canonical_definition")),
            requires_explicit_opt_in=bool(payload.get("requires_explicit_opt_in")),
            spec_golden_vectors=vector_paths("spec_golden_vectors"),
            runtime_golden_vectors=vector_paths("runtime_golden_vectors"),
            test_vector_version=str(payload.get("test_vector_version") or ""),
            validation_status=str(payload.get("validation_status") or ""),
            validated_against=tuple(
                str(item) for item in payload.get("validated_against", ())
            ),
            reviewer=str(payload.get("reviewer") or ""),
            last_reviewed_at=str(payload.get("last_reviewed_at") or ""),
            reference_implementation=(
                str(payload["reference_implementation"])
                if payload.get("reference_implementation")
                else None
            ),
            production_executor=(
                str(payload["production_executor"])
                if payload.get("production_executor")
                else None
            ),
            production_callback=(
                str(payload["production_callback"])
                if payload.get("production_callback")
                else None
            ),
            runtime_inputs={
                str(key): str(value)
                for key, value in (payload.get("runtime_inputs") or {}).items()
            },
            reference_commit=(
                str(payload["reference_commit"])
                if payload.get("reference_commit")
                else None
            ),
            depends_on_contracts=tuple(
                str(item) for item in payload.get("depends_on_contracts", ())
            ),
            ascertainment_limitations=tuple(
                str(item) for item in payload.get("ascertainment_limitations", ())
            ),
            database_conformance={
                str(key): str(value)
                for key, value in (payload.get("database_conformance") or {}).items()
            },
        )


def load_clinical_contracts(path: Optional[Path] = None) -> dict[str, ClinicalConceptContract]:
    source = path or _CONTRACT_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("clinical_contract_registry_not_object")
    return {
        str(contract_id): ClinicalConceptContract.from_mapping(str(contract_id), row)
        for contract_id, row in payload.items()
        if isinstance(row, Mapping)
    }


def validate_clinical_contracts(
    dictionary: ConceptDictionary,
    *,
    repo_root: Path,
    contracts_path: Optional[Path] = None,
) -> list[str]:
    """Return stable findings; an empty list is the formal coverage gate."""

    contracts = load_clinical_contracts(contracts_path)
    findings: list[str] = []
    for contract_id, contract in contracts.items():
        if not contract.concepts:
            findings.append(f"{contract_id}:concepts_missing")
        if not contract.definition or not contract.version or not contract.source_id:
            findings.append(f"{contract_id}:definition_provenance_incomplete")
        if contract.definition_time_anchor is not None and not re.fullmatch(
            r"[a-z][a-z0-9_]{1,79}", contract.definition_time_anchor
        ):
            findings.append(f"{contract_id}:definition_time_anchor_invalid")
        if not contract.spec_golden_vectors:
            findings.append(f"{contract_id}:spec_golden_vectors_missing")
        for vector_path in contract.spec_golden_vectors:
            if not (repo_root / vector_path).is_file():
                findings.append(
                    f"{contract_id}:spec_golden_vector_missing:{vector_path}"
                )
        if not contract.runtime_golden_vectors:
            findings.append(f"{contract_id}:runtime_golden_vectors_missing")
        for vector_path in contract.runtime_golden_vectors:
            if not (repo_root / vector_path).is_file():
                findings.append(
                    f"{contract_id}:runtime_golden_vector_missing:{vector_path}"
                )
        if (
            not contract.test_vector_version
            or not contract.validation_status
            or not contract.validated_against
            or not contract.reviewer
            or not contract.last_reviewed_at
        ):
            findings.append(f"{contract_id}:validation_provenance_incomplete")
        if contract.status not in _CLINICAL_STATUS_RANK:
            findings.append(f"{contract_id}:clinical_status_invalid")
        for dependency_id in contract.depends_on_contracts:
            if dependency_id not in contracts:
                findings.append(f"{contract_id}:dependency_unknown:{dependency_id}")
        for database, level in contract.database_conformance.items():
            if database not in _DATABASES or level not in _CONFORMANCE_LEVELS:
                findings.append(f"{contract_id}:database_conformance_invalid:{database}")
        for concept_id in contract.concepts:
            definition = dictionary.get(concept_id)
            if definition is None:
                findings.append(f"{contract_id}:concept_missing:{concept_id}")
            elif definition.clinical_contract_id != contract_id:
                findings.append(f"{contract_id}:concept_binding_mismatch:{concept_id}")
            elif definition.canonical_definition != contract.canonical_definition:
                findings.append(
                    f"{contract_id}:canonical_definition_mismatch:{concept_id}"
                )

    for contract_id, contract in contracts.items():
        dependency_contracts = [
            contracts[dependency_id]
            for dependency_id in contract.depends_on_contracts
            if dependency_id in contracts
        ]
        if dependency_contracts and contract.status in _CLINICAL_STATUS_RANK:
            weakest = min(
                _CLINICAL_STATUS_RANK.get(dependency.status, -1)
                for dependency in dependency_contracts
            )
            if _CLINICAL_STATUS_RANK[contract.status] > weakest:
                findings.append(f"{contract_id}:status_exceeds_weakest_dependency")

    aggregate = contracts.get("sofa2_aggregate_2025")
    if aggregate is None:
        findings.append("sofa2_aggregate_2025:contract_missing")
    elif set(aggregate.depends_on_contracts) != set(_SOFA2_COMPONENT_CONTRACTS.values()):
        findings.append("sofa2_aggregate_2025:component_dependencies_incomplete")

    for concept_id, contract_id in _SOFA2_COMPONENT_CONTRACTS.items():
        contract = contracts.get(contract_id)
        if contract is None:
            findings.append(f"{contract_id}:contract_missing")
            continue
        if contract.production_executor != "easyicu.concept.ConceptResolver.load_concepts":
            findings.append(f"{contract_id}:production_executor_unbound")
        if contract.production_callback != "easyicu.concept.callbacks._callback_sofa_component":
            findings.append(f"{contract_id}:production_callback_unbound")
        definition = dictionary.get(concept_id)
        if definition is not None:
            if definition.clinical_status != contract.status:
                findings.append(f"{contract_id}:dictionary_status_mismatch:{concept_id}")
            declared_inputs = set(contract.runtime_inputs)
            dictionary_inputs = set(definition.sub_concepts)
            if declared_inputs != dictionary_inputs:
                findings.append(f"{contract_id}:runtime_inputs_mismatch:{concept_id}")
            for input_name, owner in contract.runtime_inputs.items():
                if owner != f"concept:{input_name}":
                    findings.append(
                        f"{contract_id}:runtime_input_owner_invalid:{input_name}"
                    )
                    continue
                input_definition = dictionary.get(input_name)
                if input_definition is None or not (
                    input_definition.sources
                    or input_definition.callback
                    or input_definition.sub_concepts
                ):
                    findings.append(
                        f"{contract_id}:runtime_input_unresolvable:{input_name}"
                    )

            for vector_path in contract.runtime_golden_vectors:
                fixture_path = repo_root / vector_path
                if fixture_path.is_file():
                    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                    fixture_inputs = set((fixture.get("inputs") or {}).keys())
                    for input_name in fixture_inputs - declared_inputs:
                        findings.append(
                            f"{contract_id}:runtime_golden_input_unowned:{input_name}"
                        )
    if aggregate is not None:
        if aggregate.production_executor != "easyicu.concept.ConceptResolver.load_concepts":
            findings.append("sofa2_aggregate_2025:production_executor_unbound")
        if aggregate.production_callback != "easyicu.concept.callbacks._callback_sofa2_score":
            findings.append("sofa2_aggregate_2025:production_callback_unbound")
        definition = dictionary.get("sofa2")
        if definition is not None:
            if definition.clinical_status != aggregate.status:
                findings.append("sofa2_aggregate_2025:dictionary_status_mismatch:sofa2")
            declared_inputs = set(aggregate.runtime_inputs)
            dictionary_inputs = set(definition.sub_concepts)
            if declared_inputs != dictionary_inputs:
                findings.append("sofa2_aggregate_2025:runtime_inputs_mismatch:sofa2")
            for input_name, owner in aggregate.runtime_inputs.items():
                if owner != f"concept:{input_name}":
                    findings.append(
                        f"sofa2_aggregate_2025:runtime_input_owner_invalid:{input_name}"
                    )
                    continue
                if dictionary.get(input_name) is None:
                    findings.append(
                        f"sofa2_aggregate_2025:runtime_input_unresolvable:{input_name}"
                    )
            for vector_path in aggregate.runtime_golden_vectors:
                fixture_path = repo_root / vector_path
                if fixture_path.is_file():
                    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                    fixture_inputs = set((fixture.get("components") or {}).keys())
                    for input_name in fixture_inputs - declared_inputs:
                        findings.append(
                            "sofa2_aggregate_2025:"
                            f"runtime_golden_input_unowned:{input_name}"
                        )

    for concept_id, definition in dictionary.items():
        if definition.clinical_status and not definition.clinical_contract_id:
            findings.append(f"{concept_id}:clinical_contract_missing")
        elif (
            definition.clinical_contract_id
            and definition.clinical_contract_id not in contracts
        ):
            findings.append(f"{concept_id}:clinical_contract_unknown")
    return sorted(set(findings))


def render_clinical_conformance_matrix_markdown() -> str:
    contracts = load_clinical_contracts()
    lines = [
        "# EasyICU clinical conformance matrix",
        "",
        "_Generated from `easyicu/data/clinical-contracts.json`. `mapping_only` means extraction wiring is covered; it does not claim that a database-specific clinical result has an independent gold-standard validation._",
        "",
        "| Contract | Concepts | Definition/version | Source | Status | Validation | Spec vectors | Runtime vectors | Production binding | Dependencies / limitations | "
        + " | ".join(_DATABASES)
        + " |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in _DATABASES) + " |",
    ]
    for contract in contracts.values():
        source = contract.source_id + (f" ({contract.source_table})" if contract.source_table else "")
        row = [
            f"`{contract.contract_id}`",
            ", ".join(f"`{item}`" for item in contract.concepts),
            f"{contract.definition} / {contract.version}",
            source,
            contract.status,
            contract.validation_status,
            ", ".join(f"`{item}`" for item in contract.spec_golden_vectors),
            ", ".join(f"`{item}`" for item in contract.runtime_golden_vectors),
            " → ".join(
                f"`{item}`"
                for item in (contract.production_executor, contract.production_callback)
                if item
            ) or "not bound",
            "; ".join(
                [
                    *(
                        f"runtime `{name}` owned by `{owner}`"
                        for name, owner in contract.runtime_inputs.items()
                    ),
                    *(f"depends on `{item}`" for item in contract.depends_on_contracts),
                    *contract.ascertainment_limitations,
                ]
            ) or "none declared",
        ]
        row.extend(contract.database_conformance.get(db, "not_assessed") for db in _DATABASES)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


__all__ = [
    "ClinicalConceptContract",
    "load_clinical_contracts",
    "render_clinical_conformance_matrix_markdown",
    "validate_clinical_contracts",
]


if __name__ == "__main__":  # pragma: no cover - documentation generator
    print(render_clinical_conformance_matrix_markdown(), end="")
