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
from typing import Any, Mapping, Optional

from .concept.schema import ConceptDictionary


_CONTRACT_PATH = Path(__file__).resolve().parent / "data" / "clinical-contracts.json"
_DATABASES = ("mimic", "miiv", "eicu", "aumc", "hirid", "sic")
_CONFORMANCE_LEVELS = {"not_assessed", "mapping_only", "algorithm_golden"}


@dataclass(frozen=True)
class ClinicalConceptContract:
    contract_id: str
    concepts: tuple[str, ...]
    definition: str
    version: str
    source_id: str
    source_table: Optional[str]
    status: str
    canonical_definition: bool
    requires_explicit_opt_in: bool
    golden_vector: str
    test_vector_version: str
    validation_status: str
    validated_against: tuple[str, ...]
    reviewer: str
    last_reviewed_at: str
    reference_implementation: Optional[str]
    reference_commit: Optional[str]
    database_conformance: Mapping[str, str]

    @classmethod
    def from_mapping(cls, contract_id: str, payload: Mapping[str, Any]) -> "ClinicalConceptContract":
        return cls(
            contract_id=contract_id,
            concepts=tuple(str(item) for item in payload.get("concepts", ())),
            definition=str(payload.get("definition") or ""),
            version=str(payload.get("version") or ""),
            source_id=str(payload.get("source_id") or ""),
            source_table=(str(payload["source_table"]) if payload.get("source_table") else None),
            status=str(payload.get("status") or ""),
            canonical_definition=bool(payload.get("canonical_definition")),
            requires_explicit_opt_in=bool(payload.get("requires_explicit_opt_in")),
            golden_vector=str(payload.get("golden_vector") or ""),
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
            reference_commit=(
                str(payload["reference_commit"])
                if payload.get("reference_commit")
                else None
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
) -> list[str]:
    """Return stable findings; an empty list is the formal coverage gate."""

    contracts = load_clinical_contracts()
    findings: list[str] = []
    for contract_id, contract in contracts.items():
        if not contract.concepts:
            findings.append(f"{contract_id}:concepts_missing")
        if not contract.definition or not contract.version or not contract.source_id:
            findings.append(f"{contract_id}:definition_provenance_incomplete")
        if not contract.golden_vector or not (repo_root / contract.golden_vector).is_file():
            findings.append(f"{contract_id}:golden_vector_missing")
        if (
            not contract.test_vector_version
            or not contract.validation_status
            or not contract.validated_against
            or not contract.reviewer
            or not contract.last_reviewed_at
        ):
            findings.append(f"{contract_id}:validation_provenance_incomplete")
        for database, level in contract.database_conformance.items():
            if database not in _DATABASES or level not in _CONFORMANCE_LEVELS:
                findings.append(f"{contract_id}:database_conformance_invalid:{database}")
        for concept_id in contract.concepts:
            definition = dictionary.get(concept_id)
            if definition is None:
                findings.append(f"{contract_id}:concept_missing:{concept_id}")
            elif definition.clinical_contract_id != contract_id:
                findings.append(f"{contract_id}:concept_binding_mismatch:{concept_id}")

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
        "| Contract | Concepts | Definition/version | Source | Status | Validation | Golden vector | "
        + " | ".join(_DATABASES)
        + " |",
        "| --- | --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in _DATABASES) + " |",
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
            f"`{contract.golden_vector}`",
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
