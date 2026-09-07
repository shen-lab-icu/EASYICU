"""Assemble the existing numeric reader, exact-plan tables and cited metadata.

This is a display projection, not a Writer, statistical executor or literature
review. Callers supply current evidence membership and source-bound inputs.
"""

from dataclasses import asdict
import re
from typing import Any, Mapping, Sequence

from ..literature import LiteratureBundle, manuscript_citable_records
from ..schema import AnalysisPlan, EvidenceRecord
from .manuscript_provenance import ManuscriptProvenanceError, build_manuscript_provenance
from .manuscript_tables import ManuscriptTableProjectionError, build_manuscript_tables


def build_manuscript_reader(
    *,
    manuscript: str,
    evidence: Any,
    plan: AnalysisPlan | None = None,
    literature: LiteratureBundle | None = None,
    evidence_records: Sequence[EvidenceRecord] | None = None,
    binding_map: Mapping | None = None,
) -> dict[str, Any]:
    """Keep source numbers unchanged and order references by first citation."""

    payload = build_manuscript_provenance(
        manuscript=manuscript, evidence=evidence, binding_map=binding_map,
    )
    try:
        tables = build_manuscript_tables(
            plan=plan,
            evidence_records=evidence.records() if evidence_records is None else evidence_records,
            run_dir=evidence.root,
        ) if plan is not None else ()
    except ManuscriptTableProjectionError as exc:
        raise ManuscriptProvenanceError(str(exc)) from exc
    payload["tables"] = [
        {"label": f"Table {index}", **asdict(table)}
        for index, table in enumerate(tables, 1)
    ]
    references = []
    if literature is not None:
        records = manuscript_citable_records(literature)
        by_key = {record.key: record for record in records}
        if len(by_key) != len(records):
            raise ManuscriptProvenanceError("Ambiguous manuscript citation identity")
        keys = dict.fromkeys(
            key for block in re.findall(r"\[@[^\[\]\n]+\]", manuscript)
            for key in re.findall(r"@([A-Za-z0-9_.:-]+)", block)
        )
        for number, key in enumerate(keys, 1):
            if key not in by_key:
                raise ManuscriptProvenanceError(f"Manuscript citation is not citable: {key}")
            references.append({"number": number, **by_key[key].model_dump(mode="json")})
    payload["references"] = references
    return payload
