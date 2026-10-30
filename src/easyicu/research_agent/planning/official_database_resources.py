"""Versioned database resource identities; identifier matching is not appraisal.

Catalog entries are curated from the source-provider pages referenced in each
entry. The publication is not a dataset version, and a version-specific DOI
cannot establish the version of locally registered files. Matching does not
fetch or independently authenticate source content.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Literal

from easyicu.databases.profiles import normalize_database_key
from .literature_contract import SourceResourceIdentity


@dataclass(frozen=True)
class OfficialResourceEntry:
    database: str
    kind: Literal["source_publication", "dataset_release"]
    title: str
    doi: str
    pmid: str | None
    dataset_version: str | None
    reference: str


# Registry versions are immutable: /1 remains available to replay old receipts.
_SICDB_REFERENCE = "https://physionet.org/content/sicdb/1.0.6/"
_MIMIC_31_REFERENCE = "https://physionet.org/content/mimiciv/3.1/"
_MIMIC_22_REFERENCE = "https://physionet.org/content/mimiciv/2.2/"
_VERSION_1: tuple[OfficialResourceEntry, ...] = (
    OfficialResourceEntry("sic", "source_publication",
                          "The Salzburg Intensive Care database (SICdb): an openly available critical care dataset",
                          "10.1007/s00134-023-07046-3", "37052626", None, _SICDB_REFERENCE),
    OfficialResourceEntry("sic", "dataset_release",
                          "Salzburg Intensive Care database (SICdb), a freely accessible intensive care database",
                          "10.13026/dm9f-dm56", None, "1.0.6", _SICDB_REFERENCE),
)
_VERSION_2: tuple[OfficialResourceEntry, ...] = _VERSION_1 + (
    OfficialResourceEntry("miiv", "source_publication",
                          "MIMIC-IV, a freely accessible electronic health record dataset",
                          "10.1038/s41597-022-01899-x", "36596836", None, _MIMIC_31_REFERENCE),
    OfficialResourceEntry("miiv", "dataset_release", "MIMIC-IV",
                          "10.13026/kpb9-mt58", None, "3.1", _MIMIC_31_REFERENCE),
    OfficialResourceEntry("miiv", "dataset_release", "MIMIC-IV",
                          "10.13026/6mm1-ek67", None, "2.2", _MIMIC_22_REFERENCE),
)
OFFICIAL_RESOURCE_REGISTRIES = MappingProxyType({
    "easyicu.official_source_identity/1": _VERSION_1,
    "easyicu.official_source_identity/2": _VERSION_2,
})
CURRENT_OFFICIAL_SOURCE_REGISTRY_VERSION = "easyicu.official_source_identity/2"


def _words(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _doi(value: str | None) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(value or "").strip(), flags=re.I).casefold()


def canonical_resource_database(database: str | None) -> str | None:
    try:
        return normalize_database_key(str(database or "")).removesuffix("_demo")
    except KeyError:
        return None


def match_official_source_identity(
    database: str | None, record: Any, *, registry_version: str | None = None,
) -> SourceResourceIdentity | None:
    """Match registered source coordinates, rejecting contradictions.

    This proves only that supplied coordinates match the named local catalog.
    It is not an authenticated retrieval, content review, study applicability,
    or proof that the registered dataset is the version named by a DOI.
    """
    source = canonical_resource_database(database)
    version = registry_version or CURRENT_OFFICIAL_SOURCE_REGISTRY_VERSION
    entries = OFFICIAL_RESOURCE_REGISTRIES.get(version)
    if source is None or entries is None:
        return None
    pmid = str(getattr(record, "pmid", "") or "").strip()
    doi = _doi(getattr(record, "doi", ""))
    title = _words(str(getattr(record, "title", "") or ""))
    declared_version = str(getattr(record, "dataset_version", "") or "").strip()
    for entry in entries:
        if entry.database != source or title != _words(entry.title):
            continue
        if not ((pmid and entry.pmid and pmid == entry.pmid) or (doi and doi == entry.doi)):
            continue
        if (pmid and pmid != entry.pmid) or (doi and doi != entry.doi):
            continue
        if declared_version and declared_version != entry.dataset_version:
            continue
        return SourceResourceIdentity(
            database=entry.database, kind=entry.kind, pmid=entry.pmid, doi=entry.doi,
            dataset_version=entry.dataset_version, registry_reference=entry.reference,
            registry_version=version,
        )
    return None
