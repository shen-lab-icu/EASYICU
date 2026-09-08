"""Source-verified author metadata, independent of scientific citation eligibility.

PubMed AuthorList and Crossref author metadata, checked 2026-09-08.
The offline snapshot only fills empty authors on an exact bibliographic match.
It never selects papers, changes screening, or grants full-text/claim authority.
"""

from hashlib import sha256
import json
import re
from typing import Any, Mapping


PUBMED_SNAPSHOT_SHA256 = "3fb8330bbe3a874d9f52db3cec04f1712c4d37eebbd944c6b385b97cd05ddd52"
_SOURCES = (
    {
        "pmid": "26903338", "doi": "10.1001/jama.2016.0287", "year": "2016", "venue": "JAMA",
        "title": "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).",
        "authors": (
            "Mervyn Singer", "Clifford S Deutschman", "Christopher Warren Seymour",
            "Manu Shankar-Hari", "Djillali Annane", "Michael Bauer", "Rinaldo Bellomo",
            "Gordon R Bernard", "Jean-Daniel Chiche", "Craig M Coopersmith",
            "Richard S Hotchkiss", "Mitchell M Levy", "John C Marshall", "Greg S Martin",
            "Steven M Opal", "Gordon D Rubenfeld", "Tom van der Poll", "Jean-Louis Vincent",
            "Derek C Angus",
        ),
    },
    {
        "pmid": "18056625", "doi": "10.1093/aje/kwm324", "year": "2008",
        "venue": "American Journal of Epidemiology",
        "title": "Immortal time bias in pharmaco-epidemiology.",
        "authors": ("Samy Suissa",),
    },
    {
        "pmid": "17938396", "doi": "10.7326/0003-4819-147-8-200710160-00010", "year": "2007",
        "venue": "Annals of Internal Medicine",
        "title": "The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies.",
        "authors": (
            "Erik von Elm", "Douglas G Altman", "Matthias Egger", "Stuart J Pocock",
            "Peter C Gøtzsche", "Jan P Vandenbroucke", "STROBE Initiative",
        ),
    },
    {
        "pmid": "26440803", "doi": "10.1371/journal.pmed.1001885", "year": "2015",
        "venue": "PLoS Medicine",
        "title": "The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) statement.",
        "authors": (
            "Eric I Benchimol", "Liam Smeeth", "Astrid Guttmann", "Katie Harron",
            "David Moher", "Irene Petersen", "Henrik T Sørensen", "Erik von Elm",
            "Sinéad M Langan", "RECORD Working Committee",
        ),
    },
    {
        "pmid": "19564179",
        "title": "Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls.",
        "venue": "BMJ",
        "year": "2009",
        "doi": "10.1136/bmj.b2393",
        "authors": [
            "Jonathan A C Sterne",
            "Ian R White",
            "John B Carlin",
            "Michael Spratt",
            "Patrick Royston",
            "Michael G Kenward",
            "Angela M Wood",
            "James R Carpenter",
        ],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/19564179/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "6668489",
        "title": "Analysis of survival by tumor response.",
        "venue": "Journal of Clinical Oncology",
        "year": "1983",
        "doi": "10.1200/JCO.1983.1.11.710",
        "authors": ["J R Anderson", "K C Cain", "R D Gelber"],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/6668489/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "21611958",
        "title": "The use of restricted mean survival time to estimate the treatment effect in randomized clinical trials when the proportional hazards assumption is in doubt.",
        "venue": "Statistics in medicine",
        "year": "2011",
        "doi": "10.1002/sim.4274",
        "authors": ["Patrick Royston", "Mahesh K B Parmar"],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/21611958/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "2657958",
        "title": "Flexible regression models with cubic splines.",
        "venue": "Statistics in medicine",
        "year": "1989",
        "doi": "10.1002/sim.4780080504",
        "authors": ["S Durrleman", "R Simon"],
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/2657958/",
        "snapshot_sha256": "1443819ed4b4f81454b96e614bfdfe0a687cc14020487f9f501bec40bc6cc3a8",
    },
    {
        "pmid": "",
        "doi": "10.1093/biomet/81.3.515",
        "title": "Proportional hazards tests and diagnostics based on weighted residuals",
        "venue": "Biometrika",
        "year": "1994",
        "authors": ["PATRICIA M. GRAMBSCH", "TERRY M. THERNEAU"],
        "source_url": "https://api.crossref.org/works/10.1093/biomet/81.3.515",
        "snapshot_sha256": "90f3d350e2fda0727ee80a13c56a64649a35ff02b3fd134e6154319c83d3c40d",
    },
)


def _text_identity(value: Any) -> str:
    # Punctuation/capitalisation only; no fuzzy title or author-key guessing.
    return re.sub(r"[\W_]", "", str(value or "").casefold())


def complete_missing_authors(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a copy plus a versioned metadata-only receipt; leave conflicts alone."""
    copied = dict(record)
    if record.get("authors"):
        return copied, None
    for source in _SOURCES:
        if not all(
            str(record.get(key) or "").strip().casefold() == source[key].casefold()
            for key in ("pmid", "doi", "year")
        ) or not all(
            _text_identity(record.get(key)) == _text_identity(source[key])
            for key in ("title", "venue")
        ):
            continue
        copied["authors"] = list(source["authors"])
        return copied, {
            "schema_version": "easyicu.bibliographic_metadata_completion/1",
            "scope": "bibliographic_metadata_only",
            "fields": ["authors"], "verified_on": "2026-09-08",
            "source_url": source.get("source_url", f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/"),
            "snapshot_sha256": source.get("snapshot_sha256", PUBMED_SNAPSHOT_SHA256),
            "record_sha256": sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        }
    return copied, None
