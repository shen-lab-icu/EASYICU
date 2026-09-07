"""Source-verified author metadata, independent of scientific citation eligibility.

PubMed efetch AuthorList (including collective authors), checked 2026-09-08.
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
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/",
            "snapshot_sha256": PUBMED_SNAPSHOT_SHA256,
            "record_sha256": sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        }
    return copied, None
