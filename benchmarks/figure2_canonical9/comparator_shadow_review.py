"""Evaluator-only published-anchor hydration and Dev9 shadow-review contracts.

The Research Agent must not see the fixed anchors before execution. This owner
therefore lives under the benchmark evaluator, fetches only the protocol's
exact PMID/PMCID identifiers, and records source coverage plus content digests
without persisting article text. A reviewer separately compares run evidence
against the hydrated anchors; numeric agreement is never an acceptance rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComparatorShadowReviewError(RuntimeError):
    """The evaluator-only anchor or review contract failed closed."""


REVIEW_DIMENSIONS = (
    "study_population",
    "time_zero_and_windows",
    "variable_operationalization",
    "missingness_and_censoring",
    "primary_model_and_sensitivities",
    "table_and_figure_completeness",
    "conclusion_boundaries",
)

REVIEW_STATES = (
    "meets_anchor",
    "stronger_than_anchor",
    "actionable_gap",
    "not_applicable",
    "fail_closed_appropriate",
)


class AnchorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str
    url: str


class AnchorAccessPolicy(BaseModel):
    """Access requirements for evaluator-only comparison sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    full_text_required_for_every_anchor: bool
    supplement_handling: Literal["record_and_review_if_published"]
    inaccessible_anchor_action: Literal["replace_anchor"]


class TaskAnchorProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    anchors: tuple[AnchorSpec, ...] = Field(min_length=1)
    focus: tuple[str, ...] = Field(min_length=1)


class ComparatorShadowReviewProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.dev9_comparator_shadow_review/1"]
    protocol_ref: str
    audience: Literal["evaluator_only"]
    agent_visibility: Literal["forbidden_before_execution"]
    purpose: str
    use_policy: dict[str, tuple[str, ...]]
    dimensions: tuple[str, ...]
    review_states: tuple[str, ...]
    required_review_fields: tuple[str, ...]
    acceptance_rule: str
    anchor_access_policy: AnchorAccessPolicy
    tasks: tuple[TaskAnchorProtocol, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_frozen_review_axes(self) -> "ComparatorShadowReviewProtocol":
        if self.dimensions != REVIEW_DIMENSIONS:
            raise ValueError("shadow-review dimensions differ from the frozen contract")
        if self.review_states != REVIEW_STATES:
            raise ValueError("shadow-review states differ from the frozen contract")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("shadow-review task ids must be unique")
        citation_ids = [
            anchor.citation_id for task in self.tasks for anchor in task.anchors
        ]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("shadow-review anchor citation ids must be unique")
        forbidden = set(self.use_policy.get("forbidden", ()))
        required_forbidden = {
            "numeric_gold_answer",
            "expected_effect_direction",
            "result_similarity_pass_fail",
        }
        if not required_forbidden <= forbidden:
            raise ValueError("shadow-review policy does not forbid result leakage")
        if not self.anchor_access_policy.full_text_required_for_every_anchor:
            raise ValueError("every comparison anchor must require accessible full text")
        return self

    def task(self, task_id: str) -> TaskAnchorProtocol:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise ComparatorShadowReviewError(f"unknown shadow-review task: {task_id}")


class AnchorSourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str
    task_id: str
    protocol_url: str
    pmid: str
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: str
    journal: Optional[str] = None
    year: Optional[str] = None
    publication_types: tuple[str, ...] = ()
    abstract_available: bool
    abstract_word_count: int = Field(ge=0)
    pmc_full_text_available: bool
    pmc_section_titles: tuple[str, ...] = ()
    figure_caption_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    supplementary_material_count: int = Field(default=0, ge=0)
    abstract_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    pmc_xml_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnchorSourcePack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.dev9_anchor_source_pack/1"] = (
        "easyicu.dev9_anchor_source_pack/1"
    )
    protocol_ref: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fetched_at: str
    source: Literal["ncbi_eutils"] = "ncbi_eutils"
    records: tuple[AnchorSourceRecord, ...] = Field(min_length=1)


class ReviewDimension(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: Literal[
        "study_population",
        "time_zero_and_windows",
        "variable_operationalization",
        "missingness_and_censoring",
        "primary_model_and_sensitivities",
        "table_and_figure_completeness",
        "conclusion_boundaries",
    ]
    state: Literal[
        "meets_anchor",
        "stronger_than_anchor",
        "actionable_gap",
        "not_applicable",
        "fail_closed_appropriate",
    ]
    anchor_source_refs: tuple[str, ...] = Field(min_length=1)
    run_evidence_paths: tuple[str, ...] = Field(min_length=1)
    gap_or_rationale: str = Field(min_length=1)
    owner_module: str = Field(min_length=1)
    next_action: str = Field(min_length=1)
    supports: str = Field(min_length=1)
    cannot_prove: str = Field(min_length=1)


class RunBoundShadowReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.dev9_run_bound_shadow_review/1"] = (
        "easyicu.dev9_run_bound_shadow_review/1"
    )
    protocol_ref: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchor_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supplement_review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str
    run_head: str = Field(min_length=7)
    run_image: str = Field(min_length=1)
    run_path: str = Field(min_length=1)
    anchors: tuple[str, ...] = Field(min_length=1)
    dimensions: tuple[ReviewDimension, ...] = Field(min_length=7, max_length=7)
    overall_status: Literal["accepted", "changes_required"]
    review_authority: Literal["ai_development_review"] = "ai_development_review"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dimension_closure(self) -> "RunBoundShadowReview":
        dimensions = [row.dimension for row in self.dimensions]
        if tuple(dimensions) != REVIEW_DIMENSIONS:
            raise ValueError("run-bound review must contain every dimension in order")
        expected_status = (
            "changes_required"
            if any(row.state == "actionable_gap" for row in self.dimensions)
            else "accepted"
        )
        if self.overall_status != expected_status:
            raise ValueError("overall status disagrees with dimension findings")
        if any(not set(row.anchor_source_refs) <= set(self.anchors) for row in self.dimensions):
            raise ValueError("dimension cites an anchor outside the task review")
        return self


FetchText = Callable[[str, Sequence[str]], str]


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_shadow_review_protocol(path: Path | str) -> ComparatorShadowReviewProtocol:
    return ComparatorShadowReviewProtocol.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def protocol_content_sha256(protocol: ComparatorShadowReviewProtocol) -> str:
    return canonical_json_sha256(protocol.model_dump(mode="json"))


def _ncbi_request_ids(db: str, ids: Sequence[str]) -> tuple[str, ...]:
    if db != "pmc":
        return tuple(ids)
    return tuple(
        value[3:] if str(value).upper().startswith("PMC") else str(value)
        for value in ids
    )


def _default_fetch_text(db: str, ids: Sequence[str]) -> str:
    request_ids = _ncbi_request_ids(db, ids)
    query = urllib.parse.urlencode(
        {
            "db": db,
            "id": ",".join(request_ids),
            "retmode": "xml",
            "tool": "easyicu-dev9-shadow-review",
        }
    )
    request = urllib.request.Request(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}",
        headers={"User-Agent": "EasyICU-Dev9-ShadowReview/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _article_ids(article: ET.Element) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    # PubMed records may embed ArticleIdList nodes for cited references. Only
    # the current record's top-level PubmedData identifier list is authoritative.
    for node in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        kind = str(node.attrib.get("IdType") or "").casefold()
        value = _text(node)
        if kind and value:
            identifiers[kind] = value
    return identifiers


def _parse_pubmed(xml_text: str) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue
        ids = _article_ids(article)
        abstract = " ".join(
            _text(node) for node in article.findall(".//Article/Abstract/AbstractText")
        ).strip()
        year = _text(article.find(".//Article/Journal/JournalIssue/PubDate/Year"))
        if not year:
            medline_date = _text(
                article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
            )
            match = re.search(r"\b(19|20)\d{2}\b", medline_date)
            year = match.group(0) if match else ""
        records[pmid] = {
            "pmid": pmid,
            "pmcid": ids.get("pmc"),
            "doi": ids.get("doi"),
            "title": _text(article.find(".//Article/ArticleTitle")),
            "journal": _text(article.find(".//Article/Journal/Title")),
            "year": year,
            "publication_types": tuple(
                dict.fromkeys(
                    _text(node)
                    for node in article.findall(".//Article/PublicationTypeList/PublicationType")
                    if _text(node)
                )
            ),
            "abstract": abstract,
        }
    return records


def _parse_pmc(xml_text: str) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records: dict[str, dict[str, Any]] = {}
    articles = root.findall(".//article")
    if root.tag == "article":
        articles = [root]
    for article in articles:
        identifiers = {
            str(node.attrib.get("pub-id-type") or "").casefold(): _text(node)
            for node in article.findall(".//article-meta/article-id")
            if _text(node)
        }
        pmcid = (
            identifiers.get("pmcid")
            or identifiers.get("pmc")
            or identifiers.get("pmcaid")
        )
        if not pmcid:
            continue
        normalized_pmcid = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        section_titles = tuple(
            dict.fromkeys(
                _text(node)
                for node in article.findall(".//body//sec/title")
                if _text(node)
            )
        )
        records[normalized_pmcid] = {
            "pmid": identifiers.get("pmid"),
            "pmcid": normalized_pmcid,
            "section_titles": section_titles[:80],
            "figure_caption_count": len(article.findall(".//body//fig")),
            "table_count": len(article.findall(".//body//table-wrap")),
            "supplementary_material_count": len(
                article.findall(".//supplementary-material")
            ),
            "xml_sha256": hashlib.sha256(
                ET.tostring(article, encoding="utf-8")
            ).hexdigest(),
        }
    return records


def _citation_identifier(citation_id: str) -> tuple[str, str]:
    pmid_match = re.fullmatch(r"pmid_(\d+)", citation_id)
    if pmid_match:
        return "pmid", pmid_match.group(1)
    pmcid_match = re.fullmatch(r"pmcid_(PMC\d+)", citation_id, flags=re.IGNORECASE)
    if pmcid_match:
        return "pmcid", pmcid_match.group(1).upper()
    raise ComparatorShadowReviewError(
        f"unsupported exact anchor identifier: {citation_id!r}"
    )


def hydrate_anchor_source_pack(
    protocol: ComparatorShadowReviewProtocol,
    *,
    fetch_text: FetchText = _default_fetch_text,
    fetched_at: Optional[str] = None,
) -> AnchorSourcePack:
    """Fetch every exact protocol anchor and return metadata-only evidence."""

    anchor_rows = [
        (task, anchor, *_citation_identifier(anchor.citation_id))
        for task in protocol.tasks
        for anchor in task.anchors
    ]
    explicit_pmcids = [value for _, _, kind, value in anchor_rows if kind == "pmcid"]
    pmc_records: dict[str, dict[str, Any]] = {}
    if explicit_pmcids:
        pmc_records.update(_parse_pmc(fetch_text("pmc", explicit_pmcids)))

    pmids = [value for _, _, kind, value in anchor_rows if kind == "pmid"]
    for pmcid in explicit_pmcids:
        pmid = str((pmc_records.get(pmcid) or {}).get("pmid") or "")
        if not pmid:
            raise ComparatorShadowReviewError(
                f"PMCID anchor did not resolve to a PMID: {pmcid}"
            )
        pmids.append(pmid)
    pubmed_records = _parse_pubmed(fetch_text("pubmed", tuple(dict.fromkeys(pmids))))
    discovered_pmcids = [
        str(record.get("pmcid"))
        for record in pubmed_records.values()
        if record.get("pmcid") and str(record.get("pmcid")) not in pmc_records
    ]
    if discovered_pmcids:
        pmc_records.update(_parse_pmc(fetch_text("pmc", discovered_pmcids)))

    records: list[AnchorSourceRecord] = []
    for task, anchor, kind, value in anchor_rows:
        if kind == "pmid":
            pmid = value
        else:
            pmid = str((pmc_records.get(value) or {}).get("pmid") or "")
        metadata = pubmed_records.get(pmid)
        if metadata is None:
            raise ComparatorShadowReviewError(
                f"exact anchor missing from PubMed response: {anchor.citation_id}"
            )
        abstract = str(metadata.get("abstract") or "")
        pmcid = str(metadata.get("pmcid") or "") or None
        pmc = pmc_records.get(pmcid or "")
        source_payload = {
            "citation_id": anchor.citation_id,
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": metadata.get("doi"),
            "title": metadata.get("title"),
            "year": metadata.get("year"),
            "publication_types": metadata.get("publication_types"),
            "abstract_sha256": (
                hashlib.sha256(abstract.encode("utf-8")).hexdigest()
                if abstract
                else None
            ),
            "pmc_xml_sha256": pmc.get("xml_sha256") if pmc else None,
        }
        records.append(
            AnchorSourceRecord(
                citation_id=anchor.citation_id,
                task_id=task.task_id,
                protocol_url=anchor.url,
                pmid=pmid,
                pmcid=pmcid,
                doi=metadata.get("doi"),
                title=str(metadata.get("title") or ""),
                journal=str(metadata.get("journal") or "") or None,
                year=str(metadata.get("year") or "") or None,
                publication_types=tuple(metadata.get("publication_types") or ()),
                abstract_available=bool(abstract),
                abstract_word_count=len(abstract.split()),
                pmc_full_text_available=pmc is not None,
                pmc_section_titles=tuple((pmc or {}).get("section_titles") or ()),
                figure_caption_count=int((pmc or {}).get("figure_caption_count") or 0),
                table_count=int((pmc or {}).get("table_count") or 0),
                supplementary_material_count=int(
                    (pmc or {}).get("supplementary_material_count") or 0
                ),
                abstract_sha256=source_payload["abstract_sha256"],
                pmc_xml_sha256=source_payload["pmc_xml_sha256"],
                source_fingerprint_sha256=canonical_json_sha256(source_payload),
            )
        )
    pack = AnchorSourcePack(
        protocol_ref=protocol.protocol_ref,
        protocol_sha256=protocol_content_sha256(protocol),
        fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
        records=tuple(records),
    )
    if protocol.anchor_access_policy.full_text_required_for_every_anchor:
        inaccessible = tuple(
            record.citation_id
            for record in pack.records
            if not record.pmc_full_text_available
        )
        if inaccessible:
            raise ComparatorShadowReviewError(
                "comparison anchors lack accessible PMC full text and must be replaced: "
                + ", ".join(inaccessible)
            )
    return pack


def validate_run_bound_review(
    review: RunBoundShadowReview,
    *,
    protocol: ComparatorShadowReviewProtocol,
    anchor_pack: AnchorSourcePack,
) -> None:
    task = protocol.task(review.task_id)
    expected_anchors = tuple(anchor.citation_id for anchor in task.anchors)
    if review.protocol_ref != protocol.protocol_ref:
        raise ComparatorShadowReviewError("review protocol ref mismatch")
    if review.protocol_sha256 != protocol_content_sha256(protocol):
        raise ComparatorShadowReviewError("review protocol digest mismatch")
    if review.anchor_pack_sha256 != canonical_json_sha256(
        anchor_pack.model_dump(mode="json")
    ):
        raise ComparatorShadowReviewError("review anchor-pack digest mismatch")
    if review.anchors != expected_anchors:
        raise ComparatorShadowReviewError("review anchors differ from task protocol")
    hydrated = {
        record.citation_id
        for record in anchor_pack.records
        if record.task_id == review.task_id
    }
    if set(review.anchors) != hydrated:
        raise ComparatorShadowReviewError("review anchors are not fully hydrated")
    validate_run_bound_review_artifacts(review)


def validate_run_bound_review_artifacts(review: RunBoundShadowReview) -> None:
    """Bind a review to one exact run and files that actually exist there."""

    run_path = Path(review.run_path)
    if not run_path.is_absolute() or not run_path.is_dir():
        raise ComparatorShadowReviewError("review run_path is not an existing directory")
    resolved_run = run_path.resolve(strict=True)
    status_path = resolved_run / "run_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ComparatorShadowReviewError(
            "review run_status.json is missing or invalid"
        ) from exc
    code_version = status.get("code_version")
    actual_head = (
        str(code_version.get("git_sha") or "")
        if isinstance(code_version, dict)
        else ""
    )
    if actual_head != review.run_head:
        raise ComparatorShadowReviewError("review run_head differs from run_status")

    lineage_path = resolved_run / "development_runtime_lineage.json"
    if lineage_path.exists():
        try:
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ComparatorShadowReviewError(
                "review development runtime lineage is invalid"
            ) from exc
        image_ids = {
            str(provenance.get("image_id") or "")
            for step in lineage.get("steps", ())
            if isinstance(step, dict)
            for provenance in (step.get("provenance"),)
            if isinstance(provenance, dict)
        }
        image_ids.discard("")
        if image_ids and review.run_image not in image_ids:
            raise ComparatorShadowReviewError(
                "review run_image differs from development runtime lineage"
            )

    for dimension in review.dimensions:
        for relative in dimension.run_evidence_paths:
            candidate = Path(relative)
            if candidate.is_absolute():
                raise ComparatorShadowReviewError(
                    "review evidence paths must be relative to run_path"
                )
            try:
                resolved = (resolved_run / candidate).resolve(strict=True)
                resolved.relative_to(resolved_run)
            except (OSError, ValueError) as exc:
                raise ComparatorShadowReviewError(
                    f"review evidence path is missing or escapes run_path: {relative}"
                ) from exc
            if not resolved.is_file():
                raise ComparatorShadowReviewError(
                    f"review evidence path is not a file: {relative}"
                )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_shadow_review_protocol(args.protocol)
    pack = hydrate_anchor_source_pack(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(canonical_json_sha256(pack.model_dump(mode="json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "AnchorAccessPolicy",
    "AnchorSourcePack",
    "AnchorSourceRecord",
    "ComparatorShadowReviewError",
    "ComparatorShadowReviewProtocol",
    "REVIEW_DIMENSIONS",
    "REVIEW_STATES",
    "ReviewDimension",
    "RunBoundShadowReview",
    "canonical_json_sha256",
    "hydrate_anchor_source_pack",
    "load_shadow_review_protocol",
    "protocol_content_sha256",
    "validate_run_bound_review",
    "validate_run_bound_review_artifacts",
]
