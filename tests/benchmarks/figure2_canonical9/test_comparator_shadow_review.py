from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.comparator_shadow_review import (
    _ncbi_request_ids,
    ComparatorShadowReviewError,
    REVIEW_DIMENSIONS,
    ReviewDimension,
    RunBoundShadowReview,
    canonical_json_sha256,
    hydrate_anchor_source_pack,
    load_shadow_review_protocol,
    protocol_content_sha256,
    validate_run_bound_review,
)


_PROTOCOL = Path("benchmarks/figure2_canonical9/dev9_comparator_shadow_review_v1.json")


def _pubmed_xml(pmids: list[str]) -> str:
    articles = []
    for pmid in pmids:
        articles.append(
            f"""
            <PubmedArticle>
              <MedlineCitation><PMID>{pmid}</PMID><Article>
                <ArticleTitle>Anchor {pmid}</ArticleTitle>
                <Abstract><AbstractText>Adult ICU cohort methods and results.</AbstractText></Abstract>
                <Journal><Title>ICU Journal</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
                <PublicationTypeList><PublicationType>Observational Study</PublicationType></PublicationTypeList>
              </Article><ReferenceList><Reference><ArticleIdList>
                <ArticleId IdType="pmc">PMC999999</ArticleId>
              </ArticleIdList></Reference></ReferenceList></MedlineCitation>
              <PubmedData><ArticleIdList>
                <ArticleId IdType="pubmed">{pmid}</ArticleId>
                <ArticleId IdType="doi">10.1/{pmid}</ArticleId>
                <ArticleId IdType="pmc">PMC{pmid}</ArticleId>
              </ArticleIdList></PubmedData>
            </PubmedArticle>
            """
        )
    return "<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"


def _pmc_xml(pmcids: list[str], pmid_by_pmc: dict[str, str]) -> str:
    articles = []
    for pmcid in pmcids:
        pmid = pmid_by_pmc.get(pmcid, pmcid.removeprefix("PMC"))
        articles.append(
            f"""
            <article><front><article-meta>
              <article-id pub-id-type="pmcid">{pmcid}</article-id>
              <article-id pub-id-type="pmid">{pmid}</article-id>
            </article-meta></front><body>
              <sec><title>Methods</title><p>Methods text.</p></sec>
              <fig><caption><p>Figure.</p></caption></fig>
              <table-wrap><table/></table-wrap>
            </body></article>
            """
        )
    return "<pmc-articleset>" + "".join(articles) + "</pmc-articleset>"


def test_protocol_is_evaluator_only_and_forbids_result_leakage() -> None:
    protocol = load_shadow_review_protocol(_PROTOCOL)
    assert protocol.audience == "evaluator_only"
    assert protocol.agent_visibility == "forbidden_before_execution"
    assert protocol.dimensions == REVIEW_DIMENSIONS
    assert len(protocol.tasks) == 9
    assert sum(len(task.anchors) for task in protocol.tasks) == 14
    assert len(protocol_content_sha256(protocol)) == 64


def test_ncbi_transport_normalizes_pmc_ids_at_the_external_boundary() -> None:
    assert _ncbi_request_ids("pmc", ("PMC9322581", "123")) == ("9322581", "123")
    assert _ncbi_request_ids("pubmed", ("38905261",)) == ("38905261",)


def test_exact_anchor_hydration_records_source_coverage_without_article_text() -> None:
    protocol = load_shadow_review_protocol(_PROTOCOL)
    explicit_pmc = "PMC9322581"
    explicit_pmid = "35938334"

    def fetch(db: str, ids: tuple[str, ...] | list[str]) -> str:
        values = list(ids)
        if db == "pmc":
            mapping = {explicit_pmc: explicit_pmid}
            return _pmc_xml(values, mapping)
        assert db == "pubmed"
        return _pubmed_xml(values)

    pack = hydrate_anchor_source_pack(
        protocol,
        fetch_text=fetch,
        fetched_at="2026-08-24T00:00:00+00:00",
    )
    assert len(pack.records) == 14
    assert {record.citation_id for record in pack.records} == {
        anchor.citation_id for task in protocol.tasks for anchor in task.anchors
    }
    assert all(record.abstract_available for record in pack.records)
    assert all(record.pmc_full_text_available for record in pack.records)
    assert all(record.figure_caption_count == 1 for record in pack.records)
    assert all(record.supplementary_material_count == 0 for record in pack.records)
    assert all(record.pmcid != "PMC999999" for record in pack.records)
    payload = pack.model_dump(mode="json")
    encoded = json.dumps(payload)
    assert "Adult ICU cohort methods" not in encoded
    assert "Methods text" not in encoded


def test_exact_anchor_hydration_fails_when_protocol_id_is_absent() -> None:
    protocol = load_shadow_review_protocol(_PROTOCOL)

    def fetch(db: str, ids: tuple[str, ...] | list[str]) -> str:
        if db == "pmc":
            return _pmc_xml(list(ids), {"PMC9322581": "35938334"})
        return "<PubmedArticleSet/>"

    with pytest.raises(ComparatorShadowReviewError, match="missing from PubMed"):
        hydrate_anchor_source_pack(protocol, fetch_text=fetch)


def test_exact_anchor_hydration_requires_accessible_full_text() -> None:
    protocol = load_shadow_review_protocol(_PROTOCOL)

    def fetch(db: str, ids: tuple[str, ...] | list[str]) -> str:
        values = list(ids)
        if db == "pmc":
            available = [value for value in values if value != "PMC38905261"]
            return _pmc_xml(available, {"PMC9322581": "35938334"})
        return _pubmed_xml(values)

    with pytest.raises(ComparatorShadowReviewError, match="must be replaced"):
        hydrate_anchor_source_pack(protocol, fetch_text=fetch)


def test_run_bound_review_requires_all_dimensions_and_exact_task_anchors(
    tmp_path: Path,
) -> None:
    protocol = load_shadow_review_protocol(_PROTOCOL)

    def fetch(db: str, ids: tuple[str, ...] | list[str]) -> str:
        values = list(ids)
        if db == "pmc":
            return _pmc_xml(values, {"PMC9322581": "35938334"})
        return _pubmed_xml(values)

    pack = hydrate_anchor_source_pack(protocol, fetch_text=fetch)
    task = protocol.task("e2_lactate_mortality")
    anchor_ids = tuple(anchor.citation_id for anchor in task.anchors)
    run_path = tmp_path / "run_e2"
    (run_path / "run").mkdir(parents=True)
    (run_path / "run" / "evidence.json").write_text("{}\n", encoding="utf-8")
    (run_path / "run_status.json").write_text(
        json.dumps({"code_version": {"git_sha": "abcdef1"}}) + "\n",
        encoding="utf-8",
    )
    dimensions = tuple(
        ReviewDimension(
            dimension=dimension,
            state="meets_anchor",
            anchor_source_refs=anchor_ids,
            run_evidence_paths=("run/evidence.json",),
            gap_or_rationale="Run evidence closes this dimension.",
            owner_module="benchmark_evaluator",
            next_action="No action.",
            supports="Development quality comparison.",
            cannot_prove="Numeric agreement or publication readiness.",
        )
        for dimension in REVIEW_DIMENSIONS
    )
    review = RunBoundShadowReview(
        protocol_ref=protocol.protocol_ref,
        protocol_sha256=protocol_content_sha256(protocol),
        anchor_pack_sha256=canonical_json_sha256(pack.model_dump(mode="json")),
        supplement_review_sha256="a" * 64,
        task_id=task.task_id,
        run_head="abcdef1",
        run_image="sha256:image",
        run_path=str(run_path),
        anchors=anchor_ids,
        dimensions=dimensions,
        overall_status="accepted",
        claim_boundary="Development-only external comparison; no numeric gold answer.",
    )
    validate_run_bound_review(review, protocol=protocol, anchor_pack=pack)
    drifted = review.model_copy(update={"anchors": ("pmid_wrong",)})
    with pytest.raises(ComparatorShadowReviewError, match="anchors differ"):
        validate_run_bound_review(drifted, protocol=protocol, anchor_pack=pack)
    wrong_head = review.model_copy(update={"run_head": "deadbee"})
    with pytest.raises(ComparatorShadowReviewError, match="run_head differs"):
        validate_run_bound_review(wrong_head, protocol=protocol, anchor_pack=pack)
    (run_path / "run" / "evidence.json").unlink()
    with pytest.raises(ComparatorShadowReviewError, match="evidence path is missing"):
        validate_run_bound_review(review, protocol=protocol, anchor_pack=pack)
