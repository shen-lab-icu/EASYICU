#!/usr/bin/env python3
"""Local S6 discovery validation harness.

This script is intentionally not a pytest target. It performs live PubMed
queries and optional OpenRouter same-topic screening, then freezes the result
under ``EASYICU/.tmp``. API keys are read from the environment or stdin and are
never written to output files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from easyicu.research_agent.discovery.idea_mining import (  # noqa: E402
    LiteratureIdeaCandidate,
    OutcomeDeterminability,
    SourceMaterial,
    assess_prior_art_for_idea,
    map_literature_idea_to_executable_candidate,
    run_idea_mining_dry_run,
)
from easyicu.research_agent.literature import CitationRecord  # noqa: E402
from easyicu.research_agent.providers.protocol import LLMMessage  # noqa: E402
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-4.5-air:free"
DEFAULT_SEEDS = REPO_ROOT / "benchmark" / "idea_mining_s6_screen_seed_controls.json"
DEFAULT_OUT = REPO_ROOT / ".tmp" / "idea_mining_s6_validation"
DEFAULT_MIIV = REPO_ROOT.parent / "其他文件" / "miiv_20260420"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_payload(value: Any) -> str:
    return "sha256:" + sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "EasyICU-S6-validation/1.0"})
    last: Exception | None = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last = exc
            time.sleep(min(2.0 * (attempt + 1), 20.0))
    raise RuntimeError(f"PubMed JSON request failed: {url}") from last


def _request_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "EasyICU-S6-validation/1.0"})
    last: Exception | None = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            last = exc
            time.sleep(min(2.0 * (attempt + 1), 20.0))
    raise RuntimeError(f"PubMed text request failed: {url}") from last


def pubmed_search(term: str, *, retmax: int) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        }
    )
    raw = _request_json(f"{EUTILS}/esearch.fcgi?{params}")
    result = raw.get("esearchresult") or {}
    return {
        "count": int(result.get("count", 0) or 0),
        "pmids": [str(item) for item in result.get("idlist", [])],
    }


def _text_content(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _article_year(article: ET.Element) -> str:
    for path in (".//ArticleDate/Year", ".//PubDate/Year", ".//DateCompleted/Year"):
        node = article.find(path)
        if node is not None and node.text:
            return node.text.strip()
    return "unknown"


def pubmed_fetch(pmids: Iterable[str]) -> list[dict[str, str]]:
    ids = [str(pmid) for pmid in pmids if str(pmid).strip()]
    if not ids:
        return []
    params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
    root = ET.fromstring(_request_text(f"{EUTILS}/efetch.fcgi?{params}"))
    out: list[dict[str, str]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text_content(article.find(".//PMID"))
        title = _text_content(article.find(".//ArticleTitle"))
        journal = _text_content(article.find(".//Journal/Title")) or _text_content(
            article.find(".//Journal/ISOAbbreviation")
        )
        abstract = " ".join(
            _text_content(node)
            for node in article.findall(".//Abstract/AbstractText")
            if _text_content(node)
        )
        if not pmid or not title:
            continue
        out.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": _article_year(article),
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return out


# --- PMC open-access full-text adapter (Discussion / Limitations mining) ----
# Abstracts state findings, not open questions. The S4 extractor is asked for
# "unresolved question / future direction / limitation / uncertainty / evidence
# gap" (idea_mining.py), but a PubMed abstract rarely contains that language, so
# the model regresses to the headline association. The genuine gap language
# lives in the Discussion / Limitations / Future-directions sections of the full
# text. For the PubMed Central open-access subset we can fetch that legally and
# reproducibly via E-utilities and feed it instead of the abstract.

PMC_GAP_SECTION_RE = re.compile(
    r"(discussion|limitation|future|perspective|unanswered|unresolved|"
    r"research agenda|implication|conclusion|outlook|knowledge gap)",
    re.I,
)


def pmids_to_pmcids(pmids: Iterable[str]) -> dict[str, str]:
    """Map PubMed IDs to PMC IDs via the NCBI ID-converter (OA subset only).

    Returns ``{pmid: pmcid}`` for the articles that have a PMC record; PMIDs
    with no PMC counterpart (paywalled / not deposited) are simply absent.
    """
    ids = [str(p) for p in pmids if str(p).strip()]
    if not ids:
        return {}
    mapping: dict[str, str] = {}
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        params = urllib.parse.urlencode(
            {
                "ids": ",".join(batch),
                "format": "json",
                "tool": "EasyICU-S6-validation",
                "email": "easyicu@example.org",
            }
        )
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?{params}"
        try:
            payload = _request_json(url)
        except RuntimeError:
            continue
        for record in payload.get("records", []) or []:
            pmid = str(record.get("pmid") or "").strip()
            pmcid = str(record.get("pmcid") or "").strip()
            if pmid and pmcid and not record.get("errmsg"):
                mapping[pmid] = pmcid
    return mapping


def _extract_pmc_gap_text(article: ET.Element, *, max_chars: int) -> tuple[str, bool]:
    """Pull Discussion / Limitations / future-direction text from a JATS body.

    Returns ``(text, matched)`` where ``matched`` is True when at least one
    top-level section title matched the gap-section pattern. When nothing
    matches (e.g. an editorial with free-form body), falls back to the whole
    body text so editorial argument prose is still mined.
    """
    body = article.find(".//body")
    if body is None:
        return "", False
    matched_blocks: list[str] = []
    for sec in body.findall("sec"):
        title = _text_content(sec.find("title"))
        sec_type = str(sec.get("sec-type") or "")
        if title and PMC_GAP_SECTION_RE.search(title):
            matched_blocks.append(_text_content(sec))
        elif sec_type and PMC_GAP_SECTION_RE.search(sec_type):
            matched_blocks.append(_text_content(sec))
    if matched_blocks:
        return (" ".join(matched_blocks)[:max_chars], True)
    return (_text_content(body)[:max_chars], False)


def pmc_fetch_gap_sections(
    pmcids: Iterable[str], *, max_chars: int = 9000
) -> dict[str, dict[str, Any]]:
    """Fetch PMC full text and extract gap-bearing sections, keyed by PMID.

    Only the open-access subset returns a parseable ``<body>``; closed records
    come back without one and are reported as ``matched=False, text=""`` so the
    caller can fall back to the abstract.
    """
    ids = [str(p).replace("PMC", "") for p in pmcids if str(p).strip()]
    if not ids:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 20):
        batch = ids[start : start + 20]
        params = urllib.parse.urlencode(
            {"db": "pmc", "id": ",".join(batch), "retmode": "xml"}
        )
        try:
            root = ET.fromstring(_request_text(f"{EUTILS}/efetch.fcgi?{params}"))
        except (ET.ParseError, RuntimeError):
            continue
        for article in root.findall(".//article"):
            pmid = ""
            for aid in article.findall(".//article-id"):
                if (aid.get("pub-id-type") or "") == "pmid":
                    pmid = _text_content(aid)
                    break
            text, matched = _extract_pmc_gap_text(article, max_chars=max_chars)
            if pmid:
                out[pmid] = {"text": text, "matched": matched}
        time.sleep(0.34)  # E-utilities courtesy rate limit
    return out


def build_fulltext_materials(
    articles: list[dict[str, str]],
    *,
    max_chars: int = 9000,
) -> tuple[list[SourceMaterial], dict[str, Any]]:
    """Build extraction materials preferring PMC OA Discussion/Limitations text.

    Falls back to the abstract for any article without OA full text so coverage
    never drops below the abstract-only path. Returns ``(materials, report)``
    where ``report`` records how many materials carried real gap-section text vs
    fell back to the abstract — so the discovery output can state honestly how
    much of the run was mined from full text.
    """
    pmid_to_pmcid = pmids_to_pmcids(a["pmid"] for a in articles)
    gap_by_pmid = pmc_fetch_gap_sections(
        pmid_to_pmcid.values(), max_chars=max_chars
    )
    materials: list[SourceMaterial] = []
    n_fulltext_gap = n_fulltext_body = n_abstract = 0
    for article in articles:
        pmid = article["pmid"]
        gap = gap_by_pmid.get(pmid)
        if gap and gap.get("text"):
            body_text = str(gap["text"]).strip()
            text = f"{article['title']} {body_text}".strip()
            locator = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid_to_pmcid[pmid]}/"
            if gap.get("matched"):
                n_fulltext_gap += 1
            else:
                n_fulltext_body += 1
        else:
            text = f"{article['title']} {article['abstract']}".strip()
            locator = article["url"]
            n_abstract += 1
        if len(text) < 120:
            continue
        citation = CitationRecord(
            key=f"pubmed_{pmid}",
            title=article["title"],
            year=article["year"],
            venue=article["journal"],
            relevance=article["abstract"],
            url=article["url"],
            pmid=pmid,
        )
        materials.append(
            SourceMaterial(
                citation=citation,
                source_adapter_level="user_supplied_excerpt",
                locator=locator,
                source_text=text,
            )
        )
    report = {
        "articles": len(articles),
        "pmc_mapped": len(pmid_to_pmcid),
        "materials": len(materials),
        "from_pmc_gap_section": n_fulltext_gap,
        "from_pmc_body": n_fulltext_body,
        "from_abstract_fallback": n_abstract,
    }
    return materials, report


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response was not a JSON object")
    return payload


class OpenRouterSameTopicScreener:
    """OpenRouter-backed top-hit same-topic screener."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        sleep_seconds: float = 0.35,
        timeout_seconds: float = 45.0,
        screen_cache_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.timeout_seconds = timeout_seconds
        self.screen_cache_dir = screen_cache_dir
        if self.screen_cache_dir:
            self.screen_cache_dir.mkdir(parents=True, exist_ok=True)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def screen(self, *, idea: LiteratureIdeaCandidate, hit: MappingLike) -> dict[str, Any]:
        cache_path = self._screen_cache_path(idea=idea, hit=hit)
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return {**dict(hit), **cached}

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return compact JSON only. Do not infer results, effect "
                        "sizes, p-values, or novelty."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": (
                                "Decide whether this PubMed hit is a direct "
                                "same-topic prior study for the candidate."
                            ),
                            "candidate": {
                                "population": idea.population,
                                "predictor_or_exposure": idea.exposure_or_predictor,
                                "outcome": idea.outcome,
                                "analysis_family": idea.analysis_family,
                                "time_window_hint": idea.time_window_hint,
                                "aggregation_hint": idea.aggregation_hint,
                            },
                            "pubmed_hit": {
                                "pmid": hit.get("pmid"),
                                "title": hit.get("title"),
                                "abstract": str(hit.get("abstract") or "")[:2600],
                            },
                            "decision_schema": {
                                "direct_same_topic": "boolean",
                                "rationale": "one concise sentence",
                            },
                            "rules": [
                                (
                                    "direct_same_topic=true only if the hit studies "
                                    "substantially the same predictor/exposure, "
                                    "outcome, and differentiating design element"
                                ),
                                (
                                    "same broad field, same outcome alone, or adjacent "
                                    "background is not direct same-topic"
                                ),
                                "when uncertain, choose false and explain the mismatch",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 360,
            "reasoning": {"effort": "none", "exclude": True},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
            "X-Title": "EasyICU S6 validation",
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        screened = dict(hit)
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            message = ((data.get("choices") or [{}])[0].get("message") or {})
            content = str(
                message.get("content")
                or message.get("reasoning")
                or message.get("reasoning_content")
                or ""
            ).strip()
            decision = _parse_json_object(content)
            screened["same_topic_screened"] = True
            screened["direct_same_topic"] = bool(decision.get("direct_same_topic"))
            screened["direct_same_topic_rationale"] = str(
                decision.get("rationale") or "LLM same-topic screen completed"
            )
        except Exception as exc:  # noqa: BLE001
            screened["same_topic_screened"] = False
            screened["direct_same_topic"] = False
            screened["direct_same_topic_rationale"] = (
                f"LLM screen failed: {type(exc).__name__}: {str(exc)[:180]}"
            )
        if cache_path and screened.get("same_topic_screened"):
            cache_payload = {
                "same_topic_screened": bool(screened.get("same_topic_screened")),
                "direct_same_topic": bool(screened.get("direct_same_topic")),
                "direct_same_topic_rationale": screened.get(
                    "direct_same_topic_rationale"
                ),
                "screen_model": self.model,
            }
            cache_path.write_text(
                json.dumps(cache_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        time.sleep(self.sleep_seconds)
        return screened

    def _screen_cache_path(
        self,
        *,
        idea: LiteratureIdeaCandidate,
        hit: MappingLike,
    ) -> Optional[Path]:
        if self.screen_cache_dir is None:
            return None
        key = _hash_payload(
            {
                "model": self.model,
                "idea": {
                    "predictor": idea.exposure_or_predictor,
                    "outcome": idea.outcome,
                    "analysis_family": idea.analysis_family,
                    "time_window_hint": idea.time_window_hint,
                    "aggregation_hint": idea.aggregation_hint,
                },
                "hit": {
                    "pmid": hit.get("pmid"),
                    "title": hit.get("title"),
                    "abstract_sha256": sha256(
                        str(hit.get("abstract") or "").encode("utf-8")
                    ).hexdigest(),
                },
            }
        ).replace("sha256:", "")
        return self.screen_cache_dir / f"{key}.json"


MappingLike = dict[str, Any]


class PubMedPriorArtScreenClient:
    """PubMed client returning S6-compatible frozen prior-art records."""

    def __init__(
        self,
        *,
        screener: Optional[OpenRouterSameTopicScreener] = None,
        cache_dir: Optional[Path] = None,
        top_n_screen: int = 3,
    ) -> None:
        self.screener = screener
        self.top_n_screen = top_n_screen
        self.cache_dir = cache_dir
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def search_prior_art(
        self,
        query: str,
        *,
        max_results: int = 3,
        idea: Optional[LiteratureIdeaCandidate] = None,
    ) -> dict[str, Any]:
        cache_key = _hash_payload(
            {
                "query": query,
                "max_results": max_results,
                "idea_id": idea.literature_idea_id if idea else None,
                "screen": bool(self.screener),
            }
        ).replace("sha256:", "")
        if cache_key in self.cache:
            return self.cache[cache_key]
        cache_path = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cache[cache_key] = record
            return record

        try:
            search = pubmed_search(query, retmax=max_results)
            time.sleep(0.34)
            fetched = pubmed_fetch(search["pmids"])
        except Exception as exc:  # noqa: BLE001
            record = {
                "query": query,
                "hit_count": 0,
                "pmids": [],
                "top_hits": [],
                "search_error": f"{type(exc).__name__}: {str(exc)[:220]}",
            }
            self.cache[cache_key] = record
            if cache_path:
                cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            return record

        top_hits: list[dict[str, Any]] = []
        for hit in fetched[:max_results]:
            raw_hit: dict[str, Any] = {
                "pmid": hit["pmid"],
                "title": hit["title"],
                "venue": hit["journal"],
                "year": hit["year"],
                "abstract": hit["abstract"],
                "relevance": hit["abstract"],
                "same_topic_screened": False,
                "direct_same_topic": False,
            }
            if self.screener is not None and idea is not None:
                raw_hit = self.screener.screen(idea=idea, hit=raw_hit)
            top_hits.append(raw_hit)
        record = {
            "query": query,
            "hit_count": int(search["count"]),
            "pmids": search["pmids"],
            "top_hits": top_hits[: self.top_n_screen],
            "searched_at": _utc_now(),
        }
        self.cache[cache_key] = record
        if cache_path:
            cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


def _idea_from_control(control: dict[str, Any]) -> LiteratureIdeaCandidate:
    return LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:seed-controls",
        citation_key=control["control_id"],
        source_adapter_level="metadata_only",
        population=control["population"],
        exposure_or_predictor=control["exposure_or_predictor"],
        outcome=control["outcome"],
        rationale=control["rationale"],
        source_quote=control["source_quote"],
        analysis_family=control.get("analysis_family") or "association",
        time_window_hint=control.get("time_window_hint"),
        aggregation_hint=control.get("aggregation_hint"),
    )


def run_screen_validation(
    *,
    seeds_path: Path,
    output_dir: Path,
    search_client: PubMedPriorArtScreenClient,
    top_n: int,
) -> dict[str, Any]:
    seed_payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    controls = list(seed_payload["controls"])
    searched_at = _utc_now()
    rows: list[dict[str, Any]] = []
    for control in controls:
        idea = _idea_from_control(control)
        assessment = assess_prior_art_for_idea(
            idea,
            search_client=search_client,
            searched_at=searched_at,
            top_n=top_n,
        )
        predicted_positive = assessment.novelty_label == "already_done"
        truth_positive = control["truth"] == "known_published"
        rows.append(
            {
                "control": control,
                "assessment": assessment.model_dump(mode="json"),
                "truth_positive": truth_positive,
                "predicted_positive": predicted_positive,
            }
        )
    metrics = _classification_metrics(rows)
    payload = {
        "schema_version": "easyicu.idea_mining_s6_screen_validation/1",
        "searched_at": searched_at,
        "seeds_path": str(seeds_path),
        "top_n": top_n,
        "metrics": metrics,
        "rows": rows,
    }
    payload["snapshot_hash"] = _hash_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "screen_validation_results.json"
    md_path = output_dir / "screen_validation_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_screen_validation_report(payload), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path), **payload}


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["truth_positive"] and row["predicted_positive"] for row in rows)
    fp = sum((not row["truth_positive"]) and row["predicted_positive"] for row in rows)
    tn = sum((not row["truth_positive"]) and (not row["predicted_positive"]) for row in rows)
    fn = sum(row["truth_positive"] and (not row["predicted_positive"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    accuracy = (tp + tn) / len(rows) if rows else None
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "already_done_precision": precision,
        "already_done_recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def _pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100 * float(value):.1f}%"


def _render_screen_validation_report(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# S6 novelty-screen validation",
        "",
        "This is a local PubMed + LLM same-topic screening validation artifact. "
        "It is a triage check, not a novelty claim.",
        "",
        "Interpretation guard: the screen is asymmetric. An `already_done` "
        "label is the stronger signal when precision is high; an "
        "`apparently_gap` label is weak when recall is incomplete and must be "
        "treated only as a human prior-art review trigger.",
        "",
        f"- searched_at: `{payload['searched_at']}`",
        f"- top_n: `{payload['top_n']}`",
        f"- snapshot_hash: `{payload['snapshot_hash']}`",
        "",
        "## Confusion Matrix",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| TP known-published -> already_done | {m['tp']} |",
        f"| FP known-gap -> already_done | {m['fp']} |",
        f"| TN known-gap -> not already_done | {m['tn']} |",
        f"| FN known-published -> not already_done | {m['fn']} |",
        f"| already_done precision | {_pct(m['already_done_precision'])} |",
        f"| already_done recall | {_pct(m['already_done_recall'])} |",
        f"| specificity | {_pct(m['specificity'])} |",
        f"| accuracy | {_pct(m['accuracy'])} |",
        "",
        "## Per-control labels",
        "",
        "| control | truth | predicted | label | direct PMIDs | screen status |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        control = row["control"]
        assessment = row["assessment"]
        predicted = "already_done" if row["predicted_positive"] else "not_already_done"
        lines.append(
            "| "
            + " | ".join(
                [
                    control["control_id"],
                    control["truth"],
                    predicted,
                    assessment["novelty_label"],
                    ", ".join(assessment["direct_same_topic_pmids"]) or "none",
                    assessment["same_topic_screen_status"],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _recall_curve(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for item in rows:
        metrics = item["metrics"]
        curve.append(
            {
                "top_n": item["top_n"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "already_done_precision": metrics["already_done_precision"],
                "already_done_recall": metrics["already_done_recall"],
                "specificity": metrics["specificity"],
                "accuracy": metrics["accuracy"],
            }
        )
    return sorted(curve, key=lambda item: item["top_n"])


def _render_recall_curve_report(payload: dict[str, Any]) -> str:
    lines = [
        "# S6 novelty-screen recall curve",
        "",
        "This local harness artifact measures same-topic screen depth on seed "
        "controls. It is not a novelty claim.",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- snapshot_hash: `{payload['snapshot_hash']}`",
        "",
        "## Recall At Depth",
        "",
        "| top_n | TP | FP | TN | FN | precision | recall | specificity | accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["recall_curve"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["top_n"]),
                    str(row["tp"]),
                    str(row["fp"]),
                    str(row["tn"]),
                    str(row["fn"]),
                    _pct(row["already_done_precision"]),
                    _pct(row["already_done_recall"]),
                    _pct(row["specificity"]),
                    _pct(row["accuracy"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Scope note: `already_done` is the high-confidence negative screen "
            "when precision remains high. `apparently_gap` and downstream "
            "`recommend` remain human prior-art review triggers, especially at "
            "depths where recall is low.",
            "",
        ]
    )
    return "\n".join(lines)


def fetch_review_editorial_articles(*, retmax: int) -> list[dict[str, str]]:
    term = (
        '("Intensive Care Med"[Journal] OR "Crit Care"[Journal]) '
        'AND (review[Publication Type] OR editorial[Publication Type]) '
        'AND (future[Title/Abstract] OR "future research"[Title/Abstract] '
        'OR "further studies"[Title/Abstract] OR unresolved[Title/Abstract] '
        'OR uncertainty[Title/Abstract] OR "research agenda"[Title/Abstract] '
        'OR needs[Title/Abstract]) '
        'AND ("critical care"[Title/Abstract] OR ICU[Title/Abstract] '
        'OR "intensive care"[Title/Abstract])'
    )
    search = pubmed_search(term, retmax=retmax)
    articles = pubmed_fetch(search["pmids"])
    if len(articles) >= retmax:
        return articles[:retmax]
    fallback = (
        '("Intensive Care Med"[Journal] OR "Crit Care"[Journal]) '
        'AND (review[Publication Type] OR editorial[Publication Type]) '
        'AND ("critical care"[Title/Abstract] OR ICU[Title/Abstract] '
        'OR "intensive care"[Title/Abstract])'
    )
    more = [pmid for pmid in pubmed_search(fallback, retmax=retmax * 2)["pmids"]]
    existing = {article["pmid"] for article in articles}
    articles.extend(pubmed_fetch([pmid for pmid in more if pmid not in existing]))
    return articles[:retmax]


def build_materials(articles: list[dict[str, str]]) -> list[SourceMaterial]:
    materials: list[SourceMaterial] = []
    for article in articles:
        citation = CitationRecord(
            key=f"pubmed_{article['pmid']}",
            title=article["title"],
            year=article["year"],
            venue=article["journal"],
            relevance=article["abstract"],
            url=article["url"],
            pmid=article["pmid"],
        )
        text = f"{article['title']} {article['abstract']}".strip()
        if len(text) < 120:
            continue
        materials.append(
            SourceMaterial(
                citation=citation,
                source_adapter_level="user_supplied_excerpt",
                locator=article["url"],
                source_text=text,
            )
        )
    return materials


GAP_RE = re.compile(
    r"\b(future|further|needed|needs|should|unresolved|unknown|unclear|"
    r"uncertainty|research|studies|trials|evidence|investigate|determine|"
    r"identify|agenda|validated|validation)\b",
    re.I,
)
CONCEPT_RE = re.compile(
    r"\b(lactate|vasopressor|norepinephrine|ventilation|driving pressure|"
    r"creatinine|kidney|aki|sofa|organ dysfunction|shock|oxygenation|fluid|"
    r"delirium|sedation|weakness|sepsis)\b",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if len(piece.strip()) > 30]


def _topic_from_sentence(
    sentence: str,
) -> Optional[tuple[str, str, str, Optional[str], Optional[str]]]:
    s = sentence.lower()
    if "lactate" in s:
        return ("lactate clearance trajectory", "mortality", "trajectory", "first 24 hours", "clearance")
    if "vasopressor" in s or "norepinephrine" in s:
        if "load" in s or "dose" in s:
            return ("vasopressor load", "mortality", "association", "shock resuscitation", "load")
        return ("vasopressor exposure", "mortality", "association", "shock resuscitation", None)
    if "driving pressure" in s:
        return ("driving pressure", "mortality", "association", "mechanical ventilation", None)
    if "ventilat" in s:
        return ("mechanical ventilation", "mortality", "association", None, None)
    if "kidney" in s or "aki" in s or "creatinine" in s:
        return ("creatinine trajectory", "mortality", "trajectory", "first 24 hours", "trajectory")
    if "sofa" in s or "organ dysfunction" in s:
        return ("SOFA score trajectory", "mortality", "trajectory", "first 24 hours", "trajectory")
    if "fluid" in s:
        return ("fluid balance", "mortality", "association", "first 24 hours", "balance")
    if "delirium" in s or "sedation" in s:
        return ("delirium", "mortality", "association", None, None)
    if "weakness" in s:
        return ("ICU-acquired weakness", "mortality", "association", None, None)
    return None


class DeterministicGapExtractor:
    """Deterministic local extractor used only by this validation harness."""

    name = "deterministic_gap_extractor"

    def __init__(self) -> None:
        self.skipped_generic_gap_sentences: list[str] = []

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        payload = json.loads(messages[1].content)
        ideas: list[dict[str, Any]] = []
        for source in payload["sources"]:
            text = source.get("available_source_text") or ""
            scored: list[tuple[int, str]] = []
            for sent in split_sentences(text):
                has_gap = bool(GAP_RE.search(sent))
                has_concept = bool(CONCEPT_RE.search(sent))
                if has_gap and not has_concept:
                    self.skipped_generic_gap_sentences.append(sent[:300])
                    continue
                score = 0
                if has_gap:
                    score += 3
                if has_concept:
                    score += 2
                if len(sent) > 380:
                    score -= 1
                if score > 0:
                    scored.append((score, sent))
            scored.sort(key=lambda item: (-item[0], len(item[1])))
            for _, sent in scored[:2]:
                topic = _topic_from_sentence(sent)
                if topic is None:
                    self.skipped_generic_gap_sentences.append(sent[:300])
                    continue
                predictor, outcome, family, window, aggregation = topic
                ideas.append(
                    {
                        "citation_key": source["citation_key"],
                        "population": "adult ICU patients",
                        "exposure_or_predictor": predictor,
                        "outcome": outcome,
                        "rationale": (
                            "The abstract contains gap, uncertainty, validation, "
                            "or future-direction language."
                        ),
                        "source_quote": sent[:800],
                        "analysis_family": family,
                        "time_window_hint": window,
                        "aggregation_hint": aggregation,
                    }
                )
        return json.dumps(ideas, ensure_ascii=False)


def _aggregate_column(data_dir: Path, file_name: str, column: str, method: str) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / file_name, columns=["stay_id", column])
    if method == "median":
        return df.groupby("stay_id", as_index=False)[column].median()
    if method == "first":
        return df.groupby("stay_id", as_index=False)[column].first()
    return df.groupby("stay_id", as_index=False)[column].max()


def build_wide_cohort(data_dir: Path, output_path: Path) -> dict[str, Any]:
    outcome = pd.read_parquet(data_dir / "outcome_death_los_hosp_los_icu.parquet")
    wide = outcome[["stay_id", "death"]].drop_duplicates("stay_id").copy()
    wide["death"] = wide["death"].fillna(0).astype(int)
    specs = [
        ("vitals_dbp_hr_map_resp_etc7.parquet", "map", "median"),
        ("blood_gas_be_cai_lact_methb_etc8.parquet", "lact", "max"),
        ("chemistry_alb_alp_alt_ast_etc20.parquet", "crea", "max"),
        ("sofa2_score_sofa2_sofa2_cardio_sofa2_cns_sofa2_coag_etc7.parquet", "sofa2", "max"),
        ("vasopressors_adh_rate_dobu60_dobu_dur_dobu_rate_etc17.parquet", "vaso_ind", "max"),
        ("vasopressors_adh_rate_dobu60_dobu_dur_dobu_rate_etc17.parquet", "norepi_equiv", "max"),
        ("ventilator_compliance_driving_pres_etco2_mean_airway_pres_etc12.parquet", "driving_pres", "median"),
        ("respiratory_adv_resp_ecmo_ecmo_indication_ett_gcs_etc12.parquet", "vent_ind", "max"),
        ("respiratory_adv_resp_ecmo_ecmo_indication_ett_gcs_etc12.parquet", "fio2", "median"),
        (
            "renal_aki_aki_stage_aki_stage_creat_aki_stage_rrt_aki_stage_uo_creat_low_past_48hr_creat_low_past_7day_uo_rt_12hr_uo_rt_24hr_uo_rt_6hr.parquet",
            "aki",
            "max",
        ),
    ]
    for file_name, column, method in specs:
        wide = wide.merge(
            _aggregate_column(data_dir, file_name, column, method),
            on="stay_id",
            how="left",
        )
    wide.to_parquet(output_path, index=False)
    return {
        "n_stays": int(len(wide)),
        "columns": list(wide.columns),
        "nonmissing_fraction": {
            column: float(wide[column].notna().mean())
            for column in wide.columns
            if column != "stay_id"
        },
    }


def base_concepts() -> list[ConceptDescriptor]:
    return [
        ConceptDescriptor(name="death", source_concept="death", role=VariableRole.OUTCOME, dtype="int64"),
        ConceptDescriptor(name="lact", source_concept="lact", role=VariableRole.LAB, dtype="float64"),
        ConceptDescriptor(name="crea", source_concept="crea", role=VariableRole.LAB, dtype="float64"),
        ConceptDescriptor(name="map", source_concept="map", role=VariableRole.VITAL, dtype="float64"),
        ConceptDescriptor(name="sofa2", source_concept="sofa2", role=VariableRole.COMPOSITE_SCORE, dtype="float64"),
        ConceptDescriptor(name="vaso_ind", source_concept="vaso_ind", role=VariableRole.INTERVENTION, dtype="float64"),
        ConceptDescriptor(name="norepi_equiv", source_concept="norepi_equiv", role=VariableRole.INTERVENTION, dtype="float64"),
        ConceptDescriptor(name="driving_pres", source_concept="driving_pres", role=VariableRole.VITAL, dtype="float64"),
        ConceptDescriptor(name="vent_ind", source_concept="vent_ind", role=VariableRole.INTERVENTION, dtype="float64"),
        ConceptDescriptor(name="fio2", source_concept="fio2", role=VariableRole.VITAL, dtype="float64"),
        ConceptDescriptor(name="aki", source_concept="aki", role=VariableRole.ORDINAL_SCORE, dtype="float64"),
    ]


def dictionary_aliases_for_concepts(
    concepts: Sequence[ConceptDescriptor],
) -> dict[str, list[str]]:
    """Build phrase aliases from the EasyICU concept dictionary.

    This intentionally avoids a new S6 clinical synonym table. It only exposes
    strings already present in EasyICU's concept definitions: concept names,
    descriptions, callbacks, component concepts and dependency concepts.
    """

    from easyicu.concept_loader import _load_concept_dict_cached

    concept_dict = _load_concept_dict_cached()
    aliases: dict[str, list[str]] = {}
    for concept in concepts:
        key = concept.source_concept or concept.name
        payloads: list[dict[str, Any]] = []
        for candidate_key in _dictionary_lookup_keys(key):
            payload = concept_dict.get(candidate_key)
            if isinstance(payload, dict) and payload not in payloads:
                payloads.append(payload)
        values: list[str] = [key, concept.name, concept.source_concept or ""]
        values.extend(concept.derived_from_concepts)
        for payload in payloads:
            values.extend(
                str(payload.get(field) or "")
                for field in (
                    "description",
                    "category",
                    "target",
                    "callback",
                    "class",
                    "class_name",
                )
            )
            for field in ("concepts", "depends_on"):
                raw = payload.get(field) or []
                if isinstance(raw, str):
                    values.append(raw)
                elif isinstance(raw, Iterable):
                    values.extend(str(item) for item in raw)
            for entries in (payload.get("sources") or {}).values():
                for entry in entries or []:
                    if not isinstance(entry, dict):
                        continue
                    for field in ("ids", "regex", "callback", "class", "class_name"):
                        raw = entry.get(field)
                        if raw is None:
                            continue
                        if isinstance(raw, (list, tuple)):
                            values.extend(str(item) for item in raw)
                        else:
                            values.append(str(raw))
        cleaned = _ordered_unique_local(
            value.replace("_", " ")
            for value in values
            if str(value or "").strip()
        )
        if cleaned:
            aliases[key] = cleaned
    return aliases


def _dictionary_lookup_keys(key: str) -> list[str]:
    normalised = str(key or "").strip()
    keys = [normalised]
    without_digits = re.sub(r"\d+$", "", normalised)
    if without_digits and without_digits != normalised:
        keys.append(without_digits)
    return _ordered_unique_local(keys)


def _ordered_unique_local(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def resolution_summary(
    ideas: Sequence[LiteratureIdeaCandidate],
    *,
    concepts: Sequence[ConceptDescriptor],
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict[str, Any]:
    candidates = [
        map_literature_idea_to_executable_candidate(
            idea,
            available_concepts=concepts,
            concept_aliases=concept_aliases,
            outcome_determinability={
                "death": OutcomeDeterminability(
                    outcome="death",
                    status="known_0_1",
                ),
                "mortality": OutcomeDeterminability(
                    outcome="mortality",
                    status="known_0_1",
                    normalized_outcome_concept="death",
                ),
            },
        )
        for idea in ideas
    ]
    return {
        "n": len(candidates),
        "resolved_predictor": sum(
            1 for candidate in candidates if candidate.resolved_predictor_concept
        ),
        "resolved_outcome": sum(
            1 for candidate in candidates if candidate.resolved_outcome_concept
        ),
        "executable": sum(1 for candidate in candidates if candidate.executable),
        "feature_derivation_status_counts": _status_counts(candidates),
        "top_unresolved_predictors": dict(
            Counter(
                candidate.predictor_label
                for candidate in candidates
                if candidate.resolved_predictor_concept is None
            ).most_common(8)
        ),
    }


def _status_counts(candidates: list[Any]) -> dict[str, int]:
    return dict(Counter(candidate.feature_derivation_status for candidate in candidates))


def run_yield_smoke(
    *,
    output_dir: Path,
    search_client: PubMedPriorArtScreenClient,
    data_dir: Path,
    article_count: int,
    top_n: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = fetch_review_editorial_articles(retmax=article_count)
    materials = build_materials(articles)
    wide_path = output_dir / "miiv_wide_cohort.parquet"
    cohort_summary = build_wide_cohort(data_dir, wide_path)
    concepts = base_concepts()
    concept_aliases = dictionary_aliases_for_concepts(concepts)
    extractor = DeterministicGapExtractor()
    result = run_idea_mining_dry_run(
        materials=materials,
        llm=extractor,
        available_concepts=concepts,
        concept_aliases=concept_aliases,
        outcome_determinability={
            "death": OutcomeDeterminability(outcome="death", status="known_0_1"),
            "mortality": OutcomeDeterminability(
                outcome="mortality",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
        },
        output_dir=output_dir / "dry_run",
        database="miiv",
        data_path=wide_path,
        analytic_unit="stay",
        top_k=12,
        prior_art_search_client=search_client,
        prior_art_searched_at=_utc_now(),
        prior_art_top_n=top_n,
    )
    payload = result.model_dump(mode="json")
    triage_payload = json.loads(
        Path(payload["triage_report_path"]).read_text(encoding="utf-8")
    )
    novelty_labels = Counter(
        record["prior_art"]["novelty_label"]
        for record in payload["discovery_records"]
    )
    decisions = Counter(record["go_no_go"] for record in payload["discovery_records"])
    screen_status = Counter(
        record["prior_art"]["same_topic_screen_status"]
        for record in payload["discovery_records"]
    )
    resolution_recall_comparison = {
        "baseline_no_dictionary_aliases": resolution_summary(
            result.literature_ideas,
            concepts=concepts,
        ),
        "dictionary_backed_aliases": resolution_summary(
            result.literature_ideas,
            concepts=concepts,
            concept_aliases=concept_aliases,
        ),
    }
    summary = {
        "schema_version": "easyicu.idea_mining_s6_real_yield_validation/1",
        "searched_at": _utc_now(),
        "article_count_requested": article_count,
        "article_count_materials": len(materials),
        "article_pmids": [material.citation.pmid for material in materials],
        "cohort_summary": cohort_summary,
        "extraction_specificity": {
            "skipped_generic_gap_sentence_count": len(
                extractor.skipped_generic_gap_sentences
            ),
            "skipped_generic_gap_sentence_examples": extractor.skipped_generic_gap_sentences[:8],
        },
        "yield_report": payload["yield_report"],
        "resolution_recall_comparison": resolution_recall_comparison,
        "feature_derivation_status_counts": _status_counts(result.executable_candidates),
        "novelty_label_distribution": dict(novelty_labels),
        "go_no_go_distribution": dict(decisions),
        "same_topic_screen_status_distribution": dict(screen_status),
        "discovery_counts": triage_payload.get("discovery_counts", {}),
        "warnings": payload["warnings"],
        "paths": {
            "triage_report": payload["triage_report_path"],
            "discovery_report": payload["discovery_report_path"],
            "novelty_snapshot": payload["novelty_snapshot_path"],
        },
    }
    summary["snapshot_hash"] = _hash_payload(summary)
    summary_path = output_dir / "real_yield_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "real_yield_report.md").write_text(
        _render_real_yield_report(summary),
        encoding="utf-8",
    )
    return {"summary_path": str(summary_path), **summary}


def _render_real_yield_report(summary: dict[str, Any]) -> str:
    y = summary["yield_report"]
    lines = [
        "# S6 real-yield validation",
        "",
        "Local PubMed abstract + MIIV wide-cohort discovery smoke. "
        "All candidates remain triage-only and require human review.",
        "",
        f"- searched_at: `{summary['searched_at']}`",
        f"- materials: `{summary['article_count_materials']}`",
        f"- snapshot_hash: `{summary['snapshot_hash']}`",
        "",
        "## Extraction specificity",
        "",
        f"- skipped_generic_gap_sentence_count: `{summary.get('extraction_specificity', {}).get('skipped_generic_gap_sentence_count', 0)}`",
        "",
        "## Four-stage conversion",
        "",
        "| stage | n |",
        "|---|---:|",
        f"| extracted literature rows | {y['n_literature_ideas']} |",
        f"| resolved predictor | {y['n_resolved_predictor']} |",
        f"| resolved outcome | {y['n_resolved_outcome']} |",
        f"| executable | {y['n_executable']} |",
        f"| non-executable | {y['n_non_executable']} |",
        "",
        "## Distributions",
        "",
        f"- feature_derivation_status: `{summary['feature_derivation_status_counts']}`",
        f"- novelty_label: `{summary['novelty_label_distribution']}`",
        f"- go_no_go: `{summary['go_no_go_distribution']}`",
        f"- discovery_counts: `{summary['discovery_counts']}`",
        "",
        "## Resolution recall comparison",
        "",
        "| resolver | resolved predictor | resolved outcome | executable | status counts |",
        "|---|---:|---:|---:|---|",
    ]
    for label, metrics in summary.get("resolution_recall_comparison", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(metrics["resolved_predictor"]),
                    str(metrics["resolved_outcome"]),
                    str(metrics["executable"]),
                    str(metrics["feature_derivation_status_counts"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
        ]
    )
    return "\n".join(lines)


def _read_key_from_env_or_stdin(use_stdin: bool) -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if use_stdin:
        return sys.stdin.readline().strip() or None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["screen-validation", "real-yield", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / datetime.now().strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--miiv-dir", type=Path, default=DEFAULT_MIIV)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--recall-depths",
        default=None,
        help="Comma-separated top-N depths for screen-validation recall curves.",
    )
    parser.add_argument("--article-count", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--openrouter-timeout", type=float, default=45.0)
    parser.add_argument("--screen-cache-dir", type=Path, default=None)
    parser.add_argument("--openrouter-key-stdin", action="store_true")
    parser.add_argument("--no-llm-screen", action="store_true")
    return parser.parse_args()


def _parse_depths(raw: Optional[str], fallback: int) -> list[int]:
    if not raw:
        return [fallback]
    depths: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        depth = int(item)
        if depth <= 0:
            raise ValueError("--recall-depths values must be positive")
        depths.append(depth)
    return sorted(set(depths)) or [fallback]


def main() -> None:
    args = parse_args()
    api_key = _read_key_from_env_or_stdin(args.openrouter_key_stdin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    screener = None
    if api_key and not args.no_llm_screen:
        screener = OpenRouterSameTopicScreener(
            api_key=api_key,
            model=args.model,
            base_url=args.openrouter_base_url,
            timeout_seconds=args.openrouter_timeout,
            screen_cache_dir=args.screen_cache_dir
            or args.out_dir / "openrouter_screen_cache",
        )
    elif not args.no_llm_screen:
        raise SystemExit(
            "OPENROUTER_API_KEY missing. Set env var or pass --openrouter-key-stdin."
        )

    outputs: dict[str, Any] = {
        "schema_version": "easyicu.idea_mining_s6_validation_harness/1",
        "created_at": _utc_now(),
        "model": args.model if screener else None,
        "top_n": args.top_n,
        "mode": args.mode,
    }
    if args.mode in {"screen-validation", "all"}:
        depths = _parse_depths(args.recall_depths, args.top_n)
        screen_outputs = []
        for depth in depths:
            search_client = PubMedPriorArtScreenClient(
                screener=screener,
                cache_dir=args.out_dir / "pubmed_prior_art_cache",
                top_n_screen=depth,
            )
            screen_outputs.append(
                run_screen_validation(
                    seeds_path=args.seeds,
                    output_dir=args.out_dir / f"screen_validation_top{depth}",
                    search_client=search_client,
                    top_n=depth,
                )
            )
        outputs["screen_validation_by_depth"] = screen_outputs
        outputs["screen_validation_recall_curve"] = _recall_curve(screen_outputs)
        curve_payload = {
            "schema_version": "easyicu.idea_mining_s6_recall_curve/1",
            "created_at": _utc_now(),
            "model": args.model if screener else None,
            "recall_curve": outputs["screen_validation_recall_curve"],
        }
        curve_payload["snapshot_hash"] = _hash_payload(curve_payload)
        curve_json = args.out_dir / "screen_validation_recall_curve.json"
        curve_md = args.out_dir / "screen_validation_recall_curve.md"
        curve_json.write_text(
            json.dumps(curve_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        curve_md.write_text(
            _render_recall_curve_report(curve_payload),
            encoding="utf-8",
        )
        outputs["screen_validation_recall_curve_path"] = str(curve_json)
        outputs["screen_validation_recall_curve_report"] = str(curve_md)
    if args.mode in {"real-yield", "all"}:
        search_client = PubMedPriorArtScreenClient(
            screener=screener,
            cache_dir=args.out_dir / "pubmed_prior_art_cache",
            top_n_screen=args.top_n,
        )
        outputs["real_yield"] = run_yield_smoke(
            output_dir=args.out_dir / "real_yield",
            search_client=search_client,
            data_dir=args.miiv_dir,
            article_count=args.article_count,
            top_n=args.top_n,
        )
    outputs["snapshot_hash"] = _hash_payload(outputs)
    summary_path = args.out_dir / "harness_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {
        "summary_path": str(summary_path),
        "snapshot_hash": outputs["snapshot_hash"],
        "mode": outputs["mode"],
        "top_n": outputs["top_n"],
    }
    if "screen_validation_by_depth" in outputs:
        compact["screen_validation_recall_curve"] = outputs[
            "screen_validation_recall_curve"
        ]
        compact["screen_validation_recall_curve_report"] = outputs[
            "screen_validation_recall_curve_report"
        ]
    if "real_yield" in outputs:
        compact["real_yield_summary"] = outputs["real_yield"]["summary_path"]
        compact["real_yield_conversion"] = outputs["real_yield"]["yield_report"]
        compact["real_yield_novelty_labels"] = outputs["real_yield"][
            "novelty_label_distribution"
        ]
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
