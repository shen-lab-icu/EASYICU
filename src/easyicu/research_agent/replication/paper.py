"""Paper-aware replication helpers for EasyICU research-agent runs.

The base pipeline answers one research question over one cohort. This module
adds a deterministic-first layer for "replicate a published paper" mode:

1. parse a paper-like text/PDF extract into a typed profile;
2. map the paper design onto EasyICU-supported concepts;
3. compare paper claims against structured EasyICU outputs; and
4. render explicit replication/deviation artefacts.
"""

from __future__ import annotations

import csv
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .metrics import compare_metric_values
from ..schema import (
    AnalysisManifest,
    AnalysisPlan,
    PaperClaimRecord,
    PaperProfile,
    PaperReplicationSpec,
    PaperResultLedger,
    PipelineResult,
    ReplicationDeviationItem,
    ReplicationDeviationReport,
    ResearchContext,
)

SUPPORTED_CONCEPT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "sofa2": ("sofa-2", "sofa 2", "sofa2", "sofa"),
    "age": ("age",),
    "sex": ("sex", "gender"),
    "lact": ("lactate", "lact"),
    "map": ("mean arterial pressure", "map"),
    "vaso": ("vasopressor", "vasopressor exposure", "norepinephrine", "vaso"),
    "creat": ("creatinine", "creat"),
    "death": (
        "icu mortality",
        "hospital mortality",
        "28-day mortality",
        "28 day mortality",
        "mortality",
        "death",
    ),
    "los_icu": ("icu length of stay", "icu los", "length of stay"),
}

UNSUPPORTED_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bwaveform\b", "Paper requires bedside waveform data, which this mode does not support."),
    (r"\bimaging\b|\bct\b|\bmr[i]?\b", "Paper requires imaging-derived features."),
    (r"\bomics\b|\bproteom|\bmetabolom|\btranscriptom", "Paper requires omics features."),
    (r"\bfree text\b|\bclinical notes?\b|\bnlp\b", "Paper relies on unstructured text/NLP features."),
    (r"\btrial emulation\b|\btarget trial\b", "Paper requires strong causal emulation design support."),
    (r"\brandomi[sz]ed\b|\brct\b", "Paper is an interventional/randomised design, not the initial replication target."),
)


def load_paper_source(paper_or_text: str | Path) -> tuple[str, str]:
    """Return a display name and paper text.

    Accepts either raw text or a path to ``.txt``, ``.md`` or ``.pdf``. PDF
    extraction is best-effort and intentionally optional so test runs remain
    lightweight.
    """
    if isinstance(paper_or_text, Path):
        paper_path = paper_or_text
    else:
        raw = str(paper_or_text)
        if "\n" in raw or len(raw) > 240:
            paper_path = None
        else:
            candidate = Path(raw).expanduser()
            try:
                paper_path = candidate if candidate.exists() else None
            except OSError:
                paper_path = None
    if paper_path is not None and paper_path.exists():
        suffix = paper_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return str(paper_path), paper_path.read_text(encoding="utf-8")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(str(paper_path))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
                return str(paper_path), text
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ValueError(
                    f"Could not extract text from PDF {paper_path}: {type(exc).__name__}: {exc}"
                ) from exc
        return str(paper_path), paper_path.read_text(encoding="utf-8", errors="replace")
    return "inline_text", str(paper_or_text)


def parse_paper_profile(paper_or_text: str | Path) -> PaperProfile:
    """Parse a paper-like source into a typed profile."""
    source_name, text = load_paper_source(paper_or_text)
    norm = _normalise_whitespace(text)
    lower = norm.lower()

    unsupported_reasons = [
        reason for pattern, reason in UNSUPPORTED_PATTERNS if re.search(pattern, lower)
    ]
    title = _extract_title(norm)
    target_outcome = _extract_target_outcome(lower)
    paper_type = _infer_paper_type(lower)
    primary_exposure = _extract_primary_signal(lower, exclude=(target_outcome or "",))
    primary_predictor = primary_exposure if paper_type == "prediction" else None
    covariates = _extract_covariates(norm)
    primary_method = _extract_primary_method(norm)
    key_claims = _extract_key_claims(norm, target_outcome=target_outcome, predictor=primary_exposure)
    research_question = _extract_research_question(
        title=title,
        predictor=primary_exposure,
        outcome=target_outcome,
        paper_type=paper_type,
    )
    table_figure_inventory = sorted(
        {
            match.group(0).strip()
            for match in re.finditer(r"\b(?:Table|Figure)\s+[A-Za-z0-9]+\b", norm)
        }
    )
    inclusion = _extract_bullets_after_heading(norm, ("inclusion", "included"))
    exclusion = _extract_bullets_after_heading(norm, ("exclusion", "excluded"))
    secondary = _extract_secondary_analyses(lower)
    cohort_definition = _extract_cohort_definition(norm)

    if not target_outcome:
        unsupported_reasons.append("Could not identify a supported outcome from the paper text.")
    if not primary_exposure and paper_type in {"association", "prediction", "survival", "fairness"}:
        unsupported_reasons.append("Could not identify a primary exposure/predictor from the paper text.")
    if not key_claims:
        unsupported_reasons.append("Could not parse any key numeric claims from the paper text.")

    if unsupported_reasons:
        paper_type = "unsupported_or_underspecified"

    return PaperProfile(
        paper_source=source_name,
        paper_title=title,
        paper_type=paper_type,
        research_question=research_question,
        target_outcome=target_outcome,
        cohort_definition=cohort_definition,
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        primary_exposure=primary_exposure,
        primary_predictor=primary_predictor,
        covariates=covariates,
        primary_analysis_method=primary_method,
        secondary_analyses=secondary,
        table_figure_inventory=table_figure_inventory,
        key_claims=key_claims,
        unsupported_reasons=unsupported_reasons,
    )


def build_paper_replication_spec(profile: PaperProfile) -> tuple[PaperReplicationSpec, ReplicationDeviationReport]:
    """Map a parsed paper profile onto EasyICU-supported execution contracts."""
    mapped: Dict[str, str] = {}
    unmappable: List[str] = []
    approximations: Dict[str, str] = {}
    notes: List[str] = []
    deviations: List[ReplicationDeviationItem] = []

    for label, concept in _iter_profile_targets(profile):
        mapped_name = map_text_to_easyicu_concept(concept)
        if mapped_name is not None:
            mapped[label] = mapped_name
            if concept.lower() != mapped_name.lower():
                approximations[label] = f"{concept} -> {mapped_name}"
                deviations.append(
                    ReplicationDeviationItem(
                        item=label,
                        severity="info",
                        original=concept,
                        easyicu_proxy=mapped_name,
                        reason="Mapped to the closest supported EasyICU concept alias.",
                    )
                )
        else:
            unmappable.append(f"{label}: {concept}")
            deviations.append(
                ReplicationDeviationItem(
                    item=label,
                    severity="error",
                    original=concept,
                    reason="No supported EasyICU concept mapping was found.",
                )
            )

    if profile.target_outcome and "mortality" in profile.target_outcome.lower():
        notes.append("Treat mortality claims as EasyICU `death` unless the paper requires a more specific endpoint.")
    if profile.primary_analysis_method:
        notes.append(f"Primary analysis method parsed as: {profile.primary_analysis_method}")

    required_outputs = ["table_one", "outcome_incidence", "figure_bundle"]
    if profile.paper_type in {"association", "survival"}:
        required_outputs.extend(["primary_model", "subgroup", "sensitivity"])
    elif profile.paper_type == "prediction":
        required_outputs.extend(["primary_model", "calibration", "subgroup"])
    elif profile.paper_type == "fairness":
        required_outputs.extend(["primary_model", "subgroup", "sensitivity"])

    alignment_targets = []
    if profile.primary_exposure:
        alignment_targets.append(f"primary effect direction for {profile.primary_exposure}")
    if profile.target_outcome:
        alignment_targets.append(f"outcome definition aligned to {profile.target_outcome}")
    for claim in profile.key_claims:
        if claim.metric:
            alignment_targets.append(claim.metric)

    if profile.paper_type == "unsupported_or_underspecified":
        notes.extend(profile.unsupported_reasons)

    supported = profile.paper_type != "unsupported_or_underspecified" and not any(
        item.severity == "error" for item in deviations
    )
    summary = (
        "Paper can be replicated to design-and-conclusion alignment in EasyICU."
        if supported
        else "Paper is unsupported or underspecified for strict EasyICU replication."
    )
    if profile.unsupported_reasons:
        deviations.extend(
            ReplicationDeviationItem(
                item="paper_support",
                severity="error",
                original=profile.paper_title or profile.paper_source,
                reason=reason,
            )
            for reason in profile.unsupported_reasons
        )

    spec = PaperReplicationSpec(
        paper_title=profile.paper_title,
        paper_type=profile.paper_type,
        mapped_concepts=mapped,
        unmappable_items=unmappable,
        approximate_substitutions=approximations,
        time_windows=_infer_time_windows(profile),
        required_outputs=sorted(set(required_outputs)),
        alignment_targets=sorted(set(alignment_targets)),
        notes=notes,
    )
    deviation_report = ReplicationDeviationReport(
        supported=supported,
        summary=summary,
        items=deviations,
    )
    return spec, deviation_report


def build_paper_result_ledger(
    *,
    paper_profile: PaperProfile,
    manifest: Dict[str, Any],
    context_payload: Optional[Dict[str, Any]] = None,
) -> PaperResultLedger:
    """Combine parsed paper claims with EasyICU structured metrics."""
    return PaperResultLedger(
        paper_claims=paper_profile.key_claims,
        easyicu_metrics=collect_easyicu_metrics(
            manifest=manifest,
            context_payload=context_payload,
        ),
    )


def collect_easyicu_metrics(
    *,
    manifest: Dict[str, Any],
    context_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract a compact EasyICU metric dictionary from manifest records."""
    metrics: Dict[str, Any] = {}
    per_step_records = manifest.get("per_step_records") or []
    for record in per_step_records:
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        _update_metric(metrics, "outcome_rate", summary, ("outcome_rate", "statistic:outcome_rate"))
        _update_metric(metrics, "primary_or", summary, ("primary_or", "odds_ratio", "statistic:primary_or"))
        _update_metric(metrics, "primary_pvalue", summary, ("primary_pvalue", "p_value", "pvalue"))
        _update_metric(metrics, "primary_ci_low", summary, ("primary_ci_low", "ci_low", "conf_low"))
        _update_metric(metrics, "primary_ci_high", summary, ("primary_ci_high", "ci_high", "conf_high"))
        _update_metric(metrics, "auroc", summary, ("auroc", "auc", "statistic:auroc", "held_out_auroc"))
        _update_metric(metrics, "brier_score", summary, ("brier_score", "statistic:brier_score", "held_out_brier"))
        predictor = summary.get("predictor") or summary.get("primary_predictor")
        if predictor and "predictor" not in metrics:
            metrics["predictor"] = predictor
        outcome = summary.get("outcome") or summary.get("target_outcome")
        if outcome and "target_outcome" not in metrics:
            metrics["target_outcome"] = outcome
        if "primary_or" not in metrics and summary.get("estimate") is not None:
            try:
                metrics["primary_or"] = float(math.exp(float(summary["estimate"])))
            except Exception:
                pass
        if "primary_ci_low" not in metrics and summary.get("ci_lower") is not None:
            try:
                metrics["primary_ci_low"] = float(math.exp(float(summary["ci_lower"])))
            except Exception:
                pass
        if "primary_ci_high" not in metrics and summary.get("ci_upper") is not None:
            try:
                metrics["primary_ci_high"] = float(math.exp(float(summary["ci_upper"])))
            except Exception:
                pass
    if context_payload:
        cohort = context_payload.get("cohort") if isinstance(context_payload, dict) else None
        if isinstance(cohort, dict):
            metrics.setdefault("n_stays", cohort.get("n_stays"))
            metrics.setdefault("n_patients", cohort.get("n_patients"))
        if isinstance(context_payload, dict):
            metrics.setdefault("target_outcome", context_payload.get("target_outcome"))
    return {k: v for k, v in metrics.items() if v is not None}


def compare_paper_to_easyicu(
    *,
    paper_profile: PaperProfile,
    ledger: PaperResultLedger,
) -> List[Dict[str, Any]]:
    """Generate a claim-by-claim comparison table."""
    rows: List[Dict[str, Any]] = []
    for claim in paper_profile.key_claims:
        metric_key = _map_claim_metric_to_easyicu_key(claim.metric)
        easyicu_value = ledger.easyicu_metrics.get(metric_key) if metric_key else None
        alignment, reason = compare_metric_values(
            metric=claim.metric,
            paper_value=claim.numeric_value,
            paper_direction=claim.direction,
            easyicu_value=easyicu_value,
        )
        rows.append(
            {
                "claim_id": claim.claim_id,
                "paper_claim": claim.sentence,
                "paper_value": claim.paper_value or ("" if claim.numeric_value is None else str(claim.numeric_value)),
                "easyicu_value": "" if easyicu_value is None else str(easyicu_value),
                "alignment_status": alignment,
                "reason_if_mismatch": reason,
                "metric": claim.metric or "",
            }
        )
    return rows


def render_replication_report(
    *,
    paper_profile: PaperProfile,
    spec: PaperReplicationSpec,
    deviation_report: ReplicationDeviationReport,
    comparison_rows: Sequence[Dict[str, Any]],
    ledger: PaperResultLedger,
) -> str:
    """Render a paper-aware replication report."""
    lines = [
        "# EasyICU replication report",
        "",
        f"- Paper: {paper_profile.paper_title or paper_profile.paper_source}",
        f"- Paper type: `{paper_profile.paper_type}`",
        f"- Replication goal: `{spec.replication_goal}`",
        f"- Supported in EasyICU: `{deviation_report.supported}`",
        "",
        "## Study design",
        "",
        f"- Research question: {paper_profile.research_question or '(not parsed)'}",
        f"- Cohort definition: {paper_profile.cohort_definition or '(not parsed)'}",
        f"- Primary exposure/predictor: {paper_profile.primary_exposure or paper_profile.primary_predictor or '(not parsed)'}",
        f"- Target outcome: {paper_profile.target_outcome or '(not parsed)'}",
        f"- Primary analysis: {paper_profile.primary_analysis_method or '(not parsed)'}",
        "",
        "## EasyICU mapping",
        "",
    ]
    if spec.mapped_concepts:
        for label, concept in sorted(spec.mapped_concepts.items()):
            lines.append(f"- `{label}` -> `{concept}`")
    else:
        lines.append("- No supported concept mappings were resolved.")
    lines.append("")
    lines.extend(["## Deviation summary", "", deviation_report.summary, ""])
    if deviation_report.items:
        lines.extend(["### Deviation table", "", "| Item | Severity | Original | EasyICU proxy | Reason |", "|---|---|---|---|---|"])
        for item in deviation_report.items:
            lines.append(
                "| {item} | {severity} | {original} | {proxy} | {reason} |".format(
                    item=item.item,
                    severity=item.severity,
                    original=(item.original or "").replace("|", "/"),
                    proxy=(item.easyicu_proxy or "").replace("|", "/"),
                    reason=item.reason.replace("|", "/"),
                )
            )
        lines.append("")
    lines.extend(["## Claim-by-claim comparison", ""])
    if comparison_rows:
        lines.append("| Claim | Paper value | EasyICU value | Alignment | Note |")
        lines.append("|---|---|---|---|---|")
        for row in comparison_rows:
            lines.append(
                "| {claim} | {paper} | {easyicu} | {status} | {note} |".format(
                    claim=row["paper_claim"].replace("|", "/"),
                    paper=str(row["paper_value"]).replace("|", "/"),
                    easyicu=str(row["easyicu_value"]).replace("|", "/"),
                    status=row["alignment_status"],
                    note=str(row["reason_if_mismatch"]).replace("|", "/"),
                )
            )
    else:
        lines.append("No claim-by-claim comparison rows were produced.")
    lines.append("")
    lines.extend(["## EasyICU result ledger", ""])
    if ledger.easyicu_metrics:
        for key, value in sorted(ledger.easyicu_metrics.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No structured EasyICU metrics were extracted.")
    lines.append("")
    return "\n".join(lines)


def render_deviation_report(report: ReplicationDeviationReport) -> str:
    """Render the deviation report as Markdown."""
    lines = [
        "# Replication deviation report",
        "",
        f"- Supported: `{report.supported}`",
        f"- Summary: {report.summary}",
        "",
    ]
    if report.items:
        for item in report.items:
            lines.append(
                f"- `{item.item}` [{item.severity}] {item.reason}"
                + (f" (original: {item.original})" if item.original else "")
                + (f" (EasyICU proxy: {item.easyicu_proxy})" if item.easyicu_proxy else "")
            )
    else:
        lines.append("- No deviations were recorded.")
    lines.append("")
    return "\n".join(lines)


def render_showcase_manuscript(
    *,
    bound_manuscript: str,
    paper_profile: PaperProfile,
    deviation_report: ReplicationDeviationReport,
) -> str:
    """Post-process a bound manuscript into replication-study framing."""
    header = [
        "# EasyICU-based replication manuscript",
        "",
        f"_This manuscript reports an EasyICU-based replication study of "
        f"{paper_profile.paper_title or 'the source paper'}. It does not present "
        "the original paper's dataset as if it were re-analysed directly._",
        "",
    ]
    body = bound_manuscript.strip()
    body = re.sub(
        r"(?im)^#\s+.*manuscript.*$",
        "",
        body,
    ).strip()
    if "replication study" not in body.lower():
        body = (
            "This is a replication study using an EasyICU cohort, with variable harmonisation "
            "and explicit deviation tracking relative to the source paper.\n\n"
            + body
        )
    if deviation_report.items and not re.search(r"\bdeviation|differ|limitation\b", body, flags=re.I):
        body += (
            "\n\n## Limitations\n\n"
            "This replication includes explicit deviations between the original paper "
            "and EasyICU-supported concepts/time windows; see `deviation_report.md`."
        )
    return "\n".join(header) + body + "\n"


def write_claim_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    """Write a UTF-8 CSV file with stable fieldnames."""
    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out


def map_text_to_easyicu_concept(text: str) -> Optional[str]:
    lowered = text.lower()
    for canonical, aliases in SUPPORTED_CONCEPT_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return canonical
    return None


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _extract_title(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0].lower().startswith("title:"):
        return lines[0].split(":", 1)[1].strip()
    return lines[0][:240]


def _infer_paper_type(lower: str) -> str:
    if "auroc" in lower or "auc" in lower or "prediction model" in lower:
        return "prediction"
    if "hazard ratio" in lower or "cox" in lower:
        return "survival"
    if "subgroup" in lower or "interaction" in lower or "fairness" in lower:
        return "fairness"
    if "odds ratio" in lower or "logistic regression" in lower or "associated with" in lower:
        return "association"
    if "descriptive" in lower or "incidence" in lower:
        return "descriptive"
    return "unsupported_or_underspecified"


def _extract_target_outcome(lower: str) -> Optional[str]:
    if "28-day mortality" in lower or "28 day mortality" in lower:
        return "28-day mortality"
    if "icu mortality" in lower:
        return "ICU mortality"
    if "hospital mortality" in lower:
        return "hospital mortality"
    if "mortality" in lower or "death" in lower:
        return "mortality"
    if "length of stay" in lower or "los" in lower:
        return "length of stay"
    return None


def _extract_primary_signal(lower: str, *, exclude: Sequence[str]) -> Optional[str]:
    candidates = [
        "sofa-2",
        "sofa score",
        "sofa",
        "lactate",
        "mean arterial pressure",
        "vasopressor exposure",
        "creatinine",
        "age",
        "sex",
    ]
    excluded = {item.lower() for item in exclude if item}
    for candidate in candidates:
        if candidate in lower and candidate not in excluded:
            return candidate
    match = re.search(
        r"(?:primary exposure|primary predictor|exposure|predictor)\s*(?:was|were|:)?\s*([a-z0-9\- /]+)",
        lower,
    )
    if match:
        return match.group(1).strip(" .,:;")
    return None


def _extract_covariates(text: str) -> List[str]:
    match = re.search(
        r"adjusted for ([^.]+)\.",
        text,
        flags=re.I,
    )
    if not match:
        return []
    block = match.group(1)
    return [item.strip(" .") for item in re.split(r",| and ", block) if item.strip()]


def _extract_primary_method(text: str) -> str:
    for pattern in (
        r"(multivariable logistic regression[^.]*)\.",
        r"(logistic regression[^.]*)\.",
        r"(cox proportional hazards[^.]*)\.",
        r"(random forest[^.]*)\.",
        r"(xgboost[^.]*)\.",
    ):
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip()
    return ""


def _extract_secondary_analyses(lower: str) -> List[str]:
    hits = []
    for key in ("subgroup", "interaction", "sensitivity", "calibration", "decision curve", "fairness"):
        if key in lower:
            hits.append(key)
    return hits


def _extract_cohort_definition(text: str) -> str:
    for pattern in (
        r"(retrospective [^.]+ cohort[^.]*)\.",
        r"(adult icu patients[^.]*)\.",
        r"(patients admitted to the icu[^.]*)\.",
    ):
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip()
    return ""


def _extract_research_question(
    *,
    title: Optional[str],
    predictor: Optional[str],
    outcome: Optional[str],
    paper_type: str,
) -> str:
    if predictor and outcome:
        if paper_type == "prediction":
            return f"Can admission {predictor} predict {outcome} in ICU patients?"
        return f"Is {predictor} associated with {outcome} in ICU patients?"
    return title or ""


def _extract_bullets_after_heading(text: str, keywords: Sequence[str]) -> List[str]:
    items: List[str] = []
    for line in text.splitlines():
        lower = line.lower().strip()
        if any(word in lower for word in keywords):
            cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
            if cleaned:
                items.append(cleaned)
    return items


def _extract_key_claims(text: str, *, target_outcome: Optional[str], predictor: Optional[str]) -> List[PaperClaimRecord]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    rows: List[PaperClaimRecord] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        metric = None
        direction = None
        paper_value = None
        numeric_value = None
        if match := re.search(r"\bOR\b\s*=?\s*([0-9.]+)", stripped, flags=re.I):
            metric = "OR"
            paper_value = match.group(1)
            numeric_value = float(match.group(1))
            direction = "positive" if numeric_value > 1 else "negative"
        elif match := re.search(r"\bHR\b\s*=?\s*([0-9.]+)", stripped, flags=re.I):
            metric = "HR"
            paper_value = match.group(1)
            numeric_value = float(match.group(1))
            direction = "positive" if numeric_value > 1 else "negative"
        elif match := re.search(r"\b(?:AUROC|AUC)\b\s*=?\s*([0-9.]+)", stripped, flags=re.I):
            metric = "AUROC"
            paper_value = match.group(1)
            numeric_value = float(match.group(1))
        elif match := re.search(r"\bBrier(?: score)?\b\s*=?\s*([0-9.]+)", stripped, flags=re.I):
            metric = "Brier_score"
            paper_value = match.group(1)
            numeric_value = float(match.group(1))
        elif match := re.search(r"\bp\s*[<=>]\s*([0-9.eE\-]+)", stripped, flags=re.I):
            metric = "p_value"
            paper_value = match.group(1)
            numeric_value = float(match.group(1))
        elif match := re.search(r"\b(?:n|N)\s*=\s*([0-9]+)", stripped):
            metric = "n"
            paper_value = match.group(1)
            numeric_value = float(match.group(1))
        if metric is None:
            continue
        rows.append(
            PaperClaimRecord(
                claim_id=f"paper_claim_{len(rows) + 1:03d}",
                section=_infer_section(stripped),
                sentence=stripped,
                metric=metric,
                paper_value=paper_value,
                numeric_value=numeric_value,
                direction=direction,
                predictor=predictor,
                outcome=target_outcome,
            )
        )
    return rows


def _infer_section(sentence: str) -> str:
    lower = sentence.lower()
    if any(token in lower for token in ("methods", "we included", "we used")):
        return "methods"
    if any(token in lower for token in ("conclusion", "suggest", "implication")):
        return "discussion"
    return "results"


def _iter_profile_targets(profile: PaperProfile) -> Iterable[tuple[str, str]]:
    if profile.primary_exposure:
        yield "primary_exposure", profile.primary_exposure
    if profile.primary_predictor:
        yield "primary_predictor", profile.primary_predictor
    if profile.target_outcome:
        yield "target_outcome", profile.target_outcome
    for idx, cov in enumerate(profile.covariates, start=1):
        yield f"covariate_{idx}", cov


def _infer_time_windows(profile: PaperProfile) -> List[str]:
    text = " ".join(filter(None, [profile.cohort_definition, profile.primary_analysis_method, *profile.secondary_analyses])).lower()
    windows = []
    if "24h" in text or "24 h" in text or "24-hour" in text or "first 24" in text:
        windows.append("first_24h")
    if "48h" in text or "48 h" in text or "48-hour" in text:
        windows.append("first_48h")
    return windows or ["first_24h"]


def _update_metric(metrics: Dict[str, Any], target_key: str, summary: Dict[str, Any], candidates: Sequence[str]) -> None:
    if target_key in metrics:
        return
    for candidate in candidates:
        if candidate in summary and summary[candidate] is not None:
            metrics[target_key] = summary[candidate]
            return


def _map_claim_metric_to_easyicu_key(metric: Optional[str]) -> Optional[str]:
    mapping = {
        "or": "primary_or",
        "hr": "primary_or",
        "rr": "primary_or",
        "auroc": "auroc",
        "auc": "auroc",
        "brier_score": "brier_score",
        "p_value": "primary_pvalue",
        "p": "primary_pvalue",
        "n": "n_stays",
    }
    if metric is None:
        return None
    return mapping.get(metric.lower())


# ---------------------------------------------------------------------------
# Pipeline post-processing
# ---------------------------------------------------------------------------


def postprocess_paper_replication(
    *,
    result: "PipelineResult",
    paper_profile: PaperProfile,
    replication_spec: PaperReplicationSpec,
    deviation_report: ReplicationDeviationReport,
    mode: str,
) -> "PipelineResult":
    """Post-process a completed analysis run into paper-replication artefacts.

    Reads the run's ``manifest.json`` produced by the main pipeline,
    derives an EasyICU result ledger, compares it against the parsed
    paper claims, runs design / comparison / publication-claim audits,
    and writes the canonical replication outputs:
    ``paper_profile.json``, ``replication_spec.json``,
    ``paper_claim_ledger.csv``, ``replication_comparison.csv``,
    ``deviation_report.md``, ``replication_report.md`` and an updated
    ``manuscript_ready.md`` (in showcase manuscript mode).

    Audits are imported lazily to break the ``replication.paper`` ↔
    ``audits.validators`` import cycle.
    """
    # Local imports to avoid the audits ↔ replication import cycle.
    from ..audits.validators import (
        PublicationClaimAuditor,
        ReplicationDesignAuditor,
        ReplicationResultComparator,
    )
    from ..evidence import EvidenceStore
    from ..schema import PipelineResult

    run_dir = Path(result.workdir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context_payload: Optional[Dict[str, Any]] = None
    context_rel = manifest.get("context_path")
    if context_rel:
        context_path = run_dir / str(context_rel)
        if context_path.exists():
            context_payload = json.loads(context_path.read_text(encoding="utf-8"))

    ledger = build_paper_result_ledger(
        paper_profile=paper_profile,
        manifest=manifest,
        context_payload=context_payload,
    )
    comparator = ReplicationResultComparator()
    comparison_rows = comparator.compare(
        paper_profile=paper_profile,
        ledger=ledger,
    )
    design_findings = ReplicationDesignAuditor().audit(
        paper_profile=paper_profile,
        deviation_report=deviation_report,
    )
    comparison_findings = comparator.findings_from_rows(comparison_rows)

    evidence = EvidenceStore(run_dir)
    profile_path = run_dir / "paper_profile.json"
    profile_path.write_text(
        paper_profile.model_dump_json(indent=2),
        encoding="utf-8",
    )
    spec_path = run_dir / "replication_spec.json"
    spec_path.write_text(
        replication_spec.model_dump_json(indent=2),
        encoding="utf-8",
    )
    claim_rows = [
        {
            "claim_id": claim.claim_id,
            "section": claim.section,
            "sentence": claim.sentence,
            "metric": claim.metric or "",
            "paper_value": claim.paper_value or "",
            "numeric_value": "" if claim.numeric_value is None else claim.numeric_value,
            "direction": claim.direction or "",
            "predictor": claim.predictor or "",
            "outcome": claim.outcome or "",
        }
        for claim in paper_profile.key_claims
    ]
    claim_csv_path = write_claim_csv(
        run_dir / "paper_claim_ledger.csv",
        claim_rows,
        [
            "claim_id",
            "section",
            "sentence",
            "metric",
            "paper_value",
            "numeric_value",
            "direction",
            "predictor",
            "outcome",
        ],
    )
    comparison_csv_path = write_claim_csv(
        run_dir / "replication_comparison.csv",
        comparison_rows,
        [
            "claim_id",
            "paper_claim",
            "paper_value",
            "easyicu_value",
            "alignment_status",
            "reason_if_mismatch",
            "metric",
        ],
    )
    deviation_md_path = run_dir / "deviation_report.md"
    deviation_md_path.write_text(
        render_deviation_report(deviation_report),
        encoding="utf-8",
    )
    replication_report_path = run_dir / "replication_report.md"
    replication_report_path.write_text(
        render_replication_report(
            paper_profile=paper_profile,
            spec=replication_spec,
            deviation_report=deviation_report,
            comparison_rows=comparison_rows,
            ledger=ledger,
        ),
        encoding="utf-8",
    )

    manuscript_bound_path = run_dir / manifest.get("manuscript_path", "manuscript_scaffold_bound.md")
    if not manuscript_bound_path.exists():
        manuscript_bound_path = run_dir / "manuscript_scaffold_bound.md"
    bound_text = manuscript_bound_path.read_text(encoding="utf-8") if manuscript_bound_path.exists() else ""
    showcase_text = render_showcase_manuscript(
        bound_manuscript=bound_text,
        paper_profile=paper_profile,
        deviation_report=deviation_report,
    )
    publication_claim_findings = PublicationClaimAuditor().audit(
        manuscript_text=showcase_text,
        deviation_report=deviation_report,
    )

    all_findings = list(manifest.get("findings") or [])
    all_findings.extend(f.model_dump(mode="json") for f in design_findings)
    all_findings.extend(f.model_dump(mode="json") for f in comparison_findings)
    all_findings.extend(f.model_dump(mode="json") for f in publication_claim_findings)
    manifest["findings"] = all_findings

    readiness = dict(manifest.get("readiness") or {})
    design_reproduced = bool(
        readiness.get("execution_complete")
        and paper_profile.paper_type != "unsupported_or_underspecified"
        and not any(f.severity == "error" for f in design_findings)
    )
    paper_claims_parsed = bool(paper_profile.key_claims)
    result_alignment_audited = bool(comparison_rows)
    replication_report_ready = bool(
        readiness.get("execution_complete")
        and design_reproduced
        and paper_claims_parsed
        and replication_report_path.exists()
    )
    showcase_errors = [
        f for f in publication_claim_findings if f.severity == "error"
    ]
    showcase_manuscript_ready = bool(
        mode == "manuscript"
        and readiness.get("manuscript_ready")
        and design_reproduced
        and paper_claims_parsed
        and result_alignment_audited
        and not showcase_errors
    )
    readiness.update(
        {
            "design_reproduced": design_reproduced,
            "paper_claims_parsed": paper_claims_parsed,
            "result_alignment_audited": result_alignment_audited,
            "replication_report_ready": replication_report_ready,
            "showcase_manuscript_ready": showcase_manuscript_ready,
        }
    )
    manifest["readiness"] = readiness

    manuscript_ready_path = run_dir / "manuscript_ready.md"
    artifact_paths = dict(manifest.get("artifact_paths") or {})
    if showcase_manuscript_ready:
        manuscript_ready_path.write_text(showcase_text, encoding="utf-8")
        artifact_paths["manuscript_ready"] = "manuscript_ready.md"
    elif manuscript_ready_path.exists():
        manuscript_ready_path.unlink()
        artifact_paths.pop("manuscript_ready", None)
    artifact_paths.update(
        {
            "paper_profile": "paper_profile.json",
            "replication_spec": "replication_spec.json",
            "paper_claim_ledger": "paper_claim_ledger.csv",
            "replication_comparison": "replication_comparison.csv",
            "replication_report": "replication_report.md",
            "deviation_report": "deviation_report.md",
        }
    )
    manifest["artifact_paths"] = artifact_paths

    run_status_path = run_dir / "run_status.json"
    status_payload = (
        json.loads(run_status_path.read_text(encoding="utf-8"))
        if run_status_path.exists()
        else {"schema_version": "easyicu.run_status/1"}
    )
    status_payload["status"] = (
        "publication_ready"
        if readiness.get("publication_ready") and showcase_manuscript_ready
        else "manuscript_ready"
        if showcase_manuscript_ready
        else "replication_ready"
        if readiness.get("replication_report_ready")
        else "analysis_only"
        if readiness.get("execution_complete")
        else "diagnostic_only"
    )
    status_payload["gates"] = readiness
    status_payload["canonical_outputs"] = artifact_paths
    run_status_path.write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    author_note_path = run_dir / "author_review_note.md"
    base_note = author_note_path.read_text(encoding="utf-8") if author_note_path.exists() else "# Author review note\n\n"
    paper_status = (
        "publication_ready"
        if readiness.get("publication_ready") and showcase_manuscript_ready
        else "manuscript_ready"
        if showcase_manuscript_ready
        else "replication_ready"
        if readiness.get("replication_report_ready")
        else "analysis_only"
        if readiness.get("execution_complete")
        else "diagnostic_only"
    )
    base_note = re.sub(
        r"(?m)^- Status: `[^`]+`$",
        f"- Status: `{paper_status}`",
        base_note,
        count=1,
    )
    if readiness.get("replication_report_ready") and "Use `replication_report.md`" not in base_note:
        base_note = base_note.rstrip() + (
            "\n\n## Replication review\n\n"
            "The analysis run completed for paper replication. Use `replication_report.md` "
            "and `replication_comparison.csv` as the canonical replication outputs. "
            "`manuscript_ready.md` is emitted only in showcase manuscript mode after the "
            "paper-aware manuscript gates pass.\n"
        )
    author_note_path.write_text(
        base_note.rstrip()
        + "\n\n## Paper replication gates\n\n"
        + f"- design_reproduced: `{design_reproduced}`\n"
        + f"- paper_claims_parsed: `{paper_claims_parsed}`\n"
        + f"- result_alignment_audited: `{result_alignment_audited}`\n"
        + f"- replication_report_ready: `{replication_report_ready}`\n"
        + f"- showcase_manuscript_ready: `{showcase_manuscript_ready}`\n",
        encoding="utf-8",
    )

    for evidence_id, kind, description, path in (
        ("paper_profile", "log", "Parsed source-paper profile for replication mode.", profile_path),
        ("replication_spec", "log", "Typed EasyICU replication specification derived from the paper.", spec_path),
        ("paper_claim_ledger", "table", "Ledger of parsed result claims from the source paper.", claim_csv_path),
        ("replication_comparison", "table", "Claim-by-claim comparison of source paper and EasyICU results.", comparison_csv_path),
        ("replication_report", "log", "Narrative EasyICU replication report.", replication_report_path),
        ("deviation_report", "log", "Structured deviation report for unsupported or approximated design elements.", deviation_md_path),
    ):
        if evidence.get(evidence_id) is None:
            evidence.register_file(
                kind=kind,
                description=description,
                source_path=path,
                evidence_id=evidence_id,
                aliases=[evidence_id],
                producer="pipeline",
                generation_mode="system",
            )

    manifest["evidence"] = [record.model_dump(mode="json") for record in evidence.records()]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return PipelineResult.model_validate(
        {
            **result.model_dump(mode="json"),
            "paper_profile_path": str(profile_path),
            "replication_spec_path": str(spec_path),
            "replication_report_path": str(replication_report_path),
        }
    )


def canonical_outcome_name(raw: Optional[str]) -> Optional[str]:
    """Map a free-text outcome label to a canonical EasyICU outcome key.

    ``death`` covers ICU/hospital/28-day mortality labels; ``los_icu``
    covers length-of-stay variants. Anything that doesn't match a
    known canonical bucket is returned unchanged so the caller can
    still observe what the paper meant.
    """
    text = (raw or "").lower()
    if not text:
        return None
    if "mortality" in text or "death" in text:
        return "death"
    if "length of stay" in text or "los" in text:
        return "los_icu"
    return raw


def write_fail_closed_paper_package(
    *,
    workdir: Path,
    llm: Optional[Any],
    materialise_cohort: Callable[[Any, Path], Path],
    paper: Union[str, Path],
    cohort: Any,
    database: str,
    cohort_name: str,
    paper_profile: PaperProfile,
    replication_spec: PaperReplicationSpec,
    deviation_report: ReplicationDeviationReport,
) -> PipelineResult:
    """Write the canonical fail-closed paper-replication package.

    Called when the source-paper profile cannot be safely replicated
    (unsupported features, missing concepts). Emits the same artefact
    set a successful replication would (``paper_profile.json``,
    ``replication_spec.json``, ``paper_claim_ledger.csv``,
    ``replication_comparison.csv``, ``replication_report.md``,
    ``deviation_report.md``, ``results_report.md``, ``manifest.json``)
    but with empty / blocked content so downstream tools still see a
    valid run directory layout.

    The caller supplies ``workdir`` (the pipeline's run-root), ``llm``
    (only used to set ``used_mock_llm`` on the manifest) and the
    ``materialise_cohort`` helper that converts the cohort (path or
    DataFrame) into a parquet file under ``run_dir``.

    Lazy imports break the ``replication.paper`` ↔ ``audits.validators``
    and ``replication.paper`` ↔ ``pipeline_report`` cycles that would
    otherwise be created.
    """
    from ..audits.validators import ReplicationDesignAuditor
    from ..research_context.builder import build_naive_research_context
    from ..evidence import EvidenceStore
    from ..providers.mocks import MockLLMClient
    from ..pipeline_report import render_report, write_readiness_artifacts
    from ..providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files

    run_id = (
        "paperrep_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )
    run_dir = workdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = materialise_cohort(cohort, run_dir)
    df = pd.read_parquet(cohort_path)
    context = ResearchContext(
        research_question=paper_profile.research_question or "Paper replication failed closed before execution.",
        cohort=build_naive_research_context(
            research_question=paper_profile.research_question or "Paper replication failed closed.",
            cohort=cohort_path,
            cohort_name=cohort_name,
            database=database,
            target_outcome=canonical_outcome_name(paper_profile.target_outcome),
        ).cohort,
        variables=[],
        target_outcome=canonical_outcome_name(paper_profile.target_outcome),
        cohort_parquet=str(cohort_path),
        notes="Strict fail-closed paper replication package.",
    )
    context_path = run_dir / "context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[],
    )
    plan_path = run_dir / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    evidence = EvidenceStore(run_dir)
    profile_path = run_dir / "paper_profile.json"
    profile_path.write_text(paper_profile.model_dump_json(indent=2), encoding="utf-8")
    spec_path = run_dir / "replication_spec.json"
    spec_path.write_text(replication_spec.model_dump_json(indent=2), encoding="utf-8")
    deviation_md_path = run_dir / "deviation_report.md"
    deviation_md_path.write_text(render_deviation_report(deviation_report), encoding="utf-8")
    claim_csv_path = write_claim_csv(
        run_dir / "paper_claim_ledger.csv",
        [
            {
                "claim_id": claim.claim_id,
                "section": claim.section,
                "sentence": claim.sentence,
                "metric": claim.metric or "",
                "paper_value": claim.paper_value or "",
                "numeric_value": "" if claim.numeric_value is None else claim.numeric_value,
                "direction": claim.direction or "",
                "predictor": claim.predictor or "",
                "outcome": claim.outcome or "",
            }
            for claim in paper_profile.key_claims
        ],
        [
            "claim_id",
            "section",
            "sentence",
            "metric",
            "paper_value",
            "numeric_value",
            "direction",
            "predictor",
            "outcome",
        ],
    )
    comparison_csv_path = write_claim_csv(
        run_dir / "replication_comparison.csv",
        [],
        [
            "claim_id",
            "paper_claim",
            "paper_value",
            "easyicu_value",
            "alignment_status",
            "reason_if_mismatch",
            "metric",
        ],
    )
    replication_report_path = run_dir / "replication_report.md"
    replication_report_path.write_text(
        render_replication_report(
            paper_profile=paper_profile,
            spec=replication_spec,
            deviation_report=deviation_report,
            comparison_rows=[],
            ledger=PaperResultLedger(paper_claims=paper_profile.key_claims, easyicu_metrics={"n_stays": int(len(df))}),
        ),
        encoding="utf-8",
    )
    manuscript_path = run_dir / "manuscript_scaffold_bound.md"
    manuscript_path.write_text(
        "# Manuscript scaffold not generated\n\nStrict fail-closed policy blocked paper replication drafting.\n",
        encoding="utf-8",
    )

    design_findings = ReplicationDesignAuditor().audit(
        paper_profile=paper_profile,
        deviation_report=deviation_report,
    )
    readiness, artifact_paths = write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=design_findings,
        per_step_records=[],
        evidence=evidence,
        run_dir=run_dir,
        manuscript_path=manuscript_path,
        stop_after_analysis=False,
    )
    readiness.update(
        {
            "design_reproduced": False,
            "paper_claims_parsed": bool(paper_profile.key_claims),
            "result_alignment_audited": False,
            "replication_report_ready": True,
            "showcase_manuscript_ready": False,
        }
    )
    artifact_paths.update(
        {
            "paper_profile": "paper_profile.json",
            "replication_spec": "replication_spec.json",
            "paper_claim_ledger": "paper_claim_ledger.csv",
            "replication_comparison": "replication_comparison.csv",
            "replication_report": "replication_report.md",
            "deviation_report": "deviation_report.md",
        }
    )
    for evidence_id, kind, description, path in (
        ("paper_profile", "log", "Parsed source-paper profile for replication mode.", profile_path),
        ("replication_spec", "log", "Typed EasyICU replication specification derived from the paper.", spec_path),
        ("paper_claim_ledger", "table", "Ledger of parsed result claims from the source paper.", claim_csv_path),
        ("replication_comparison", "table", "Claim-by-claim comparison of source paper and EasyICU results.", comparison_csv_path),
        ("replication_report", "log", "Narrative EasyICU replication report.", replication_report_path),
        ("deviation_report", "log", "Structured deviation report for unsupported or approximated design elements.", deviation_md_path),
    ):
        if evidence.get(evidence_id) is None:
            evidence.register_file(
                kind=kind,
                description=description,
                source_path=path,
                evidence_id=evidence_id,
                aliases=[evidence_id],
                producer="pipeline",
                generation_mode="system",
            )

    report_path = run_dir / "results_report.md"
    report_path.write_text(
        render_report(
            context=context,
            plan=plan,
            findings=design_findings,
            per_step_records=[],
            evidence=evidence,
            readiness=readiness,
        ),
        encoding="utf-8",
    )
    run_status_path = run_dir / "run_status.json"
    run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
    run_status["gates"] = readiness
    run_status["canonical_outputs"] = artifact_paths
    run_status_path.write_text(
        json.dumps(run_status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    manifest = AnalysisManifest(
        run_id=run_id,
        research_question=context.research_question,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        context_path="context.json",
        plan_path="plan.json",
        evidence=evidence.records(),
        findings=design_findings,
        readiness=readiness,
        artifact_paths=artifact_paths,
        report_path="results_report.md",
        manuscript_path="manuscript_scaffold_bound.md",
        used_mock_llm=isinstance(llm, MockLLMClient) if llm is not None else False,
        prompt_pack_version=PROMPT_PACK_VERSION,
        prompt_pack_files=prompt_pack_files(),
        notes="Strict fail-closed replication package generated before analysis execution.",
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return PipelineResult(
        run_id=run_id,
        workdir=str(run_dir),
        context_path=str(context_path),
        plan_path=str(plan_path),
        manifest_path=str(manifest_path),
        report_path=str(report_path),
        manuscript_path=str(manuscript_path),
        evidence_count=len(evidence.records()),
        findings_count=len(design_findings),
        paper_profile_path=str(profile_path),
        replication_spec_path=str(spec_path),
        replication_report_path=str(replication_report_path),
    )
