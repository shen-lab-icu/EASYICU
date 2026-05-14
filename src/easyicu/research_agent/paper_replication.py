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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import (
    PaperClaimRecord,
    PaperProfile,
    PaperReplicationSpec,
    PaperResultLedger,
    ReplicationDeviationItem,
    ReplicationDeviationReport,
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


def compare_metric_values(
    *,
    metric: Optional[str],
    paper_value: Optional[float],
    paper_direction: Optional[str],
    easyicu_value: Any,
) -> tuple[str, str]:
    """Classify paper vs EasyICU alignment for one claim."""
    if paper_value is None:
        if paper_direction and easyicu_value is not None:
            easy_dir = "positive" if float(easyicu_value) > 1 else "negative"
            if paper_direction == easy_dir:
                return "directionally_aligned", "Direction matched but no exact paper scalar was available."
        return "not_comparable", "Paper claim did not expose a comparable numeric value."
    if easyicu_value is None:
        return "not_comparable", "EasyICU run did not emit a comparable structured metric."
    try:
        easy = float(easyicu_value)
    except Exception:
        return "not_comparable", "EasyICU metric could not be interpreted numerically."

    metric_name = (metric or "").lower()
    if metric_name in {"p_value", "p"}:
        paper_sig = paper_value < 0.05
        easy_sig = easy < 0.05
        if paper_sig == easy_sig:
            return "directionally_aligned", "Significance state matched."
        return "not_aligned", "Significance state did not match."

    if metric_name in {"or", "hr", "rr"}:
        if (paper_value > 1 and easy > 1) or (paper_value < 1 and easy < 1):
            delta = abs(paper_value - easy) / max(abs(paper_value), 1e-6)
            if delta <= 0.25:
                return "aligned", "Effect direction and magnitude were close."
            return "directionally_aligned", "Effect direction matched but magnitude differed."
        return "not_aligned", "Effect direction did not match."

    if metric_name in {"auroc", "auc", "brier_score", "outcome_rate", "n"}:
        delta = abs(paper_value - easy)
        tol = 0.03 if metric_name in {"auroc", "auc", "brier_score", "outcome_rate"} else max(5.0, 0.05 * max(abs(paper_value), 1.0))
        if delta <= tol:
            return "aligned", "Numeric value was within tolerance."
        return "not_aligned", "Numeric value differed beyond the comparison tolerance."

    delta = abs(paper_value - easy)
    if delta <= max(0.05, 0.2 * max(abs(paper_value), 1.0)):
        return "aligned", "Generic numeric tolerance matched."
    return "not_aligned", "Generic numeric tolerance did not match."


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
