"""Publication figure skill for evidence-bound manuscript figures.

Exploratory plots may be emitted by analysis scripts, but manuscript-facing
figures should pass through a small, auditable EasyICU figure skill once the
analysis evidence is stable. This module sits between analysis and writing:
it consumes registered tables/statistics, creates a claim-first
``FigureContract``, exports journal-friendly formats through
``publication_figures``, and registers every output in the EvidenceStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .evidence import EvidenceStore
from .publication_figures import (
    add_panel_label,
    apply_publication_style,
    audit_publication_exports,
    make_figure_contract,
    save_publication_figure,
)
from .schema import AnalysisPlan, EvidenceRecord, ResearchContext, ValidationFinding


@dataclass
class PublicationFigureSkillResult:
    """Result of the manuscript-facing publication figure stage."""

    generated: bool
    skipped_reason: Optional[str] = None
    contract_evidence_id: Optional[str] = None
    figure_evidence_ids: List[str] = field(default_factory=list)
    summary_evidence_id: Optional[str] = None
    findings: List[ValidationFinding] = field(default_factory=list)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "generated": self.generated,
            "skipped_reason": self.skipped_reason,
            "contract_evidence_id": self.contract_evidence_id,
            "figure_evidence_ids": list(self.figure_evidence_ids),
            "summary_evidence_id": self.summary_evidence_id,
            "findings": [f.model_dump(mode="json") for f in self.findings],
        }


class PublicationFigureSkill:
    """Generate a formal publication figure from registered evidence.

    The skill is deterministic and intentionally conservative. It creates a
    publication figure only when analysis has produced citable source evidence
    and no publication figure bundle already exists. The first supported
    template is the common EasyICU association table -> forest/scatter panel
    path; other runs still receive a recorded skip reason instead of an ad hoc
    decorative figure.
    """

    name = "publication_figure_skill"

    def run(
        self,
        *,
        context: ResearchContext,
        plan: AnalysisPlan,
        evidence: EvidenceStore,
        run_dir: Path,
        prompt_pack_version: Optional[str] = None,
    ) -> PublicationFigureSkillResult:
        if _has_publication_figure_bundle(evidence):
            return PublicationFigureSkillResult(
                generated=False,
                skipped_reason="existing_publication_figure_bundle",
            )
        if not _plan_requests_figures(plan):
            return PublicationFigureSkillResult(
                generated=False,
                skipped_reason="plan_has_no_figure_outputs",
            )

        primary = _first_existing_record(
            evidence,
            [
                "primary_association",
                "primary_association_table",
                "table_primary_association",
            ],
        )
        if primary is None:
            return self._write_skip_summary(
                reason="no_supported_source_table",
                context=context,
                plan=plan,
                evidence=evidence,
                run_dir=run_dir,
                prompt_pack_version=prompt_pack_version,
            )

        try:
            frame = _read_table(run_dir / primary.relative_path)
            return self._render_primary_association(
                context=context,
                plan=plan,
                evidence=evidence,
                run_dir=run_dir,
                source_record=primary,
                frame=frame,
                prompt_pack_version=prompt_pack_version,
            )
        except Exception as exc:
            finding = ValidationFinding(
                validator=self.name,
                severity="warning",
                message=f"PublicationFigureSkill skipped rendering: {exc}",
                evidence_ids=[primary.evidence_id],
            )
            skipped = self._write_skip_summary(
                reason="render_failed",
                context=context,
                plan=plan,
                evidence=evidence,
                run_dir=run_dir,
                prompt_pack_version=prompt_pack_version,
                findings=[finding],
            )
            return skipped

    def _render_primary_association(
        self,
        *,
        context: ResearchContext,
        plan: AnalysisPlan,
        evidence: EvidenceStore,
        run_dir: Path,
        source_record: EvidenceRecord,
        frame: pd.DataFrame,
        prompt_pack_version: Optional[str],
    ) -> PublicationFigureSkillResult:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plot_df = _normalise_association_frame(frame)
        if plot_df.empty:
            raise ValueError("primary association table has no plottable rows")

        palette = apply_publication_style()
        out_dir = run_dir / "publication_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        source_copy = out_dir / "publication_figure_source_primary_association.csv"
        plot_df.to_csv(source_copy, index=False)

        height = max(1.8, 0.34 * len(plot_df) + 0.9)
        fig, ax = plt.subplots(figsize=(4.8, height), constrained_layout=True)
        y = np.arange(len(plot_df))
        estimate = plot_df["estimate"].astype(float)
        lower = plot_df["lower"].astype(float)
        upper = plot_df["upper"].astype(float)
        xerr = np.vstack([
            np.maximum(0.0, estimate - lower),
            np.maximum(0.0, upper - estimate),
        ])
        ax.errorbar(
            estimate,
            y,
            xerr=xerr,
            fmt="o",
            color=palette.get("blue", "#0F4D92"),
            ecolor=palette.get("neutral", "#8F8F8F"),
            elinewidth=0.9,
            capsize=2.0,
            markersize=4.0,
        )
        ax.axvline(1.0, color=palette.get("neutral", "#8F8F8F"), linestyle="--", linewidth=0.8)
        ax.set_yticks(y, plot_df["label"].astype(str).tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Odds ratio")
        ax.set_ylabel("")
        add_panel_label(ax, "A")
        ax.margins(x=0.08)

        predictor = str(plot_df["label"].iloc[0])
        outcome = context.target_outcome or "the target outcome"
        core_claim = (
            f"The primary EasyICU association estimate for {predictor} and "
            f"{outcome} is rendered from registered analysis evidence."
        )
        contract = make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=core_claim,
            panels=[
                {
                    "panel_id": "A",
                    "title": "Primary association",
                    "role": "relationship",
                    "claim": "The association estimate and interval are drawn from the registered primary association table.",
                    "evidence_ids": [source_record.evidence_id],
                    "review_risk": "Interpretability depends on the upstream model specification and validator findings.",
                }
            ],
            source_data=[source_record.evidence_id],
            statistics_note=(
                "The panel is generated after analysis validation from an "
                "EvidenceStore-registered source table; it is not drawn from "
                "writer prose."
            ),
        )
        paths = save_publication_figure(
            fig,
            out_dir / "easyicu_publication_figure",
            contract=contract,
            dpi=300,
        )
        plt.close(fig)

        audit_findings = list(audit_publication_exports(paths))
        figure_ids: List[str] = []
        contract_evidence_id: Optional[str] = None
        for key, path in paths.items():
            suffix = path.suffix.lower()
            if key == "contract" or suffix.endswith(".json"):
                record = evidence.register_file(
                    kind="log",
                    description="Publication figure contract generated from analysis evidence.",
                    source_path=path,
                    evidence_id="publication_figure_contract",
                    aliases=["publication_figure_contract", "figure_contract"],
                    producer=self.name,
                    generation_mode="deterministic_figure_skill",
                    prompt_pack_version=prompt_pack_version,
                    metadata={"source_evidence_id": source_record.evidence_id},
                )
                contract_evidence_id = record.evidence_id
                continue
            record = evidence.register_file(
                kind="figure",
                description=f"Publication figure export ({suffix.lstrip('.')}) generated from analysis evidence.",
                source_path=path,
                evidence_id=f"publication_figure_{suffix.lstrip('.')}",
                aliases=["publication_figure", f"publication_figure_{suffix.lstrip('.')}"],
                producer=self.name,
                generation_mode="deterministic_figure_skill",
                prompt_pack_version=prompt_pack_version,
                metadata={
                    "source_evidence_id": source_record.evidence_id,
                    "figure_contract": "publication_figure_contract",
                },
            )
            figure_ids.append(record.evidence_id)

        source_copy_record = evidence.register_file(
            kind="table",
            description="Source data copied for the publication figure skill.",
            source_path=source_copy,
            evidence_id="publication_figure_source_primary_association",
            aliases=["publication_figure_source_data"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
            metadata={"source_evidence_id": source_record.evidence_id},
        )

        summary = {
            "stage": self.name,
            "generated": True,
            "figure_id": contract.figure_id,
            "core_claim": contract.core_claim,
            "source_evidence_id": source_record.evidence_id,
            "source_copy_evidence_id": source_copy_record.evidence_id,
            "figure_evidence_ids": figure_ids,
            "contract_evidence_id": contract_evidence_id,
            "audit_findings": [f.model_dump(mode="json") for f in audit_findings],
        }
        summary_record = evidence.register_json(
            kind="log",
            description="PublicationFigureSkill summary.",
            payload=summary,
            filename="publication_figure_skill_summary.json",
            evidence_id="publication_figure_skill_summary",
            aliases=["publication_figure_skill_summary"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
        )
        return PublicationFigureSkillResult(
            generated=True,
            contract_evidence_id=contract_evidence_id,
            figure_evidence_ids=figure_ids,
            summary_evidence_id=summary_record.evidence_id,
            findings=audit_findings,
        )

    def _write_skip_summary(
        self,
        *,
        reason: str,
        context: ResearchContext,
        plan: AnalysisPlan,
        evidence: EvidenceStore,
        run_dir: Path,
        prompt_pack_version: Optional[str],
        findings: Optional[Sequence[ValidationFinding]] = None,
    ) -> PublicationFigureSkillResult:
        payload = {
            "stage": self.name,
            "generated": False,
            "skipped_reason": reason,
            "research_question": context.research_question,
            "figure_steps": [
                step.step_id
                for step in plan.steps
                if any(str(out).startswith("figure:") for out in step.expected_outputs)
            ],
        }
        record = evidence.register_json(
            kind="log",
            description="PublicationFigureSkill skip summary.",
            payload=payload,
            filename="publication_figure_skill_summary.json",
            evidence_id="publication_figure_skill_summary",
            aliases=["publication_figure_skill_summary"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
        )
        return PublicationFigureSkillResult(
            generated=False,
            skipped_reason=reason,
            summary_evidence_id=record.evidence_id,
            findings=list(findings or []),
        )


def _has_publication_figure_bundle(evidence: EvidenceStore) -> bool:
    for record in evidence.records():
        haystack = f"{record.evidence_id} {record.relative_path}".lower()
        if record.kind == "figure" and "publication_figure" in haystack:
            return True
    return False


def _plan_requests_figures(plan: AnalysisPlan) -> bool:
    return any(
        any(str(out).startswith("figure:") for out in step.expected_outputs)
        for step in plan.steps
    )


def _first_existing_record(
    evidence: EvidenceStore,
    names: Sequence[str],
) -> Optional[EvidenceRecord]:
    name_set = {str(name).lower() for name in names}
    for name in names:
        record = evidence.get(name)
        if record is not None:
            return record
    for record in evidence.records():
        if record.kind != "table":
            continue
        basename = Path(record.relative_path).stem.lower()
        if any(token in basename for token in name_set):
            return record
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"unsupported table format for figure skill: {path.name}")


def _normalise_association_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["label", "estimate", "lower", "upper"])
    cols = {str(c).lower(): c for c in frame.columns}
    label_col = _first_col(cols, ["variable", "predictor", "term", "feature"])
    estimate_col = _first_col(cols, ["odds_ratio", "or", "estimate", "effect"])
    lower_col = _first_col(cols, ["or_lower", "ci_lower", "lower", "estimate_lower"])
    upper_col = _first_col(cols, ["or_upper", "ci_upper", "upper", "estimate_upper"])
    if estimate_col is None:
        numeric_cols = [
            c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
        ]
        estimate_col = numeric_cols[0] if numeric_cols else None
    if label_col is None:
        label_col = estimate_col
    if estimate_col is None or label_col is None:
        return pd.DataFrame(columns=["label", "estimate", "lower", "upper"])

    out = pd.DataFrame({
        "label": frame[label_col].astype(str),
        "estimate": pd.to_numeric(frame[estimate_col], errors="coerce"),
    })
    if lower_col is not None:
        out["lower"] = pd.to_numeric(frame[lower_col], errors="coerce")
    else:
        out["lower"] = out["estimate"]
    if upper_col is not None:
        out["upper"] = pd.to_numeric(frame[upper_col], errors="coerce")
    else:
        out["upper"] = out["estimate"]
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["estimate"]
    )
    out["lower"] = out["lower"].fillna(out["estimate"])
    out["upper"] = out["upper"].fillna(out["estimate"])
    return out[["label", "estimate", "lower", "upper"]]


def _first_col(cols: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in cols:
            return cols[candidate]
    return None


__all__ = ["PublicationFigureSkill", "PublicationFigureSkillResult"]
