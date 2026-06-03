"""Publication figure skill for evidence-bound manuscript figures.

Exploratory plots may be emitted by analysis scripts, but manuscript-facing
figures should pass through a small, auditable EasyICU figure skill once the
analysis evidence is stable. This module sits between analysis and writing:
it consumes registered tables/statistics, creates a claim-first
``FigureContract``, exports journal-friendly formats through
``publication_figures``, and registers every output in the EvidenceStore.
"""

from __future__ import annotations

import shutil
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
from .robustness_panel import RobustnessPanel, load_robustness_panel
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
        if _has_curated_publication_figure_bundle(evidence):
            return PublicationFigureSkillResult(
                generated=False,
                skipped_reason="existing_curated_publication_figure_bundle",
            )
        robustness_record = evidence.get("robustness_panel")
        robustness_panel = load_robustness_panel(run_dir / "robustness_panel.json")
        if robustness_record is not None and robustness_panel is not None:
            try:
                return self._render_robustness_panel(
                    context=context,
                    evidence=evidence,
                    run_dir=run_dir,
                    source_record=robustness_record,
                    panel=robustness_panel,
                    prompt_pack_version=prompt_pack_version,
                )
            except Exception as exc:
                return self._write_skip_summary(
                    reason="robustness_panel_render_failed",
                    context=context,
                    plan=plan,
                    evidence=evidence,
                    run_dir=run_dir,
                    prompt_pack_version=prompt_pack_version,
                    findings=[
                        ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message=(
                                "PublicationFigureSkill skipped robustness-panel "
                                f"rendering: {exc}"
                            ),
                            evidence_ids=[robustness_record.evidence_id],
                        )
                    ],
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
        prediction_bundle = _select_existing_prediction_figure_bundle(evidence)
        if prediction_bundle is not None:
            return self._promote_prediction_validation_figure(
                context=context,
                evidence=evidence,
                run_dir=run_dir,
                figure_records=prediction_bundle,
                summary_record=_first_existing_statistic_record(
                    evidence,
                    [
                        "01_model_training",
                        "model_performance",
                        "baseline_prevalence",
                    ],
                ),
                prompt_pack_version=prompt_pack_version,
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
            strata = _first_existing_record(
                evidence,
                [
                    "stratified_mortality",
                    "stratified_mortality_incidence",
                ],
            )
            missingness = _first_existing_record(
                evidence,
                ["missingness", "missingness_summary", "table_missingness"],
            )
            return self._render_primary_association(
                context=context,
                plan=plan,
                evidence=evidence,
                run_dir=run_dir,
                source_record=primary,
                frame=frame,
                strata_record=strata,
                missingness_record=missingness,
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
        strata_record: Optional[EvidenceRecord],
        missingness_record: Optional[EvidenceRecord],
        prompt_pack_version: Optional[str],
    ) -> PublicationFigureSkillResult:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plot_df = _normalise_association_frame(frame)
        if plot_df.empty:
            raise ValueError("primary association table has no plottable rows")
        plot_df = plot_df.reset_index(drop=True)
        plot_df["is_primary"] = False
        if not plot_df.empty:
            plot_df.loc[0, "is_primary"] = True

        palette = apply_publication_style()
        out_dir = run_dir / "publication_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        source_copy = out_dir / "publication_figure_source_primary_association.csv"
        plot_df.to_csv(source_copy, index=False)
        source_records = [source_record]
        strata_df = pd.DataFrame()
        missingness_df = pd.DataFrame()
        if strata_record is not None:
            try:
                strata_df = _normalise_strata_frame(
                    _read_table(run_dir / strata_record.relative_path)
                )
                if not strata_df.empty:
                    source_records.append(strata_record)
                    strata_df.to_csv(
                        out_dir / "publication_figure_source_stratified_mortality.csv",
                        index=False,
                    )
            except Exception:
                strata_df = pd.DataFrame()
        if missingness_record is not None:
            try:
                missingness_df = _normalise_missingness_frame(
                    _read_table(run_dir / missingness_record.relative_path)
                )
                if not missingness_df.empty:
                    source_records.append(missingness_record)
                    missingness_df.to_csv(
                        out_dir / "publication_figure_source_missingness.csv",
                        index=False,
                    )
            except Exception:
                missingness_df = pd.DataFrame()

        n_side_panels = int(not strata_df.empty) + int(not missingness_df.empty)
        if n_side_panels:
            fig = plt.figure(figsize=(183 / 25.4, 112 / 25.4), constrained_layout=False)
            grid = fig.add_gridspec(
                2,
                2,
                width_ratios=[1.45, 0.92],
                height_ratios=[1.0, 0.78],
                left=0.08,
                right=0.98,
                top=0.96,
                bottom=0.13,
                wspace=0.4,
                hspace=0.46,
            )
            ax = fig.add_subplot(grid[:, 0])
            side_axes = []
            if not strata_df.empty:
                side_axes.append(("strata", fig.add_subplot(grid[0, 1])))
            if not missingness_df.empty:
                side_axes.append(("missingness", fig.add_subplot(grid[1, 1])))
        else:
            height = max(2.3, 0.42 * len(plot_df) + 1.05)
            fig, ax = plt.subplots(figsize=(126 / 25.4, height), constrained_layout=False)
            fig.subplots_adjust(left=0.2, right=0.96, top=0.93, bottom=0.18)
            side_axes = []
        y = np.arange(len(plot_df))
        estimate = plot_df["estimate"].astype(float).to_numpy()
        lower = plot_df["lower"].astype(float).to_numpy()
        upper = plot_df["upper"].astype(float).to_numpy()
        for idx, row in plot_df.iterrows():
            center = float(row["estimate"])
            lo = float(row["lower"])
            hi = float(row["upper"])
            color = (
                palette.get("blue", "#0F4D92")
                if bool(row.get("is_primary"))
                else palette.get("baseline", "#272727")
            )
            ax.errorbar(
                center,
                idx,
                xerr=np.array(
                    [[max(0.0, center - lo)], [max(0.0, hi - center)]],
                    dtype=float,
                ),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.2,
                markersize=4.3 if bool(row.get("is_primary")) else 3.8,
                zorder=3,
            )
        ax.axvline(1.0, color=palette.get("neutral", "#8F8F8F"), linestyle="--", linewidth=0.8)
        ax.set_yticks(y, plot_df["label"].astype(str).tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Odds ratio")
        ax.set_ylabel("")
        use_log_scale = (
            float(lower.min()) > 0
            and float(upper.max()) / max(float(lower.min()), 1e-9) > 1.8
        )
        if use_log_scale:
            ax.set_xscale("log")
        right_anchor = float(max(upper.max(), estimate.max(), 1.0))
        right_pad = right_anchor * (0.6 if use_log_scale else 0.38)
        if use_log_scale:
            ax.set_xlim(max(float(lower.min()) * 0.8, 1e-3), right_anchor + right_pad)
        else:
            left_bound = min(float(lower.min()), 1.0)
            right_bound = right_anchor + right_pad
            ax.set_xlim(max(0.0, left_bound - 0.08 * max(right_bound, 1.0)), right_bound)
        text_x = right_anchor + right_pad * 0.12
        ax.text(
            text_x,
            -0.55,
            "OR (95% CI)",
            ha="left",
            va="bottom",
            fontsize=6.8,
            color=palette.get("baseline", "#272727"),
        )
        for idx, (center, lo, hi) in enumerate(zip(estimate, lower, upper)):
            ax.text(
                text_x,
                idx,
                f"{center:.2f} ({lo:.2f}-{hi:.2f})",
                ha="left",
                va="center",
                fontsize=6.5,
                color=palette.get("baseline", "#272727"),
            )
        ax.set_title("Adjusted association with ICU mortality", loc="left", pad=4)
        add_panel_label(ax, "A", x=-0.08)
        ax.margins(x=0.08)
        ax.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
            zorder=0,
        )

        next_panel = ord("B")
        for kind, side_ax in side_axes:
            label = chr(next_panel)
            next_panel += 1
            if kind == "strata":
                _draw_strata_panel(
                    side_ax,
                    strata_df,
                    palette=palette,
                    outcome=context.target_outcome or "outcome",
                )
            elif kind == "missingness":
                _draw_missingness_panel(side_ax, missingness_df, palette=palette)
            add_panel_label(side_ax, label, x=-0.12, y=1.04, fontsize=10.0)

        predictor = str(plot_df["label"].iloc[0])
        outcome = context.target_outcome or "the target outcome"
        core_claim = (
            f"The primary EasyICU association estimate for {predictor} and "
            f"{outcome} is rendered from registered analysis evidence."
        )
        panels = [
            {
                "panel_id": "A",
                "title": "Adjusted association",
                "role": "relationship",
                "claim": "The association estimate and interval are drawn from the registered primary association table.",
                "evidence_ids": [source_record.evidence_id],
                "review_risk": "Interpretability depends on the upstream model specification and validator findings.",
            }
        ]
        if not strata_df.empty and strata_record is not None:
            panels.append(
                {
                    "panel_id": "B",
                    "title": "Outcome by score",
                    "role": "audit",
                    "claim": "Outcome rates by SOFA-2 stratum are shown directly from the registered stratum audit table.",
                    "evidence_ids": [strata_record.evidence_id],
                    "review_risk": "Sparse high-score strata should be interpreted with their denominators.",
                }
            )
        if not missingness_df.empty and missingness_record is not None:
            panels.append(
                {
                    "panel_id": chr(ord("A") + len(panels)),
                    "title": "Missingness audit",
                    "role": "validation",
                    "claim": "Feature-level missingness is displayed rather than hidden from the manuscript figure.",
                    "evidence_ids": [missingness_record.evidence_id],
                    "review_risk": "Zero-missingness summaries can otherwise look like empty plots if not annotated.",
                }
            )
        contract = make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=core_claim,
            panels=panels,
            source_data=[record.evidence_id for record in source_records],
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
                    "figure_role": "publication_figure",
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
            "source_evidence_ids": [record.evidence_id for record in source_records],
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

    def _render_robustness_panel(
        self,
        *,
        context: ResearchContext,
        evidence: EvidenceStore,
        run_dir: Path,
        source_record: EvidenceRecord,
        panel: RobustnessPanel,
        prompt_pack_version: Optional[str],
    ) -> PublicationFigureSkillResult:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        rows = [
            row
            for row in panel.rows
            if row.converged
            and row.point_estimate is not None
            and row.ci_low is not None
            and row.ci_high is not None
        ]
        if not rows:
            raise ValueError("robustness panel has no converged rows to plot")
        primary_rows = [row for row in rows if row.spec_id == panel.primary_spec_id]
        other_rows = [row for row in rows if row.spec_id != panel.primary_spec_id]
        plot_rows = (primary_rows + other_rows)[:10]
        source_df = pd.DataFrame(
            [
                {
                    "spec_id": row.spec_id,
                    "axis": row.axis,
                    "n": row.n,
                    "point_estimate": row.point_estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "converged": row.converged,
                    "notes": row.notes,
                }
                for row in plot_rows
            ]
        )

        palette = apply_publication_style()
        out_dir = run_dir / "publication_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        source_copy = out_dir / "publication_figure_source_robustness_panel.csv"
        source_df.to_csv(source_copy, index=False)

        height = max(2.5, 0.42 * len(plot_rows) + 1.15)
        fig, ax = plt.subplots(figsize=(142 / 25.4, height), constrained_layout=False)
        fig.subplots_adjust(left=0.34, right=0.95, top=0.90, bottom=0.18)
        y = np.arange(len(source_df))
        estimate = source_df["point_estimate"].astype(float).to_numpy()
        lower = source_df["ci_low"].astype(float).to_numpy()
        upper = source_df["ci_high"].astype(float).to_numpy()
        labels = [
            "Primary" if str(row["spec_id"]) == panel.primary_spec_id else str(row["spec_id"]).replace("_", " ")
            for _, row in source_df.iterrows()
        ]
        for idx, row in source_df.iterrows():
            center = float(row["point_estimate"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            is_primary = str(row["spec_id"]) == panel.primary_spec_id
            color = (
                palette.get("blue", "#0F4D92")
                if is_primary
                else palette.get("baseline", "#272727")
            )
            ax.errorbar(
                center,
                idx,
                xerr=np.array(
                    [[max(0.0, center - lo)], [max(0.0, hi - center)]],
                    dtype=float,
                ),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.2,
                markersize=4.4 if is_primary else 3.6,
                zorder=3,
            )
        ax.axvline(
            1.0,
            color=palette.get("neutral", "#8F8F8F"),
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_yticks(y, labels)
        header_y = -0.75
        ax.set_ylim(len(source_df) - 0.5, header_y - 0.15)
        ax.set_xlabel("Primary effect estimate (95% CI)")
        ax.set_title("Pre-specified robustness panel", loc="left", pad=8)
        ax.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )
        right_anchor = float(max(upper.max(), estimate.max(), 1.0))
        right_pad = right_anchor * 0.46
        ax.set_xlim(
            max(0.0, min(float(lower.min()), 1.0) - 0.08 * right_anchor),
            right_anchor + right_pad,
        )
        text_x = right_anchor + right_pad * 0.08
        ax.text(
            text_x,
            header_y,
            "Estimate (95% CI)",
            ha="left",
            va="center",
            fontsize=6.8,
        )
        for idx, (center, lo, hi) in enumerate(zip(estimate, lower, upper)):
            ax.text(
                text_x,
                idx,
                f"{center:.2f} ({lo:.2f}-{hi:.2f})",
                ha="left",
                va="center",
                fontsize=6.5,
            )
        add_panel_label(ax, "A", x=-0.32)

        contract = make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=(
                "The manuscript-facing primary effect and robustness range "
                "are rendered from the registered pre-specified robustness panel."
            ),
            panels=[
                {
                    "panel_id": "A",
                    "title": "Primary effect and robustness variants",
                    "role": "robustness",
                    "claim": (
                        "The primary row and converged variants are drawn from "
                        "robustness_panel.json rather than generated figure-step files."
                    ),
                    "evidence_ids": [source_record.evidence_id],
                    "review_risk": (
                        "Non-converged variants remain visible in robustness_panel.json "
                        "and are not silently plotted."
                    ),
                }
            ],
            source_data=[source_record.evidence_id],
            statistics_note=(
                "Generated deterministically from the registered robustness panel "
                "after analysis validation; no model pickle is required."
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
        contract_evidence_id: Optional[str] = None
        figure_ids: List[str] = []
        for key, path in paths.items():
            suffix = path.suffix.lower()
            if key == "contract" or suffix.endswith(".json"):
                record = evidence.register_file(
                    kind="log",
                    description=(
                        "Publication figure contract generated from the "
                        "robustness panel."
                    ),
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
                description=(
                    f"Publication figure export ({suffix.lstrip('.')}) generated "
                    "from the robustness panel."
                ),
                source_path=path,
                evidence_id=f"publication_figure_{suffix.lstrip('.')}",
                aliases=["publication_figure", f"publication_figure_{suffix.lstrip('.')}"],
                producer=self.name,
                generation_mode="deterministic_figure_skill",
                prompt_pack_version=prompt_pack_version,
                metadata={
                    "source_evidence_id": source_record.evidence_id,
                    "figure_contract": "publication_figure_contract",
                    "figure_role": "publication_figure",
                },
            )
            figure_ids.append(record.evidence_id)

        source_copy_record = evidence.register_file(
            kind="table",
            description="Source data copied for the robustness-panel publication figure.",
            source_path=source_copy,
            evidence_id="publication_figure_source_robustness_panel",
            aliases=["publication_figure_source_data"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
            metadata={"source_evidence_id": source_record.evidence_id},
        )
        summary = {
            "stage": self.name,
            "generated": True,
            "generation_mode": "robustness_panel_publication_figure",
            "figure_id": contract.figure_id,
            "core_claim": contract.core_claim,
            "source_evidence_ids": [source_record.evidence_id],
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

    def _promote_prediction_validation_figure(
        self,
        *,
        context: ResearchContext,
        evidence: EvidenceStore,
        run_dir: Path,
        figure_records: Dict[str, EvidenceRecord],
        summary_record: Optional[EvidenceRecord],
        prompt_pack_version: Optional[str],
    ) -> PublicationFigureSkillResult:
        from PIL import Image

        svg_record = figure_records.get("svg")
        png_record = figure_records.get("png")
        if svg_record is None or png_record is None:
            return self._write_skip_summary(
                reason="prediction_figure_bundle_missing_svg_or_png",
                context=context,
                plan=AnalysisPlan(research_question=context.research_question, steps=[]),
                evidence=evidence,
                run_dir=run_dir,
                prompt_pack_version=prompt_pack_version,
            )

        out_dir = run_dir / "publication_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        svg_target = out_dir / "easyicu_publication_figure.svg"
        png_target = out_dir / "easyicu_publication_figure.png"
        pdf_target = out_dir / "easyicu_publication_figure.pdf"
        tiff_target = out_dir / "easyicu_publication_figure.tiff"

        shutil.copy2(run_dir / svg_record.relative_path, svg_target)
        shutil.copy2(run_dir / png_record.relative_path, png_target)

        image = Image.open(png_target).convert("RGB")
        image.save(pdf_target, "PDF", resolution=300.0)
        image.save(tiff_target, compression="tiff_lzw", dpi=(300, 300))
        image.close()

        source_ids = [svg_record.evidence_id, png_record.evidence_id]
        if summary_record is not None:
            source_ids.append(summary_record.evidence_id)
        contract = make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=(
                "The manuscript-facing discrimination and calibration figure is "
                "promoted directly from registered prediction-model evidence."
            ),
            panels=[
                {
                    "panel_id": "A",
                    "title": "Discrimination and calibration",
                    "role": "validation",
                    "claim": (
                        "Prediction-model discrimination and calibration are shown "
                        "from the registered validation figure rather than writer prose."
                    ),
                    "evidence_ids": source_ids,
                    "review_risk": (
                        "Interpretation depends on the upstream model-training "
                        "step summary and any validator findings."
                    ),
                }
            ],
            source_data=source_ids,
            statistics_note=(
                "This publication bundle reuses the registered validation "
                "figure exports and preserves their evidence chain."
            ),
        )
        contract_path = out_dir / "easyicu_publication_figure.figure_contract.json"
        contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")

        paths = {
            "svg": svg_target,
            "png": png_target,
            "pdf": pdf_target,
            "tiff": tiff_target,
            "contract": contract_path,
        }
        audit_findings = list(audit_publication_exports(paths.values()))

        contract_record = evidence.register_file(
            kind="log",
            description="Publication figure contract generated from prediction-model evidence.",
            source_path=contract_path,
            evidence_id="publication_figure_contract",
            aliases=["publication_figure_contract", "figure_contract"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
            metadata={
                "source_evidence_ids": source_ids,
                "figure_role": "publication_figure",
            },
        )

        figure_ids: List[str] = []
        for suffix, path in (
            ("svg", svg_target),
            ("png", png_target),
            ("pdf", pdf_target),
            ("tiff", tiff_target),
        ):
            record = evidence.register_file(
                kind="figure",
                description=(
                    "Publication figure export "
                    f"({suffix}) promoted from prediction-model evidence."
                ),
                source_path=path,
                evidence_id=f"publication_figure_{suffix}",
                aliases=["publication_figure", f"publication_figure_{suffix}"],
                producer=self.name,
                generation_mode="deterministic_figure_skill",
                prompt_pack_version=prompt_pack_version,
                metadata={
                    "source_evidence_ids": source_ids,
                    "figure_contract": contract_record.evidence_id,
                    "figure_role": "publication_figure",
                },
            )
            figure_ids.append(record.evidence_id)

        summary = {
            "stage": self.name,
            "generated": True,
            "generation_mode": "promoted_prediction_validation_figure",
            "figure_id": contract.figure_id,
            "core_claim": contract.core_claim,
            "source_evidence_ids": source_ids,
            "figure_evidence_ids": figure_ids,
            "contract_evidence_id": contract_record.evidence_id,
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
            contract_evidence_id=contract_record.evidence_id,
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


def _first_existing_statistic_record(
    evidence: EvidenceStore,
    names: Sequence[str],
) -> Optional[EvidenceRecord]:
    name_set = {str(name).lower() for name in names}
    for name in names:
        record = evidence.get(name)
        if record is not None and record.kind == "statistic":
            return record
    for record in evidence.records():
        if record.kind != "statistic":
            continue
        basename = Path(record.relative_path).stem.lower()
        if any(token in basename for token in name_set):
            return record
    return None


def _select_existing_prediction_figure_bundle(
    evidence: EvidenceStore,
) -> Optional[Dict[str, EvidenceRecord]]:
    candidates: Dict[str, Dict[str, EvidenceRecord]] = {}
    for record in evidence.records():
        if record.kind != "figure":
            continue
        haystack = " ".join(
            [
                record.evidence_id,
                record.description,
                record.relative_path,
                str((record.metadata or {}).get("figure_role") or ""),
            ]
        ).lower()
        if "publication_figure" in haystack:
            continue
        if not any(
            token in haystack
            for token in (
                "discrimination",
                "calibration",
                "model_performance",
                "roc",
                "auroc",
            )
        ):
            continue
        suffix = Path(record.relative_path).suffix.lower().lstrip(".")
        if suffix not in {"svg", "png"}:
            continue
        stem = Path(record.relative_path).stem.lower()
        group_key = stem.split("__", 1)[-1]
        candidates.setdefault(group_key, {})[suffix] = record
    if not candidates:
        return None
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            0 if "discrimination_calibration" in item[0] else 1,
            0 if {"svg", "png"} <= set(item[1]) else 1,
            item[0],
        ),
    )
    for _, bundle in ranked:
        if {"svg", "png"} <= set(bundle):
            return bundle
    return None


def _has_curated_publication_figure_bundle(evidence: EvidenceStore) -> bool:
    for record in evidence.records():
        haystack = f"{record.evidence_id} {record.relative_path}".lower()
        if (
            record.kind == "figure"
            and "publication_figure" in haystack
            and (
                record.producer == PublicationFigureSkill.name
                or record.generation_mode == "deterministic_figure_skill"
            )
        ):
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
        if record is not None and record.kind == "table":
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
        "label": frame[label_col].astype(str).map(_prettify_label),
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
    raw_label = frame[label_col].astype(str).str.strip().str.lower()
    out = out.loc[raw_label.ne("intercept")].copy()
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["estimate"]
    )
    out["lower"] = out["lower"].fillna(out["estimate"])
    out["upper"] = out["upper"].fillna(out["estimate"])
    return out[["label", "estimate", "lower", "upper"]]


def _normalise_strata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["score", "rate"])
    cols = {str(c).lower(): c for c in frame.columns}
    score_col = _first_col(cols, ["sofa2", "score", "stratum"])
    rate_col = _first_col(cols, ["death_rate", "mortality_rate", "outcome_rate", "rate"])
    n_col = _first_col(cols, ["n", "count", "n_total"])
    if score_col is None or rate_col is None:
        return pd.DataFrame(columns=["score", "rate"])
    out = pd.DataFrame(
        {
            "score": pd.to_numeric(frame[score_col], errors="coerce"),
            "rate": pd.to_numeric(frame[rate_col], errors="coerce"),
        }
    ).dropna(subset=["score", "rate"])
    if n_col is not None:
        out["n"] = pd.to_numeric(frame.loc[out.index, n_col], errors="coerce")
    if not out.empty and out["rate"].max() > 1.0:
        out["rate"] = out["rate"] / 100.0
    return out.sort_values("score").reset_index(drop=True)


def _normalise_missingness_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["variable", "missing_fraction"])
    cols = {str(c).lower(): c for c in frame.columns}
    variable_col = _first_col(cols, ["variable", "feature", "column", "name"])
    frac_col = _first_col(cols, ["missing_fraction", "missing_rate", "fraction"])
    pct_col = _first_col(cols, ["missing_percentage", "missing_percent", "percent"])
    count_col = _first_col(cols, ["missing_count", "missing_n"])
    n_col = _first_col(cols, ["n", "n_total", "total"])
    if variable_col is None:
        return pd.DataFrame(columns=["variable", "missing_fraction"])
    out = pd.DataFrame({"variable": frame[variable_col].astype(str).map(_prettify_label)})
    if frac_col is not None:
        out["missing_fraction"] = pd.to_numeric(frame[frac_col], errors="coerce")
    elif pct_col is not None:
        out["missing_fraction"] = pd.to_numeric(frame[pct_col], errors="coerce") / 100.0
    elif count_col is not None and n_col is not None:
        numerator = pd.to_numeric(frame[count_col], errors="coerce")
        denominator = pd.to_numeric(frame[n_col], errors="coerce").replace(0, pd.NA)
        out["missing_fraction"] = numerator / denominator
    else:
        return pd.DataFrame(columns=["variable", "missing_fraction"])
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["missing_fraction"]
    )
    return out.sort_values("missing_fraction", ascending=False).head(8).reset_index(drop=True)


def _draw_strata_panel(ax: Any, frame: pd.DataFrame, *, palette: Dict[str, str], outcome: str) -> None:
    import matplotlib.ticker as mticker

    x = frame["score"].astype(float)
    y = frame["rate"].astype(float)
    ax.plot(
        x,
        y,
        color=palette.get("blue", "#0F4D92"),
        linewidth=1.4,
        marker="o",
        markersize=3.4,
    )
    ax.set_title("Observed outcome by score", loc="left", pad=3)
    ax.set_xlabel("SOFA-2 score")
    ax.set_ylabel(f"{_prettify_label(outcome)} rate")
    ymax = max(0.05, min(1.0, float(y.max()) * 1.25 if len(y) else 0.05))
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color=palette.get("neutral_light", "#D8D8D8"), linewidth=0.5, alpha=0.7)


def _draw_missingness_panel(ax: Any, frame: pd.DataFrame, *, palette: Dict[str, str]) -> None:
    import matplotlib.ticker as mticker
    import numpy as np

    plot = frame.copy()
    if plot.empty:
        ax.axis("off")
        ax.text(0.0, 0.5, "Missingness not available", va="center")
        return
    y = np.arange(len(plot))
    values = plot["missing_fraction"].astype(float).clip(lower=0)
    ax.barh(y, values, color=palette.get("teal", "#42949E"), height=0.58)
    ax.set_yticks(y, plot["variable"].astype(str).tolist())
    ax.invert_yaxis()
    ax.set_title("Feature missingness", loc="left", pad=3)
    ax.set_xlabel("Missing")
    xmax = max(0.05, min(1.0, float(values.max()) * 1.35 if len(values) else 0.05))
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    if float(values.max()) == 0.0:
        ax.text(
            0.01,
            0.5,
            "0% missing across displayed variables",
            transform=ax.transAxes,
            va="center",
            ha="left",
            color=palette.get("neutral", "#8F8F8F"),
            fontsize=7.0,
        )
    ax.grid(axis="x", color=palette.get("neutral_light", "#D8D8D8"), linewidth=0.5, alpha=0.7)


def _prettify_label(value: Any) -> str:
    token = str(value or "").strip()
    mapping = {
        "sofa2": "SOFA-2",
        "sex_m": "Male sex",
        "death": "ICU mortality",
        "lact": "Lactate",
        "creat": "Creatinine",
        "map": "MAP",
        "los_icu": "ICU LOS",
        "stay_id": "ICU stay",
    }
    lower = token.lower()
    if lower in mapping:
        return mapping[lower]
    return token.replace("_", " ").strip().title()


def _first_col(cols: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in cols:
            return cols[candidate]
    return None


__all__ = ["PublicationFigureSkill", "PublicationFigureSkillResult"]
