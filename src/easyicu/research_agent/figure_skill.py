"""Publication figure skill for evidence-bound manuscript figures.

Exploratory plots may be emitted by analysis scripts, but manuscript-facing
figures should pass through a small, auditable EasyICU figure skill once the
analysis evidence is stable. This module sits between analysis and writing:
it consumes registered tables/statistics, creates a claim-first
``FigureContract``, exports journal-friendly formats through
``publication_figures``, and registers every output in the EvidenceStore.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .audits.validators import FigureContractQualityValidator
from .evidence import EvidenceStore
from .publication_figures import (
    PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
    add_panel_label,
    apply_publication_style,
    audit_publication_exports,
    make_figure_contract,
    save_publication_figure,
)
from .figures import RenderedFigure, render_family_figure
from .robustness_panel import RobustnessPanel, load_robustness_panel
from .schema import AnalysisPlan, EvidenceRecord, ResearchContext, ValidationFinding
from .planning.study_design import infer_study_design_family


def _close_leaked_figures() -> None:
    """Close matplotlib figures left open by a render that raised mid-way.

    pyplot is imported lazily inside the render methods, so only close when
    it is actually loaded — importing it here would break the skip path on
    installs without matplotlib.
    """
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass


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
        if _has_curated_publication_figure_bundle(
            evidence,
            run_dir=run_dir,
            context=context,
        ):
            return PublicationFigureSkillResult(
                generated=False,
                skipped_reason="existing_curated_publication_figure_bundle",
            )
        # Study-design-aware dispatch: survival / prediction / phenotyping /
        # causal questions render their own family figure (KM curve, ROC +
        # calibration, cluster heatmap, love plot) instead of being funnelled
        # into the association forest. The renderer returns None when its
        # source evidence is absent, so association/descriptive runs and any
        # family whose analysis did not produce its tables fall straight
        # through to the existing ladder below with no behaviour change.
        family = infer_study_design_family(context)
        family_figure = render_family_figure(
            family,
            context=context,
            plan=plan,
            evidence=evidence,
            run_dir=run_dir,
        )
        if family_figure is not None:
            try:
                return self._finalise_family_figure(
                    context=context,
                    evidence=evidence,
                    run_dir=run_dir,
                    rendered=family_figure,
                    prompt_pack_version=prompt_pack_version,
                )
            except Exception:
                # A finalisation bug must not crash the figure stage; fall
                # through to the existing association/promotion/skip ladder.
                _close_leaked_figures()
        primary = _select_primary_association_record(
            evidence,
            run_dir=run_dir,
            context=context,
            names=[
                "primary_association",
                "primary_association_table",
                "table_primary_association",
                "adjusted_association",
                "adjusted_association_death",
                "association_table",
            ],
        )
        if primary is None:
            primary = _first_existing_record(
                evidence,
                [
                    "primary_association",
                    "primary_association_table",
                    "table_primary_association",
                    "adjusted_association",
                    "adjusted_association_death",
                    "association_table",
                ],
            )
        promoted_bundle = _select_existing_step_publication_figure_bundle(evidence)
        if promoted_bundle is not None and (
            primary is None or _bundle_primary_strategy_ready(context, promoted_bundle)
        ):
            return self._promote_registered_publication_figure(
                context=context,
                evidence=evidence,
                run_dir=run_dir,
                bundle=promoted_bundle,
                prompt_pack_version=prompt_pack_version,
            )
        if primary is None:
            robustness_record = _latest_record_for_basename(
                evidence,
                "robustness_panel.json",
                kind="statistic",
            )
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
                    _close_leaked_figures()
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
        if primary is not None:
            try:
                frame = _read_table(run_dir / primary.relative_path)
                strata = _first_normalisable_record(
                    evidence,
                    [
                        "stratified_mortality",
                        "stratified_mortality_incidence",
                        "outcome_by_exposure",
                        "outcome_by_primary_exposure",
                        "outcome_by_group",
                        "outcome_by_sepsis3",
                        # Association steps commonly export the absolute
                        # outcome risk by exposure group as
                        # absolute_risk_by_<exposure>.csv; the prefix token
                        # matches any exposure spelling in the substring
                        # pass. It is the same descriptive-result content
                        # as an outcome-by-exposure table.
                        "absolute_risk_by",
                    ],
                    run_dir=run_dir,
                    normalise=_normalise_strata_frame,
                )
                missingness = _first_normalisable_record(
                    evidence,
                    [
                        "missingness",
                        "missingness_summary",
                        "table_missingness",
                        "measurement_missingness",
                        "cohort_missingness_audit",
                    ],
                    run_dir=run_dir,
                    normalise=_normalise_missingness_frame,
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
                _close_leaked_figures()
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
        robustness_record = _latest_record_for_basename(
            evidence,
            "robustness_panel.json",
            kind="statistic",
        )
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
                _close_leaked_figures()
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
        return self._write_skip_summary(
            reason="no_supported_source_table",
            context=context,
            plan=plan,
            evidence=evidence,
            run_dir=run_dir,
            prompt_pack_version=prompt_pack_version,
        )

    def _finalise_family_figure(
        self,
        *,
        context: ResearchContext,
        evidence: EvidenceStore,
        run_dir: Path,
        rendered: RenderedFigure,
        prompt_pack_version: Optional[str],
    ) -> PublicationFigureSkillResult:
        """Persist a family renderer's figure via the shared save/register path.

        Renderers stay free of EvidenceStore mechanics; this method performs
        the source-copy registration, contract build, journal-format export,
        export/contract audit, and evidence registration in one place — the
        same registration surface as ``_render_primary_association`` so the
        readiness gates and manuscript binder see an identical figure record
        regardless of which family produced it.
        """

        import matplotlib.pyplot as plt

        out_dir = run_dir / "publication_figures"
        out_dir.mkdir(parents=True, exist_ok=True)

        source_copy_ids: List[str] = []
        for name, frame in rendered.source_frames.items():
            try:
                copy_path = out_dir / f"publication_figure_source_{name}.csv"
                frame.to_csv(copy_path, index=False)
            except Exception:
                continue
            record = evidence.register_file(
                kind="table",
                description=f"Source data copied for the {rendered.generation_mode}.",
                source_path=copy_path,
                evidence_id=f"publication_figure_source_{name}",
                aliases=[
                    "publication_figure_source_data",
                    f"publication_figure_source_{name}",
                ],
                producer=self.name,
                generation_mode="deterministic_figure_skill",
                prompt_pack_version=prompt_pack_version,
                on_sha_change="new_id",
            )
            source_copy_ids.append(record.evidence_id)

        contract = make_figure_contract(
            figure_id=rendered.figure_id,
            core_claim=rendered.core_claim,
            panels=rendered.panels,
            source_data=rendered.source_evidence_ids or source_copy_ids,
            statistics_note=rendered.statistics_note,
        )
        paths = save_publication_figure(
            rendered.fig,
            out_dir / rendered.figure_id,
            contract=contract,
            dpi=300,
        )
        plt.close(rendered.fig)

        audit_findings = list(audit_publication_exports(paths))
        audit_findings.extend(
            FigureContractQualityValidator().audit_contract_file(
                paths["contract"],
                manuscript_facing=True,
            )
        )
        contract_record, figure_records = _register_publication_figure_bundle(
            evidence=evidence,
            paths=paths,
            contract=contract,
            prompt_pack_version=prompt_pack_version,
        )
        contract_evidence_id = contract_record.evidence_id
        figure_ids = [record.evidence_id for record in figure_records]

        summary = {
            "stage": self.name,
            "generated": True,
            "generation_mode": rendered.generation_mode,
            "figure_id": contract.figure_id,
            "core_claim": contract.core_claim,
            "source_evidence_ids": list(rendered.source_evidence_ids),
            "source_copy_evidence_ids": source_copy_ids,
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
            on_sha_change="new_id",
        )
        return PublicationFigureSkillResult(
            generated=True,
            contract_evidence_id=contract_evidence_id,
            figure_evidence_ids=figure_ids,
            summary_evidence_id=summary_record.evidence_id,
            findings=audit_findings,
        )

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

        plot_df = _normalise_association_frame(
            frame,
            primary_exposure=context.primary_exposure,
        )
        if plot_df.empty:
            raise ValueError("primary association table has no plottable rows")
        axis_meta = _association_axis_metadata(plot_df)
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
                        out_dir / "publication_figure_source_stratified_outcome.csv",
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
            compact_primary = len(plot_df) <= 2 and n_side_panels >= 2
            if compact_primary:
                grid = fig.add_gridspec(
                    2,
                    2,
                    width_ratios=[1.16, 1.0],
                    height_ratios=[0.86, 1.0],
                    left=0.16,
                    right=0.98,
                    top=0.94,
                    bottom=0.14,
                    wspace=0.42,
                    hspace=0.72,
                )
                ax = fig.add_subplot(grid[0, 0])
                side_axes = []
                if not strata_df.empty:
                    side_axes.append(("strata", fig.add_subplot(grid[0, 1])))
                if not missingness_df.empty:
                    side_axes.append(("missingness", fig.add_subplot(grid[1, :])))
            else:
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
            fig, ax = plt.subplots(
                figsize=(126 / 25.4, height), constrained_layout=False
            )
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
        null_value = axis_meta.get("null_value")
        if null_value is not None:
            ax.axvline(
                float(null_value),
                color=palette.get("neutral", "#8F8F8F"),
                linestyle="--",
                linewidth=0.8,
            )
        ax.set_yticks(y, plot_df["label"].astype(str).tolist())
        ax.invert_yaxis()
        ax.set_xlabel(str(axis_meta["xlabel"]))
        ax.set_ylabel("")
        use_log_scale = (
            bool(axis_meta.get("ratio_scale"))
            and float(lower.min()) > 0
            and float(upper.max()) / max(float(lower.min()), 1e-9) > 1.8
        )
        if use_log_scale:
            ax.set_xscale("log")
        anchors = [float(upper.max()), float(estimate.max())]
        left_anchors = [float(lower.min()), float(estimate.min())]
        if null_value is not None:
            anchors.append(float(null_value))
            left_anchors.append(float(null_value))
        right_anchor = max(anchors)
        if use_log_scale:
            right_pad = right_anchor * 0.6
            ax.set_xlim(max(float(lower.min()) * 0.8, 1e-3), right_anchor + right_pad)
        else:
            left_anchor = min(left_anchors)
            span = max(right_anchor - left_anchor, 1e-6)
            right_pad = max(span * 0.38, 0.1)
            left_pad = max(span * 0.12, 0.05)
            ax.set_xlim(left_anchor - left_pad, right_anchor + right_pad)
        text_x = right_anchor + right_pad * 0.12
        # The annotation-column header lives in the title band as a
        # right-anchored title. Earlier placements collided in the SVG QA:
        # inside the axes at y=0.96 it overlapped the row-0 annotation
        # (inverted y-axis puts row 0 on top), and free-floating above the
        # axes at y=1.02 it overlapped long left titles on short axes.
        # Left/right titles share one band with opposite anchors, so they
        # stay apart for realistic title lengths.
        ax.set_title(
            str(axis_meta["header"]),
            loc="right",
            pad=4,
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
        outcome_label = _prettify_label(context.target_outcome or "target outcome")
        ax.set_title(f"Adjusted estimate for {outcome_label}", loc="left", pad=4)
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
                "role": "primary_estimand",
                "chart_type": "dot_interval",
                "claim": "The association estimate and interval are drawn from the registered primary association table.",
                "evidence_ids": [source_record.evidence_id],
                "review_risk": "Interpretability depends on the upstream model specification and validator findings.",
            }
        ]
        if not strata_df.empty and strata_record is not None:
            score_label = _strata_score_label(strata_df)
            panels.append(
                {
                    "panel_id": "B",
                    "title": f"Outcome by {score_label}",
                    "role": "descriptive_result",
                    "chart_type": "event_rate_panel",
                    "claim": f"Observed outcome risk by {score_label} is shown before adjusted relative estimates.",
                    "evidence_ids": [strata_record.evidence_id],
                    "review_risk": "Sparse high-score strata should be interpreted with their denominators.",
                }
            )
        if not missingness_df.empty and missingness_record is not None:
            panels.append(
                {
                    "panel_id": chr(ord("A") + len(panels)),
                    "title": "Missingness audit",
                    "role": "data_quality",
                    "chart_type": "availability_panel",
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
        audit_findings.extend(
            FigureContractQualityValidator().audit_contract_file(
                paths["contract"],
                manuscript_facing=True,
            )
        )
        contract_record, figure_records = _register_publication_figure_bundle(
            evidence=evidence,
            paths=paths,
            contract=contract,
            prompt_pack_version=prompt_pack_version,
        )
        contract_evidence_id = contract_record.evidence_id
        figure_ids = [record.evidence_id for record in figure_records]
        source_metadata = _source_fingerprint_metadata(
            evidence,
            _figure_contract_source_ids(contract),
        )

        source_copy_record = evidence.register_file(
            kind="table",
            description="Source data copied for the publication figure skill.",
            source_path=source_copy,
            evidence_id="publication_figure_source_primary_association",
            aliases=["publication_figure_source_data"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
            metadata=source_metadata,
            on_sha_change="new_id",
        )

        summary = {
            "stage": self.name,
            "generated": True,
            "generation_mode": "primary_association_publication_figure",
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
            on_sha_change="new_id",
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
        primary = primary_rows[0] if primary_rows else None
        other_rows = [
            row
            for row in rows
            if row.spec_id != panel.primary_spec_id
            and not _duplicates_primary_row(row, primary)
        ]
        plot_rows = (primary_rows + other_rows)[:10]
        source_df = pd.DataFrame(
            [
                {
                    "spec_id": row.spec_id,
                    "display_label": _robustness_spec_label(row.spec_id),
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
        all_rows_df = pd.DataFrame(
            [
                {
                    "spec_id": row.spec_id,
                    "axis": row.axis or "unspecified",
                    "n": row.n,
                    "converged": bool(row.converged),
                    "point_estimate": row.point_estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "notes": row.notes,
                }
                for row in panel.rows
            ]
        )
        if all_rows_df.empty:
            all_rows_df = source_df[["spec_id", "axis", "n", "converged"]].copy()
        all_rows_df["axis"] = all_rows_df["axis"].fillna("unspecified").astype(str)
        all_rows_df["n"] = pd.to_numeric(all_rows_df["n"], errors="coerce")
        all_rows_df.loc[all_rows_df["n"] <= 0, "n"] = pd.NA
        all_rows_df["converged"] = all_rows_df["converged"].astype(bool)
        axis_summary = (
            all_rows_df.groupby("axis", dropna=False)
            .agg(
                total_specs=("spec_id", "count"),
                converged_specs=("converged", "sum"),
                median_n=("n", "median"),
                min_n=("n", "min"),
                max_n=("n", "max"),
            )
            .reset_index()
        )
        axis_summary["not_converged_specs"] = (
            axis_summary["total_specs"] - axis_summary["converged_specs"]
        )
        axis_order = {"primary": 0, "cohort": 1, "missing": 2, "outcome": 3}
        axis_summary["axis_order"] = axis_summary["axis"].map(
            lambda value: axis_order.get(str(value).lower(), 99)
        )
        axis_summary = axis_summary.sort_values(
            ["axis_order", "axis"],
            kind="stable",
        ).reset_index(drop=True)
        axis_summary["axis_label"] = axis_summary["axis"].map(_robustness_axis_label)

        palette = apply_publication_style()
        out_dir = run_dir / "publication_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        source_copy = out_dir / "publication_figure_source_robustness_panel.csv"
        source_df.to_csv(source_copy, index=False)
        axis_summary_copy = (
            out_dir / "publication_figure_source_robustness_axis_summary.csv"
        )
        axis_summary.drop(columns=["axis_order"]).to_csv(
            axis_summary_copy,
            index=False,
        )

        height = max(4.35, 0.38 * len(plot_rows) + 1.85)
        fig = plt.figure(figsize=(183 / 25.4, height), constrained_layout=False)
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.55, 0.95],
            height_ratios=[1.0, 1.0],
            left=0.27,
            right=0.98,
            top=0.93,
            bottom=0.14,
            wspace=0.42,
            hspace=0.55,
        )
        ax = fig.add_subplot(grid[:, 0])
        ax_counts = fig.add_subplot(grid[0, 1])
        ax_n = fig.add_subplot(grid[1, 1])
        y = np.arange(len(source_df))
        estimate = source_df["point_estimate"].astype(float).to_numpy()
        lower = source_df["ci_low"].astype(float).to_numpy()
        upper = source_df["ci_high"].astype(float).to_numpy()
        labels = [
            (
                "Primary"
                if str(row["spec_id"]) == panel.primary_spec_id
                else str(row["display_label"])
            )
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
        add_panel_label(ax, "A", x=-0.36)

        axis_y = np.arange(len(axis_summary))
        axis_labels = axis_summary["axis_label"].astype(str).tolist()
        converged = axis_summary["converged_specs"].astype(float).to_numpy()
        not_converged = axis_summary["not_converged_specs"].astype(float).to_numpy()
        ax_counts.barh(
            axis_y,
            converged,
            color=palette.get("green", "#008B5E"),
            height=0.58,
            label="Converged",
        )
        if float(not_converged.max()) > 0:
            ax_counts.barh(
                axis_y,
                not_converged,
                left=converged,
                color=palette.get("neutral_light", "#D8D8D8"),
                height=0.58,
                label="Not converged",
            )
        for idx, row in axis_summary.iterrows():
            total = int(row["total_specs"])
            conv = int(row["converged_specs"])
            ax_counts.text(
                float(total) + 0.08,
                idx,
                f"{conv}/{total}",
                va="center",
                ha="left",
                fontsize=6.5,
                color=palette.get("baseline", "#272727"),
            )
        ax_counts.set_yticks(axis_y, axis_labels)
        ax_counts.invert_yaxis()
        ax_counts.set_xlabel("Specifications, n")
        ax_counts.set_title("Convergence", loc="left", pad=7)
        ax_counts.set_xlim(0, max(1.0, float(axis_summary["total_specs"].max()) + 1.0))
        ax_counts.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.5,
            alpha=0.75,
        )
        if float(not_converged.max()) > 0:
            ax_counts.legend(
                frameon=False,
                fontsize=5.6,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.02),
                ncol=1,
                handlelength=1.2,
                borderaxespad=0.0,
            )
        add_panel_label(ax_counts, "B", x=-0.18, y=1.06, fontsize=10.0)

        n_summary = axis_summary.dropna(subset=["median_n", "min_n", "max_n"]).copy()
        if n_summary.empty:
            ax_n.text(
                0.5,
                0.5,
                "Analytic n not reported",
                ha="center",
                va="center",
                fontsize=7.0,
                transform=ax_n.transAxes,
            )
            ax_n.set_axis_off()
        else:
            n_y = np.arange(len(n_summary))
            median_n = n_summary["median_n"].astype(float).to_numpy()
            min_n = n_summary["min_n"].astype(float).to_numpy()
            max_n = n_summary["max_n"].astype(float).to_numpy()
            ax_n.errorbar(
                median_n,
                n_y,
                xerr=np.vstack(
                    [
                        np.maximum(0.0, median_n - min_n),
                        np.maximum(0.0, max_n - median_n),
                    ]
                ),
                fmt="o",
                color=palette.get("blue", "#0F4D92"),
                ecolor=palette.get("blue", "#0F4D92"),
                elinewidth=1.0,
                capsize=2.0,
                markersize=3.8,
            )
            ax_n.set_yticks(n_y, n_summary["axis_label"].astype(str).tolist())
            ax_n.invert_yaxis()
            ax_n.set_xlabel("Analytic sample size (n)")
            ax_n.set_title("Sample-size range", loc="left", pad=4)
            ax_n.grid(
                axis="x",
                color=palette.get("neutral_light", "#D8D8D8"),
                linewidth=0.5,
                alpha=0.75,
            )
            if float(max_n.max()) > 0:
                ax_n.set_xlim(0, float(max_n.max()) * 1.12)
        add_panel_label(ax_n, "C", x=-0.18, y=1.06, fontsize=10.0)

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
                        "registered robustness evidence rather than generated "
                        "figure-step files."
                    ),
                    "evidence_ids": [source_record.evidence_id],
                    "review_risk": (
                        "Non-converged variants remain recorded in the registered "
                        "robustness evidence and are not silently plotted."
                    ),
                },
                {
                    "panel_id": "B",
                    "title": "Variant convergence by axis",
                    "role": "validation",
                    "claim": (
                        "The number of converged and non-converged robustness "
                        "specifications is summarised by pre-specified analysis axis."
                    ),
                    "evidence_ids": [source_record.evidence_id],
                    "review_risk": (
                        "A small number of variants limits how strongly the "
                        "robustness range can be interpreted."
                    ),
                },
                {
                    "panel_id": "C",
                    "title": "Analytic sample-size range",
                    "role": "audit",
                    "claim": (
                        "Sample-size ranges are shown by robustness axis so "
                        "estimate shifts can be read with denominator context."
                    ),
                    "evidence_ids": [source_record.evidence_id],
                    "review_risk": (
                        "Large denominator changes can indicate that a robustness "
                        "variant is testing both definition and selection effects."
                    ),
                },
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
        audit_findings.extend(
            FigureContractQualityValidator().audit_contract_file(
                paths["contract"],
                manuscript_facing=True,
            )
        )
        contract_record, figure_records = _register_publication_figure_bundle(
            evidence=evidence,
            paths=paths,
            contract=contract,
            prompt_pack_version=prompt_pack_version,
        )
        contract_evidence_id = contract_record.evidence_id
        figure_ids = [record.evidence_id for record in figure_records]
        source_metadata = _source_fingerprint_metadata(
            evidence,
            _figure_contract_source_ids(contract),
        )

        source_copy_record = evidence.register_file(
            kind="table",
            description="Source data copied for the robustness-panel publication figure.",
            source_path=source_copy,
            evidence_id="publication_figure_source_robustness_panel",
            aliases=["publication_figure_source_data"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
            metadata=source_metadata,
            on_sha_change="new_id",
        )
        axis_summary_record = evidence.register_file(
            kind="table",
            description=(
                "Axis-level source data copied for the robustness-panel "
                "publication figure."
            ),
            source_path=axis_summary_copy,
            evidence_id="publication_figure_source_robustness_axis_summary",
            aliases=["publication_figure_axis_summary_source_data"],
            producer=self.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
            metadata=source_metadata,
            on_sha_change="new_id",
        )
        summary = {
            "stage": self.name,
            "generated": True,
            "generation_mode": "robustness_panel_publication_figure",
            "figure_id": contract.figure_id,
            "core_claim": contract.core_claim,
            "source_evidence_ids": [source_record.evidence_id],
            "source_copy_evidence_id": source_copy_record.evidence_id,
            "axis_summary_source_evidence_id": axis_summary_record.evidence_id,
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
            on_sha_change="new_id",
        )
        return PublicationFigureSkillResult(
            generated=True,
            contract_evidence_id=contract_evidence_id,
            figure_evidence_ids=figure_ids,
            summary_evidence_id=summary_record.evidence_id,
            findings=audit_findings,
        )

    def _promote_registered_publication_figure(
        self,
        *,
        context: ResearchContext,
        evidence: EvidenceStore,
        run_dir: Path,
        bundle: Dict[str, Any],
        prompt_pack_version: Optional[str],
    ) -> PublicationFigureSkillResult:
        """Promote a step-level manuscript figure bundle to the run-level bundle."""

        from PIL import Image

        figure_records: Dict[str, EvidenceRecord] = bundle["figures"]
        contract_source: EvidenceRecord = bundle["contract"]
        source_records: List[EvidenceRecord] = list(bundle.get("source_records") or [])
        svg_record = figure_records.get("svg")
        png_record = figure_records.get("png")
        if svg_record is None or png_record is None:
            return self._write_skip_summary(
                reason="registered_publication_bundle_missing_svg_or_png",
                context=context,
                plan=AnalysisPlan(
                    research_question=context.research_question,
                    steps=[],
                ),
                evidence=evidence,
                run_dir=run_dir,
                prompt_pack_version=prompt_pack_version,
            )

        out_dir = run_dir / "publication_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = {
            "svg": out_dir / "easyicu_publication_figure.svg",
            "png": out_dir / "easyicu_publication_figure.png",
            "pdf": out_dir / "easyicu_publication_figure.pdf",
            "tiff": out_dir / "easyicu_publication_figure.tiff",
        }
        for suffix, target in targets.items():
            source = figure_records.get(suffix)
            if source is not None:
                shutil.copy2(run_dir / source.relative_path, target)
        if not targets["pdf"].exists() or not targets["tiff"].exists():
            image = Image.open(targets["png"]).convert("RGB")
            if not targets["pdf"].exists():
                image.save(targets["pdf"], "PDF", resolution=300.0)
            if not targets["tiff"].exists():
                image.save(targets["tiff"], compression="tiff_lzw", dpi=(300, 300))
            image.close()

        source_ids = _unique_evidence_ids(
            [
                *figure_records.values(),
                contract_source,
                *source_records,
            ]
        )
        contract = _contract_promoted_from_source(
            run_dir / contract_source.relative_path,
            source_ids=source_ids,
        )
        contract_path = out_dir / "easyicu_publication_figure.figure_contract.json"
        contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")

        paths = {
            "svg": targets["svg"],
            "png": targets["png"],
            "pdf": targets["pdf"],
            "tiff": targets["tiff"],
            "contract": contract_path,
        }
        audit_findings = list(audit_publication_exports(paths.values()))
        audit_findings.extend(
            FigureContractQualityValidator().audit_contract_file(
                contract_path,
                manuscript_facing=True,
            )
        )

        provenance_metadata = {
            "promoted_from_step_id": bundle.get("step_id"),
            "promoted_from_stem": bundle.get("stem"),
        }
        contract_record, registered_figures = _register_publication_figure_bundle(
            evidence=evidence,
            paths=paths,
            contract=contract,
            prompt_pack_version=prompt_pack_version,
            contract_metadata=provenance_metadata,
            figure_metadata=provenance_metadata,
        )
        figure_ids = [record.evidence_id for record in registered_figures]

        summary = {
            "stage": self.name,
            "generated": True,
            "generation_mode": "promoted_step_publication_figure",
            "figure_id": contract.figure_id,
            "core_claim": contract.core_claim,
            "source_evidence_ids": source_ids,
            "figure_evidence_ids": figure_ids,
            "contract_evidence_id": contract_record.evidence_id,
            "promoted_from_step_id": bundle.get("step_id"),
            "promoted_from_stem": bundle.get("stem"),
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
            on_sha_change="new_id",
        )
        return PublicationFigureSkillResult(
            generated=True,
            contract_evidence_id=contract_record.evidence_id,
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
                plan=AnalysisPlan(
                    research_question=context.research_question, steps=[]
                ),
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
        audit_findings.extend(
            FigureContractQualityValidator().audit_contract_file(
                contract_path,
                manuscript_facing=True,
            )
        )

        contract_record, registered_figures = _register_publication_figure_bundle(
            evidence=evidence,
            paths=paths,
            contract=contract,
            prompt_pack_version=prompt_pack_version,
        )
        figure_ids = [record.evidence_id for record in registered_figures]

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
            on_sha_change="new_id",
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
            on_sha_change="new_id",
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


def _select_existing_step_publication_figure_bundle(
    evidence: EvidenceStore,
) -> Optional[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    records = evidence.records()
    for order, record in enumerate(records):
        metadata = record.metadata or {}
        # Step association lives on the record itself (produced_by_step);
        # metadata["step_id"] is never populated by the runner registration
        # path, and keying on it alone collapsed every step's bundle into the
        # ("", stem) group so contracts could attach to another step's figure.
        step_id = str(
            metadata.get("step_id") or getattr(record, "produced_by_step", "") or ""
        )
        if record.kind == "figure":
            if _is_run_level_publication_figure(record):
                continue
            if str(metadata.get("figure_role") or "").lower() != "publication_figure":
                continue
            suffix = Path(record.relative_path).suffix.lower().lstrip(".")
            if suffix not in {"svg", "png", "pdf", "tiff", "tif"}:
                continue
            stem = _record_artifact_stem(record)
            key = (step_id, stem)
            group = groups.setdefault(
                key,
                {
                    "step_id": step_id,
                    "stem": stem,
                    "figures": {},
                    "contract": None,
                    "contract_payload": {},
                    "source_records": [],
                    "order": order,
                },
            )
            group["figures"]["tiff" if suffix == "tif" else suffix] = record
            group["order"] = max(int(group.get("order", 0)), order)
            continue
        if record.kind == "log" and _record_artifact_basename(record).endswith(
            ".figure_contract.json"
        ):
            stem = _record_artifact_basename(record).removesuffix(
                ".figure_contract.json"
            )
            key = (step_id, stem)
            group = groups.setdefault(
                key,
                {
                    "step_id": step_id,
                    "stem": stem,
                    "figures": {},
                    "contract": None,
                    "contract_payload": {},
                    "source_records": [],
                    "order": order,
                },
            )
            group["contract"] = record
            group["contract_payload"] = _read_contract_payload(evidence, record)
            group["order"] = max(int(group.get("order", 0)), order)
            continue
        if record.kind == "table":
            basename = _record_artifact_basename(record).lower()
            if "source_data" not in basename:
                continue
            for key, group in groups.items():
                if key[0] == step_id:
                    group.setdefault("source_records", []).append(record)
                    group["order"] = max(int(group.get("order", 0)), order)

    # Content-addressed evidence registration can deduplicate a copied source
    # table back to its original parent-step record.  In that common case the
    # publication-figure child has valid contract references but no table whose
    # ``produced_by_step`` equals the child step, so the one-pass association
    # above incorrectly declares the bundle source-less.  Resolve every local
    # table/evidence reference named by the contract across the whole store;
    # exact basename/stem/id matching keeps the promotion evidence-bound.
    table_records = [record for record in records if record.kind == "table"]
    record_order = {
        record.evidence_id: order for order, record in enumerate(records)
    }
    for group in groups.values():
        refs = _contract_payload_source_references(
            group.get("contract_payload") or {}
        )
        attached = list(group.get("source_records") or [])
        attached_ids = {record.evidence_id for record in attached}

        def record_step_id(record: EvidenceRecord) -> str:
            return str(
                (record.metadata or {}).get("step_id")
                or getattr(record, "produced_by_step", "")
                or ""
            )

        def record_tokens(record: EvidenceRecord) -> set[str]:
            basename = _record_artifact_basename(record).lower()
            return {
                record.evidence_id.lower(),
                basename,
                Path(basename).stem,
                Path(record.relative_path).name.lower(),
                Path(record.relative_path).stem.lower(),
            }

        group_step_id = str(group.get("step_id") or "")
        direct_parent_id = group_step_id.removesuffix("_figure")
        for ref in refs:
            token = str(ref).strip().lower()
            if not token:
                continue
            ref_tokens = {token, Path(token).name, Path(token).stem}
            candidates = [
                record
                for record in table_records
                if ref_tokens & record_tokens(record)
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda record: (
                    2 if record_step_id(record) == group_step_id else 0,
                    1 if record_step_id(record) == direct_parent_id else 0,
                    record_order.get(record.evidence_id, -1),
                ),
                reverse=True,
            )
            record = candidates[0]
            if record.evidence_id in attached_ids:
                continue
            attached.append(record)
            attached_ids.add(record.evidence_id)
        group["source_records"] = attached

    viable = [
        group
        for group in groups.values()
        if {"svg", "png"} <= set(group.get("figures", {}))
        and group.get("contract") is not None
        and group.get("source_records")
    ]
    if not viable:
        return None
    ranked = sorted(viable, key=_step_publication_bundle_rank)
    return ranked[0]


def _contract_payload_source_references(payload: Any) -> List[str]:
    """Collect source/evidence references from a raw figure contract."""

    references: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            token = value.strip()
            if token:
                references.append(token)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)

    if not isinstance(payload, dict):
        return references
    collect(payload.get("source_data"))
    for panel in payload.get("panels") or []:
        if isinstance(panel, dict):
            collect(panel.get("evidence_ids"))
    return list(dict.fromkeys(references))


_PRIMARY_RESULT_PANEL_ROLES = {
    "descriptive_result",
    "primary_estimand",
    "temporal_absolute_risk",
    "survival_effect",
    "clinical_utility",
}
_VALIDATION_PANEL_ROLES = {
    "model_performance",
    "calibration",
    "validation",
    "explainability",
    "transportability",
}
_CONTEXT_PANEL_ROLES = {
    "cohort_accounting",
    "baseline_context",
    "data_quality",
    "overview",
    "relationship",
    "heterogeneity",
    "distribution",
}
_SUPPLEMENTAL_PANEL_ROLES = {
    "robustness",
    "audit",
    "diagnostics",
    "stability",
    "supplementary_provenance",
}
_PRIMARY_CONTEXT_CHART_TOKENS = (
    "absolute_risk",
    "event_rate",
    "prevalence",
    "incidence",
    "survival",
)
_PRIMARY_PUBLICATION_ROLE_POOLS: Dict[str, set[str]] = {
    "association": {
        "descriptive_result",
        "primary_estimand",
        "robustness",
        "data_quality",
    },
    "prediction": {
        "model_performance",
        "calibration",
        "validation",
        "data_quality",
    },
    "time_to_event": {
        "temporal_absolute_risk",
        "survival_effect",
        "diagnostics",
    },
    "phenotyping": {
        "phenotype_structure",
        "phenotype_profile",
        "stability",
        "data_quality",
    },
    "causal_emulation": {
        "causal_protocol",
        "balance_positivity",
        "causal_contrast",
        "robustness",
    },
    "descriptive": {
        "distribution",
        "cohort_accounting",
        "data_quality",
    },
}
_PRIMARY_PUBLICATION_HERO_ROLES: Dict[str, str] = {
    "association": "descriptive_result",
    "prediction": "calibration",
    "time_to_event": "temporal_absolute_risk",
    "phenotyping": "phenotype_structure",
    "causal_emulation": "causal_protocol",
    "descriptive": "distribution",
}
_PRIMARY_PUBLICATION_MIN_ROLE_COUNTS: Dict[str, int] = {
    "association": 3,
    "prediction": 3,
    "time_to_event": 2,
    "phenotyping": 3,
    "causal_emulation": 3,
    "descriptive": 2,
}


def _bundle_primary_strategy_ready(
    context: ResearchContext,
    bundle: Dict[str, Any],
) -> bool:
    """Return True when a step-level bundle is rich enough for the main figure."""

    payload = bundle.get("contract_payload")
    if not isinstance(payload, dict):
        return False
    return _contract_primary_strategy_ready(context, payload)


def _contract_primary_strategy_ready(
    context: ResearchContext,
    payload: Dict[str, Any],
) -> bool:
    """Return True when a figure contract is rich enough for the main figure."""

    family = str(infer_study_design_family(context))
    role_pool = _PRIMARY_PUBLICATION_ROLE_POOLS.get(family)
    if not role_pool:
        return True
    roles = _contract_payload_roles(payload)
    hero_role = _PRIMARY_PUBLICATION_HERO_ROLES.get(family)
    if hero_role and hero_role not in roles:
        return False
    minimum = min(
        len(role_pool),
        _PRIMARY_PUBLICATION_MIN_ROLE_COUNTS.get(family, min(3, len(role_pool))),
    )
    return len(roles & role_pool) >= minimum


def _step_publication_bundle_rank(
    bundle: Dict[str, Any],
) -> Tuple[int, int, int, int, str]:
    roles = _bundle_contract_roles(bundle)
    chart_types = _bundle_contract_chart_types(bundle)
    step_text = str(bundle.get("step_id") or "").lower()
    stem_text = str(bundle.get("stem") or "").lower()
    text = f"{step_text} {stem_text}"
    generic_penalty = 1 if stem_text in {"publication_figure", "figure"} else 0
    primary_role_count = len(roles & _PRIMARY_RESULT_PANEL_ROLES)
    has_absolute_context = any(
        any(token in chart_type for token in _PRIMARY_CONTEXT_CHART_TOKENS)
        for chart_type in chart_types
    )
    supplemental_only = bool(roles) and roles <= _SUPPLEMENTAL_PANEL_ROLES
    sensitivity_or_robustness = "sensitivity" in text or "robust" in text

    if {"descriptive_result", "primary_estimand"} <= roles:
        family_rank = 0
    elif primary_role_count and has_absolute_context:
        family_rank = 1
    elif primary_role_count:
        family_rank = 2
    elif "prediction" in text or "calibration" in text or "discrimination" in text:
        family_rank = 3
    elif roles & _VALIDATION_PANEL_ROLES:
        family_rank = 4
    elif "overlap" in text or "eligibility" in text or "definition" in text:
        family_rank = 5
    elif roles & _CONTEXT_PANEL_ROLES:
        family_rank = 6
    elif sensitivity_or_robustness or supplemental_only:
        family_rank = 7
    elif "primary" in text or "association" in text:
        family_rank = 8
    else:
        family_rank = 9
    return (
        family_rank,
        1 if supplemental_only else 0,
        generic_penalty,
        -int(bundle.get("order", 0)),
        str(bundle.get("stem") or ""),
    )


def _read_contract_payload(
    evidence: EvidenceStore,
    record: EvidenceRecord,
) -> Dict[str, Any]:
    try:
        payload = json.loads(
            (evidence.root / record.relative_path).read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _bundle_contract_roles(bundle: Dict[str, Any]) -> set[str]:
    payload = bundle.get("contract_payload")
    if not isinstance(payload, dict):
        return set()
    return _contract_payload_roles(payload)


def _contract_payload_roles(payload: Dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for panel in payload.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        role = str(panel.get("role") or "").strip().lower()
        if role:
            roles.add(role)
    for key in ("hero_role", "primary_role", "figure_role"):
        role = str(payload.get(key) or "").strip().lower()
        if role and role != "publication_figure":
            roles.add(role)
    return roles


def _bundle_contract_chart_types(bundle: Dict[str, Any]) -> set[str]:
    payload = bundle.get("contract_payload")
    if not isinstance(payload, dict):
        return set()
    return _contract_payload_chart_types(payload)


def _contract_payload_chart_types(payload: Dict[str, Any]) -> set[str]:
    chart_types: set[str] = set()
    for panel in payload.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        candidates = [panel.get("chart_type")]
        metadata = panel.get("metadata")
        if isinstance(metadata, dict):
            candidates.append(metadata.get("chart_type"))
        for value in candidates:
            chart_type = str(value or "").strip().lower()
            if chart_type:
                chart_types.add(chart_type)
    return chart_types


def _promoted_contract_preserves_preferred_chart_types(
    contract_path: Path,
    preferred_step_bundle: Optional[Dict[str, Any]],
) -> bool:
    if preferred_step_bundle is None:
        return True
    expected = _bundle_contract_chart_types(preferred_step_bundle)
    if not expected:
        return True
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    observed = _contract_payload_chart_types(payload)
    return expected <= observed


def _record_artifact_basename(record: EvidenceRecord) -> str:
    return Path(record.relative_path).name.split("__", 1)[-1]


def _record_artifact_stem(record: EvidenceRecord) -> str:
    basename = _record_artifact_basename(record)
    return Path(basename).with_suffix("").name


def _is_run_level_publication_figure(record: EvidenceRecord) -> bool:
    basename = _record_artifact_basename(record)
    return record.producer == PublicationFigureSkill.name and basename.startswith(
        "easyicu_publication_figure."
    )


def _unique_evidence_ids(records: Sequence[EvidenceRecord]) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for record in records:
        if record.evidence_id in seen:
            continue
        seen.add(record.evidence_id)
        ids.append(record.evidence_id)
    return ids


def _source_fingerprint_metadata(
    evidence: EvidenceStore,
    source_ids: Sequence[str],
) -> Dict[str, Any]:
    ids = list(dict.fromkeys(str(eid) for eid in source_ids if str(eid)))
    fingerprints: Dict[str, str] = {}
    for evidence_id in ids:
        record = evidence.get(evidence_id)
        if record is not None:
            fingerprints[evidence_id] = record.sha256
    metadata: Dict[str, Any] = {
        "figure_skill_policy_version": PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
        "source_evidence_ids": ids,
        "source_evidence_sha256": fingerprints,
    }
    if len(ids) == 1:
        metadata["source_evidence_id"] = ids[0]
    return metadata


def _figure_contract_source_ids(contract: Any) -> List[str]:
    """Return every EvidenceStore id named by a figure contract."""

    ids = [str(item) for item in contract.source_data if str(item)]
    for panel in contract.panels:
        ids.extend(str(item) for item in panel.evidence_ids if str(item))
    return list(dict.fromkeys(ids))


def _register_publication_figure_bundle(
    *,
    evidence: EvidenceStore,
    paths: Dict[str, Path],
    contract: Any,
    prompt_pack_version: Optional[str],
    contract_metadata: Optional[Dict[str, Any]] = None,
    figure_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[EvidenceRecord, List[EvidenceRecord]]:
    """Register a contract and every export with strict provenance links."""

    source_ids = _figure_contract_source_ids(contract)
    source_metadata = _source_fingerprint_metadata(evidence, source_ids)
    script_record = evidence.register_file(
        kind="code",
        description="Deterministic PublicationFigureSkill renderer source.",
        source_path=Path(__file__).resolve(),
        evidence_id="publication_figure_skill_renderer",
        aliases=["publication_figure_renderer_code"],
        producer=PublicationFigureSkill.name,
        generation_mode="deterministic_figure_skill",
        prompt_pack_version=prompt_pack_version,
        metadata={
            "artifact_role": "figure_renderer",
            "figure_id": contract.figure_id,
        },
        on_sha_change="new_id",
    )

    contract_path = paths.get("contract")
    if contract_path is None:
        raise ValueError("publication figure bundle has no contract export")
    contract_record = evidence.register_file(
        kind="log",
        description="Publication figure contract generated from analysis evidence.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        aliases=["publication_figure_contract", "figure_contract"],
        inputs=source_ids,
        script_evidence_id=script_record.evidence_id,
        producer=PublicationFigureSkill.name,
        generation_mode="deterministic_figure_skill",
        prompt_pack_version=prompt_pack_version,
        metadata={
            **source_metadata,
            **dict(contract_metadata or {}),
            "artifact_role": "figure_contract",
            "figure_id": contract.figure_id,
            "source_evidence_ids": source_ids,
        },
        on_sha_change="new_id",
    )

    figure_records: List[EvidenceRecord] = []
    for key, path in paths.items():
        suffix = path.suffix.lower()
        if key == "contract" or suffix.endswith(".json"):
            continue
        record = evidence.register_file(
            kind="figure",
            description=(
                f"Publication figure export ({suffix.lstrip('.')}) generated "
                "from analysis evidence."
            ),
            source_path=path,
            evidence_id=f"publication_figure_{suffix.lstrip('.')}",
            aliases=[
                "publication_figure",
                f"publication_figure_{suffix.lstrip('.')}",
            ],
            inputs=[*source_ids, contract_record.evidence_id],
            script_evidence_id=script_record.evidence_id,
            producer=PublicationFigureSkill.name,
            generation_mode="deterministic_figure_skill",
            prompt_pack_version=prompt_pack_version,
            metadata={
                **source_metadata,
                **dict(figure_metadata or {}),
                "artifact_role": "manuscript_figure",
                "figure_id": contract.figure_id,
                "contract_evidence_id": contract_record.evidence_id,
                "source_evidence_ids": source_ids,
                "figure_contract": contract_record.evidence_id,
                "figure_role": "publication_figure",
            },
            on_sha_change="new_id",
        )
        figure_records.append(record)
    return contract_record, figure_records


def _bundle_source_ids(bundle: Dict[str, Any]) -> List[str]:
    return _unique_evidence_ids(
        [
            *dict(bundle.get("figures") or {}).values(),
            bundle["contract"],
            *list(bundle.get("source_records") or []),
        ]
    )


def _source_fingerprints_match(
    evidence: EvidenceStore,
    metadata: Dict[str, Any],
) -> bool:
    source_ids = metadata.get("source_evidence_ids")
    if isinstance(source_ids, str):
        ids = [source_ids]
    elif isinstance(source_ids, (list, tuple, set)):
        ids = [str(eid) for eid in source_ids if str(eid)]
    else:
        ids = []
    single = metadata.get("source_evidence_id")
    if single and str(single) not in ids:
        ids.append(str(single))
    fingerprints = metadata.get("source_evidence_sha256")
    if not ids or not isinstance(fingerprints, dict) or not fingerprints:
        return False
    for evidence_id in ids:
        record = evidence.get(evidence_id)
        if record is None or fingerprints.get(evidence_id) != record.sha256:
            return False
    return True


def _figure_skill_policy_matches(metadata: Dict[str, Any]) -> bool:
    return (
        metadata.get("figure_skill_policy_version")
        == PUBLICATION_FIGURE_SKILL_POLICY_VERSION
    )


def _latest_record_for_basename(
    evidence: EvidenceStore,
    basename: str,
    *,
    kind: Optional[str] = None,
) -> Optional[EvidenceRecord]:
    matches = [
        record
        for record in evidence.records()
        if _record_artifact_basename(record) == basename
        and (kind is None or record.kind == kind)
    ]
    return matches[-1] if matches else None


def _contract_promoted_from_source(
    contract_path: Path,
    *,
    source_ids: Sequence[str],
):
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    source_panels = payload.get("panels")
    panels: List[Dict[str, Any]] = []
    if isinstance(source_panels, list):
        for idx, panel in enumerate(source_panels, start=1):
            if not isinstance(panel, dict):
                continue
            panel_id = str(panel.get("panel_id") or panel.get("id") or idx)
            metadata = dict(panel.get("metadata") or {})
            for key in ("chart_type", "scale", "visual_role"):
                if panel.get(key) is not None and key not in metadata:
                    metadata[key] = panel.get(key)
            promoted_panel = {
                "panel_id": panel_id,
                "title": str(panel.get("title") or f"Panel {panel_id}"),
                "role": panel.get("role") or "validation",
                "claim": str(
                    panel.get("claim")
                    or panel.get("purpose")
                    or "This panel is promoted from registered figure evidence."
                ),
                "evidence_ids": list(source_ids),
                "review_risk": panel.get("review_risk"),
            }
            if metadata:
                promoted_panel["metadata"] = metadata
            panels.append(promoted_panel)
    if not panels:
        panels = [
            {
                "panel_id": "A",
                "title": "Registered manuscript figure",
                "role": "validation",
                "claim": (
                    "The run-level figure is promoted from registered step-level "
                    "figure evidence with source data."
                ),
                "evidence_ids": list(source_ids),
                "review_risk": (
                    "Interpretation depends on the upstream figure contract and "
                    "source-data table."
                ),
            }
        ]
    try:
        return make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=str(
                payload.get("core_claim")
                or "The manuscript-facing figure is promoted from registered step-level evidence."
            ),
            panels=panels,
            source_data=list(source_ids),
            statistics_note=(
                "This run-level bundle promotes a registered step-level "
                "publication figure and preserves its source evidence."
            ),
        )
    except Exception:
        return make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=(
                "The manuscript-facing figure is promoted from registered "
                "step-level evidence."
            ),
            panels=[
                {
                    "panel_id": "A",
                    "title": "Registered manuscript figure",
                    "role": "validation",
                    "claim": "The figure is copied from a registered step-level figure bundle.",
                    "evidence_ids": list(source_ids),
                }
            ],
            source_data=list(source_ids),
            statistics_note=(
                "This run-level bundle promotes a registered step-level "
                "publication figure and preserves its source evidence."
            ),
        )


def _has_curated_publication_figure_bundle(
    evidence: EvidenceStore,
    *,
    run_dir: Path,
    context: Optional[ResearchContext] = None,
) -> bool:
    preferred_step_bundle = _select_existing_step_publication_figure_bundle(evidence)
    preferred_source_ids = (
        set(_bundle_source_ids(preferred_step_bundle))
        if preferred_step_bundle is not None
        else set()
    )
    fresh_bundle = False
    for record in evidence.records():
        metadata = record.metadata or {}
        if (
            record.kind == "figure"
            and _is_run_level_publication_figure(record)
            and _source_fingerprints_match(evidence, metadata)
            and _figure_skill_policy_matches(metadata)
        ):
            if preferred_step_bundle is not None and (
                metadata.get("promoted_from_step_id")
                != preferred_step_bundle.get("step_id")
                or metadata.get("promoted_from_stem")
                != preferred_step_bundle.get("stem")
            ):
                continue
            if preferred_source_ids and not preferred_source_ids <= set(
                metadata.get("source_evidence_ids") or []
            ):
                continue
            fresh_bundle = True
            break
    if not fresh_bundle:
        return False

    contract_candidates: List[Path] = []
    contract_record = evidence.get("publication_figure_contract")
    if contract_record is not None:
        contract_candidates.append(run_dir / contract_record.relative_path)
    contract_candidates.append(
        run_dir
        / "publication_figures"
        / "easyicu_publication_figure.figure_contract.json"
    )
    for contract_path in contract_candidates:
        if not contract_path.exists():
            continue
        findings = FigureContractQualityValidator().audit_contract_file(
            contract_path,
            manuscript_facing=True,
        )
        if any(finding.severity == "error" for finding in findings):
            continue
        if context is not None:
            try:
                payload = json.loads(contract_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict) and not _contract_primary_strategy_ready(
                context,
                payload,
            ):
                continue
        if not _promoted_contract_preserves_preferred_chart_types(
            contract_path,
            preferred_step_bundle,
        ):
            continue
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


def _first_normalisable_record(
    evidence: EvidenceStore,
    names: Sequence[str],
    *,
    run_dir: Path,
    normalise: Callable[[pd.DataFrame], pd.DataFrame],
) -> Optional[EvidenceRecord]:
    """First matching table record whose normalized frame is non-empty.

    ``_first_existing_record`` returns the first *name* match, but an
    alias can resolve to a semantically different table whose columns the
    panel normalizer rejects (e.g. ``missingness_summary`` binding to a
    numeric-coercion audit with ``original_missing_*`` columns). Stopping
    there silently drops the side panel even when a perfectly renderable
    sibling table is registered, so this variant keeps scanning until a
    candidate actually normalizes.
    """

    name_set = {str(name).lower() for name in names}
    candidates: List[EvidenceRecord] = []
    seen: set[str] = set()
    for name in names:
        record = evidence.get(name)
        if (
            record is not None
            and record.kind == "table"
            and record.evidence_id not in seen
        ):
            candidates.append(record)
            seen.add(record.evidence_id)
    for record in evidence.records():
        if record.kind != "table" or record.evidence_id in seen:
            continue
        basename = Path(record.relative_path).stem.lower()
        if any(token in basename for token in name_set):
            candidates.append(record)
            seen.add(record.evidence_id)
    for record in candidates:
        try:
            frame = normalise(_read_table(run_dir / record.relative_path))
        except Exception:
            continue
        if not frame.empty:
            return record
    return None


def _select_primary_association_record(
    evidence: EvidenceStore,
    *,
    run_dir: Path,
    context: ResearchContext,
    names: Sequence[str],
) -> Optional[EvidenceRecord]:
    name_set = {str(name).lower() for name in names}
    candidates: List[tuple[float, int, EvidenceRecord]] = []
    seen: set[str] = set()

    def consider(record: Optional[EvidenceRecord], order: int) -> None:
        if record is None or record.kind != "table" or record.evidence_id in seen:
            return
        seen.add(record.evidence_id)
        basename = Path(record.relative_path).stem.lower()
        if (
            not any(token in basename for token in name_set)
            and record.evidence_id.lower() not in name_set
        ):
            return
        score = float(order) * 0.001
        severity = str(getattr(record, "finding_severity", "") or "").lower()
        if severity == "error":
            score -= 100.0
        elif severity == "warning":
            score -= 2.0
        if "full_coefficients" in basename or "all_coefficients" in basename:
            score -= 25.0
        try:
            frame = _read_table(run_dir / record.relative_path)
            cols = {str(c).lower(): c for c in frame.columns}
            if "point_estimate" in cols:
                score += 18.0
            if "exposure" in cols or "primary_exposure" in cols:
                score += 8.0
            normalised = _normalise_association_frame(
                frame,
                primary_exposure=context.primary_exposure,
            )
            if not normalised.empty:
                score += 20.0
                if len(normalised) == 1:
                    score += 10.0
                if str(context.primary_exposure or "").strip():
                    labels = " ".join(normalised["label"].astype(str).tolist())
                    if _match_token(context.primary_exposure) in _match_token(labels):
                        score += 6.0
        except Exception:
            score -= 50.0
        candidates.append((score, order, record))

    for order, name in enumerate(names):
        consider(evidence.get(name), order)
    base_order = len(names)
    for offset, record in enumerate(evidence.records()):
        consider(record, base_order + offset)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best = candidates[0]
    if best_score < -20.0:
        return None
    return best


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


def _primary_match_rank(label: Any, primary_token: str) -> Tuple[int, int]:
    """Rank a forest row by how well its label matches the primary exposure.

    Lower is better. Exact token match wins; then a substring match preferring the
    smallest length gap (so the main-effect term beats an interaction / derived
    term like ``exposure_x_age``); then no match. Used to put the true primary
    exposure at row 0 instead of whatever coefficient happened to be first in a
    long per-coefficient table (false-pass audit #15/#16).
    """
    lab = _match_token(label)
    if not lab or not primary_token:
        return (3, 0)
    if lab == primary_token:
        return (0, 0)
    if primary_token in lab or lab in primary_token:
        return (1, abs(len(lab) - len(primary_token)))
    return (2, 0)


def _order_primary_first(result: pd.DataFrame, primary_token: str) -> pd.DataFrame:
    """Stable-sort so the best primary-exposure match is row 0.

    Rows that do not match the exposure keep their original relative order, so a
    table with no exposure match is returned unchanged (row 0 as before).
    """
    if result.empty or not primary_token:
        return result
    ranks = [_primary_match_rank(lab, primary_token) for lab in result["label"]]
    order = sorted(range(len(result)), key=lambda i: ranks[i])
    if order == list(range(len(result))):
        return result
    return result.iloc[order].reset_index(drop=True)


def _normalise_association_frame(
    frame: pd.DataFrame,
    *,
    primary_exposure: Optional[str] = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["label", "estimate", "lower", "upper"])
    cols = {str(c).lower(): c for c in frame.columns}
    primary_token = _match_token(primary_exposure)
    if primary_token:
        match_cols = [
            col
            for name, col in cols.items()
            if name
            in {
                "exposure",
                "primary_exposure",
                "predictor",
                "variable",
                "term",
                "feature",
                "reader_label",
                "label",
                "contrast",
            }
        ]
        primary_mask = pd.Series(False, index=frame.index)
        for col in match_cols:
            primary_mask = primary_mask | frame[col].map(
                lambda value: primary_token in _match_token(value)
            )
        if primary_mask.any():
            frame = frame.loc[primary_mask].copy()
            cols = {str(c).lower(): c for c in frame.columns}
    label_col = _first_col(
        cols,
        [
            "exposure",
            "primary_exposure",
            "reader_label",
            "label",
            "model_label",
            "model",
            "specification",
            "contrast",
            "variable",
            "predictor",
            "term",
            "feature",
        ],
    )
    estimate_col = _first_col(
        cols,
        [
            "point_estimate",
            "odds_ratio",
            "adjusted_or",
            "or",
            "hazard_ratio",
            "adjusted_hr",
            "hr",
            "risk_ratio",
            "relative_risk",
            "rr",
            "average_treatment_effect",
            "ate",
            "treatment_effect",
            "risk_difference",
            "mean_difference",
            "coefficient",
            "coef",
            "beta",
            "estimate",
            "effect",
        ],
    )
    lower_col = _first_col(
        cols,
        [
            "or_lower",
            "hr_lower",
            "rr_lower",
            "ate_lower",
            "effect_lower",
            "coefficient_lower",
            "coef_lower",
            "beta_lower",
            "ci_low",
            "ci_lower",
            "lower_ci",
            "lower",
            "estimate_lower",
            "estimate_ci_low",
        ],
    )
    upper_col = _first_col(
        cols,
        [
            "or_upper",
            "hr_upper",
            "rr_upper",
            "ate_upper",
            "effect_upper",
            "coefficient_upper",
            "coef_upper",
            "beta_upper",
            "ci_high",
            "ci_upper",
            "upper_ci",
            "upper",
            "estimate_upper",
            "estimate_ci_high",
        ],
    )
    if estimate_col is None:
        numeric_cols = [
            c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
        ]
        estimate_col = numeric_cols[0] if numeric_cols else None
    if label_col is None:
        label_col = estimate_col
    if estimate_col is None or label_col is None:
        return pd.DataFrame(columns=["label", "estimate", "lower", "upper"])

    out = pd.DataFrame(
        {
            "label": frame[label_col].astype(str).map(_prettify_label),
            "estimate": pd.to_numeric(frame[estimate_col], errors="coerce"),
        }
    )
    if lower_col is not None:
        out["lower"] = pd.to_numeric(frame[lower_col], errors="coerce")
    else:
        out["lower"] = out["estimate"]
    if upper_col is not None:
        out["upper"] = pd.to_numeric(frame[upper_col], errors="coerce")
    else:
        out["upper"] = out["estimate"]
    raw_label = frame[label_col].astype(str).str.strip().str.lower()
    out = out.loc[~raw_label.isin({"intercept", "const", "constant"})].copy()
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["estimate"])
    out["lower"] = out["lower"].fillna(out["estimate"])
    out["upper"] = out["upper"].fillna(out["estimate"])
    result = out[["label", "estimate", "lower", "upper"]].reset_index(drop=True)
    # Put the true primary-exposure coefficient at row 0 so the downstream forest
    # (which marks row 0 as the blue "primary" estimand named in the core claim)
    # highlights the exposure, not whatever coefficient a long per-coefficient
    # table happened to list first (false-pass audit #15/#16).
    result = _order_primary_first(result, primary_token)
    result.attrs.update(
        _association_axis_from_token(_effect_measure_token(frame, cols, estimate_col))
    )
    return result


def _effect_measure_token(
    frame: pd.DataFrame,
    cols: Dict[str, str],
    estimate_col: Any,
) -> str:
    for meta_name in (
        "effect_type",
        "effect_scale",
        "estimate_type",
        "measure",
        "metric",
        "scale",
    ):
        meta_col = cols.get(meta_name)
        if meta_col is None:
            continue
        values = frame[meta_col].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        if not values.empty:
            return values.iloc[0]
    return str(estimate_col or "")


def _match_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _association_axis_from_token(token: str) -> Dict[str, Any]:
    normalized = _match_token(token)
    if normalized in {"or", "oddsratio", "adjustedor"} or "oddsratio" in normalized:
        return {
            "xlabel": "Odds ratio",
            "header": "OR (95% CI)",
            "null_value": 1.0,
            "ratio_scale": True,
        }
    if (
        normalized in {"hr", "hazardratio", "adjustedhr"}
        or "hazardratio" in normalized
        or "hazard" in normalized
    ):
        return {
            "xlabel": "Hazard ratio",
            "header": "HR (95% CI)",
            "null_value": 1.0,
            "ratio_scale": True,
        }
    if normalized in {"rr", "riskratio", "relativerisk"} or "riskratio" in normalized:
        return {
            "xlabel": "Risk ratio",
            "header": "RR (95% CI)",
            "null_value": 1.0,
            "ratio_scale": True,
        }
    if (
        normalized in {"ate", "averagetreatmenteffect"}
        or "treatmenteffect" in normalized
    ):
        return {
            "xlabel": "Average treatment effect",
            "header": "ATE (95% CI)",
            "null_value": 0.0,
            "ratio_scale": False,
        }
    if "riskdifference" in normalized:
        return {
            "xlabel": "Risk difference",
            "header": "Risk difference (95% CI)",
            "null_value": 0.0,
            "ratio_scale": False,
        }
    if "meandifference" in normalized:
        return {
            "xlabel": "Mean difference",
            "header": "Mean difference (95% CI)",
            "null_value": 0.0,
            "ratio_scale": False,
        }
    if normalized in {"coef", "coefficient", "beta"} or any(
        marker in normalized for marker in ("coefficient", "coef", "beta")
    ):
        return {
            "xlabel": "Coefficient",
            "header": "Coefficient (95% CI)",
            "null_value": 0.0,
            "ratio_scale": False,
        }
    return {
        "xlabel": "Effect estimate",
        "header": "Estimate (95% CI)",
        "null_value": None,
        "ratio_scale": False,
    }


def _association_axis_metadata(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.attrs:
        return {
            **_association_axis_from_token(""),
            **frame.attrs,
        }
    return _association_axis_from_token("")


def _normalise_strata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["score", "rate"])
    cols = {str(c).lower(): c for c in frame.columns}
    score_col = _first_col(
        cols,
        [
            "score",
            "stratum",
            "severity_score",
            "risk_score",
            "sofa2",
            "sofa_2",
            "gcs",
            "gcs_score",
            "kdigo",
            "kdigo_stage",
            "exposure",
            "exposure_label",
            "exposure_group",
            "group",
            "group_label",
            "status",
            "category",
            "level",
            "sepsis3",
            "sepsis_3",
            "exposure_status",
        ],
    )
    rate_col = _first_col(
        cols,
        [
            "death_rate",
            "mortality_rate",
            "outcome_rate",
            "outcome_risk",
            "event_rate",
            "death_pct",
            "mortality_pct",
            "event_pct",
            "outcome_pct",
            "risk",
            "rate",
            "death_risk",
            "mortality_risk",
        ],
    )
    if score_col is None:
        # Exposure/severity strata are often named after the predictor
        # (``lactate_group``, ``sofa2_stratum``, ``lactate_quartile``),
        # so an exact-name list can never enumerate them. Fall back to the
        # first column whose name ends in a grouping suffix, preferring a
        # sibling ``*_order`` column's source when present. General on
        # purpose — do NOT add case-specific names like ``lactate_group``.
        _GROUP_SUFFIXES = (
            "_group",
            "_stratum",
            "_strata",
            "_bin",
            "_band",
            "_category",
            "_class",
            "_quartile",
            "_quintile",
            "_decile",
            "_tertile",
            "_level",
        )
        for name, col in cols.items():
            if name.endswith(_GROUP_SUFFIXES) and not name.endswith("_order"):
                score_col = col
                break
    n_col = _first_col(cols, ["n", "count", "n_total"])
    if score_col is None or rate_col is None:
        return pd.DataFrame(columns=["score", "rate"])
    raw_score = frame[score_col]
    numeric_score = pd.to_numeric(raw_score, errors="coerce")
    semantic_category = _score_column_is_semantic_category(score_col)
    score_is_numeric = bool(numeric_score.notna().all()) and not semantic_category
    score_values = (
        numeric_score
        if score_is_numeric
        else raw_score.map(lambda value: _score_category_label(score_col, value))
    )
    score_order = (
        numeric_score
        if numeric_score.notna().any()
        else pd.Series(range(len(frame)), index=frame.index)
    )
    out = pd.DataFrame(
        {
            "score": score_values,
            "rate": pd.to_numeric(frame[rate_col], errors="coerce"),
            "_score_order": score_order,
        }
    ).dropna(subset=["score", "rate"])
    if n_col is not None:
        out["n"] = pd.to_numeric(frame.loc[out.index, n_col], errors="coerce")
    if not out.empty and out["rate"].max() > 1.0:
        out["rate"] = out["rate"] / 100.0
    result = (
        out.sort_values("score").drop(columns=["_score_order"]).reset_index(drop=True)
        if score_is_numeric
        else out.sort_values("_score_order")
        .drop(columns=["_score_order"])
        .reset_index(drop=True)
    )
    result.attrs["score_label"] = _score_axis_label(score_col)
    result.attrs["score_is_numeric"] = score_is_numeric
    return result


def _score_column_is_semantic_category(column: Any) -> bool:
    normalized = str(column or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        "exposure",
        "exposure_label",
        "group",
        "group_label",
        "status",
        "category",
        "level",
        "sepsis3",
        "sepsis_3",
        "exposure_status",
    }


def _score_category_label(column: Any, value: Any) -> str:
    normalized_col = (
        str(column or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    state = _binary_state_label(value)
    if normalized_col in {"sepsis3", "sepsis_3"} and state is not None:
        return f"Sepsis-3 {state}"
    if normalized_col in {"exposure", "exposure_status"} and state is not None:
        return "Exposed" if state == "positive" else "Unexposed"
    if normalized_col == "status" and state is not None:
        return state.capitalize()
    if normalized_col in {"group", "group_label"}:
        return f"Group {_prettify_label(value)}"
    return _prettify_label(value)


def _binary_state_label(value: Any) -> Optional[str]:
    token = str(value).strip().lower()
    if token in {"1", "1.0", "true", "yes", "y", "positive", "present"}:
        return "positive"
    if token in {"0", "0.0", "false", "no", "n", "negative", "absent"}:
        return "negative"
    return None


def _normalise_missingness_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["variable", "missing_fraction"])
    cols = {str(c).lower(): c for c in frame.columns}
    variable_col = _first_col(cols, ["variable", "feature", "column", "name"])
    # Accept both word orders for each metric: coder-generated audits write
    # `frac_missing`/`n_missing` about as often as `missing_fraction`/
    # `missing_n`, and rejecting one spelling silently drops the data-quality
    # panel from the publication bundle.
    frac_col = _first_col(
        cols, ["missing_fraction", "missing_rate", "fraction", "frac_missing"]
    )
    pct_col = _first_col(
        cols, ["missing_percentage", "missing_percent", "percent", "pct_missing"]
    )
    count_col = _first_col(cols, ["missing_count", "missing_n", "n_missing"])
    n_col = _first_col(cols, ["n", "n_total", "total"])
    if variable_col is None:
        return pd.DataFrame(columns=["variable", "missing_fraction"])
    raw_variable = frame[variable_col].astype(str)
    out = pd.DataFrame(
        {
            "source_variable": raw_variable.map(_prettify_label),
            "measurement_family": raw_variable.map(_missingness_family_label),
        }
    )
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
    if out.empty:
        return pd.DataFrame(columns=["variable", "missing_fraction"])
    grouped = (
        out.groupby("measurement_family", dropna=False)
        .agg(
            missing_fraction=("missing_fraction", "max"),
            feature_count=("source_variable", "nunique"),
            source_variable=("source_variable", "first"),
        )
        .reset_index()
    )
    grouped["variable"] = grouped.apply(
        lambda row: (
            row["measurement_family"]
            if int(row["feature_count"]) > 1
            else row["source_variable"]
        ),
        axis=1,
    )
    if (grouped["missing_fraction"] > 0).any():
        grouped = grouped.loc[grouped["missing_fraction"] > 0].copy()
    return (
        grouped[["variable", "missing_fraction", "feature_count"]]
        .sort_values("missing_fraction", ascending=False)
        .head(8)
        .reset_index(drop=True)
    )


def _missingness_family_label(value: Any) -> str:
    raw = str(value or "").strip()
    tokens = [token for token in re.split(r"[^a-z0-9]+", raw.lower()) if token]
    suffixes = {
        "first",
        "last",
        "min",
        "max",
        "mean",
        "median",
        "avg",
        "average",
        "sd",
        "std",
        "value",
        "val",
    }
    while len(tokens) > 1 and tokens[-1] in suffixes:
        tokens.pop()
    if not tokens:
        return _prettify_label(raw)
    base = "_".join(tokens)
    family_labels = {
        "lact": "Lactate",
        "lactate": "Lactate",
        "temp": "Temperature",
        "temperature": "Temperature",
        "bun": "BUN",
        "creat": "Creatinine",
        "creatinine": "Creatinine",
        "wbc": "WBC",
        "hr": "Heart rate",
        "heart_rate": "Heart rate",
        "rr": "Resp. rate",
        "resp": "Resp. rate",
        "resp_rate": "Resp. rate",
        "spo2": "SpO2",
    }
    if base in family_labels:
        return family_labels[base]
    return _prettify_label(base)


def _draw_strata_panel(
    ax: Any, frame: pd.DataFrame, *, palette: Dict[str, str], outcome: str
) -> None:
    import matplotlib.ticker as mticker
    import numpy as np

    y_values = frame["rate"].astype(float)
    score_label = _strata_score_label(frame)
    if bool(frame.attrs.get("score_is_numeric", True)):
        x = frame["score"].astype(float)
        ax.plot(
            x,
            y_values,
            color=palette.get("blue", "#0F4D92"),
            linewidth=1.4,
            marker="o",
            markersize=3.4,
        )
        ax.set_xlabel(score_label)
        ax.set_ylabel(f"{_prettify_label(outcome)} rate")
        ymax = max(
            0.05, min(1.0, float(y_values.max()) * 1.25 if len(y_values) else 0.05)
        )
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(
            axis="y",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.5,
            alpha=0.7,
        )
    else:
        y = np.arange(len(frame))
        ax.hlines(
            y,
            xmin=0,
            xmax=y_values,
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=1.2,
        )
        ax.plot(
            y_values,
            y,
            "o",
            color=palette.get("blue", "#0F4D92"),
            markersize=3.8,
        )
        ax.set_yticks(y, frame["score"].astype(str).tolist())
        ax.invert_yaxis()
        ax.set_xlabel(f"{_prettify_label(outcome)} rate")
        xmax = max(
            0.05, min(1.0, float(y_values.max()) * 1.35 if len(y_values) else 0.05)
        )
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.5,
            alpha=0.7,
        )
    ax.set_title(f"Observed outcome by {score_label}", loc="left", pad=3)


def _strata_score_label(frame: pd.DataFrame) -> str:
    label = frame.attrs.get("score_label")
    if label:
        return str(label)
    return "Score"


def _score_axis_label(column: Any) -> str:
    raw = str(column or "").strip()
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "score": "Score",
        "stratum": "Stratum",
        "severity_score": "Severity score",
        "risk_score": "Risk score",
        "sofa2": "SOFA-2 score",
        "sofa_2": "SOFA-2 score",
        "gcs": "GCS score",
        "gcs_score": "GCS score",
        "kdigo": "KDIGO stage",
        "kdigo_stage": "KDIGO stage",
        "exposure": "Exposure group",
        "exposure_label": "Exposure group",
        "group": "Group",
        "group_label": "Group",
        "status": "Status",
        "category": "Category",
        "level": "Level",
        "sepsis3": "Sepsis-3 status",
        "sepsis_3": "Sepsis-3 status",
        "exposure_status": "Exposure status",
    }
    if normalized in mapping:
        return mapping[normalized]
    pretty = _prettify_label(raw)
    if any(
        word in pretty.lower()
        for word in ("score", "stratum", "stage", "group", "status", "category")
    ):
        return pretty
    return f"{pretty} score"


def _draw_missingness_panel(
    ax: Any, frame: pd.DataFrame, *, palette: Dict[str, str]
) -> None:
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
    labels: List[str] = []
    for _, row in plot.iterrows():
        label = str(row["variable"])
        try:
            feature_count = int(row.get("feature_count", 1))
        except (TypeError, ValueError):
            feature_count = 1
        if feature_count > 1:
            label = f"{label} ({feature_count})"
        labels.append(label)
    ax.set_yticks(y, labels)
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
    ax.grid(
        axis="x",
        color=palette.get("neutral_light", "#D8D8D8"),
        linewidth=0.5,
        alpha=0.7,
    )


def _duplicates_primary_row(row: Any, primary: Any) -> bool:
    if primary is None:
        return False
    spec_id = str(getattr(row, "spec_id", "") or "").lower()
    if "primary" not in spec_id:
        return False
    return (
        _same_float(
            getattr(row, "point_estimate", None),
            getattr(primary, "point_estimate", None),
        )
        and _same_float(getattr(row, "ci_low", None), getattr(primary, "ci_low", None))
        and _same_float(
            getattr(row, "ci_high", None), getattr(primary, "ci_high", None)
        )
    )


def _same_float(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _robustness_spec_label(spec_id: Any) -> str:
    raw = str(spec_id or "").strip()
    if not raw:
        return "Variant"
    lowered = raw.lower()
    explicit = {
        "primary": "Primary",
        "alt_cohort_no_los_restriction": "No ICU LOS restriction",
        "cohort_no_los_restriction": "No ICU LOS restriction",
        "alt_cohort_stricter_los_2d": "ICU LOS >=2 d",
        "cohort_los_ge_2d": "ICU LOS >=2 d",
        "alt_cohort_core_physiology_present": "Core physiology present",
        "cohort_core_physiology_present": "Core physiology present",
        "alt_missing_complete_case": "Complete-case",
        "missing_raw_complete_case": "Complete-case",
        "missing_drop_lactate": "Drop lactate",
        "alt_outcome_rr_scale": "Risk-ratio scale",
        "effect_robust_poisson_rr": "Risk-ratio scale",
        "alt_outcome_rd_scale": "Risk-difference scale",
        "effect_marginal_standardized_rd": "Risk-difference scale",
        "primary_los_ge_1d": "Primary ICU LOS >=1 d",
    }
    if lowered in explicit:
        return explicit[lowered]
    replacements = {
        "adult": "Adult",
        "any": "any",
        "los": "ICU LOS",
        "ge": ">=",
        "1d": "1 day",
        "2": "2",
        "cc": "complete case",
        "frozen": "locked",
        "lact": "lactate",
        "lactate": "lactate",
        "measured": "measured",
        "missing": "missing",
        "indicator": "indicator",
        "offprotocol": "off-protocol",
        "primary": "primary",
        "cohort": "cohort",
    }
    tokens = [token for token in lowered.replace("-", "_").split("_") if token]
    if not tokens:
        return _prettify_label(raw)
    words = [replacements.get(token, token) for token in tokens]
    label = " ".join(words)
    label = label.replace("ICU LOS >= 1 day", "ICU LOS >=1 d")
    label = label.replace("ICU LOS >= 2", "ICU LOS >=2 d")
    return label[0].upper() + label[1:] if label else _prettify_label(raw)


def _robustness_axis_label(axis: Any) -> str:
    mapping = {
        "primary": "Primary",
        "cohort": "Cohort",
        "missing": "Missing data",
        "outcome": "Outcome",
        "unspecified": "Unspecified",
    }
    return mapping.get(str(axis or "").strip().lower(), _prettify_label(axis))


def _prettify_label(value: Any) -> str:
    token = str(value or "").strip()
    mapping = {
        "sofa2": "SOFA-2",
        "sepsis3": "Sepsis-3",
        "sepsis_3": "Sepsis-3",
        "sex_m": "Male sex",
        "death": "ICU mortality",
        "icu_mortality": "ICU mortality",
        "lact": "Lactate",
        "creat": "Creatinine",
        "resp": "Respiratory rate",
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
