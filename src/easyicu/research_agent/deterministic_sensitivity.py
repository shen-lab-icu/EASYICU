"""Deterministic code templates for standard sensitivity-analysis steps."""

from __future__ import annotations

import textwrap


def cohort_definition_overlap_code() -> str:
    """Return a runner script for standard eligibility-overlap analyses."""

    return textwrap.dedent(
        r'''
        import json
        import os
        from pathlib import Path

        import numpy as np
        import pandas as pd

        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        current_step_id = out_dir.parent.name
        cohort_path = Path(os.environ["COHORT_PARQUET"])
        df = pd.read_parquet(cohort_path).copy()

        # --- config-first: the study's declared exposure drives one eligibility
        # axis (exposure derivability). Read it from research_context.json; the
        # Sepsis-3 flag names below are only last-resort aliases so the skill is
        # not bound to the sepsis benchmark case.
        primary_exposure = ""
        target_outcome = ""
        try:
            _run_dir = out_dir.parents[2]
            _ctx = json.loads((_run_dir / "research_context.json").read_text("utf-8"))
            primary_exposure = str(_ctx.get("primary_exposure") or "").strip()
            target_outcome = str(_ctx.get("target_outcome") or "").strip()
        except Exception:
            pass


        def _numeric(col):
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index, dtype="float64")
            return pd.to_numeric(df[col], errors="coerce")


        def _measured(col):
            if col in df.columns:
                return _numeric(col).fillna(0).eq(1)
            raw = col.removesuffix("_measured")
            if raw in df.columns:
                return df[raw].notna()
            return pd.Series(False, index=df.index)


        def _pct(num, den):
            return float(num / den * 100.0) if den else np.nan


        required_for_overlap = ["stay_id", "age", "los_icu"]
        missing_required = [col for col in required_for_overlap if col not in df.columns]

        # Resolve the exposure column config-first (declared exposure wins; a
        # Sepsis-3 flag is only a fallback). "Derivable" == the exposure is
        # observable for that stay; the eligibility axis is the same shape for
        # any exposure.
        exposure_col = None
        for _cand in (primary_exposure, "sep3_sofa2_max", "sepsis3"):
            if _cand and _cand in df.columns:
                exposure_col = _cand
                break
        exposure_label = primary_exposure or exposure_col or "exposure"
        if exposure_col is None:
            exposure_derivable = pd.Series(False, index=df.index)
            exposure_source = None
            missing_required.append("primary_exposure_column")
        elif exposure_col == "sep3_sofa2_max":
            exposure_derivable = _numeric("sep3_sofa2_max").notna()
            exposure_source = "sep3_sofa2_max"
        elif exposure_col == "sepsis3":
            exposure_derivable = _numeric("sepsis3").isin([0, 1])
            exposure_source = "sepsis3"
        else:
            exposure_derivable = df[exposure_col].notna()
            exposure_source = exposure_col

        measured_flag = (
            _numeric("sep3_sofa2_measured")
            if exposure_col == "sep3_sofa2_max" and "sep3_sofa2_measured" in df.columns
            else pd.Series(np.nan, index=df.index, dtype="float64")
        )
        if exposure_col == "sep3_sofa2_max":
            exposure_binary = (
                (_numeric("sep3_sofa2_max") >= 1)
                .astype(float)
                .where(_numeric("sep3_sofa2_max").notna())
            )
        elif exposure_col is not None:
            _raw = _numeric(exposure_col)
            _vals = set(_raw.dropna().unique().tolist())
            if _vals <= {0.0, 1.0}:
                exposure_binary = _raw.where(_raw.isin([0, 1]))
            else:
                exposure_binary = (_raw >= 1).astype(float).where(_raw.notna())
        else:
            exposure_binary = pd.Series(np.nan, index=df.index, dtype="float64")
        if missing_required:
            pd.DataFrame(
                [
                    {
                        "definition_id": "unavailable",
                        "definition_label": "Required columns missing",
                        "definition_type": "blocked",
                        "criteria": "Not computable",
                        "n_included": 0,
                        "n_excluded": len(df),
                        "included_pct_of_rows": np.nan,
                        "overlap_with_primary_n": np.nan,
                        "overlap_with_primary_pct_of_primary": np.nan,
                        "overlap_with_primary_pct_of_definition": np.nan,
                        "moved_in_vs_primary_n": np.nan,
                        "moved_out_vs_primary_n": np.nan,
                        "missing_required_columns": "|".join(missing_required),
                    }
                ]
            ).to_csv(out_dir / "alternative_cohort_attrition.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "cohort_overlap_matrix.csv", index=False)
            pd.DataFrame().to_csv(
                out_dir / "cohort_definition_empirical_equivalence_audit.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "semantic_check": "exposure_derivability",
                        "status": "blocked",
                        "evidence": f"Required exposure source column ({exposure_label}) missing.",
                    }
                ]
            ).to_csv(out_dir / "cohort_definition_semantics_audit.csv", index=False)
            summary = {
                "step_id": current_step_id,
                "analysis_family": "cohort_definition_sensitivity",
                "status": "blocked",
                "missing_required_columns": missing_required,
                "outputs": ["alternative_cohort_attrition.csv"],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            raise SystemExit(0)

        adult_mask = _numeric("age").ge(18)
        los = _numeric("los_icu")
        los_ge_1 = los.ge(1)
        los_ge_2 = los.ge(2)
        map_measured = _measured("map_measured")
        hr_measured = _measured("hr_measured")
        resp_measured = _measured("resp_measured")
        temp_measured = _measured("temp_measured")
        vital_count = (
            map_measured.astype(int)
            + hr_measured.astype(int)
            + resp_measured.astype(int)
            + temp_measured.astype(int)
        )
        all_vitals = map_measured & hr_measured & resp_measured & temp_measured
        three_of_four = vital_count.ge(3)

        # Important: sep3_sofa2_measured can encode a positive source record in
        # some exported wide tables. For eligibility, use exposure derivability
        # from the source value instead of measured_flag == 1 so binary negatives
        # remain in the risk set.
        definitions = [
            {
                "definition_id": "primary_adult_los1_all_vitals_exposure_derivable",
                "definition_label": "Primary cohort",
                "definition_type": "primary",
                "criteria": (
                    "age>=18 AND los_icu>=1 day AND map/hr/resp/temp measured "
                    "AND exposure derivable"
                ),
                "mask": adult_mask & los_ge_1 & all_vitals & exposure_derivable,
            },
            {
                "definition_id": "alt_adult_no_los_all_vitals_exposure_derivable",
                "definition_label": "Relax ICU length-of-stay threshold",
                "definition_type": "alternative",
                "criteria": (
                    "age>=18 AND map/hr/resp/temp measured AND exposure derivable"
                ),
                "mask": adult_mask & all_vitals & exposure_derivable,
            },
            {
                "definition_id": "alt_adult_los1_three_of_four_vitals_exposure_derivable",
                "definition_label": "Relax vital completeness to >=3 of 4",
                "definition_type": "alternative",
                "criteria": (
                    "age>=18 AND los_icu>=1 day AND at least 3 of map/hr/resp/temp "
                    "measured AND exposure derivable"
                ),
                "mask": adult_mask & los_ge_1 & three_of_four & exposure_derivable,
            },
            {
                "definition_id": "alt_adult_los1_no_temp_requirement_exposure_derivable",
                "definition_label": "Relax temperature requirement",
                "definition_type": "alternative",
                "criteria": (
                    "age>=18 AND los_icu>=1 day AND map/hr/resp measured "
                    "AND exposure derivable"
                ),
                "mask": adult_mask & los_ge_1 & map_measured & hr_measured & resp_measured & exposure_derivable,
            },
            {
                "definition_id": "alt_adult_los2_all_vitals_exposure_derivable",
                "definition_label": "Tighten ICU length-of-stay threshold",
                "definition_type": "alternative",
                "criteria": (
                    "age>=18 AND los_icu>=2 days AND map/hr/resp/temp measured "
                    "AND exposure derivable"
                ),
                "mask": adult_mask & los_ge_2 & all_vitals & exposure_derivable,
            },
        ]

        stay_id = _numeric("stay_id")
        primary = definitions[0]
        primary_ids = set(stay_id.loc[primary["mask"]].dropna().astype(int).tolist())
        primary_n = int(primary["mask"].sum())
        n_rows = int(len(df))

        attrition_rows = []
        definition_sets = {}
        for item in definitions:
            mask = item["mask"].fillna(False)
            ids = set(stay_id.loc[mask].dropna().astype(int).tolist())
            definition_sets[item["definition_id"]] = ids
            included = int(mask.sum())
            overlap = int(len(ids & primary_ids))
            attrition_rows.append(
                {
                    "definition_id": item["definition_id"],
                    "definition_label": item["definition_label"],
                    "definition_type": item["definition_type"],
                    "criteria": item["criteria"],
                    "n_included": included,
                    "n_excluded": int(n_rows - included),
                    "included_pct_of_rows": _pct(included, n_rows),
                    "overlap_with_primary_n": overlap,
                    "overlap_with_primary_pct_of_primary": _pct(overlap, primary_n),
                    "overlap_with_primary_pct_of_definition": _pct(overlap, included),
                    "moved_in_vs_primary_n": int(len(ids - primary_ids)),
                    "moved_out_vs_primary_n": int(len(primary_ids - ids)),
                }
            )

        overlap_rows = []
        for left in definitions:
            ids_left = definition_sets[left["definition_id"]]
            for right in definitions:
                ids_right = definition_sets[right["definition_id"]]
                intersection = len(ids_left & ids_right)
                union = len(ids_left | ids_right)
                overlap_rows.append(
                    {
                        "definition_a": left["definition_id"],
                        "definition_b": right["definition_id"],
                        "n_a": len(ids_left),
                        "n_b": len(ids_right),
                        "intersection_n": int(intersection),
                        "union_n": int(union),
                        "jaccard": float(intersection / union) if union else np.nan,
                        "a_in_b_pct": _pct(intersection, len(ids_left)),
                        "b_in_a_pct": _pct(intersection, len(ids_right)),
                    }
                )

        equivalence_rows = []
        for item in definitions[1:]:
            ids = definition_sets[item["definition_id"]]
            primary_only = len(primary_ids - ids)
            comparison_only = len(ids - primary_ids)
            shared = len(primary_ids & ids)
            identical = primary_only == 0 and comparison_only == 0
            if identical:
                reason = (
                    "Empirically identical to the primary cohort in this export."
                )
            elif comparison_only > 0 and primary_only == 0:
                reason = (
                    f"Relaxing the definition added {comparison_only} ICU stays "
                    "relative to primary."
                )
            elif primary_only > 0 and comparison_only == 0:
                reason = (
                    f"Tightening the definition removed {primary_only} ICU stays "
                    "from primary."
                )
            else:
                reason = (
                    f"Definition exchanged {primary_only} primary-only and "
                    f"{comparison_only} comparison-only ICU stays."
                )
            equivalence_rows.append(
                {
                    "comparison_definition": item["definition_id"],
                    "comparison_label": item["definition_label"],
                    "primary_definition": primary["definition_id"],
                    "identical_to_primary": bool(identical),
                    "primary_only_n": int(primary_only),
                    "comparison_only_n": int(comparison_only),
                    "shared_n": int(shared),
                    "reason": reason,
                }
            )

        semantics_rows = [
            {
                "semantic_check": "exposure_derivability",
                "status": "passed",
                "source_column": exposure_source,
                "n_derivable": int(exposure_derivable.sum()),
                "n_positive": int((exposure_binary == 1).sum()),
                "n_negative": int((exposure_binary == 0).sum()),
                "measured_flag_positive_only": bool(
                    measured_flag.notna().any()
                    and int(measured_flag.eq(1).sum()) == int((exposure_binary == 1).sum())
                    and int(measured_flag.eq(0).sum()) == int((exposure_binary == 0).sum())
                ),
                "action": (
                    (
                        "Eligibility used source-value derivability, not "
                        "sep3_sofa2_measured == 1, so binary negatives were retained."
                    )
                    if exposure_col == "sep3_sofa2_max"
                    else (
                        f"Eligibility used source-value derivability of "
                        f"`{exposure_label}`; binary negatives were retained."
                    )
                ),
            }
        ]

        pd.DataFrame(attrition_rows).to_csv(
            out_dir / "alternative_cohort_attrition.csv", index=False
        )
        pd.DataFrame(overlap_rows).to_csv(
            out_dir / "cohort_overlap_matrix.csv", index=False
        )
        pd.DataFrame(equivalence_rows).to_csv(
            out_dir / "cohort_definition_empirical_equivalence_audit.csv",
            index=False,
        )
        pd.DataFrame(semantics_rows).to_csv(
            out_dir / "cohort_definition_semantics_audit.csv", index=False
        )
        note = (
            "No patient identifier is available in this export. Overlap is assessed "
            "at the ICU-stay level using stay_id; first-stay selection and "
            "within-patient correlation cannot be evaluated."
        )
        (out_dir / "patient_identifier_limitation_note.txt").write_text(
            note, encoding="utf-8"
        )

        summary = {
            "step_id": current_step_id,
            "analysis_family": "cohort_definition_sensitivity",
            "method": "deterministic_cohort_definition_overlap",
            "status": "ok",
            "n_rows": n_rows,
            "n_unique_stays": int(stay_id.nunique(dropna=True)),
            "primary_definition": {
                key: value
                for key, value in attrition_rows[0].items()
                if key in ["definition_id", "criteria", "n_included", "n_excluded"]
            },
            "definitions_evaluated": [
                {
                    "definition_id": row["definition_id"],
                    "definition_label": row["definition_label"],
                    "definition_type": row["definition_type"],
                    "criteria": row["criteria"],
                    "n_included": row["n_included"],
                }
                for row in attrition_rows
            ],
            "semantic_guard": semantics_rows[0],
            "overlap_unit": "ICU-stay level using stay_id",
            "limitations": {
                "patient_identifier_absent": True,
                "note": note,
            },
            "output_files": {
                "alternative_cohort_attrition": "alternative_cohort_attrition.csv",
                "cohort_overlap_matrix": "cohort_overlap_matrix.csv",
                "cohort_definition_empirical_equivalence_audit": (
                    "cohort_definition_empirical_equivalence_audit.csv"
                ),
                "cohort_definition_semantics_audit": (
                    "cohort_definition_semantics_audit.csv"
                ),
                "patient_identifier_limitation_note": (
                    "patient_identifier_limitation_note.txt"
                ),
            },
            "notes": [
                "Exposure binary negatives were retained when the source value was derivable.",
                "Overlap is computed at the ICU-stay level using stay_id.",
            ],
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps({"cohort_definition_overlap": "ok"}))
        '''
    )


def cohort_definition_sensitivity_comparison_code() -> str:
    """Return a runner script for cohort-definition sensitivity comparisons.

    This is intentionally narrow: it handles the common manuscript pattern
    where an upstream step has registered alternative eligibility definitions
    and the next step needs to re-fit the same exposure/outcome association
    under each definition. The generated script derives all numbers from the
    run's cohort parquet and prior step outputs.
    """

    return textwrap.dedent(
        r'''
        import json
        import math
        import os
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir.parents[2]
        current_step_id = out_dir.parent.name
        cohort_path = Path(os.environ["COHORT_PARQUET"])

        df = pd.read_parquet(cohort_path).copy()


        def _find_parent_outputs():
            steps_dir = run_dir / "steps"
            if not steps_dir.exists():
                return None
            candidates = []
            for step_dir in sorted(steps_dir.iterdir()):
                if not step_dir.is_dir() or step_dir.name == current_step_id:
                    continue
                outputs_dir = step_dir / "outputs"
                attrition = outputs_dir / "alternative_cohort_attrition.csv"
                overlap = outputs_dir / "cohort_overlap_matrix.csv"
                if attrition.exists():
                    score = 0
                    name = step_dir.name.lower()
                    if "alternative" in name or "eligibility" in name:
                        score += 4
                    if "cohort" in name or "definition" in name:
                        score += 3
                    if overlap.exists():
                        score += 2
                    candidates.append((score, step_dir.name, outputs_dir))
            if not candidates:
                return None
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2]


        parent_outputs = _find_parent_outputs()
        if parent_outputs is None:
            summary = {
                "step_id": current_step_id,
                "analysis_family": "cohort_definition_sensitivity",
                "status": "blocked",
                "blocking_reason": (
                    "No upstream alternative_cohort_attrition.csv was available "
                    "for a cohort-definition sensitivity comparison."
                ),
                "outputs": [],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            raise SystemExit(0)

        attrition_path = parent_outputs / "alternative_cohort_attrition.csv"
        attrition = pd.read_csv(attrition_path)


        def _first_existing(candidates):
            for col in candidates:
                if col and col in df.columns:
                    return col
            return None


        # --- config-first exposure / outcome / covariate resolution -----------
        # The study's declared primary_exposure / target_outcome / covariates
        # (from research_context.json, written by the pipeline from the study or
        # benchmark config) always win. The Sepsis-3 / mortality names below are
        # kept only as last-resort aliases so the skill still runs standalone.
        # This keeps the sensitivity re-fit tied to the QUESTION being asked
        # rather than to one benchmark case.
        primary_exposure = ""
        target_outcome = ""
        req_covariates = []
        try:
            _ctx = json.loads((run_dir / "research_context.json").read_text("utf-8"))
            primary_exposure = str(_ctx.get("primary_exposure") or "").strip()
            target_outcome = str(_ctx.get("target_outcome") or "").strip()
            _prefs = _ctx.get("user_preferences") or {}
            if isinstance(_prefs, dict):
                req_covariates = [
                    str(c).strip()
                    for c in (_prefs.get("covariates") or [])
                    if str(c).strip()
                ]
        except Exception:
            pass

        age_col = _first_existing(["age", "admission_age", "anchor_age"])
        los_col = _first_existing(["los_icu", "icu_los_days", "los_days"])
        stay_col = _first_existing(["stay_id", "icustay_id", "icu_stay_id"])
        outcome_col = _first_existing(
            [target_outcome, "death", "hospital_mortality", "mortality"]
        )
        # Exposure: prefer the DECLARED primary exposure column. Only when the
        # study declares none do we fall back to a Sepsis-3 flag; a raw SOFA-2
        # total is still never silently relabelled as the exposure.
        exposure_col = _first_existing(
            [primary_exposure, "sep3_sofa2_max", "sepsis3"]
        )
        outcome_label = target_outcome or outcome_col or "outcome"
        exposure_label = primary_exposure or exposure_col or "exposure"
        if outcome_col is None or exposure_col is None:
            summary = {
                "step_id": current_step_id,
                "analysis_family": "cohort_definition_sensitivity",
                "status": "blocked",
                "blocking_reason": (
                    "Required outcome or exposure column was absent for the "
                    f"declared association ({exposure_label} -> {outcome_label})."
                ),
                "cohort_path": str(cohort_path),
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            raise SystemExit(0)

        outcome = pd.to_numeric(df[outcome_col], errors="coerce")
        exposure_raw = pd.to_numeric(df[exposure_col], errors="coerce")
        _exp_vals = set(exposure_raw.dropna().unique().tolist())
        if exposure_col == "sep3_sofa2_max":
            # SOFA-2 organ-dysfunction score -> Sepsis-3 positive at >=1.
            exposure = (exposure_raw >= 1).astype(float).where(exposure_raw.notna())
        elif _exp_vals <= {0.0, 1.0}:
            # already a binary indicator (declared exposure or 'sepsis3' flag)
            exposure = exposure_raw.where(exposure_raw.isin([0, 1]))
        else:
            # generic continuous exposure -> binarise at >=1 (any)
            exposure = (exposure_raw >= 1).astype(float).where(exposure_raw.notna())
        df["_det_outcome"] = outcome
        df["_det_exposure"] = exposure


        def _series_numeric(col):
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index, dtype="float64")
            return pd.to_numeric(df[col], errors="coerce")


        def _measured(col):
            if col in df.columns:
                return _series_numeric(col).fillna(0).astype(bool)
            raw = col.removesuffix("_measured")
            if raw in df.columns:
                return df[raw].notna()
            return pd.Series(False, index=df.index)


        adult_mask = pd.Series(True, index=df.index)
        if age_col is not None:
            adult_mask = _series_numeric(age_col) >= 18
        valid_mask = (
            adult_mask.fillna(False)
            & df["_det_outcome"].isin([0, 1])
            & df["_det_exposure"].isin([0, 1])
        )
        if stay_col is not None:
            valid_mask = valid_mask & df[stay_col].notna()

        los = _series_numeric(los_col) if los_col is not None else pd.Series(np.nan, index=df.index)
        map_measured = _measured("map_measured")
        hr_measured = _measured("hr_measured")
        resp_measured = _measured("resp_measured")
        temp_measured = _measured("temp_measured")
        exposure_derivable = exposure_raw.notna()
        all_vitals = map_measured & hr_measured & resp_measured & temp_measured
        three_of_four = (
            map_measured.astype(int)
            + hr_measured.astype(int)
            + resp_measured.astype(int)
            + temp_measured.astype(int)
        ) >= 3


        def _definition_mask(definition_id, criteria):
            token = f"{definition_id} {criteria}".lower()
            mask = valid_mask.copy()
            if "los2" in token or "los >=2" in token or ">=2" in token:
                mask = mask & (los >= 2)
            elif "no_los" in token or "no los" in token:
                pass
            else:
                mask = mask & (los >= 1)

            if "3of4" in token or "3 of 4" in token or "three" in token:
                mask = mask & three_of_four
            elif "no_temp" in token or "no temperature" in token or "without temp" in token:
                mask = mask & map_measured & hr_measured & resp_measured
            else:
                mask = mask & all_vitals

            if "no_sofa" not in token and "no_exposure" not in token:
                mask = mask & exposure_derivable
            return mask.fillna(False)


        def _scaled_feature_from_frame(frame, source, scale):
            raw = pd.to_numeric(frame[source], errors="coerce")
            missing = raw.isna().astype(int)
            if raw.notna().any():
                filled = raw.fillna(raw.median())
            else:
                filled = raw.fillna(0.0)
            return filled.astype(float) / float(scale), missing


        def _first_available_in_frame(frame, candidates):
            for col in candidates:
                if col is not None and col in frame.columns:
                    return col
            return None


        def _fit_adjusted_or(sub, label, *, exclude_features=None):
            exclude_features = set(exclude_features or [])
            # Internal, exposure/outcome-agnostic model keys: the primary term
            # is always ``_exposure`` and the response is ``_outcome``, whatever
            # the study's declared exposure/outcome happen to be.
            model_df = pd.DataFrame(
                {
                    "_outcome": sub["_det_outcome"].astype(float),
                    "_exposure": sub["_det_exposure"].astype(float),
                },
                index=sub.index,
            )
            covariates = ["_exposure"]
            dropped = []

            if req_covariates:
                # Study-declared adjustment set (per-unit; each covariate scaled
                # by 1.0). Config wins over the clinical default below.
                feature_specs = [(cov, [cov], 1.0) for cov in req_covariates]
                add_clinical_companions = False
            else:
                # Case-neutral clinical default adjustment set (used only when no
                # covariates were declared in research_context.json).
                feature_specs = [
                    ("age", [age_col, "age"], 10.0),
                    ("hr_max", ["hr_max", "heart_rate_max"], 10.0),
                    ("resp_max", ["resp_max", "respiratory_rate_max"], 5.0),
                    ("temp_max", ["temp_max", "temperature_max"], 1.0),
                    ("lact_max", ["lact_max_mmol_l", "lact_max", "lactate_max"], 1.0),
                    ("bun_max", ["bun_max", "bun_max_mg_dl"], 10.0),
                    ("wbc_max", ["wbc_max", "wbc_max_10e9_l"], 10.0),
                ]
                add_clinical_companions = True

            for feature_name, sources, scale in feature_specs:
                if feature_name in exclude_features:
                    continue
                source = _first_available_in_frame(sub, sources)
                if source is None:
                    continue
                values, missing = _scaled_feature_from_frame(sub, source, scale)
                value_col = f"{feature_name}_scaled"
                miss_col = f"{feature_name}_missing_indicator"
                model_df[value_col] = values
                model_df[miss_col] = missing
                covariates.extend([value_col, miss_col])

            if add_clinical_companions:
                for measured_col in (
                    "hr_measured",
                    "resp_measured",
                    "temp_measured",
                    "lact_measured",
                    "bun_measured",
                    "wbc_measured",
                ):
                    if measured_col.startswith("lact") and "lact_max" in exclude_features:
                        continue
                    if measured_col in sub.columns:
                        model_df[measured_col] = pd.to_numeric(
                            sub[measured_col], errors="coerce"
                        ).fillna(0).astype(float)
                        covariates.append(measured_col)

                if "sex" in sub.columns:
                    sex_dummies = pd.get_dummies(
                        sub["sex"].astype(str), prefix="sex", drop_first=True, dtype=float
                    )
                    for col in sex_dummies.columns:
                        model_df[col] = sex_dummies[col]
                        covariates.append(col)

            model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
            unique_covariates = []
            for col in covariates:
                if col not in model_df.columns or col in unique_covariates:
                    continue
                if col != "_exposure" and model_df[col].nunique(dropna=True) <= 1:
                    dropped.append(col)
                    continue
                unique_covariates.append(col)
            covariates = unique_covariates

            result = {
                "point_estimate": None,
                "ci_low": None,
                "ci_high": None,
                "se": None,
                "p_value": None,
                "modeled_analytic_n": int(len(model_df)),
                "events": int(model_df["_outcome"].sum()) if len(model_df) else 0,
                "converged": False,
                "model_message": "",
                "covariates": covariates,
                "dropped_covariates": dropped,
            }
            if (
                len(model_df) < 50
                or model_df["_outcome"].nunique(dropna=True) < 2
                or model_df["_exposure"].nunique(dropna=True) < 2
            ):
                result["model_message"] = "Insufficient outcome or exposure variation."
                return result

            try:
                x = sm.add_constant(model_df[covariates], has_constant="add")
                y = model_df["_outcome"].astype(float)
                fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
                    cov_type="HC1", maxiter=100
                )
                beta = float(fit.params["_exposure"])
                se = float(fit.bse["_exposure"])
                ci_low = beta - 1.96 * se
                ci_high = beta + 1.96 * se
                result.update(
                    {
                        "point_estimate": math.exp(beta),
                        "ci_low": math.exp(ci_low),
                        "ci_high": math.exp(ci_high),
                        "se": se,
                        "p_value": float(fit.pvalues["_exposure"]),
                        "converged": bool(getattr(fit, "converged", True)),
                        "model_message": "GLM Binomial with HC1 robust SE.",
                    }
                )
            except Exception as exc:
                result["model_message"] = f"{type(exc).__name__}: {exc}"
            return result


        def _outcome_summary(sub, definition_id, label):
            rows = []
            total_n = int(len(sub))
            total_events = int(sub["_det_outcome"].sum()) if total_n else 0
            exposed_n = int((sub["_det_exposure"] == 1).sum()) if total_n else 0
            rows.append(
                {
                    "definition_id": definition_id,
                    "definition_label": label,
                    "stratum": "all",
                    "n": total_n,
                    "events": total_events,
                    "event_rate": total_events / total_n if total_n else np.nan,
                    "exposure_prevalence": exposed_n / total_n if total_n else np.nan,
                }
            )
            for level in [0.0, 1.0]:
                ss = sub[sub["_det_exposure"] == level]
                n = int(len(ss))
                events = int(ss["_det_outcome"].sum()) if n else 0
                rows.append(
                    {
                        "definition_id": definition_id,
                        "definition_label": label,
                        "stratum": f"exposure_{int(level)}",
                        "n": n,
                        "events": events,
                        "event_rate": events / n if n else np.nan,
                        "exposure_prevalence": np.nan,
                    }
                )
            return rows


        def _risk_difference_from_outcome_rows(all_rows):
            unexposed = next(item for item in all_rows if item["stratum"] == "exposure_0")
            exposed = next(item for item in all_rows if item["stratum"] == "exposure_1")
            n0 = int(unexposed["n"])
            n1 = int(exposed["n"])
            p0 = unexposed["event_rate"]
            p1 = exposed["event_rate"]
            if n0 <= 0 or n1 <= 0 or pd.isna(p0) or pd.isna(p1):
                return None
            rd = float(p1 - p0)
            se = math.sqrt((p1 * (1.0 - p1) / n1) + (p0 * (1.0 - p0) / n0))
            return {
                "point_estimate": rd,
                "ci_low": rd - 1.96 * se,
                "ci_high": rd + 1.96 * se,
                "se": se,
                "p_value": None,
                "n0": n0,
                "n1": n1,
            }


        comparison_rows = []
        summary_rows = []
        outcome_rows = []
        covariate_rows = []

        definitions = [
            {
                "definition_id": "full_export_step03_scope",
                "definition_label": "Full export (step 03 scope)",
                "definition_type": "reference",
                "criteria": "adult/full export with valid outcome and exposure",
                "n_included": int(valid_mask.sum()),
            }
        ]
        definitions.extend(attrition.to_dict(orient="records"))
        lactate_col = _first_available_in_frame(
            df, ["lact_max_mmol_l", "lact_max", "lactate_max"]
        )
        primary_attrition = next(
            (
                row
                for row in attrition.to_dict(orient="records")
                if str(row.get("definition_type", "")).lower() == "primary"
                or "primary" in str(row.get("definition_id", "")).lower()
            ),
            None,
        )
        if lactate_col is not None and primary_attrition is not None:
            definitions.append(
                {
                    "definition_id": "primary_lactate_complete_case",
                    "definition_label": "Primary + lactate observed",
                    "definition_type": "missingness_sensitivity",
                    "criteria": primary_attrition.get("criteria", ""),
                    "n_included": None,
                    "special_mask": "primary_lactate_complete_case",
                    "base_definition_id": primary_attrition.get("definition_id"),
                    "base_criteria": primary_attrition.get("criteria", ""),
                }
            )
            definitions.append(
                {
                    "definition_id": "primary_without_lactate_adjustment",
                    "definition_label": "Primary without lactate covariate",
                    "definition_type": "missingness_sensitivity",
                    "criteria": primary_attrition.get("criteria", ""),
                    "n_included": primary_attrition.get("n_included"),
                    "special_mask": "primary_without_lactate_adjustment",
                    "base_definition_id": primary_attrition.get("definition_id"),
                    "base_criteria": primary_attrition.get("criteria", ""),
                    "exclude_features": ["lact_max"],
                }
            )

        for row in definitions:
            definition_id = str(row.get("definition_id") or row.get("spec_id") or "").strip()
            if not definition_id:
                continue
            label = str(
                row.get("definition_label")
                or row.get("display_label")
                or definition_id.replace("_", " ")
            )
            criteria = str(row.get("criteria") or "")
            if definition_id == "full_export_step03_scope":
                mask = valid_mask
            elif row.get("special_mask") == "primary_lactate_complete_case":
                mask = _definition_mask(
                    row.get("base_definition_id") or definition_id,
                    row.get("base_criteria") or criteria,
                ) & pd.to_numeric(df[lactate_col], errors="coerce").notna()
            elif row.get("special_mask") == "primary_without_lactate_adjustment":
                mask = _definition_mask(
                    row.get("base_definition_id") or definition_id,
                    row.get("base_criteria") or criteria,
                )
            else:
                mask = _definition_mask(definition_id, criteria)
            sub = df.loc[mask].copy()
            fit = _fit_adjusted_or(
                sub,
                label,
                exclude_features=row.get("exclude_features") or [],
            )
            parent_n = row.get("n_included")
            try:
                parent_n_value = int(parent_n)
            except Exception:
                parent_n_value = None
            modeled_n = fit["modeled_analytic_n"]
            if definition_id == "full_export_step03_scope":
                scope_note = (
                    "Reference row uses the full step-03 export scope; it is "
                    "not the stricter primary eligibility definition."
                )
            else:
                scope_note = "Re-fit under the registered eligibility definition."

            comparison_rows.append(
                {
                    "spec_id": definition_id,
                    "axis": (
                        "reference_scope"
                        if definition_id == "full_export_step03_scope"
                        else "missingness_strategy"
                        if str(row.get("definition_type", "")).lower()
                        == "missingness_sensitivity"
                        else "cohort_definition"
                    ),
                    "display_label": label,
                    "effect_scale": "OR",
                    "point_estimate": fit["point_estimate"],
                    "ci_low": fit["ci_low"],
                    "ci_high": fit["ci_high"],
                    "se": fit["se"],
                    "p_value": fit["p_value"],
                    "modeled_analytic_n": modeled_n,
                    "events": fit["events"],
                    "converged": fit["converged"],
                    "definition_id": definition_id,
                    "definition_label": label,
                    "definition_type": row.get("definition_type"),
                    "n_included_from_mask": int(mask.sum()),
                    "n_parent_attrition": parent_n_value,
                    "n_delta_vs_parent": (
                        int(mask.sum()) - parent_n_value
                        if parent_n_value is not None
                        else None
                    ),
                    "model_message": fit["model_message"],
                    "notes": scope_note,
                }
            )
            all_rows = _outcome_summary(sub, definition_id, label)
            outcome_rows.extend(all_rows)
            all_summary = all_rows[0]
            unexposed = next(item for item in all_rows if item["stratum"] == "exposure_0")
            exposed = next(item for item in all_rows if item["stratum"] == "exposure_1")
            rd = _risk_difference_from_outcome_rows(all_rows)
            if rd is not None:
                comparison_rows.append(
                    {
                        "spec_id": f"{definition_id}_crude_rd",
                        "axis": "descriptive_outcome",
                        "display_label": label,
                        "effect_scale": "RD",
                        "point_estimate": rd["point_estimate"],
                        "ci_low": rd["ci_low"],
                        "ci_high": rd["ci_high"],
                        "se": rd["se"],
                        "p_value": rd["p_value"],
                        "modeled_analytic_n": None,
                        "events": None,
                        "converged": True,
                        "definition_id": definition_id,
                        "definition_label": label,
                        "definition_type": row.get("definition_type"),
                        "n_included_from_mask": int(mask.sum()),
                        "n_parent_attrition": parent_n_value,
                        "n_delta_vs_parent": (
                            int(mask.sum()) - parent_n_value
                            if parent_n_value is not None
                            else None
                        ),
                        "model_message": (
                            f"Crude {exposure_label}-positive minus -negative "
                            f"{outcome_label} risk; descriptive, not adjusted."
                        ),
                        "notes": "Descriptive risk difference from outcome table.",
                    }
                )
            summary_rows.append(
                {
                    "definition_id": definition_id,
                    "definition_label": label,
                    "n_included_from_mask": int(mask.sum()),
                    "n_parent_attrition": parent_n_value,
                    "n_delta_vs_parent": (
                        int(mask.sum()) - parent_n_value
                        if parent_n_value is not None
                        else None
                    ),
                    "events": all_summary["events"],
                    "event_rate": all_summary["event_rate"],
                    "exposure_prevalence": all_summary["exposure_prevalence"],
                    "outcome_risk_unexposed": unexposed["event_rate"],
                    "outcome_risk_exposed": exposed["event_rate"],
                    "crude_risk_difference": (
                        exposed["event_rate"] - unexposed["event_rate"]
                        if pd.notna(exposed["event_rate"]) and pd.notna(unexposed["event_rate"])
                        else np.nan
                    ),
                }
            )
            covariate_rows.append(
                {
                    "definition_id": definition_id,
                    "definition_label": label,
                    "modeled_analytic_n": modeled_n,
                    "covariates_used": "|".join(fit["covariates"]),
                    "dropped_zero_variance_covariates": "|".join(
                        fit["dropped_covariates"]
                    ),
                    "excluded_by_design": (
                        "map_min/map_first and MAP measured indicators were excluded "
                        "to avoid adjusting for a variable used in eligibility."
                    ),
                }
            )

        comparison = pd.DataFrame(comparison_rows)
        definition_summary = pd.DataFrame(summary_rows)
        outcome_by_definition = pd.DataFrame(outcome_rows)
        covariates = pd.DataFrame(covariate_rows)
        _outcome_coding_cols = [
            col
            for col in [
                outcome_col,
                "death",
                "hospital_mortality",
                "mortality",
                "death_icu",
                "death_hosp",
            ]
            if col and col in df.columns
        ]
        _audit_rows = [
            {
                "sensitivity_axis": "outcome_coding",
                "status": "noninformative",
                "evidence": (
                    f"Only one binary {outcome_label} outcome column was available "
                    "in the analysis export; no alternative outcome-coding "
                    "specification could be fit without inventing a new endpoint."
                ),
                "columns_checked": "|".join(dict.fromkeys(_outcome_coding_cols)),
            }
        ]
        if exposure_col in ("sep3_sofa2_max", "sepsis3"):
            _audit_rows.append(
                {
                    "sensitivity_axis": "exposure_measurement_semantics",
                    "status": "audited",
                    "evidence": (
                        "Binary exposure negatives were derived from sep3_sofa2 "
                        "source values and not treated as unmeasured cases; "
                        "eligibility definitions used source-value derivability, "
                        "while sep3_sofa2_measured was retained as a semantics audit."
                    ),
                    "columns_checked": "|".join(
                        [
                            col
                            for col in [
                                "sep3_sofa2_max",
                                "sep3_sofa2_n",
                                "sep3_sofa2_measured",
                            ]
                            if col in df.columns
                        ]
                    ),
                }
            )
        else:
            _audit_rows.append(
                {
                    "sensitivity_axis": "exposure_measurement_semantics",
                    "status": "audited",
                    "evidence": (
                        f"Declared exposure `{exposure_label}` (column "
                        f"`{exposure_col}`) was used directly; binary negatives "
                        "were retained rather than dropped as unmeasured."
                    ),
                    "columns_checked": exposure_col,
                }
            )
        audit = pd.DataFrame(_audit_rows)

        comparison.to_csv(out_dir / "sensitivity_comparison.csv", index=False)
        definition_summary.to_csv(
            out_dir / "sensitivity_definition_summary.csv", index=False
        )
        outcome_by_definition.to_csv(
            out_dir / "sensitivity_outcome_by_definition.csv", index=False
        )
        covariates.to_csv(out_dir / "sensitivity_model_covariates.csv", index=False)
        audit.to_csv(out_dir / "noninformative_sensitivity_audit.csv", index=False)

        converged = comparison[
            comparison["converged"].astype(bool)
            & comparison["effect_scale"].astype(str).str.upper().isin(["OR", "RR", "HR"])
        ].copy()
        primary_rows = comparison[
            comparison["definition_id"].astype(str).str.contains("primary", case=False, na=False)
            & comparison["effect_scale"].astype(str).str.upper().eq("OR")
        ]
        full_row = comparison[
            (comparison["definition_id"] == "full_export_step03_scope")
            & comparison["effect_scale"].astype(str).str.upper().eq("OR")
        ]
        complete_case_n = comparison.loc[
            comparison["definition_id"] == "primary_lactate_complete_case",
            "modeled_analytic_n",
        ]
        scope_warning = (
            "The step-03 full-export reference is not identical to the stricter "
            "primary eligibility definition from the cohort-definition step; "
            "this sensitivity table should be interpreted as a scope audit before "
            "claiming a single primary estimand."
        )
        summary = {
            "step_id": current_step_id,
            "analysis_family": "cohort_definition_sensitivity",
            "method": (
                "Deterministic standard cohort-definition sensitivity comparison: "
                f"re-fit the adjusted {exposure_label} -> {outcome_label} "
                "association under registered eligibility definitions with HC1 "
                "robust SE."
            ),
            "primary_exposure": exposure_label,
            "target_outcome": outcome_label,
            "status": "ok" if not comparison.empty else "blocked",
            "cohort_path": str(cohort_path),
            "source_parent_outputs": str(parent_outputs),
            "source_attrition_table": str(attrition_path),
            "n_effect_rows": int(len(comparison)),
            "n_model_specs": int(
                comparison["effect_scale"].astype(str).str.upper().isin(["OR", "RR", "HR"]).sum()
            ),
            "n_converged_models": int(len(converged)),
            "complete_case_n": (
                int(complete_case_n.iloc[0]) if len(complete_case_n) else None
            ),
            "quality_warning": scope_warning,
            "primary_definition_rows": primary_rows.to_dict(orient="records"),
            "full_export_reference_rows": full_row.to_dict(orient="records"),
            "outputs": [
                "sensitivity_comparison.csv",
                "sensitivity_definition_summary.csv",
                "sensitivity_outcome_by_definition.csv",
                "sensitivity_model_covariates.csv",
                "noninformative_sensitivity_audit.csv",
            ],
            "limitations": [
                "Rows compare eligibility scopes and should not be pooled as independent studies.",
                "The table reports adjusted ORs; crude risk differences are descriptive only.",
                "MAP covariates are excluded because MAP is part of the eligibility definition.",
            ],
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps({"cohort_definition_sensitivity": summary["status"]}))
        '''
    )
