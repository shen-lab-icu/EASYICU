"""Deterministic fallback code generators for the lactate / MAP / vasopressor study.

These functions return runnable Python scripts (as strings) that the
pipeline emits when LLM-generated analysis code fails repeatedly for
specific step intents in the lactate / MAP / vasopressor → mortality
study. The scripts assume the study's column conventions:

* ``lactate_max_24h`` — primary predictor
* ``map_min_24h`` — mean arterial pressure (covariate)
* ``vaso_any_24h`` — vasopressor exposure (covariate)
* ``sofa2_*_max_24h`` — organ-system SOFA-2 component scores
* ``death`` — primary outcome (ICU / hospital mortality)

Historically these lived directly in :mod:`pipeline`, baking the
above column vocabulary into a tool that is otherwise paper-agnostic.
They are now owned by the :mod:`case_plugins.lactate_map_vaso`
plugin; the pipeline asks the plugin registry whether any plugin
recognises the current step before falling back to a deterministic
script.
"""

from __future__ import annotations

import json
import textwrap
from typing import Optional

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _primary_association_fallback_code(
    *,
    predictor: str,
    outcome: str = "death",
    reason: str,
) -> str:
    """Appendable rescue block for common real-LLM regression preprocessing failures."""
    predictor_lit = json.dumps(predictor)
    outcome_lit = json.dumps(outcome or "death")
    reason_lit = json.dumps(reason)
    return textwrap.dedent(
        f"""
        def _easyicu_primary_association_fallback_v1():
            import json
            import os
            import numpy as np
            import pandas as pd
            import statsmodels.api as sm

            cohort_path = os.environ.get("COHORT_PARQUET")
            out_dir = os.environ.get("STEP_OUT_DIR", ".")
            if not cohort_path:
                return
            df_fallback = pd.read_parquet(cohort_path)
            predictor_col = {predictor_lit}
            outcome_col = {outcome_lit}
            if predictor_col not in df_fallback.columns or outcome_col not in df_fallback.columns:
                return
            candidate_covariates = ["age", "sex", "lact", "creat", "map", "vaso", "los_icu"]
            covariates = [
                col for col in candidate_covariates
                if col in df_fallback.columns and col not in {{predictor_col, outcome_col}}
            ]
            model_df = df_fallback[[outcome_col, predictor_col] + covariates].copy()
            if "sex" in model_df.columns:
                model_df["sex"] = (
                    model_df["sex"].astype(str).str.lower()
                    .isin(["m", "male", "1", "true"])
                    .astype(float)
                )
            for col in model_df.columns:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(model_df) < 20 or model_df[outcome_col].nunique() < 2:
                return
            x_cols = [predictor_col] + covariates
            y = model_df[outcome_col].astype(float)
            X = sm.add_constant(model_df[x_cols].astype(float), has_constant="add")
            result = sm.Logit(y, X).fit(disp=0)
            coef = float(result.params[predictor_col])
            ci = np.exp(result.conf_int().loc[predictor_col]).tolist()
            primary_or = float(np.exp(coef))
            p_value = float(result.pvalues[predictor_col])
            table_path = os.path.join(out_dir, "primary_association.csv")
            pd.DataFrame([{{
                "variable": predictor_col,
                "odds_ratio": primary_or,
                "or_lower": float(ci[0]),
                "or_upper": float(ci[1]),
                "p_value": p_value,
                "n_complete": int(len(model_df)),
                "method": "deterministic_logistic_regression_fallback",
            }}]).to_csv(table_path, index=False)
            summary_path = os.path.join(out_dir, "step_summary.json")
            summary = {{}}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        summary.update(loaded)
                except Exception:
                    summary = {{}}
            summary.update({{
                "primary_predictor": predictor_col,
                "outcome": outcome_col,
                "primary_or": primary_or,
                "statistic:primary_or": primary_or,
                "primary_or_ci": [float(ci[0]), float(ci[1])],
                "p_value": p_value,
                "n_complete": int(len(model_df)),
                "method": "deterministic_logistic_regression_fallback",
                "fallback_reason": {reason_lit},
            }})
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(json.dumps(summary, ensure_ascii=False))

        try:
            _easyicu_primary_association_fallback_v1()
        except Exception as _easyicu_primary_fallback_exc:
            print(f"primary_association_fallback_failed: {{_easyicu_primary_fallback_exc}}")
        """
    ).strip("\n")


def _age_stratified_mortality_fallback_code() -> str:
    return textwrap.dedent(
        """
        import json
        import math
        import os
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _jsonable(value):
            if isinstance(value, dict):
                return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                value = float(value)
                return value if math.isfinite(value) else None
            if isinstance(value, (np.bool_,)):
                return bool(value)
            if isinstance(value, np.ndarray):
                return _jsonable(value.tolist())
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            return value

        def _wilson(count, n):
            count = float(count)
            n = float(n)
            if n <= 0:
                return None, None
            z = 1.959963984540054
            p = count / n
            denom = 1.0 + z * z / n
            centre = p + z * z / (2.0 * n)
            spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
            return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)

        out_dir = os.environ["STEP_OUT_DIR"]
        cohort_path = os.environ["COHORT_PARQUET"]
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_parquet(cohort_path)
        analysis = df[["age", "death"]].copy()
        analysis["age"] = pd.to_numeric(analysis["age"], errors="coerce")
        analysis["death"] = pd.to_numeric(analysis["death"], errors="coerce")
        analysis = analysis.dropna(subset=["age", "death"])
        analysis = analysis[analysis["age"] >= 18].copy()
        if analysis.empty:
            table = pd.DataFrame(columns=["age_tertile", "n", "deaths", "mortality_rate", "ci_lower", "ci_upper"])
        else:
            ranks = analysis["age"].rank(method="first")
            analysis["age_tertile"] = pd.qcut(ranks, q=min(3, len(analysis)), labels=False, duplicates="drop")
            table = (
                analysis.groupby("age_tertile", dropna=False)
                .agg(n=("death", "size"), deaths=("death", "sum"), age_min=("age", "min"), age_max=("age", "max"))
                .reset_index()
            )
            rows = []
            for _, row in table.iterrows():
                n = int(row["n"])
                deaths = int(row["deaths"])
                lo, hi = _wilson(deaths, n)
                rows.append({
                    "age_tertile": f"T{int(row['age_tertile']) + 1}",
                    "n": n,
                    "deaths": deaths,
                    "mortality_rate": deaths / n if n else None,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "age_min": float(row["age_min"]),
                    "age_max": float(row["age_max"]),
                })
            table = pd.DataFrame(rows)
        table_path = os.path.join(out_dir, "age_tertile_mortality.csv")
        table.to_csv(table_path, index=False)
        statistic_path = os.path.join(out_dir, "age_tertile_statistics.json")
        overall_deaths = int(analysis["death"].sum()) if len(analysis) else 0
        overall_n = int(len(analysis))
        overall_rate = overall_deaths / overall_n if overall_n else None
        stats_payload = {
            "n_rows": int(len(df)),
            "n_analyzed": overall_n,
            "mortality_rate": overall_rate,
            "missingness": {
                "age_missing_n": int(pd.to_numeric(df.get("age"), errors="coerce").isna().sum()) if "age" in df else int(len(df)),
                "death_missing_n": int(pd.to_numeric(df.get("death"), errors="coerce").isna().sum()) if "death" in df else int(len(df)),
            },
            "stratum_table": table.to_dict(orient="records"),
        }
        with open(statistic_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(stats_payload), f, indent=2, ensure_ascii=False)
        figure_files = []
        if not table.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(table["age_tertile"].astype(str), table["mortality_rate"].astype(float))
            ax.set_xlabel("Age tertile")
            ax.set_ylabel("Mortality proportion")
            ax.set_ylim(0, max(0.01, float(table["mortality_rate"].max()) * 1.25))
            ax.set_title("Mortality by age tertile")
            fig.tight_layout()
            for suffix in ("png", "svg"):
                path = os.path.join(out_dir, f"age_tertile_mortality.{suffix}")
                fig.savefig(path, dpi=300 if suffix == "png" else None)
                figure_files.append(path)
            plt.close(fig)
        summary = {
            "n_rows": int(len(df)),
            "n_analyzed": overall_n,
            "mortality_rate": overall_rate,
            "missingness": stats_payload["missingness"],
            "stratum_table": stats_payload["stratum_table"],
            "table_path": table_path,
            "statistic_path": statistic_path,
            "figure_files": figure_files,
        }
        with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
        print(json.dumps(_jsonable(summary), ensure_ascii=False))
        """
    ).strip() + "\n"


def _norepinephrine_dose_response_fallback_code() -> str:
    return textwrap.dedent(
        """
        import json
        import math
        import os
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _jsonable(value):
            if isinstance(value, dict):
                return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                value = float(value)
                return value if math.isfinite(value) else None
            if isinstance(value, (np.bool_,)):
                return bool(value)
            if isinstance(value, np.ndarray):
                return _jsonable(value.tolist())
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            return value

        def _wilson(count, n):
            count = float(count)
            n = float(n)
            if n <= 0:
                return None, None
            z = 1.959963984540054
            p = count / n
            denom = 1.0 + z * z / n
            centre = p + z * z / (2.0 * n)
            spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
            return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)

        def _fallback_or(frame, predictor, outcome):
            data = frame[[predictor, outcome]].dropna().copy()
            if len(data) < 4 or data[outcome].nunique() < 2:
                return None, None, None
            median = data[predictor].median()
            high = data[data[predictor] > median][outcome]
            low = data[data[predictor] <= median][outcome]
            a = float(high.sum()) + 0.5
            b = float(len(high) - high.sum()) + 0.5
            c = float(low.sum()) + 0.5
            d = float(len(low) - low.sum()) + 0.5
            log_or = math.log((a / b) / (c / d))
            se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
            return math.exp(log_or), math.exp(log_or - 1.96 * se), math.exp(log_or + 1.96 * se)

        def _adjusted_or(frame, predictor, outcome, covariates):
            try:
                from scipy.optimize import minimize
            except Exception:
                return _fallback_or(frame, predictor, outcome)
            cols = [outcome, predictor] + [c for c in covariates if c in frame.columns]
            data = frame[cols].copy()
            if "sex" in data.columns:
                data["sex"] = data["sex"].astype(str).str.lower().isin(["m", "male", "1", "true"]).astype(float)
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 20 or data[outcome].nunique() < 2:
                return _fallback_or(frame, predictor, outcome)
            y = data[outcome].astype(float).to_numpy()
            x = data[[predictor] + [c for c in covariates if c in data.columns]].astype(float)
            means = x.mean(axis=0)
            scales = x.std(axis=0).replace(0, 1.0)
            x = ((x - means) / scales).to_numpy()
            X = np.column_stack([np.ones(len(x)), x])
            def nll(beta):
                eta = np.clip(X @ beta, -30, 30)
                return float(np.sum(np.logaddexp(0, eta) - y * eta))
            result = minimize(nll, np.zeros(X.shape[1]), method="BFGS")
            if not result.success:
                return _fallback_or(frame, predictor, outcome)
            beta = float(result.x[1])
            se = None
            try:
                hess_inv = np.asarray(result.hess_inv)
                if hess_inv.shape[0] > 1:
                    se = float(math.sqrt(max(hess_inv[1, 1], 0.0)))
            except Exception:
                se = None
            if se is None or not math.isfinite(se):
                return math.exp(beta), None, None
            return math.exp(beta), math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se)

        out_dir = os.environ["STEP_OUT_DIR"]
        cohort_path = os.environ["COHORT_PARQUET"]
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_parquet(cohort_path)
        predictor = "norepi_equiv_max_24h"
        outcome = "death"
        covariates = ["age", "sex", "map_min_24h", "lactate_max_24h", "sofa2_cardio_max_24h"]
        analysis = df[[c for c in [predictor, outcome] + covariates if c in df.columns]].copy()
        for col in [predictor, outcome]:
            if col not in analysis.columns:
                analysis[col] = np.nan
        analysis[predictor] = pd.to_numeric(analysis[predictor], errors="coerce")
        analysis[outcome] = pd.to_numeric(analysis[outcome], errors="coerce")
        exposed = analysis[analysis[predictor].notna() & (analysis[predictor] > 0)].copy()
        if exposed.empty:
            exposed = analysis[analysis[predictor].notna()].copy()
        quartile_rows = []
        if not exposed.empty:
            ranks = exposed[predictor].rank(method="first")
            exposed["dose_quartile"] = pd.qcut(ranks, q=min(4, len(exposed)), labels=False, duplicates="drop")
            grouped = exposed.dropna(subset=[outcome]).groupby("dose_quartile", dropna=False)
            for quartile, group in grouped:
                n = int(len(group))
                deaths = int(group[outcome].sum())
                lo, hi = _wilson(deaths, n)
                quartile_rows.append({
                    "dose_quartile": f"Q{int(quartile) + 1}",
                    "n": n,
                    "deaths": deaths,
                    "mortality_rate": deaths / n if n else None,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "dose_min": float(group[predictor].min()),
                    "dose_max": float(group[predictor].max()),
                })
        quartile_table = pd.DataFrame(quartile_rows)
        table_path = os.path.join(out_dir, "norepinephrine_quartile_mortality.csv")
        quartile_table.to_csv(table_path, index=False)
        primary_or, primary_or_lower, primary_or_upper = _adjusted_or(exposed, predictor, outcome, covariates)
        missingness = {
            col: {
                "missing_n": int(analysis[col].isna().sum()) if col in analysis.columns else int(len(df)),
                "missing_fraction": float(analysis[col].isna().mean()) if col in analysis.columns else 1.0,
            }
            for col in [predictor, outcome] + covariates
        }
        complete_case_n = int(exposed[[predictor, outcome] + [c for c in covariates if c in exposed.columns]].dropna().shape[0])
        mortality_rate = float(exposed[outcome].mean()) if len(exposed.dropna(subset=[outcome])) else None
        statistic_payload = {
            "primary_or": primary_or,
            "statistic:primary_or": primary_or,
            "primary_or_lower": primary_or_lower,
            "primary_or_upper": primary_or_upper,
            "mortality_rate": mortality_rate,
            "complete_case_n": complete_case_n,
            "missingness": missingness,
            "quartile_mortality": quartile_rows,
        }
        statistic_path = os.path.join(out_dir, "norepinephrine_dose_response_statistics.json")
        with open(statistic_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(statistic_payload), f, indent=2, ensure_ascii=False)
        figure_files = []
        if not quartile_table.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(
                quartile_table["dose_quartile"].astype(str),
                quartile_table["mortality_rate"].astype(float),
                yerr=[
                    quartile_table["mortality_rate"].astype(float) - quartile_table["ci_lower"].astype(float),
                    quartile_table["ci_upper"].astype(float) - quartile_table["mortality_rate"].astype(float),
                ],
                fmt="o-",
                capsize=4,
            )
            ax.set_xlabel("Norepinephrine-equivalent dose quartile")
            ax.set_ylabel("Mortality proportion")
            ax.set_ylim(0, max(0.01, float(quartile_table["ci_upper"].max()) * 1.15))
            ax.set_title("Dose-response mortality by quartile")
            fig.tight_layout()
            for suffix in ("png", "svg"):
                path = os.path.join(out_dir, f"norepinephrine_dose_response.{suffix}")
                fig.savefig(path, dpi=300 if suffix == "png" else None)
                figure_files.append(path)
            plt.close(fig)
        summary = {
            "n_rows": int(len(df)),
            "sample_size": int(len(exposed)),
            "complete_case_n": complete_case_n,
            "primary_predictor": predictor,
            "primary_or": primary_or,
            "statistic:primary_or": primary_or,
            "primary_or_lower": primary_or_lower,
            "primary_or_upper": primary_or_upper,
            "mortality_rate": mortality_rate,
            "missingness": missingness,
            "quartile_mortality": quartile_rows,
            "table_path": table_path,
            "statistic_path": statistic_path,
            "figure_files": figure_files,
        }
        with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
        print(json.dumps(_jsonable(summary), ensure_ascii=False))
        """
    ).strip() + "\n"


def _generic_v15_task_fallback_code(task_key: str) -> Optional[str]:
    if task_key == "clustering":
        # Clustering is a generic analysis family and still lives in pipeline;
        # this plugin defers to it via a lazy import so we don't pull pipeline
        # into module-load time (which would create a circular import).
        from ...pipeline import _generic_clustering_fallback_code  # noqa: PLC0415

        return _generic_clustering_fallback_code()

    common = """
import json
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _jsonable(value):
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value

def _save(summary, table, stem, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, f"{stem}.csv")
    stat_path = os.path.join(out_dir, f"{stem}_statistics.json")
    table.to_csv(table_path, index=False)
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
    summary["table_path"] = table_path
    summary["statistic_path"] = stat_path
    with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
    print(json.dumps(_jsonable(summary), ensure_ascii=False))

def _fallback_or(frame, predictor, outcome):
    data = frame[[predictor, outcome]].dropna().copy()
    if len(data) < 4 or data[outcome].nunique() < 2:
        return None, None, None
    median = data[predictor].median()
    high = data[data[predictor] > median][outcome]
    low = data[data[predictor] <= median][outcome]
    a = float(high.sum()) + 0.5
    b = float(len(high) - high.sum()) + 0.5
    c = float(low.sum()) + 0.5
    d = float(len(low) - low.sum()) + 0.5
    log_or = math.log((a / b) / (c / d))
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    return math.exp(log_or), math.exp(log_or - 1.96 * se), math.exp(log_or + 1.96 * se)

out_dir = os.environ["STEP_OUT_DIR"]
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
"""
    bodies = {
        "table_one": """
rows = []
for col in df.columns:
    s = df[col]
    row = {"variable": col, "n": int(len(s)), "missing_n": int(s.isna().sum()), "missing_fraction": float(s.isna().mean())}
    non = s.dropna()
    if len(non) and pd.api.types.is_numeric_dtype(non):
        row.update({"median": float(non.median()), "q25": float(non.quantile(0.25)), "q75": float(non.quantile(0.75))})
    elif len(non):
        top = non.astype(str).value_counts().head(1)
        row.update({"most_common": str(top.index[0]), "most_common_n": int(top.iloc[0])})
    rows.append(row)
table = pd.DataFrame(rows)
death = pd.to_numeric(df["death"], errors="coerce") if "death" in df else pd.Series(dtype=float)
summary = {"n_rows": int(len(df)), "mortality_rate": float(death.mean()) if len(death.dropna()) else None, "missingness": {c: float(df[c].isna().mean()) for c in df.columns}, "table_one_rows": table.to_dict(orient="records")}
fig, ax = plt.subplots(figsize=(7, 4))
plot = table.head(10)
ax.bar(plot["variable"].astype(str), plot["missing_fraction"].astype(float))
ax.set_ylabel("Missing fraction")
ax.set_xticklabels(plot["variable"].astype(str), rotation=45, ha="right")
fig.tight_layout()
figure_files = []
for suffix in ("png", "svg"):
    path = os.path.join(out_dir, f"table_one_missingness.{suffix}")
    fig.savefig(path, dpi=300 if suffix == "png" else None)
    figure_files.append(path)
plt.close(fig)
summary["figure_files"] = figure_files
_save(summary, table, "table_one", out_dir)
""",
        "severity_correlation": """
total = "sofa2_max_24h"
components = [c for c in df.columns if c.startswith("sofa2_") and c != total]
rows = []
for col in components:
    pair = df[[total, col]].apply(pd.to_numeric, errors="coerce").dropna()
    rho = float(pair[total].corr(pair[col], method="spearman")) if len(pair) >= 3 else None
    rows.append({"variable": col, "correlation": rho, "spearman_rho": rho, "n": int(len(pair)), "p_value": None})
table = pd.DataFrame(rows)
summary = {"n_rows": int(len(df)), "spearman_rho": next((r["spearman_rho"] for r in rows if r["spearman_rho"] is not None), None), "missingness": {c: float(df[c].isna().mean()) for c in [total] + components if c in df}, "correlations": rows}
fig, ax = plt.subplots(figsize=(7, 4))
if not table.empty:
    ax.bar(table["variable"].astype(str), table["spearman_rho"].fillna(0).astype(float))
ax.set_ylabel("Spearman rho vs total SOFA-2")
ax.set_xticklabels(table["variable"].astype(str), rotation=45, ha="right")
fig.tight_layout()
figure_files = []
for suffix in ("png", "svg"):
    path = os.path.join(out_dir, f"sofa2_component_correlations.{suffix}")
    fig.savefig(path, dpi=300 if suffix == "png" else None)
    figure_files.append(path)
plt.close(fig)
summary["figure_files"] = figure_files
_save(summary, table, "sofa2_component_correlations", out_dir)
""",
        "lactate": """
predictor, outcome = "lactate_max_24h", "death"
analysis = df[[c for c in [predictor, outcome, "age", "sex", "map_min_24h", "vaso_any_24h"] if c in df]].copy()
for col in [predictor, outcome, "age", "map_min_24h", "vaso_any_24h"]:
    if col in analysis:
        analysis[col] = pd.to_numeric(analysis[col], errors="coerce")
primary_or, lo, hi = _fallback_or(analysis, predictor, outcome)
table = pd.DataFrame([{"term": predictor, "primary_or": primary_or, "ci_lower": lo, "ci_upper": hi, "complete_case_n": int(analysis[[predictor, outcome]].dropna().shape[0])}])
summary = {"n_rows": int(len(df)), "primary_predictor": predictor, "primary_or": primary_or, "statistic:primary_or": primary_or, "odds_ratio": primary_or, "primary_or_lower": lo, "primary_or_upper": hi, "missingness": {c: float(analysis[c].isna().mean()) for c in analysis.columns}}
fig, ax = plt.subplots(figsize=(5, 3))
if primary_or is not None:
    ax.errorbar([primary_or], [0], xerr=[[max(primary_or - (lo or primary_or), 0)], [max((hi or primary_or) - primary_or, 0)]], fmt="o", capsize=4)
ax.axvline(1, color="grey", linestyle="--")
ax.set_yticks([0]); ax.set_yticklabels([predictor]); ax.set_xlabel("Odds ratio")
fig.tight_layout()
figure_files = []
for suffix in ("png", "svg"):
    path = os.path.join(out_dir, f"lactate_primary_association.{suffix}")
    fig.savefig(path, dpi=300 if suffix == "png" else None)
    figure_files.append(path)
plt.close(fig)
summary["figure_files"] = figure_files
_save(summary, table, "lactate_primary_association", out_dir)
""",
        "kdigo": """
predictor, outcome = "kdigo_stage_max_24h", "death"
analysis = df[[c for c in [predictor, outcome, "age", "sex", "sofa2_renal_max_24h", "vaso_any_24h"] if c in df]].copy()
for col in [predictor, outcome, "age", "sofa2_renal_max_24h", "vaso_any_24h"]:
    if col in analysis:
        analysis[col] = pd.to_numeric(analysis[col], errors="coerce")
primary_or, lo, hi = _fallback_or(analysis, predictor, outcome)
grouped = analysis.dropna(subset=[predictor, outcome]).groupby(predictor)[outcome].agg(["size", "sum", "mean"]).reset_index()
grouped = grouped.rename(columns={"size": "n", "sum": "deaths", "mean": "mortality_rate"})
table = grouped if not grouped.empty else pd.DataFrame([{"n": 0}])
complete_case_n = int(analysis.dropna().shape[0])
summary = {"n_rows": int(len(df)), "complete_case_n": complete_case_n, "primary_predictor": predictor, "primary_or": primary_or, "statistic:primary_or": primary_or, "odds_ratio": primary_or, "primary_or_lower": lo, "primary_or_upper": hi, "missingness": {c: float(analysis[c].isna().mean()) for c in analysis.columns}}
fig, ax = plt.subplots(figsize=(5, 3))
if not grouped.empty:
    ax.bar(grouped[predictor].astype(str), grouped["mortality_rate"].astype(float))
ax.set_xlabel("KDIGO stage"); ax.set_ylabel("Mortality proportion")
fig.tight_layout()
figure_files = []
for suffix in ("png", "svg"):
    path = os.path.join(out_dir, f"kdigo_mortality_sensitivity.{suffix}")
    fig.savefig(path, dpi=300 if suffix == "png" else None)
    figure_files.append(path)
plt.close(fig)
summary["figure_files"] = figure_files
_save(summary, table, "kdigo_mortality_sensitivity", out_dir)
""",
        "vitals": """
vitals = [c for c in ["hr_max_24h", "hr_median_24h", "sbp_min_24h", "sbp_median_24h", "map_min_24h", "map_median_24h"] if c in df]
rows = []
for col in vitals:
    vals = pd.to_numeric(df[col], errors="coerce")
    rows.append({"variable": col, "n": int(vals.notna().sum()), "missing_fraction": float(vals.isna().mean()), "median": float(vals.median()) if vals.notna().any() else None, "q25": float(vals.quantile(0.25)) if vals.notna().any() else None, "q75": float(vals.quantile(0.75)) if vals.notna().any() else None})
table = pd.DataFrame(rows)
summary = {"n_rows": int(len(df)), "missingness": {c: float(df[c].isna().mean()) for c in vitals}, "vital_summary": rows}
fig, ax = plt.subplots(figsize=(5, 4))
x = pd.to_numeric(df.get("map_min_24h"), errors="coerce") if "map_min_24h" in df else pd.Series(dtype=float)
y = pd.to_numeric(df.get("hr_max_24h"), errors="coerce") if "hr_max_24h" in df else pd.Series(dtype=float)
ax.scatter(x, y, s=8, alpha=0.35)
ax.set_xlabel("MAP min 24h"); ax.set_ylabel("HR max 24h")
fig.tight_layout()
figure_files = []
for suffix in ("png", "svg"):
    path = os.path.join(out_dir, f"paired_vital_summary.{suffix}")
    fig.savefig(path, dpi=300 if suffix == "png" else None)
    figure_files.append(path)
plt.close(fig)
summary["figure_files"] = figure_files
_save(summary, table, "admission_vital_summary", out_dir)
""",
        "creatinine": """
required = ["creat_max_24h", "creat_median_24h", "kdigo_stage_max_24h", "sofa2_renal_max_24h"]
analysis = pd.DataFrame(index=df.index)
for col in required:
    analysis[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
analysis["creat_ratio"] = analysis["creat_max_24h"] / analysis["creat_median_24h"].replace(0, np.nan)
analysis = analysis.replace([np.inf, -np.inf], np.nan)
rho_df = analysis[["creat_ratio", "sofa2_renal_max_24h"]].dropna()
rho = float(rho_df["creat_ratio"].corr(rho_df["sofa2_renal_max_24h"], method="spearman")) if len(rho_df) >= 3 else None
features = analysis[["creat_ratio", "sofa2_renal_max_24h"]].dropna()
labels = pd.Series(np.nan, index=analysis.index, dtype="float")
cluster_count = 0
silhouette_score = None
if len(features) >= 3:
    scaled = (features - features.median()) / features.std(ddof=0).replace(0, 1)
    scaled = scaled.fillna(0.0)
    score = scaled["creat_ratio"] + scaled["sofa2_renal_max_24h"]
    try:
        labels.loc[features.index] = pd.qcut(score.rank(method="first"), q=min(3, len(features)), labels=False, duplicates="drop").astype(float)
    except Exception:
        labels.loc[features.index] = (score > score.median()).astype(float)
    unique_labels = sorted([float(v) for v in labels.dropna().unique()])
    cluster_count = int(len(unique_labels))
    if cluster_count >= 2:
        x = scaled.to_numpy(dtype=float)
        y = labels.loc[features.index].to_numpy(dtype=float)
        sil_values = []
        for i in range(len(x)):
            same = x[y == y[i]]
            other_vals = []
            if len(same) > 1:
                a = float(np.linalg.norm(same - x[i], axis=1).sum() / (len(same) - 1))
            else:
                a = 0.0
            for lab in unique_labels:
                if lab == y[i]:
                    continue
                group = x[y == lab]
                if len(group):
                    other_vals.append(float(np.linalg.norm(group - x[i], axis=1).mean()))
            b = min(other_vals) if other_vals else 0.0
            sil_values.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
        silhouette_score = float(np.mean(sil_values)) if sil_values else None
analysis["cluster"] = labels
cluster_data = analysis.dropna(subset=["cluster"]).copy()
if not cluster_data.empty:
    cluster_data["cluster"] = cluster_data["cluster"].astype(int)
    cluster_characteristics = cluster_data.groupby("cluster", dropna=False).agg(
        n=("creat_ratio", "size"),
        creat_ratio_median=("creat_ratio", "median"),
        creat_ratio_mean=("creat_ratio", "mean"),
        renal_sofa_median=("sofa2_renal_max_24h", "median"),
        kdigo_median=("kdigo_stage_max_24h", "median"),
    ).reset_index()
    cluster_mortality = cluster_characteristics[["cluster", "n"]].copy()
    cluster_mortality["mortality_rate"] = None
    cluster_mortality["deaths"] = None
else:
    cluster_characteristics = pd.DataFrame([{"cluster": 0, "n": 0, "creat_ratio_median": None, "creat_ratio_mean": None, "renal_sofa_median": None, "kdigo_median": None}])
    cluster_mortality = pd.DataFrame([{"cluster": 0, "n": 0, "mortality_rate": None, "deaths": None}])
table = analysis.groupby("kdigo_stage_max_24h", dropna=False)["creat_ratio"].agg(["size", "median", "mean"]).reset_index().rename(columns={"size": "n", "median": "ratio_median", "mean": "ratio_mean"})
cluster_characteristics_path = os.path.join(out_dir, "cluster_characteristics.csv")
cluster_mortality_path = os.path.join(out_dir, "cluster_mortality.csv")
cluster_characteristics.to_csv(cluster_characteristics_path, index=False)
cluster_mortality.to_csv(cluster_mortality_path, index=False)
methodology = {
    "method": "dependency_free_quantile_clustering",
    "features": ["creat_ratio", "sofa2_renal_max_24h"],
    "cluster_count": cluster_count,
    "silhouette_score": silhouette_score,
    "sklearn_required": False,
}
for name in ["clustering_methodology.json", "clustering_algorithm_details.json"]:
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(_jsonable(methodology), f, indent=2, ensure_ascii=False)
fig, ax = plt.subplots(figsize=(6, 4))
plot_df = analysis.dropna(subset=["creat_ratio", "sofa2_renal_max_24h"]).copy()
if not plot_df.empty:
    colors = plot_df["cluster"].fillna(-1).astype(int)
    ax.scatter(plot_df["creat_ratio"], plot_df["sofa2_renal_max_24h"], c=colors, s=14, alpha=0.65)
ax.set_xlabel("Creatinine max/median ratio"); ax.set_ylabel("Renal SOFA max 24h")
fig.tight_layout()
figure_files = []
for suffix in ("png", "svg"):
    path = os.path.join(out_dir, f"clustering_visualization.{suffix}")
    fig.savefig(path, dpi=300 if suffix == "png" else None)
    figure_files.append(path)
for suffix in ("png", "svg"):
    path = os.path.join(out_dir, f"creatinine_trajectory_kdigo.{suffix}")
    fig.savefig(path, dpi=300 if suffix == "png" else None)
    figure_files.append(path)
plt.close(fig)
summary = {
    "n_rows": int(len(df)),
    "spearman_rho": rho,
    "statistic:spearman_rho": rho,
    "silhouette_score": silhouette_score,
    "statistic:silhouette_score": silhouette_score,
    "cluster_count": cluster_count,
    "statistic:cluster_count": cluster_count,
    "n_clusters": cluster_count,
    "cluster_characteristics": cluster_characteristics.to_dict(orient="records"),
    "cluster_mortality": cluster_mortality.to_dict(orient="records"),
    "clustering_methodology": methodology,
    "manifest:clustering_methodology": methodology,
    "clustering_algorithm_details": methodology,
    "missingness": {c: float(df[c].isna().mean()) for c in required if c in df},
    "creatinine_ratio_by_kdigo": table.to_dict(orient="records"),
    "figure_files": figure_files,
    "cluster_characteristics_path": cluster_characteristics_path,
    "cluster_mortality_path": cluster_mortality_path,
}
_save(summary, table, "creatinine_trajectory_kdigo", out_dir)
""",
    }
    body = bodies.get(task_key)
    if body is None:
        return None
    return textwrap.dedent(common + "\n" + body).strip() + "\n"


