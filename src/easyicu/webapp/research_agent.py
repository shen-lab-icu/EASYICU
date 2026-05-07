"""Streamlit page for the EasyICU research agent (ROADMAP T1.7).

This module hosts a single, self-contained page that lets a reviewer
or end user click through the full ICU-aware research-agent pipeline:

1. Pick a cohort — upload a parquet/CSV, point at one of the user's
   prior ``extract_database`` outputs in the workspace, or fall back
   to the synthetic SOFA cohort baked into ``examples/``.
2. Pick a :class:`ClinicalSkill` from the registry (or the canonical
   "free-form question" mode).
3. Optionally turn on the ICU-aware context (default) or the naive
   ablation arm (T1.4) for live A/B comparison.
4. Hit *Run* — the page invokes
   :class:`ResearchAgentPipeline.run` and streams progress + the
   resulting ``results_report.md``, bound manuscript scaffold,
   LaTeX export, and links to every registered evidence artefact.

The page can be used two ways:

* **Embedded**: import :func:`render_research_agent_page` and call it
  from inside the main webapp — this is what
  ``app.py`` does when the user clicks the "Research Agent" tab.

* **Standalone**: ``streamlit run src/easyicu/webapp/research_agent.py``
  — handy for the paper's reviewer demo because it boots without the
  rest of the webapp's session state.

Design choices worth documenting:

* The pipeline runs in-process, not in a thread or subprocess. The
  agent loop is fast under the MockLLMClient (≈2 s end-to-end on the
  synthetic cohort) and Streamlit's reactive model makes background
  threads fragile; we trade a few seconds of UI freeze for
  determinism.
* All run artefacts land under ``./research_output/webapp/<run_id>/``
  inside the user's current working directory, mirroring the
  ``examples/`` script. Reviewers can ``open`` the folder afterwards
  to inspect the raw evidence_index.json / sha256 hashes.
* Heavy imports (``streamlit``, ``pandas``) are intentionally
  performed at module top so the page is fast to import once. The
  agent imports themselves stay lazy, in line with the
  ``research_agent`` package's own import-cost rule.
"""

from __future__ import annotations

import html
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import streamlit as st

from easyicu.webapp.i18n import get_text
from easyicu.webapp.llm_config import (
    agent_config_from_shared_settings,
    ensure_llm_config_state,
    is_configured as is_shared_llm_configured,
    provider_defaults as shared_llm_provider_defaults,
)
from easyicu.webapp.page_header import render_page_header
from easyicu.webapp.session_state import clear_run_state


def _ra_text(key: str, **kwargs: Any) -> str:
    """Return localized Research Agent copy."""
    text = get_text(f"ra_{key}")
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


# ---------------------------------------------------------------------------
# Lazy-import gates (so a webapp without ``research_agent`` deps still loads)
# ---------------------------------------------------------------------------


def _import_agent_layer():
    """Import the research agent on demand and return its handles.

    Importing the parent ``easyicu`` package is expensive in some
    environments (downstream notebook integrations etc.); deferring
    avoids paying that cost just to render the page header.
    """
    from easyicu.research_agent import (  # type: ignore
        ResearchAgentPipeline,
        list_skills,
        get_skill,
    )
    from easyicu.research_agent.llm import MockLLMClient  # type: ignore

    try:
        from easyicu.research_agent.llm import OpenAIClient  # type: ignore
    except Exception:  # pragma: no cover - optional path
        OpenAIClient = None  # type: ignore

    return {
        "ResearchAgentPipeline": ResearchAgentPipeline,
        "MockLLMClient": MockLLMClient,
        "OpenAIClient": OpenAIClient,
        "list_skills": list_skills,
        "get_skill": get_skill,
    }


# ---------------------------------------------------------------------------
# Cohort-source helpers
# ---------------------------------------------------------------------------


def _build_synthetic_cohort(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """A trimmed copy of the demo synthetic cohort.

    Duplicated here (rather than imported from ``examples/``) so the
    page works when the source checkout is not on ``sys.path``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    age = rng.normal(65, 15, n).clip(18, 95)
    base = rng.integers(1, 14, size=n, endpoint=False)
    miss = rng.random(n) < 0.10
    truly_low = rng.random(n) < 0.05
    sofa2 = np.where(miss, 0, np.where(truly_low, 0, base))
    logit = -3.5 + 0.18 * sofa2 + 0.012 * (age - 65) + np.where(miss, 1.5, 0.0)
    p = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n) < p).astype(int)
    los = rng.gamma(2.0, 1.5 + 0.15 * sofa2, size=n).clip(0.1, 60)
    lact = rng.lognormal(0.4 + 0.08 * sofa2, 0.6, size=n).clip(0.5, 25)
    creat = rng.lognormal(0.05 + 0.04 * sofa2, 0.4, size=n).clip(0.1, 12)
    map_v = rng.normal(85 - 1.6 * sofa2, 12, size=n).clip(40, 130)
    vaso = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.5 + 0.20 * sofa2)))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age,
        "sex": rng.choice(["M", "F"], size=n),
        "sofa2": sofa2,
        "lact": lact,
        "creat": creat,
        "map": map_v,
        "vaso": vaso,
        "los_icu": los,
        "death": death,
    })


def _read_cohort_upload(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit ``UploadedFile`` into a DataFrame."""
    name = (uploaded_file.name or "").lower()
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    raise ValueError(f"Unsupported upload extension: {name}")


def _scan_workspace_for_cohorts(roots: List[Path]) -> List[Path]:
    """Walk a few candidate roots and return parquet files that look
    like aggregated cohorts (one row per stay)."""
    found: List[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        # Don't recurse the entire repo — limit to the depths that
        # ``extract_database`` writes to.
        for pq in root.glob("**/*.parquet"):
            try:
                # tiny heuristic: must have a stay_id-ish column
                head = pd.read_parquet(pq).head(1)
                cols = {c.lower() for c in head.columns}
                if cols & {"stay_id", "icustay_id", "patient_id", "subject_id"}:
                    found.append(pq.resolve())
            except Exception:
                continue
            if len(found) >= 30:
                return found
    return found


def _candidate_cohort_roots() -> List[Path]:
    cwd = Path.cwd()
    return [
        cwd / "research_output",
        cwd / "extracted",
        cwd / "outputs",
        cwd / "data" / "extracted",
    ]


_ID_COLUMN_CANDIDATES = (
    "stay_id", "icustay_id", "icu_stay_id", "patientunitstayid",
    "hadm_id", "subject_id", "patient_id", "uniquepid", "admissionid",
    "patientid", "CaseID",
)

_TIME_COLUMN_NAMES = {
    "charttime", "storetime", "starttime", "endtime", "intime", "outtime",
    "time", "timestamp", "event_time", "stay_id_time", "_time",
    "time_to_event",
}

_KNOWN_MODULE_DIR_NAMES = {
    "demographics", "outcome", "sofa2_score", "sofa1_score",
    "sepsis3_sofa2", "sepsis3_sofa1", "sepsis_shared", "vitals",
    "respiratory", "ventilator", "blood_gas", "chemistry",
    "hematology", "vasopressors", "medications", "renal",
    "neurological", "circulatory", "other_scores",
}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def _detect_id_columns(columns: Sequence[str]) -> List[str]:
    by_lower = {str(c).lower(): str(c) for c in columns}
    found: List[str] = []
    for c in _ID_COLUMN_CANDIDATES:
        hit = by_lower.get(c.lower())
        if hit and hit not in found:
            found.append(hit)
    return found


def _scan_workspace_for_module_dirs(roots: List[Path]) -> List[Path]:
    """Return candidate EasyICU export folders containing parquet modules."""
    out: List[Path] = []
    seen: Set[Path] = set()

    def _has_parquet(folder: Path) -> bool:
        try:
            if any(folder.glob("*.parquet")):
                return True
            return any(folder.glob("*/*.parquet"))
        except Exception:
            return False

    def _add(folder: Path) -> None:
        try:
            resolved = folder.resolve()
        except Exception:
            return
        if resolved in seen or not resolved.is_dir() or not _has_parquet(resolved):
            return
        seen.add(resolved)
        out.append(resolved)

    for root in roots:
        if not root:
            continue
        try:
            root = root.expanduser().resolve()
        except Exception:
            continue
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            continue
        child_export_dirs: List[Path] = []
        try:
            for child in sorted(root.iterdir()):
                if len(out) >= 50:
                    return out
                if child.is_dir() and child.name not in _KNOWN_MODULE_DIR_NAMES and _has_parquet(child):
                    child_export_dirs.append(child)
        except Exception:
            continue
        if child_export_dirs:
            for child in child_export_dirs:
                _add(child)
            if any(root.glob("*.parquet")):
                _add(root)
        else:
            _add(root)
    return out


def _list_module_parquets(folder: Path) -> List[Path]:
    try:
        return sorted(
            (p.resolve() for p in folder.rglob("*.parquet") if p.is_file()),
            key=lambda p: str(p.relative_to(folder)) if p.is_relative_to(folder) else str(p),
        )
    except Exception:
        return []


def _parquet_file_summary(path: Path) -> Dict[str, Any]:
    """Small metadata summary without loading the whole parquet when possible."""
    rows: Optional[int] = None
    columns: List[str] = []
    error: Optional[str] = None
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        rows = int(pf.metadata.num_rows) if pf.metadata is not None else None
        columns = [str(c) for c in pf.schema_arrow.names]
    except Exception as exc:
        try:
            df = pd.read_parquet(path)
            rows = int(len(df))
            columns = [str(c) for c in df.columns]
        except Exception as read_exc:
            error = f"{type(read_exc).__name__}: {read_exc}"
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    return {
        "path": path,
        "rows": rows,
        "columns": columns,
        "id_columns": _detect_id_columns(columns),
        "error": error,
    }


def _read_parquet_columns(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if columns:
        cols = list(dict.fromkeys(str(c) for c in columns if c))
        try:
            return pd.read_parquet(path, columns=cols)
        except Exception:
            df = pd.read_parquet(path)
            keep = [c for c in cols if c in df.columns]
            return df[keep].copy() if keep else df
    return pd.read_parquet(path)


def _safe_column_prefix(path: Path, folder: Path) -> str:
    try:
        stem = path.relative_to(folder).with_suffix("").as_posix()
    except Exception:
        stem = path.stem
    out = []
    for ch in stem:
        out.append(ch if ch.isalnum() else "_")
    prefix = "".join(out).strip("_")
    while "__" in prefix:
        prefix = prefix.replace("__", "_")
    return prefix or path.stem


def _unique_column_name(name: str, used: Set[str]) -> str:
    candidate = name
    i = 2
    while candidate in used:
        candidate = f"{name}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _truthy_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    true_text = {"1", "true", "t", "yes", "y", "positive", "pos"}
    return series.notna() & (
        (numeric.notna() & (numeric != 0)) | text.isin(true_text)
    )


def _filter_mask(series: pd.Series, mode: str, value: str) -> pd.Series:
    if mode == "nonzero / true":
        return _truthy_mask(series)
    if mode == "not null":
        return series.notna()
    if mode == "> 0":
        return pd.to_numeric(series, errors="coerce").fillna(0) > 0
    if mode == "contains":
        return series.astype(str).str.contains(value or "", case=False, na=False)
    # equals
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    numeric_series = pd.to_numeric(series, errors="coerce")
    if pd.notna(numeric_value):
        return numeric_series == numeric_value
    return series.astype(str).str.strip().str.lower() == (value or "").strip().lower()


def _infer_filter_defaults(
    summaries: Sequence[Dict[str, Any]],
    *,
    question: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    q = (question or "").lower()
    wants_sepsis = "sepsis" in q or "脓毒" in q or "感染" in q
    ranked: List[Tuple[int, Path, str]] = []
    for s in summaries:
        path = s.get("path")
        if not isinstance(path, Path):
            continue
        rel = str(path).lower()
        for col in s.get("columns") or []:
            lc = str(col).lower()
            score = 0
            if any(token in rel for token in ("sep3", "sepsis", "susp_inf")):
                score += 4
            if any(token in lc for token in ("sep3", "sepsis", "susp_inf", "infection")):
                score += 4
            if lc in {"sep3_sofa2", "sep3_sofa1", "has_sepsis", "sepsis"}:
                score += 3
            if wants_sepsis:
                score += 2
            if score:
                ranked.append((score, path, str(col)))
    if not ranked:
        return None, None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1], ranked[0][2]


def _filter_ids_from_module_file(
    *,
    path: Path,
    id_col: str,
    filter_col: str,
    mode: str,
    value: str,
) -> Set[Any]:
    summary = _parquet_file_summary(path)
    file_id = id_col if id_col in summary["columns"] else None
    if file_id is None:
        ids = summary.get("id_columns") or []
        file_id = ids[0] if ids else None
    if file_id is None or filter_col not in summary["columns"]:
        return set()
    df = _read_parquet_columns(path, [file_id, filter_col])
    mask = _filter_mask(df[filter_col], mode, value)
    return set(df.loc[mask, file_id].dropna().tolist())


def _reduce_module_file_to_stay_level(
    *,
    path: Path,
    folder: Path,
    id_col: str,
    keep_ids: Optional[Set[Any]],
    used_columns: Set[str],
) -> Optional[pd.DataFrame]:
    df = pd.read_parquet(path)
    if df.empty:
        return None
    file_id = id_col if id_col in df.columns else None
    if file_id is None:
        ids = _detect_id_columns([str(c) for c in df.columns])
        file_id = ids[0] if ids else None
    if file_id is None:
        return None

    if keep_ids is not None:
        df = df[df[file_id].isin(keep_ids)]
        if df.empty:
            return pd.DataFrame({id_col: list(keep_ids)})

    lower = {str(c): str(c).lower() for c in df.columns}
    time_cols = [c for c in df.columns if lower[str(c)] in _TIME_COLUMN_NAMES]
    excluded = set(_detect_id_columns([str(c) for c in df.columns])) | set(time_cols)
    value_cols = [c for c in df.columns if c not in excluded and c != file_id]
    if not value_cols:
        return None

    if time_cols:
        try:
            df = df.sort_values(time_cols)
        except Exception:
            pass
    sub = (
        df[[file_id] + value_cols]
        .dropna(subset=[file_id])
        .groupby(file_id, as_index=False)
        .last()
    )
    if file_id != id_col:
        sub = sub.rename(columns={file_id: id_col})

    prefix = _safe_column_prefix(path, folder)
    rename: Dict[str, str] = {}
    generic = {"value", "valuenum", "amount", "result", "measurement"}
    for col in value_cols:
        col_text = str(col)
        if len(value_cols) == 1 and col_text.lower() in generic:
            proposed = prefix
        elif len(value_cols) == 1 and col_text.lower() == path.stem.lower():
            proposed = col_text
        elif col_text in used_columns:
            proposed = f"{prefix}__{col_text}"
        else:
            proposed = col_text
        rename[col] = _unique_column_name(proposed, used_columns)
    return sub.rename(columns=rename)


def _build_stay_level_from_module_folder(
    *,
    folder: Path,
    selected_files: Sequence[Path],
    id_col: str,
    filter_spec: Optional[Tuple[Path, str, str, str]] = None,
    join_how: str = "outer",
) -> pd.DataFrame:
    keep_ids: Optional[Set[Any]] = None
    if filter_spec is not None:
        filter_path, filter_col, mode, value = filter_spec
        keep_ids = _filter_ids_from_module_file(
            path=filter_path,
            id_col=id_col,
            filter_col=filter_col,
            mode=mode,
            value=value,
        )
        if not keep_ids:
            return pd.DataFrame(columns=[id_col])

    used_columns: Set[str] = {id_col}
    merged: Optional[pd.DataFrame] = None
    how = "inner" if join_how == "inner" else "outer"
    for path in selected_files:
        sub = _reduce_module_file_to_stay_level(
            path=path,
            folder=folder,
            id_col=id_col,
            keep_ids=keep_ids,
            used_columns=used_columns,
        )
        if sub is None:
            continue
        merged = sub if merged is None else merged.merge(sub, on=id_col, how=how)

    if merged is None:
        merged = pd.DataFrame({id_col: sorted(keep_ids) if keep_ids is not None else []})
    if keep_ids is not None:
        merged = merged[merged[id_col].isin(keep_ids)]
    cols = [id_col] + [c for c in merged.columns if c != id_col]
    return merged[cols].reset_index(drop=True)


def _default_module_selection(labels: Sequence[str]) -> List[str]:
    priority = (
        "demographics", "outcome", "death", "sofa", "sepsis", "sep3",
        "vitals", "lact", "map",
    )
    selected = [label for label in labels if any(token in label.lower() for token in priority)]
    return selected[:10] or list(labels[:6])


def _available_extract_modules() -> Dict[str, List[str]]:
    try:
        from easyicu.api import EXTRACT_MODULES  # type: ignore

        return {str(k): list(v) for k, v in EXTRACT_MODULES.items()}
    except Exception:
        return {
            "demographics": ["age", "sex", "bmi"],
            "outcome": ["death", "los_icu", "los_hosp"],
            "sofa2_score": ["sofa2"],
            "sepsis3_sofa2": ["sep3_sofa2"],
            "vitals": ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp"],
            "blood_gas": ["lact", "ph", "pco2", "po2"],
        }


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _resolve_llm(
    handles: Dict[str, Any],
    llm_choice: str,
    *,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
):
    """Return an LLM client instance based on the user's pick.

    Supports any OpenAI-compatible chat-completions endpoint via
    ``base_url`` (OpenRouter, DeepSeek-direct, Together AI, local
    vLLM, etc.). When ``base_url`` is None we hit the default OpenAI
    endpoint.
    """
    if llm_choice == "MockLLMClient (offline, deterministic)":
        return handles["MockLLMClient"]()
    if llm_choice in {"OpenAI", "OpenAIClient", "OpenRouter", "Custom OpenAI-compatible"}:
        if handles["OpenAIClient"] is None:
            raise RuntimeError(
                "openai SDK is not installed. Install with `pip install openai` "
                "or pick the MockLLMClient option."
            )
        if not api_key:
            raise RuntimeError(
                f"An API key is required for {llm_choice}. Paste it in the "
                "LLM section, or set OPENAI_API_KEY / OPENROUTER_API_KEY in "
                "your environment."
            )
        import os
        # Set env so any down-stream code that reads it directly still
        # finds the key (e.g. OpenAIClient's own env-var fallback).
        env_var = "OPENROUTER_API_KEY" if llm_choice == "OpenRouter" else "OPENAI_API_KEY"
        os.environ[env_var] = api_key
        kwargs: Dict[str, Any] = dict(model=model, api_key=api_key)
        if base_url:
            kwargs["base_url"] = base_url
        if extra_headers:
            kwargs["extra_headers"] = dict(extra_headers)
        return handles["OpenAIClient"](**kwargs)
    raise RuntimeError(f"Unknown LLM choice: {llm_choice}")


def _run_pipeline(
    *,
    handles: Dict[str, Any],
    cohort: pd.DataFrame,
    skill_key: Optional[str],
    question: Optional[str],
    target_outcome: Optional[str],
    workdir: Path,
    llm,
    disable_icu_context: bool,
    notes: Optional[str] = None,
    stop_after_analysis: bool = False,
    resume_run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Invoke the pipeline; return the :class:`PipelineResult`."""
    pipeline = handles["ResearchAgentPipeline"](
        workdir=workdir, llm=llm, disable_icu_context=disable_icu_context,
    )
    kwargs: Dict[str, Any] = dict(
        cohort=cohort,
        cohort_name="webapp_cohort",
        database="webapp",
    )
    if skill_key:
        kwargs["skill"] = skill_key
    if question:
        kwargs["question"] = question
    if target_outcome:
        kwargs["target_outcome"] = target_outcome
    if notes:
        kwargs["notes"] = notes
    if resume_run_id:
        kwargs["resume_run_id"] = resume_run_id
    kwargs["stop_after_analysis"] = stop_after_analysis
    kwargs["progress_callback"] = progress_callback
    return pipeline.run(**kwargs)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------


_FINDING_BADGE = {
    "info": "🟢",
    "warning": "🟡",
    "error": "🔴",
}


def _render_findings(manifest: Dict[str, Any]) -> None:
    findings = manifest.get("findings", [])
    if not findings:
        st.info(_ra_text("no_findings"))
        return
    counts = {"info": 0, "warning": 0, "error": 0}
    for f in findings:
        counts[f.get("severity", "info")] = counts.get(f.get("severity", "info"), 0) + 1
    cols = st.columns(3)
    cols[0].metric("Errors", counts["error"], delta=None)
    cols[1].metric("Warnings", counts["warning"], delta=None)
    cols[2].metric("Info", counts["info"], delta=None)
    for f in findings:
        sev = f.get("severity", "info")
        msg = f.get("message", "")
        validator = f.get("validator", "?")
        st.markdown(f"{_FINDING_BADGE.get(sev, '⚪')} **`{validator}`** — {msg}")


def _render_evidence_table(run_dir: Path, manifest: Dict[str, Any]) -> None:
    rows = []
    for rec in manifest.get("evidence", []):
        rows.append({
            "evidence_id": rec.get("evidence_id"),
            "kind": rec.get("kind"),
            "description": rec.get("description"),
            "sha256 (head)": (rec.get("sha256") or "")[:10] + "…",
            "path": rec.get("relative_path"),
        })
    if not rows:
        st.info(_ra_text("no_evidence"))
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_figures(run_dir: Path, manifest: Dict[str, Any]) -> None:
    figs = [r for r in manifest.get("evidence", []) if r.get("kind") == "figure"]
    if not figs:
        st.info(_ra_text("no_figures"))
        return
    cols = st.columns(2)
    for i, rec in enumerate(figs):
        path = run_dir / rec.get("relative_path", "")
        if not path.exists():
            continue
        caption = f"{rec.get('description', '')} — sha256: {rec.get('sha256', '')[:8]}"
        with cols[i % 2]:
            try:
                st.image(str(path), caption=caption, use_container_width=True)
            except Exception:
                st.image(str(path), caption=caption)


def _render_run_outputs(result, run_dir: Path) -> None:
    manifest_path = Path(result.manifest_path)
    if not manifest_path.exists():
        st.error(_ra_text("manifest_missing", path=manifest_path))
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(_ra_text("manifest_parse_failed", error=exc))
        return

    paused_after_analysis = "paused_after_analysis" in str(manifest.get("notes") or "")
    st.success(_ra_text(
        "run_complete",
        run_id=result.run_id,
        evidence=result.evidence_count,
        findings=result.findings_count,
    ))
    if paused_after_analysis:
        st.info(_ra_text("paused_notice"))

    tab_labels = [
        _ra_text("tab_report"),
        _ra_text("tab_manuscript"),
        _ra_text("tab_evidence"),
        _ra_text("tab_debug"),
    ]
    tabs = st.tabs(tab_labels)

    # 1) Report
    with tabs[0]:
        report_path = Path(result.report_path)
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning(_ra_text("report_missing"))

    # 2) Manuscript (bound)
    with tabs[1]:
        mp = Path(result.manuscript_path)
        if mp.exists():
            text = mp.read_text(encoding="utf-8")
            if paused_after_analysis:
                st.info(_ra_text("manuscript_skipped"))
                if st.button(
                    _ra_text("draft_from_analysis"),
                    key=f"research_agent_draft_from_{result.run_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["research_agent_resume_run_id"] = result.run_id
                    st.session_state["research_agent_force_manuscript"] = True
                    st.rerun()
            missing = text.count("[evidence missing:")
            if missing:
                st.warning(_ra_text("missing_placeholders", missing=missing))
            elif not paused_after_analysis:
                st.success(_ra_text("no_missing_placeholders"))
            st.markdown(text)
            if not paused_after_analysis:
                st.download_button(
                    _ra_text("download_md"), data=text,
                    file_name="manuscript_scaffold_bound.md",
                    mime="text/markdown",
                )
        else:
            st.warning(_ra_text("bound_missing"))

    # 3) Evidence pack
    with tabs[2]:
        st.markdown(f"### {_ra_text('findings')}")
        _render_findings(manifest)
        st.markdown(f"### {_ra_text('figures')}")
        _render_figures(run_dir, manifest)
        st.markdown(f"### {_ra_text('tab_evidence')}")
        _render_evidence_table(run_dir, manifest)

    # 4) Debug artefacts
    with tabs[3]:
        st.markdown(f"### {_ra_text('latex')}")
        tex_path = run_dir / "manuscript_scaffold.tex"
        if paused_after_analysis:
            st.info(_ra_text("no_latex_paused"))
        elif tex_path.exists():
            tex = tex_path.read_text(encoding="utf-8")
            st.download_button(
                _ra_text("download_tex"), data=tex,
                file_name="manuscript_scaffold.tex", mime="text/x-tex",
            )
            st.code(tex, language="latex")
        else:
            st.info(_ra_text("no_latex"))
        st.markdown(f"### {_ra_text('manifest')}")
        st.json(manifest)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _stay_level_from_loaded_concepts(
    loaded_concepts: Dict[str, "pd.DataFrame"],
    *,
    id_col: str,
) -> Optional["pd.DataFrame"]:
    """Convert ``st.session_state.loaded_concepts`` into a stay-level frame.

    Each concept dataframe is reduced to **one row per stay** by taking
    the most recent value (or the value itself if already stay-level),
    then horizontally merged on ``id_col``. Drops empty / unparseable
    concepts silently rather than blocking the handoff.
    """
    if not loaded_concepts:
        return None
    base: Optional[pd.DataFrame] = None
    for concept, df in loaded_concepts.items():
        if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns:
            continue
        # Pick the value column (first non-id, non-time-like column).
        time_cols = {"charttime", "starttime", "endtime", "time", "timestamp",
                     "stay_id_time", "_time", "time_to_event"}
        value_cols = [
            c for c in df.columns
            if c != id_col and c.lower() not in time_cols
        ]
        if not value_cols:
            continue
        # Stay-level reduction: take the last non-null value per stay
        # for each value column. groupby+last is robust to time-series
        # input.
        try:
            sub = (df[[id_col] + value_cols]
                   .dropna(subset=[id_col])
                   .groupby(id_col, as_index=False)
                   .last())
        except Exception:
            continue
        # Rename pure value column to the concept name when there's
        # only one (prevents collisions across concepts).
        if len(value_cols) == 1 and value_cols[0] != concept:
            sub = sub.rename(columns={value_cols[0]: concept})
        base = sub if base is None else base.merge(sub, on=id_col, how="outer")
    return base


def _section_cohort_picker(
    *,
    research_question: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Render the cohort source UI and return ``(cohort, label)`` or ``(None, '')``."""
    # Detect a session-level cohort handoff from the rest of the webapp
    # (Tab 1 / Tab 3). When found, surface it as the *first* option so
    # the user's flow is one click instead of "export → re-upload here".
    loaded_concepts = st.session_state.get("loaded_concepts") or {}
    id_col = st.session_state.get("id_col", "stay_id")
    inbound = st.session_state.get("research_agent_inbound_cohort")
    has_inbound = isinstance(inbound, pd.DataFrame) and not inbound.empty
    has_loaded_concepts = bool(loaded_concepts)
    source_handoff = _ra_text("source_handoff")
    source_loaded = _ra_text("source_loaded")
    source_module = _ra_text("source_module_folder")
    source_no_data = _ra_text("source_no_data")
    source_upload = _ra_text("source_upload")
    source_workspace = _ra_text("source_workspace")
    source_synthetic = _ra_text("source_synthetic")

    options: List[str] = []
    if has_inbound:
        options.append(source_handoff)
    if has_loaded_concepts and not has_inbound:
        options.append(source_loaded)
    if st.session_state.get("entry_mode") == "real":
        options += [
            source_no_data,
            source_module,
            source_upload,
            source_workspace,
            source_synthetic,
        ]
    else:
        options += [
            source_synthetic,
            source_upload,
            source_module,
            source_workspace,
            source_no_data,
        ]
    if st.session_state.get("research_agent_cohort_source") not in (None, *options):
        st.session_state.pop("research_agent_cohort_source", None)
    source = st.radio(
        _ra_text("cohort_source"),
        options=options,
        horizontal=True,
        key="research_agent_cohort_source",
    )

    if source == source_handoff:
        df = inbound  # type: ignore[assignment]
        label = st.session_state.get("research_agent_inbound_cohort_label",
                                     "session-prepared cohort")
        st.success(_ra_text("use_handoff_success", rows=len(df), label=label))
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        return df, f"session:{label}"

    if source == source_loaded:
        with st.spinner(_ra_text("pivot_spinner")):
            df = _stay_level_from_loaded_concepts(loaded_concepts, id_col=id_col)
        if df is None or df.empty:
            st.error(_ra_text("pivot_error"))
            return None, ""
        st.success(_ra_text("pivot_success", concepts=len(loaded_concepts), rows=len(df), cols=df.shape[1]))
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        return df, f"session:loaded_concepts:{len(loaded_concepts)}"

    if source == source_synthetic:
        n = st.slider(_ra_text("synthetic_n"),
                      min_value=200, max_value=4000, value=800, step=100,
                      key="research_agent_synth_n")
        seed = st.number_input(_ra_text("seed"), value=7, step=1,
                               key="research_agent_synth_seed")
        df = _build_synthetic_cohort(n=int(n), seed=int(seed))
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        return df, f"synthetic_n={n}_seed={seed}"

    if source == source_upload:
        uploaded = st.file_uploader(
            _ra_text("upload_label"),
            type=["parquet", "pq", "csv", "tsv"],
            key="research_agent_upload",
        )
        if uploaded is None:
            st.info(_ra_text("upload_info"))
            return None, ""
        try:
            df = _read_cohort_upload(uploaded)
        except Exception as exc:
            st.error(_ra_text("upload_read_failed", error=exc))
            return None, ""
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        return df, f"upload:{uploaded.name}"

    if source == source_module:
        extra_roots: List[Path] = []
        export_path = st.session_state.get("export_path") or ""
        last_export = st.session_state.get("last_export_dir") or ""
        for p in (last_export, export_path):
            if p:
                try:
                    extra_roots.append(Path(p).expanduser().resolve())
                except Exception:
                    pass
        dirs = _scan_workspace_for_module_dirs(extra_roots + _candidate_cohort_roots())
        dir_labels = [_display_path(p) for p in dirs]
        manual_default = str(extra_roots[0]) if extra_roots else ""
        picked_label = ""
        if dir_labels:
            picked_label = st.selectbox(
                _ra_text("detected_folders"),
                [_ra_text("manual_path")] + dir_labels,
                index=1,
                key="research_agent_module_dir_pick",
            )
        folder_text = st.text_input(
            _ra_text("module_folder"),
            value=manual_default if picked_label in {"", _ra_text("manual_path")} else str(dirs[dir_labels.index(picked_label)]),
            key="research_agent_module_dir_text",
            help=_ra_text("module_folder_help"),
        )
        folder = Path(folder_text).expanduser().resolve() if folder_text else None
        if folder is None or not folder.exists() or not folder.is_dir():
            st.info(_ra_text("choose_module_folder"))
            return None, ""

        module_files = _list_module_parquets(folder)
        if not module_files:
            st.warning(_ra_text("no_module_parquets", folder=folder))
            return None, ""
        summaries = [_parquet_file_summary(p) for p in module_files]
        rows = []
        for s in summaries:
            path = s["path"]
            rel = str(path.relative_to(folder)) if path.is_relative_to(folder) else path.name
            rows.append({
                "file": rel,
                "rows": s.get("rows"),
                "id": ", ".join(s.get("id_columns") or []),
                "columns": ", ".join((s.get("columns") or [])[:8]),
                "error": s.get("error") or "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        id_counts: Dict[str, int] = {}
        for s in summaries:
            for c in s.get("id_columns") or []:
                id_counts[c] = id_counts.get(c, 0) + 1
        if not id_counts:
            st.error(_ra_text("no_common_id"))
            return None, ""
        id_options = sorted(id_counts, key=lambda c: (-id_counts[c], c))
        id_col = st.selectbox(
            _ra_text("merge_id"),
            id_options,
            key="research_agent_module_id_col",
            help=_ra_text("merge_id_help"),
        )

        labels = [
            str(p.relative_to(folder)) if p.is_relative_to(folder) else p.name
            for p in module_files
        ]
        selected_labels = st.multiselect(
            _ra_text("module_files"),
            labels,
            default=_default_module_selection(labels),
            key="research_agent_module_files",
        )
        selected_files = [module_files[labels.index(label)] for label in selected_labels]
        if not selected_files:
            st.info(_ra_text("select_module_file"))
            return None, ""

        default_filter_path, default_filter_col = _infer_filter_defaults(
            summaries,
            question=research_question,
        )
        use_filter_default = default_filter_path is not None and (
            "sepsis" in (research_question or "").lower()
            or "脓毒" in (research_question or "")
            or "感染" in (research_question or "")
        )
        use_filter = st.checkbox(
            _ra_text("filter_before_merge"),
            value=use_filter_default,
            key="research_agent_module_filter_enabled",
            help=_ra_text("filter_help"),
        )
        filter_spec: Optional[Tuple[Path, str, str, str]] = None
        if use_filter:
            filter_labels = labels
            default_idx = 0
            if default_filter_path is not None and default_filter_path in module_files:
                default_idx = module_files.index(default_filter_path)
            filter_label = st.selectbox(
                _ra_text("filter_file"),
                filter_labels,
                index=default_idx,
                key="research_agent_module_filter_file",
            )
            filter_path = module_files[filter_labels.index(filter_label)]
            filter_summary = _parquet_file_summary(filter_path)
            filter_cols = [
                c for c in filter_summary.get("columns") or []
                if c != id_col and str(c).lower() not in _TIME_COLUMN_NAMES
            ]
            if not filter_cols:
                st.warning(_ra_text("filter_no_columns"))
                return None, ""
            filter_col_index = 0
            if default_filter_col in filter_cols:
                filter_col_index = filter_cols.index(default_filter_col)
            filter_col = st.selectbox(
                _ra_text("filter_column"),
                filter_cols,
                index=filter_col_index,
                key="research_agent_module_filter_col",
            )
            mode = st.selectbox(
                _ra_text("filter_condition"),
                ["nonzero / true", "equals", "> 0", "not null", "contains"],
                key="research_agent_module_filter_mode",
            )
            value = ""
            if mode in {"equals", "contains"}:
                value = st.text_input(
                    _ra_text("filter_value"),
                    value="1" if mode == "equals" else "sepsis",
                    key="research_agent_module_filter_value",
                )
            filter_spec = (filter_path, filter_col, mode, value)

        join_how = st.radio(
            _ra_text("merge_strategy"),
            ["outer", "inner"],
            horizontal=True,
            key="research_agent_module_join",
            help=_ra_text("merge_strategy_help"),
        )
        filter_signature: Optional[Tuple[str, str, str, str]] = None
        if filter_spec is not None:
            filter_signature = (
                str(filter_spec[0]),
                filter_spec[1],
                filter_spec[2],
                filter_spec[3],
            )
        build_signature = {
            "folder": str(folder),
            "files": [str(p) for p in selected_files],
            "id_col": id_col,
            "filter": filter_signature,
            "join_how": join_how,
        }
        cached_build = st.session_state.get("research_agent_module_built")
        if (
            isinstance(cached_build, dict)
            and cached_build.get("signature") == build_signature
            and isinstance(cached_build.get("df"), pd.DataFrame)
        ):
            df = cached_build["df"]
            st.success(_ra_text("cached_build", rows=len(df), cols=df.shape[1]))
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)
            return df, f"module_folder:{folder}"

        st.info(_ra_text("build_info"))
        build_clicked = st.button(
            _ra_text("build_button"),
            type="primary",
            use_container_width=True,
            key="research_agent_module_build",
        )
        if not build_clicked:
            return None, ""

        try:
            with st.spinner(_ra_text("build_spinner")):
                df = _build_stay_level_from_module_folder(
                    folder=folder,
                    selected_files=selected_files,
                    id_col=id_col,
                    filter_spec=filter_spec,
                    join_how=join_how,
                )
        except Exception as exc:
            st.error(_ra_text("build_failed", error=exc))
            st.code(traceback.format_exc())
            return None, ""
        if df.empty:
            st.warning(_ra_text("empty_build"))
            return None, ""
        st.success(_ra_text("build_success", rows=len(df), cols=df.shape[1], files=len(selected_files)))
        st.session_state["research_agent_module_built"] = {
            "signature": build_signature,
            "df": df,
        }
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        return df, f"module_folder:{folder}"

    if source == source_no_data:
        modules = _available_extract_modules()
        st.info(_ra_text("no_data_info"))
        db = st.selectbox(
            _ra_text("database"),
            ["miiv", "mimic", "eicu", "aumc", "hirid", "sic", "mock"],
            index=0,
            key="research_agent_extract_db",
        )
        data_path = st.text_input(
            _ra_text("raw_path"),
            value=st.session_state.get("data_path", ""),
            key="research_agent_extract_data_path",
        )
        output_dir = st.text_input(
            _ra_text("output_folder"),
            value=st.session_state.get("export_path", str(Path.home() / "easyicu_export" / f"{db}_research_agent")),
            key="research_agent_extract_output_dir",
        )
        default_modules = [
            m for m in ["demographics", "outcome", "sofa2_score", "sepsis3_sofa2", "vitals", "blood_gas"]
            if m in modules
        ]
        picked_modules = st.multiselect(
            _ra_text("modules_extract"),
            list(modules.keys()),
            default=default_modules,
            key="research_agent_extract_modules",
        )
        max_patients = st.selectbox(
            _ra_text("patient_limit"),
            [100, 1000, 5000, 10000, 50000, 0],
            index=1,
            format_func=lambda x: _ra_text("all_patients") if x == 0 else f"{x:,}",
            key="research_agent_extract_patient_limit",
        )
        concepts = []
        for m in picked_modules:
            concepts.extend(modules.get(m, []))
        concepts = list(dict.fromkeys(concepts))
        st.caption(_ra_text("selected_modules", modules=len(picked_modules), concepts=len(concepts)))
        if st.button(
            _ra_text("start_export"),
            type="primary",
            use_container_width=True,
            key="research_agent_start_export",
            disabled=not picked_modules or (db != "mock" and not data_path),
        ):
            Path(output_dir).expanduser().mkdir(parents=True, exist_ok=True)
            st.session_state.database = db
            st.session_state.data_path = data_path
            st.session_state.export_path = output_dir
            st.session_state.selected_concepts = concepts
            st.session_state.step3_confirmed = True
            st.session_state.export_format = "Parquet"
            st.session_state.patient_limit = max_patients
            st.session_state.trigger_export = True
            st.session_state.export_completed = False
            st.session_state["_exporting_in_progress"] = True
            st.session_state["_scroll_to_tab"] = "tutorial"
            st.success(_ra_text("export_queued"))
            st.rerun()
        return None, ""

    # Workspace pick — also include the user's configured export_path
    # so freshly-exported parquets show up without a path edit.
    extra_roots: List[Path] = []
    export_path = st.session_state.get("export_path") or ""
    last_export = st.session_state.get("last_export_dir") or ""
    for p in (export_path, last_export):
        if p:
            try:
                extra_roots.append(Path(p).resolve())
            except Exception:
                pass
    candidates = _scan_workspace_for_cohorts(_candidate_cohort_roots() + extra_roots)
    if not candidates:
        st.info(_ra_text("workspace_none"))
        return None, ""
    rels = [str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p)
            for p in candidates]
    pick = st.selectbox(_ra_text("workspace_pick"), rels, key="research_agent_workspace_pick")
    chosen = candidates[rels.index(pick)]
    try:
        df = pd.read_parquet(chosen)
    except Exception as exc:
        st.error(_ra_text("workspace_read_failed", path=chosen, error=exc))
        return None, ""
    st.caption(_ra_text("loaded_from", rows=len(df), path=chosen))
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    return df, f"workspace:{chosen}"


def _section_skill_picker(handles: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Render the skill / question / outcome inputs.

    Returns ``(skill_key, free_form_question, target_outcome)``. At
    most one of ``skill_key`` and ``free_form_question`` will be set.
    """
    skills = handles["list_skills"]()
    free_choice = _ra_text("free_question_choice")
    skill_choices = [free_choice] + [
        f"{s.key} — {s.description}" for s in skills
    ]
    pick = st.selectbox(_ra_text("skill_pick"), skill_choices, index=1 if skills else 0,
                        key="research_agent_skill_pick")
    if pick == free_choice:
        question = st.text_area(
            _ra_text("question"),
            value="Is admission SOFA-2 score associated with ICU mortality?",
            help=_ra_text("question_help"),
            key="research_agent_question",
        )
        target_outcome = st.text_input(
            _ra_text("target_outcome"), value="death",
            help=_ra_text("target_outcome_help"),
            key="research_agent_target_outcome",
        )
        return None, question.strip() or None, target_outcome.strip() or None

    skill_key = pick.split(" — ", 1)[0]
    skill = handles["get_skill"](skill_key)
    st.caption(_ra_text(
        "skill_caption",
        skill=skill.key,
        outcome=skill.target_outcome,
        predictor=skill.primary_predictor,
        variables=", ".join(skill.expected_variables),
    ))
    return skill_key, None, None


def _section_method_preferences() -> str:
    """Collect optional analysis preferences and render them as run notes."""
    method_options = [
        "Logistic regression",
        "Train/test split + AUC + Brier + calibration",
        "Random forest",
        "Cox PH",
        "Propensity score",
        "Missingness audit",
    ]
    methods = st.multiselect(
        _ra_text("methods"),
        method_options,
        default=[
            "Logistic regression",
            "Train/test split + AUC + Brier + calibration",
            "Missingness audit",
        ],
        key="research_agent_method_preferences",
        help=_ra_text("methods_help"),
    )
    covariates = st.text_input(
        _ra_text("covariates"),
        value=st.session_state.get("research_agent_covariates", "age, sex"),
        key="research_agent_covariates",
    )
    extra = st.text_area(
        _ra_text("extra_notes"),
        value="",
        height=80,
        key="research_agent_extra_notes",
    )
    notes: List[str] = []
    if methods:
        notes.append("User method preferences: " + "; ".join(methods))
    if covariates.strip():
        notes.append("User requested covariates: " + covariates.strip())
    if extra.strip():
        notes.append("User notes: " + extra.strip())
    return "\n".join(notes)


def _section_llm_picker(handles: Dict[str, Any]) -> Tuple[str, str, str, Optional[str], Optional[Dict[str, str]]]:
    """LLM choice + API key + model + base_url + extra_headers.

    Three preset paths plus a free-form custom endpoint:

    * **MockLLMClient** — deterministic offline pipeline (default).
    * **OpenAI** — OpenAI proper. Reads ``OPENAI_API_KEY`` from env /
      streamlit secrets if available.
    * **OpenRouter** — any OpenRouter model. Reads
      ``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` /
      ``EASYICU_HOSTED_DEFAULT_MODEL`` from env. Auto-attaches the
      ``HTTP-Referer`` / ``X-Title`` headers OpenRouter recommends.
    * **Custom OpenAI-compatible** — bring your own base_url
      (DeepSeek, vLLM, Together AI, Anyscale, etc.).
    """
    import os
    ensure_llm_config_state()
    mock_choice = _ra_text("llm_mock")
    sidebar_choice = _ra_text("llm_sidebar")
    override_choice = _ra_text("llm_override")
    options = [mock_choice]
    sdk_ok = handles["OpenAIClient"] is not None
    if sdk_ok:
        options = [sidebar_choice, mock_choice, override_choice]
    else:
        st.caption(_ra_text("sdk_missing"))
    if st.session_state.get("research_agent_llm_choice") not in (None, *options):
        st.session_state.pop("research_agent_llm_choice", None)
    default_index = 0 if sdk_ok and is_shared_llm_configured() else options.index(mock_choice)
    choice = st.radio(
        _ra_text("llm_client"),
        options,
        index=default_index,
        key="research_agent_llm_choice",
    )

    if choice == mock_choice:
        return "MockLLMClient (offline, deterministic)", "", "", None, None

    if choice == sidebar_choice:
        cfg = agent_config_from_shared_settings()
        provider = st.session_state.get("llm_provider", "")
        _, default_url, _default_model, _needs_key, _desc_en, _desc_zh = shared_llm_provider_defaults(provider)
        if not is_shared_llm_configured():
            st.warning(_ra_text("sidebar_not_ready"))
        st.caption(
            f"{cfg.source_label} · {cfg.model or 'model'} · "
            f"{cfg.base_url or default_url or 'default endpoint'}"
        )
        return cfg.choice, cfg.api_key, cfg.model, cfg.base_url, cfg.extra_headers

    st.caption(_ra_text("llm_override_hint"))

    # Helpers to safely read defaults from env vars or st.secrets.
    def _env_or_secret(*names: str, default: str = "") -> str:
        for n in names:
            v = os.environ.get(n)
            if v:
                return v
        try:  # pragma: no cover — streamlit-secrets path
            if hasattr(st, "secrets"):
                for n in names:
                    v = st.secrets.get(n, "") if hasattr(st.secrets, "get") else ""
                    if v:
                        return v
        except Exception:
            pass
        return default

    override_options = ["OpenAI", "OpenRouter", "Custom OpenAI-compatible"]
    override_client = st.selectbox(
        _ra_text("llm_client"),
        override_options,
        key="research_agent_llm_override_client",
    )
    api_key, model, base_url, extra_headers = "", "", None, None

    if override_client == "OpenAI":
        default_key = _env_or_secret("OPENAI_API_KEY")
        default_model = _env_or_secret("EASYICU_OPENAI_MODEL") or "gpt-4o-mini"
        api_key = st.text_input(
            _ra_text("api_key"),
            value=default_key, type="password",
            key="research_agent_openai_key",
        )
        model = st.text_input(
            _ra_text("model"), value=default_model,
            key="research_agent_openai_model",
        )

    elif override_client == "OpenRouter":
        default_key = _env_or_secret("OPENROUTER_API_KEY")
        default_url = _env_or_secret(
            "OPENROUTER_BASE_URL",
            default="https://openrouter.ai/api/v1",
        )
        default_model = _env_or_secret(
            "EASYICU_HOSTED_DEFAULT_MODEL",
            "EASYICU_SMOKE_MODEL",
            default="z-ai/glm-4.5-air:free",
        )
        st.caption(_ra_text("openrouter_caption"))
        api_key = st.text_input(
            _ra_text("api_key"),
            value=default_key, type="password",
            key="research_agent_openrouter_key",
        )
        base_url = st.text_input(
            _ra_text("base_url"),
            value=default_url,
            key="research_agent_openrouter_base_url",
        )
        # Sensible quick-pick presets users can override by typing.
        preset_models = [
            "z-ai/glm-4.5-air:free",
            "deepseek/deepseek-chat-v3.1:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "deepseek/deepseek-chat",
            "anthropic/claude-3.5-haiku",
            "openai/gpt-4o-mini",
        ]
        if default_model not in preset_models:
            preset_models = [default_model] + preset_models
        model = st.selectbox(
            _ra_text("model"),
            preset_models,
            index=preset_models.index(default_model) if default_model in preset_models else 0,
            key="research_agent_openrouter_model",
        )
        extra_headers = {
            "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
            "X-Title": "EasyICU research-agent webapp",
        }

    elif override_client == "Custom OpenAI-compatible":
        default_key = _env_or_secret("OPENAI_API_KEY", "OPENROUTER_API_KEY")
        default_url = _env_or_secret(
            "OPENAI_BASE_URL", default="https://api.deepseek.com/v1"
        )
        default_model = _env_or_secret(
            "EASYICU_CUSTOM_MODEL", default="deepseek-chat"
        )
        st.caption(_ra_text("custom_caption"))
        api_key = st.text_input(
            _ra_text("api_key"),
            value=default_key, type="password",
            key="research_agent_custom_key",
        )
        base_url = st.text_input(
            _ra_text("base_url"),
            value=default_url,
            key="research_agent_custom_base_url",
        )
        model = st.text_input(
            _ra_text("model"),
            value=default_model,
            key="research_agent_custom_model",
        )

    return override_client, api_key, model, base_url, extra_headers


def _section_options() -> Tuple[bool, str, bool]:
    cols = st.columns(3)
    with cols[0]:
        disable_icu_context = st.checkbox(
            _ra_text("disable_context"),
            value=False,
            help=_ra_text("disable_context_help"),
            key="research_agent_disable_icu",
        )
    with cols[1]:
        stop_options = [_ra_text("stop_analysis"), _ra_text("stop_manuscript")]
        if st.session_state.get("research_agent_stop_after") not in (None, *stop_options):
            st.session_state.pop("research_agent_stop_after", None)
        stop_choice = st.radio(
            _ra_text("stop_after"),
            stop_options,
            index=0,
            key="research_agent_stop_after",
        )
    with cols[2]:
        workdir_text = st.text_input(
            _ra_text("workdir"),
            value=str((Path.cwd() / "research_output" / "webapp").resolve()),
            key="research_agent_workdir",
        )
    return disable_icu_context, workdir_text, stop_choice == stop_options[0]


def _render_research_agent_demo_visuals(*, is_en: bool) -> None:
    """Render a static, non-token demo of the Research Agent outputs."""
    flow = [
        (
            "01",
            "Research question" if is_en else "研究问题",
            "Sepsis-3 ICU stays -> in-hospital mortality prediction"
            if is_en else
            "Sepsis-3 ICU 患者 -> 院内死亡预测",
            "",
        ),
        (
            "02",
            "Data recipe" if is_en else "数据配方",
            "Sepsis flag, demographics, SOFA-2, vitals, labs, outcome"
            if is_en else
            "脓毒症标记、人口学、SOFA-2、生命体征、实验室、结局",
            "",
        ),
        (
            "03",
            "Analysis pack" if is_en else "分析包",
            "Table 1, missingness audit, AUROC, Brier, calibration, findings"
            if is_en else
            "表 1、缺失审计、AUROC、Brier、校准、结果发现",
            "",
        ),
        (
            "04",
            "Review gate" if is_en else "复核关口",
            "User decides whether the evidence is strong enough to draft"
            if is_en else
            "用户判断证据是否值得继续生成文章",
            " review",
        ),
    ]
    flow_html = ""
    for idx, (label, title, body, klass) in enumerate(flow):
        if idx:
            flow_html += '<div class="ra-demo-arrow">→</div>'
        flow_html += (
            f'<div class="ra-demo-node{klass}">'
            f'<div class="ra-demo-node-label">{html.escape(label)}</div>'
            f'<div class="ra-demo-node-title">{html.escape(title)}</div>'
            f'<div class="ra-demo-node-body">{html.escape(body)}</div>'
            '</div>'
        )

    table_rows = [
        ("Age", "65.2", "68.9", "+3.7"),
        ("SOFA-2", "5.1", "8.6", "+3.5"),
        ("Lactate", "2.1", "3.8", "+1.7"),
        ("Vasopressor", "28%", "54%", "+26%"),
    ]
    table_html = "".join(
        "<tr>"
        f"<td>{html.escape(row[0])}</td>"
        f"<td>{html.escape(row[1])}</td>"
        f"<td>{html.escape(row[2])}</td>"
        f"<td>{html.escape(row[3])}</td>"
        "</tr>"
        for row in table_rows
    )
    manuscript = (
        "Among Sepsis-3 ICU stays, the analysis pack suggested clinically meaningful mortality risk separation. "
        "The draft is only generated after reviewing cohort balance, missingness, discrimination, and calibration."
        if is_en else
        "在 Sepsis-3 ICU 队列中，分析包显示死亡风险存在具有临床意义的分层。只有在复核队列构成、缺失、区分度和校准后，才进入文章生成。"
    )
    st.markdown(
        f"""
        <div class="ra-demo-hero">
            <div class="ra-demo-flow">{flow_html}</div>
        </div>
        <div class="ra-output-grid">
            <div class="ra-output-card">
                <div class="ra-output-title">{"Table 1 preview" if is_en else "表 1 预览"}</div>
                <div class="ra-output-note">{"Illustrative values only. Real runs bind tables to evidence files." if is_en else "仅为示意值；真实运行会绑定证据文件。"}</div>
                <table class="ra-mini-table">
                    <thead><tr><th>{"Feature" if is_en else "变量"}</th><th>{"Alive" if is_en else "存活"}</th><th>{"Died" if is_en else "死亡"}</th><th>Δ</th></tr></thead>
                    <tbody>{table_html}</tbody>
                </table>
            </div>
            <div class="ra-output-card">
                <div class="ra-output-title">{"Discrimination" if is_en else "区分度"}</div>
                <div class="ra-output-note">AUROC 0.82 · Brier 0.14</div>
                <svg viewBox="0 0 220 128" width="100%" height="128" role="img" aria-label="ROC curve">
                    <rect x="0" y="0" width="220" height="128" rx="10" fill="#f8fbff"/>
                    <line x1="28" y1="102" x2="196" y2="102" stroke="#cbd5e1" stroke-width="1.5"/>
                    <line x1="28" y1="102" x2="28" y2="18" stroke="#cbd5e1" stroke-width="1.5"/>
                    <line x1="28" y1="102" x2="196" y2="18" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4"/>
                    <polyline points="28,102 44,76 61,58 83,42 113,31 150,24 196,18" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round"/>
                    <text x="36" y="26" fill="#082957" font-size="12" font-weight="700">ROC</text>
                </svg>
            </div>
            <div class="ra-output-card">
                <div class="ra-output-title">{"Calibration" if is_en else "校准"}</div>
                <div class="ra-output-note">{"Review predicted vs observed risk." if is_en else "复核预测风险与实际风险。"}</div>
                <svg viewBox="0 0 220 128" width="100%" height="128" role="img" aria-label="Calibration curve">
                    <rect x="0" y="0" width="220" height="128" rx="10" fill="#f8fbff"/>
                    <line x1="28" y1="102" x2="196" y2="18" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4"/>
                    <polyline points="28,98 57,82 87,68 117,54 151,38 196,24" fill="none" stroke="#0f766e" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="57" cy="82" r="4" fill="#0f766e"/>
                    <circle cx="117" cy="54" r="4" fill="#0f766e"/>
                    <circle cx="196" cy="24" r="4" fill="#0f766e"/>
                    <text x="36" y="26" fill="#082957" font-size="12" font-weight="700">calibration</text>
                </svg>
            </div>
            <div class="ra-output-card wide">
                <div class="ra-output-title">{"Findings before manuscript" if is_en else "文章前的结果复核"}</div>
                <div class="ra-output-note">{"The agent stops here by default so users can catch wrong cohorts, weak signal, or bad calibration before spending writing tokens." if is_en else "智能体默认停在这里，便于用户在消耗写作 token 前发现队列错误、信号不足或校准较差。"}</div>
                <div class="ra-finding">{"Finding: SOFA-2 and lactate carry most of the risk signal; calibration should be checked in the high-risk decile." if is_en else "发现：SOFA-2 和乳酸贡献了主要风险信号；高风险分位的校准需要重点复核。"}</div>
                <div class="ra-finding">{"Validator: missingness acceptable for core predictors; manuscript drafting can be considered after sensitivity review." if is_en else "验证器：核心预测变量缺失可接受；完成敏感性复核后可考虑生成文章。"}</div>
            </div>
            <div class="ra-output-card">
                <div class="ra-output-title">{"Optional manuscript preview" if is_en else "可选文章预览"}</div>
                <div class="ra-manuscript-preview">{html.escape(manuscript)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_agent_demo_page() -> None:
    """Render a guide-only Research Agent page for Demo Mode.

    Demo mode should explain the research-agent value proposition without
    asking for API keys, work directories, or a real pipeline run. The full
    execution workflow remains available in Real Data Mode.
    """
    lang = st.session_state.get("language", "en")
    is_en = lang == "en"

    caption = (
        "Demo Mode is a lightweight preview. It shows what the agent can produce, "
        "without running an LLM pipeline or using tokens."
        if is_en else
        "演示模式只做轻量导览：展示智能体能产出什么，不真正运行 LLM pipeline，也不消耗 token。"
    )
    render_page_header(
        _ra_text("header"),
        caption,
        icon="🧪",
        kicker=_ra_text("kicker"),
    )

    st.info(
        "Use Real Data Mode when you are ready to connect a stay-level file, an EasyICU module export folder, "
        "or let EasyICU prepare data first."
        if is_en else
        "准备接入 stay-level 文件、EasyICU 模块导出文件夹，或让 EasyICU 先提取数据时，再进入真实数据模式。"
    )
    _render_research_agent_demo_visuals(is_en=is_en)

    overview_items = [
        (
            "Study plan" if is_en else "研究方案",
            "Question -> cohort, exposure, outcome, covariates, and analysis steps."
            if is_en else
            "把问题转成队列、暴露、结局、协变量和分析步骤。"
        ),
        (
            "Data recipe" if is_en else "数据配方",
            "Stay-level parquet, module export folder, or guided EasyICU extraction."
            if is_en else
            "支持 stay-level parquet、模块导出文件夹，或引导 EasyICU 先提取。"
        ),
        (
            "Analysis pack" if is_en else "分析包",
            "Tables, figures, missingness audit, model metrics, calibration, and findings."
            if is_en else
            "表格、图、缺失审计、模型指标、校准曲线和结果摘要。"
        ),
        (
            "Manuscript draft" if is_en else "文章初稿",
            "Generated only after the user reviews the analysis output."
            if is_en else
            "用户先看分析结果，确认后再生成文章初稿。"
        ),
    ]
    cols = st.columns(4)
    for col, (item_title, item_body) in zip(cols, overview_items):
        with col:
            st.markdown(f"**{item_title}**")
            st.caption(item_body)

    st.divider()
    example_title = (
        "Example: sepsis mortality prediction"
        if is_en else
        "示例：脓毒症患者死亡预测"
    )
    st.markdown(f"### {example_title}")

    left, right = st.columns([1.05, 1.25])
    with left:
        st.markdown(
            "\n".join([
                "**Research question**" if is_en else "**研究问题**",
                "Can we predict in-hospital mortality among Sepsis-3 ICU stays?"
                if is_en else
                "能否预测 Sepsis-3 ICU 患者的院内死亡风险？",
                "",
                "**Expected data modules**" if is_en else "**预期数据模块**",
                "- sepsis / suspicion of infection",
                "- demographics",
                "- SOFA or SOFA-2 scores",
                "- vital signs and laboratory summaries",
                "- outcome: death in hospital",
            ])
        )
    with right:
        outputs = pd.DataFrame([
            {
                "Output" if is_en else "产出": "Table 1",
                "Review target" if is_en else "复核重点": "Cohort balance, missingness, outcome rate" if is_en else "队列构成、缺失、死亡率",
            },
            {
                "Output" if is_en else "产出": "Model metrics",
                "Review target" if is_en else "复核重点": "AUROC, Brier score, calibration" if is_en else "AUROC、Brier、校准",
            },
            {
                "Output" if is_en else "产出": "Figures",
                "Review target" if is_en else "复核重点": "Calibration, ROC, feature effects" if is_en else "校准图、ROC、特征效应",
            },
            {
                "Output" if is_en else "产出": "Findings",
                "Review target" if is_en else "复核重点": "Whether the result is worth drafting" if is_en else "判断是否值得继续写文章",
            },
        ])
        st.dataframe(outputs, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown(
        "### Real-data workflow" if is_en else "### 真实数据流程"
    )
    st.markdown(
        "\n".join([
            "1. Choose an existing stay-level file, an EasyICU module export folder, or the no-data extraction path."
            if is_en else
            "1. 选择已有 stay-level 文件、EasyICU 模块导出文件夹，或走“尚未准备数据”的提取路径。",
            "2. Customize methods, covariates, cohort filters, and output stopping point."
            if is_en else
            "2. 定制方法、协变量、队列筛选和停止点。",
            "3. Run analysis first, review tables and figures, then decide whether to draft a manuscript."
            if is_en else
            "3. 先跑分析并查看表格/图，再决定是否生成文章。"
        ])
    )

    if st.button(
        "Switch to Real Data Mode" if is_en else "切换到真实数据模式",
        type="primary",
        use_container_width=True,
        key="research_agent_demo_to_real",
    ):
        clear_run_state("all")
        st.session_state.entry_mode = "real"
        st.session_state.use_mock_data = False
        st.session_state.loaded_concepts = {}
        st.session_state.loaded_data_origin = "none"
        st.session_state.patient_ids = []
        st.rerun()


def render_research_agent_page() -> None:
    """Top-level entry point used by the main webapp."""
    render_page_header(
        _ra_text("header"),
        _ra_text("subheader"),
        icon="🧪",
        kicker=_ra_text("kicker"),
    )

    try:
        handles = _import_agent_layer()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not import easyicu.research_agent: {exc}")
        st.code(traceback.format_exc())
        return

    with st.expander(_ra_text("step1_title"), expanded=True):
        skill_key, free_question, target_outcome = _section_skill_picker(handles)
    with st.expander(_ra_text("step2_title"), expanded=False):
        method_notes = _section_method_preferences()
    question_hint = free_question
    if skill_key and question_hint is None:
        try:
            question_hint = handles["get_skill"](skill_key).question_for(database="webapp")
        except Exception:
            question_hint = None
    with st.expander(_ra_text("step3_title"), expanded=True):
        cohort, cohort_label = _section_cohort_picker(research_question=question_hint)
    with st.expander(_ra_text("step4_title"), expanded=False):
        llm_choice, api_key, model, base_url, extra_headers = _section_llm_picker(handles)
    with st.expander(_ra_text("step5_title"), expanded=False):
        disable_icu_context, workdir_text, stop_after_analysis = _section_options()
    resume_run_id = st.session_state.get("research_agent_resume_run_id")
    force_manuscript = bool(st.session_state.get("research_agent_force_manuscript"))
    if force_manuscript:
        stop_after_analysis = False

    st.divider()
    run_clicked = st.button(
        "▶  " + (_ra_text("draft_button") if force_manuscript else _ra_text("run_button")),
        type="primary",
        disabled=cohort is None,
        use_container_width=True,
    ) or force_manuscript

    if cohort is None:
        st.info(_ra_text("select_cohort"))
        return

    if not run_clicked:
        # Persist the last result across reruns so users can flip tabs / themes
        # without losing their output.
        last = st.session_state.get("research_agent_last_result")
        if last:
            st.divider()
            st.markdown(f"### {_ra_text('last_run')}: `{last['run_id']}` "
                        f"({cohort_label} → {last['skill_or_question']})")
            _render_run_outputs(last["result"], Path(last["workdir"]))
        return

    # Resolve the LLM and execute the pipeline.
    workdir = Path(workdir_text).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        llm = _resolve_llm(
            handles, llm_choice,
            api_key=api_key, model=model,
            base_url=base_url, extra_headers=extra_headers,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    progress = st.empty()
    progress.info(_ra_text("running"))
    progress_bar = st.progress(0)
    progress_log = st.empty()
    progress_events: List[Dict[str, Any]] = []

    def _on_progress(event: Dict[str, Any]) -> None:
        progress_events.append(event)
        total = event.get("total_steps")
        current = event.get("current_step")
        status = event.get("status")
        if total and current:
            try:
                progress_bar.progress(min(float(current) / float(total), 1.0))
            except Exception:
                pass
        elif event.get("stage") == "run" and status == "complete":
            progress_bar.progress(1.0)
        badge = {"complete": "✅", "error": "🔴", "paused": "⏸️"}.get(status, "⚙️")
        lines = [
            f"{badge} **{e.get('stage', 'step')}** — {e.get('message', '')}"
            for e in progress_events[-8:]
        ]
        progress_log.markdown("\n".join(f"- {line}" for line in lines))

    try:
        with st.spinner(_ra_text("spinner")):
            result = _run_pipeline(
                handles=handles,
                cohort=cohort,
                skill_key=skill_key,
                question=free_question,
                target_outcome=target_outcome,
                workdir=workdir,
                llm=llm,
                disable_icu_context=disable_icu_context,
                notes=method_notes,
                stop_after_analysis=stop_after_analysis,
                resume_run_id=resume_run_id if force_manuscript else None,
                progress_callback=_on_progress,
            )
    except Exception as exc:
        progress.empty()
        st.error(_ra_text("failed", error=exc))
        st.code(traceback.format_exc())
        return
    progress.empty()
    progress_bar.empty()
    if force_manuscript:
        st.session_state.pop("research_agent_resume_run_id", None)
        st.session_state.pop("research_agent_force_manuscript", None)

    st.session_state["research_agent_last_result"] = {
        "run_id": result.run_id,
        "workdir": result.workdir,
        "result": result,
        "skill_or_question": skill_key or (free_question or "")[:60] + "…",
        "stop_after_analysis": stop_after_analysis,
    }
    _render_run_outputs(result, Path(result.workdir))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def _standalone_main() -> None:
    """Make this module runnable via ``streamlit run …/research_agent.py``."""
    st.set_page_config(
        page_title="EasyICU Research Agent",
        page_icon="🧪",
        layout="wide",
    )
    # When run standalone, allow the user to point us at a checkout root
    # in case the package was installed in editable mode.
    here = Path(__file__).resolve()
    src_root = here.parents[2]  # …/src
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    render_research_agent_page()


if __name__ == "__main__":  # pragma: no cover
    # ``streamlit run …/research_agent.py`` executes this module as
    # ``__main__``; that's the only path on which we need to call
    # set_page_config and bootstrap sys.path.
    _standalone_main()


__all__ = ["render_research_agent_demo_page", "render_research_agent_page"]
