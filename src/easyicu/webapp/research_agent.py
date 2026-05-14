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
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
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
from easyicu.webapp.data_paths import _directory_input
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
    from easyicu.research_agent.llm import (  # type: ignore
        FallbackLLMClient,
        LLMRouter,
        MockLLMClient,
    )

    try:
        from easyicu.research_agent.llm import OpenAIClient  # type: ignore
    except Exception:  # pragma: no cover - optional path
        OpenAIClient = None  # type: ignore

    return {
        "ResearchAgentPipeline": ResearchAgentPipeline,
        "MockLLMClient": MockLLMClient,
        "OpenAIClient": OpenAIClient,
        "LLMRouter": LLMRouter,
        "FallbackLLMClient": FallbackLLMClient,
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


def _placeholder_path(name: str) -> str:
    if sys.platform.startswith("win"):
        return f"D:\\path\\to\\{name.replace('/', '\\\\')}"
    return f"/path/to/{name}"


def _hide_prefilled_directory_text(input_key: str, mirrored_value: str) -> None:
    pending_key = f"{input_key}__pending_value"
    current = str(st.session_state.get(input_key, "") or "")
    if pending_key in st.session_state:
        return
    if mirrored_value and current == str(mirrored_value):
        st.session_state[input_key] = ""


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


def _read_module_file(
    *,
    path: Path,
    folder: Path,
    id_col: str,
    keep_ids: Optional[Set[Any]],
    used_columns: Set[str],
    canonical_time: str = "charttime",
) -> Optional[Tuple[pd.DataFrame, bool]]:
    """Read one module parquet and return ``(df, is_temporal)``.

    * **Temporal** files have at least one time column.  The first detected
      time column is renamed to *canonical_time* so all temporal files share
      the same key name when merged.
    * **Static** files (demographics, outcomes, …) have no time column and
      return a single row per patient.

    Returns ``None`` when the file cannot yield usable columns.
    """
    df = pd.read_parquet(path)
    if df.empty:
        return None

    # --- resolve id column ------------------------------------------------
    file_id = id_col if id_col in df.columns else None
    if file_id is None:
        ids = _detect_id_columns([str(c) for c in df.columns])
        file_id = ids[0] if ids else None
    if file_id is None:
        return None

    if keep_ids is not None:
        df = df[df[file_id].isin(keep_ids)]
        if df.empty:
            return None

    # --- detect time column -----------------------------------------------
    lower = {str(c): str(c).lower() for c in df.columns}
    time_cols = [c for c in df.columns if lower[str(c)] in _TIME_COLUMN_NAMES]
    is_temporal = len(time_cols) > 0

    # --- value columns (exclude id + all time cols) -----------------------
    excluded = set(_detect_id_columns([str(c) for c in df.columns])) | set(time_cols)
    value_cols = [c for c in df.columns if c not in excluded and c != file_id]
    if not value_cols:
        return None

    prefix = _safe_column_prefix(path, folder)
    generic = {"value", "valuenum", "amount", "result", "measurement"}

    if is_temporal:
        # Keep raw rows; rename id + first time col, then deduplicate value cols
        time_col = time_cols[0]
        keep = [file_id, time_col] + value_cols
        sub = df[keep].dropna(subset=[file_id]).copy()
        sub = sub.rename(columns={file_id: id_col, time_col: canonical_time})
        # Sort by time so later duplicates are more recent
        try:
            sub = sub.sort_values([id_col, canonical_time])
        except Exception:
            pass
    else:
        # Static: one row per patient (take last non-null)
        sub = (
            df[[file_id] + value_cols]
            .dropna(subset=[file_id])
            .groupby(file_id, as_index=False)
            .last()
        )
        if file_id != id_col:
            sub = sub.rename(columns={file_id: id_col})

    # --- rename value columns to avoid clashes ----------------------------
    rename: Dict[str, str] = {}
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
    sub = sub.rename(columns=rename)
    return sub, is_temporal


def _build_stay_level_from_module_folder(
    *,
    folder: Path,
    selected_files: Sequence[Path],
    id_col: str,
    filter_spec: Optional[Tuple[Path, str, str, str]] = None,
    join_how: str = "outer",
) -> pd.DataFrame:
    """Merge selected module parquet files into a single cohort dataframe.

    Merging strategy:
    * **Temporal files** (contain a time column such as ``charttime``,
      ``starttime``, etc.) are merged on ``[id_col, charttime]`` so that
      time-points from different files are properly aligned.  All detected
      time column names are normalised to ``charttime``.
    * **Static files** (demographics, outcomes — no time column) are
      broadcast onto the temporal result by merging on ``id_col`` alone.
    * If *all* selected files are static, they are merged purely on
      ``id_col`` and the result is a stay-level wide table.
    """
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

    canonical_time = "charttime"
    used_columns: Set[str] = {id_col, canonical_time}

    temporal_dfs: List[pd.DataFrame] = []
    static_dfs: List[pd.DataFrame] = []

    for path in selected_files:
        result = _read_module_file(
            path=path,
            folder=folder,
            id_col=id_col,
            keep_ids=keep_ids,
            used_columns=used_columns,
            canonical_time=canonical_time,
        )
        if result is None:
            continue
        sub, is_temporal = result
        if is_temporal:
            temporal_dfs.append(sub)
        else:
            static_dfs.append(sub)

    # --- merge temporal files on [id_col, canonical_time] -----------------
    temporal_merged: Optional[pd.DataFrame] = None
    for sub in temporal_dfs:
        if temporal_merged is None:
            temporal_merged = sub
        else:
            temporal_merged = temporal_merged.merge(
                sub, on=[id_col, canonical_time], how="outer"
            )

    # --- merge static files on id_col only --------------------------------
    static_merged: Optional[pd.DataFrame] = None
    for sub in static_dfs:
        if static_merged is None:
            static_merged = sub
        else:
            static_merged = static_merged.merge(sub, on=id_col, how="outer")

    # --- combine ----------------------------------------------------------
    if temporal_merged is not None and static_merged is not None:
        # Broadcast static columns onto every time-point row (left join keeps
        # all temporal rows, adding static attributes per patient)
        merged = temporal_merged.merge(static_merged, on=id_col, how="left")
    elif temporal_merged is not None:
        merged = temporal_merged
    elif static_merged is not None:
        merged = static_merged
    else:
        merged = pd.DataFrame({id_col: sorted(keep_ids) if keep_ids is not None else []})

    if keep_ids is not None:
        merged = merged[merged[id_col].isin(keep_ids)]

    # The Research Agent expects one analytical row per ICU stay. Module
    # exports can be time-series, so collapse any temporal merge to the most
    # recent observed row after static fields have been broadcast.
    if canonical_time in merged.columns:
        try:
            merged = (
                merged.sort_values([id_col, canonical_time])
                .groupby(id_col, as_index=False)
                .last()
            )
        except Exception:
            merged = merged.drop(columns=[canonical_time]).groupby(id_col, as_index=False).last()

    # Put id first, then canonical time (if present), then remaining cols
    first_cols = [id_col]
    if canonical_time in merged.columns:
        first_cols.append(canonical_time)
    rest = [c for c in merged.columns if c not in first_cols]
    return merged[first_cols + rest].reset_index(drop=True)


def _default_module_selection(labels: Sequence[str]) -> List[str]:
    # Default to all available modules so the user doesn't have to
    # manually tick every file after selecting a folder.
    return list(labels)


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
        # Qwen3 models default to thinking (chain-of-thought) mode on most
        # local servers (vLLM, SGLang, etc.).  In thinking mode the model
        # may emit only a <think>…</think> block with no trailing answer,
        # which causes the JSON parser to receive an empty string.  Disable
        # thinking at the API level so the model responds directly.
        if model and model.lower().startswith("qwen3"):
            kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        if (
            llm_choice == "OpenRouter"
            and handles.get("LLMRouter") is not None
            and handles.get("FallbackLLMClient") is not None
        ):
            return _build_openrouter_role_router(
                handles=handles,
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
                extra_headers=dict(extra_headers or {}),
                preferred_model=model,
            )
        return handles["OpenAIClient"](**kwargs)
    raise RuntimeError(f"Unknown LLM choice: {llm_choice}")


_OPENROUTER_SKIP_AS_PRIMARY = {
    "",
    "openrouter/free",
    "z-ai/glm-4.5-air:free",
}

_OPENROUTER_ROLE_MODEL_CHAINS: Dict[str, List[str]] = {
    "planner": [
        "openai/gpt-oss-120b:free",
        "google/gemma-4-31b-it:free",
        "z-ai/glm-4.5-air:free",
    ],
    "writer": [
        "openai/gpt-oss-120b:free",
        "google/gemma-4-31b-it:free",
        "z-ai/glm-4.5-air:free",
    ],
    "coder": [
        "openai/gpt-oss-120b:free",
        "google/gemma-4-31b-it:free",
        "z-ai/glm-4.5-air:free",
        "qwen/qwen3-coder:free",
    ],
    "analyzer": [
        "google/gemma-4-31b-it:free",
        "openai/gpt-oss-120b:free",
        "z-ai/glm-4.5-air:free",
    ],
    "literature": [
        "openai/gpt-oss-120b:free",
        "google/gemma-4-31b-it:free",
        "z-ai/glm-4.5-air:free",
    ],
}


def _ordered_openrouter_models(*, role: str, preferred_model: str) -> List[str]:
    preferred = (preferred_model or "").strip()
    ordered: List[str] = []
    if preferred and preferred not in _OPENROUTER_SKIP_AS_PRIMARY:
        ordered.append(preferred)
    for model_name in _OPENROUTER_ROLE_MODEL_CHAINS.get(role, []):
        if model_name not in ordered:
            ordered.append(model_name)
    return ordered


def _build_openrouter_role_router(
    *,
    handles: Dict[str, Any],
    api_key: str,
    base_url: str,
    extra_headers: Dict[str, str],
    preferred_model: str,
):
    openai_client = handles["OpenAIClient"]
    fallback_cls = handles["FallbackLLMClient"]
    router_cls = handles["LLMRouter"]

    def _chain(role: str):
        clients = [
            openai_client(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                extra_headers=extra_headers,
            )
            for model_name in _ordered_openrouter_models(
                role=role,
                preferred_model=preferred_model,
            )
        ]
        return fallback_cls(*clients, name=f"openrouter:{role}")

    planner = _chain("planner")
    analyzer = _chain("analyzer")
    return router_cls(
        default=planner,
        planner=planner,
        coder=_chain("coder"),
        analyzer=analyzer,
        writer=_chain("writer"),
        literature=_chain("literature"),
    )


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
    user_preferences: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    stop_after_analysis: bool = False,
    resume_run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Invoke the pipeline; return the :class:`PipelineResult`."""
    pipeline = handles["ResearchAgentPipeline"](
        workdir=workdir,
        llm=llm,
        disable_icu_context=disable_icu_context,
        enable_deterministic_planner_fallback=True,
        enable_deterministic_code_fallback=True,
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
    if user_preferences:
        kwargs["user_preferences"] = user_preferences
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

_RA_TABLE_EXTS = {".csv", ".tsv", ".parquet", ".pq", ".feather"}
_RA_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_RA_FIGURE_EXTS = _RA_RASTER_EXTS | {".svg", ".pdf", ".tiff", ".tif", ".pptx"}
_RA_DEBUG_EXTS = {".json", ".jsonl", ".log", ".txt", ".py", ".r"}
_RA_DEBUG_KINDS = {"log", "code"}


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_user_facing_step_artifact(rec: Dict[str, Any]) -> bool:
    """Return true for artefacts that belong in the step-results view."""
    rel = str(rec.get("relative_path") or "")
    suffix = Path(rel).suffix.lower()
    kind = str(rec.get("kind") or "").lower()
    if kind in {"figure", "table"}:
        return True
    if suffix in _RA_FIGURE_EXTS or suffix in _RA_TABLE_EXTS:
        return True
    return False


def _is_debug_artifact(rec: Dict[str, Any]) -> bool:
    """Return true for raw runtime artefacts better suited for Debug."""
    rel = str(rec.get("relative_path") or "")
    suffix = Path(rel).suffix.lower()
    kind = str(rec.get("kind") or "").lower()
    if kind in _RA_DEBUG_KINDS:
        return True
    if suffix in _RA_DEBUG_EXTS:
        return True
    return not _is_user_facing_step_artifact(rec)


def _manifest_path_for_run(run_dir: Path) -> Optional[Path]:
    final_path = run_dir / "manifest.json"
    partial_path = run_dir / "manifest_partial.json"
    if final_path.exists():
        return final_path
    if partial_path.exists():
        return partial_path
    return None


def _load_run_manifest(run_dir: Path) -> Tuple[Dict[str, Any], Optional[Path], bool]:
    manifest_path = _manifest_path_for_run(run_dir)
    if manifest_path is None:
        return {}, None, False
    manifest = _read_json_file(manifest_path)
    return manifest, manifest_path, manifest_path.name == "manifest_partial.json"


def _safe_step_status(record: Dict[str, Any]) -> str:
    status = str(record.get("status") or "").strip()
    if status:
        return status
    if record.get("step_summary"):
        return "ok"
    return "running"


_RA_MILESTONE_STAGES = {"run", "cohort", "context", "audit", "hypothesis", "plan", "step", "literature"}


def _progress_event_line(event: Dict[str, Any]) -> str:
    status = event.get("status")
    badge = {"complete": "✅", "error": "🔴", "paused": "⏸️"}.get(status, "⚙️")
    stage = str(event.get("stage") or "step")
    return f"{badge} **{stage}** — {event.get('message', '')}"


def _run_summary_from_manifest(
    run_dir: Path,
    manifest: Dict[str, Any],
    *,
    partial: bool,
) -> Dict[str, Any]:
    records = [r for r in manifest.get("per_step_records", []) if isinstance(r, dict)]
    evidence = [r for r in manifest.get("evidence", []) if isinstance(r, dict)]
    findings = [f for f in manifest.get("findings", []) if isinstance(f, dict)]
    statuses = [_safe_step_status(r) for r in records]
    return {
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "run_dir": run_dir,
        "status": "partial" if partial else "complete",
        "started_at": str(manifest.get("started_at") or ""),
        "finished_at": str(manifest.get("finished_at") or ""),
        "question": str(manifest.get("research_question") or ""),
        "step_total": len(records),
        "step_ok": sum(1 for s in statuses if s == "ok"),
        "step_failed": sum(1 for s in statuses if "fail" in s or "error" in s or "blocked" in s),
        "finding_errors": sum(1 for f in findings if f.get("severity") == "error"),
        "finding_warnings": sum(1 for f in findings if f.get("severity") == "warning"),
        "evidence_count": len(evidence),
        "figure_count": sum(1 for r in evidence if r.get("kind") == "figure"),
        "table_count": sum(1 for r in evidence if r.get("kind") == "table"),
        "manifest_partial": partial,
    }


def _scan_research_agent_runs(workdir: Path, *, limit: int = 20) -> List[Dict[str, Any]]:
    if not workdir.exists() or not workdir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(workdir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        manifest, _manifest_path, partial = _load_run_manifest(run_dir)
        if not manifest:
            continue
        rows.append(_run_summary_from_manifest(run_dir, manifest, partial=partial))
        if len(rows) >= limit:
            break
    return rows


def _evidence_by_id(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in manifest.get("evidence", []) or []:
        if isinstance(rec, dict) and rec.get("evidence_id"):
            out[str(rec["evidence_id"])] = rec
    return out


def _evidence_for_step(record: Dict[str, Any], manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    step_id = str(record.get("step_id") or "")
    by_id = _evidence_by_id(manifest)
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for evidence_id in record.get("evidence_ids") or []:
        rec = by_id.get(str(evidence_id))
        if rec is not None and rec.get("evidence_id") not in seen:
            seen.add(str(rec.get("evidence_id")))
            out.append(rec)
    for rec in manifest.get("evidence", []) or []:
        if (
            isinstance(rec, dict)
            and step_id
            and rec.get("produced_by_step") == step_id
            and rec.get("evidence_id") not in seen
        ):
            seen.add(str(rec.get("evidence_id")))
            out.append(rec)
    return out


def _json_payload_for_evidence(
    run_dir: Path,
    manifest: Dict[str, Any],
    evidence_id: str,
    *,
    fallback_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a structured JSON artefact by evidence id or fallback filename."""
    rec = _evidence_by_id(manifest).get(evidence_id)
    candidate_paths: List[Path] = []
    if rec and rec.get("relative_path"):
        candidate_paths.append(run_dir / str(rec["relative_path"]))
    if fallback_name:
        candidate_paths.append(run_dir / fallback_name)
    for path in candidate_paths:
        if path.exists():
            payload = _read_json_file(path)
            if payload:
                return payload
    return {}


def _scalar_summary_items(summary: Dict[str, Any], *, limit: int = 12) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            text = value
            if isinstance(value, str) and len(value) > 120:
                text = value[:117] + "..."
            items.append((str(key), text))
        if len(items) >= limit:
            break
    return items


def _read_table_preview(path: Path, *, n: int = 50) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).head(n)
    if suffix == ".feather":
        return pd.read_feather(path).head(n)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=n)
    return pd.read_csv(path, nrows=n)


def _render_artifact_preview(run_dir: Path, rec: Dict[str, Any], *, key_prefix: str) -> None:
    rel = rec.get("relative_path") or ""
    path = run_dir / rel
    title = rec.get("description") or rec.get("evidence_id") or Path(rel).name
    suffix = path.suffix.lower()
    if not path.exists():
        st.warning(_ra_text("artifact_missing", path=str(path)))
        return
    if rec.get("kind") == "figure" or suffix in _RA_FIGURE_EXTS:
        caption = f"{title} · {rec.get('evidence_id', '')}"
        if suffix in _RA_RASTER_EXTS:
            try:
                st.image(str(path), caption=caption, use_container_width=True)
            except Exception:
                st.image(str(path), caption=caption)
        else:
            st.markdown(f"**{html.escape(str(title))}**")
            st.caption(str(path))
        return
    if rec.get("kind") == "table" or suffix in _RA_TABLE_EXTS:
        with st.container(border=True):
            st.markdown(f"**{_ra_text('table_preview')}: `{Path(rel).name}`**")
            try:
                st.dataframe(_read_table_preview(path), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(_ra_text("table_preview_failed", error=exc))
                st.caption(str(path))
        return
    if suffix == ".json" or rec.get("kind") == "statistic":
        with st.container(border=True):
            st.markdown(f"**{_ra_text('summary_json')}: `{Path(rel).name}`**")
            payload = _read_json_file(path)
            if payload:
                st.json(payload)
            else:
                st.caption(path.read_text(encoding="utf-8", errors="replace")[:4000])
        return
    if suffix in {".md", ".txt", ".log"}:
        with st.container(border=True):
            st.markdown(f"**{_ra_text('text_artifact')}: `{Path(rel).name}`**")
            st.markdown(path.read_text(encoding="utf-8", errors="replace")[:6000])
        return
    st.caption(f"{title}: {path}")


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
    raster_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    cols = st.columns(2)
    for i, rec in enumerate(figs):
        path = run_dir / rec.get("relative_path", "")
        if not path.exists():
            continue
        caption = f"{rec.get('description', '')} — sha256: {rec.get('sha256', '')[:8]}"
        with cols[i % 2]:
            if path.suffix.lower() in raster_exts:
                try:
                    st.image(str(path), caption=caption, use_container_width=True)
                except Exception:
                    st.image(str(path), caption=caption)
            else:
                st.markdown(f"**{caption}**")
                st.caption(str(path))


def _render_literature_and_plan(run_dir: Path, manifest: Dict[str, Any]) -> None:
    """Render the pre-execution research grounding before step outputs."""
    st.markdown(f"### {_ra_text('research_grounding')}")
    st.caption(_ra_text("research_grounding_help"))

    literature = _json_payload_for_evidence(
        run_dir,
        manifest,
        "preplan_literature_bundle",
        fallback_name="preplan_literature_bundle.json",
    ) or _json_payload_for_evidence(
        run_dir,
        manifest,
        "literature_bundle",
        fallback_name="literature_bundle.json",
    )
    blueprint = _json_payload_for_evidence(
        run_dir,
        manifest,
        "hypothesis_blueprint",
        fallback_name="hypothesis_blueprint.json",
    )
    plan = _json_payload_for_evidence(
        run_dir,
        manifest,
        "analysis_plan",
        fallback_name=str(manifest.get("plan_path") or "analysis_plan.json"),
    )

    with st.container(border=True):
        st.markdown(f"**1. {_ra_text('literature_review')}**")
        citations = literature.get("citations") if isinstance(literature, dict) else None
        if citations:
            st.metric(_ra_text("citations"), len(citations))
            rows = []
            for rec in citations:
                if not isinstance(rec, dict):
                    continue
                rows.append({
                    _ra_text("citation_key"): rec.get("key"),
                    _ra_text("year"): rec.get("year"),
                    _ra_text("title"): rec.get("title"),
                    _ra_text("relevance"): rec.get("relevance"),
                    "PMID/DOI/URL": rec.get("pmid") or rec.get("doi") or rec.get("url"),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(_ra_text("no_literature_bundle"))

    with st.container(border=True):
        st.markdown(f"**2. {_ra_text('hypothesis_blueprint')}**")
        if blueprint:
            cols = st.columns(3)
            cols[0].metric(_ra_text("feasibility"), str(blueprint.get("feasibility_status") or "-"))
            cols[1].metric(_ra_text("hypothesis_type"), str(blueprint.get("hypothesis_type") or "-"))
            cols[2].metric(
                _ra_text("prior_refs"),
                len(blueprint.get("prior_literature_keys") or []),
            )
            if blueprint.get("hypothesis"):
                st.markdown(f"**{_ra_text('hypothesis')}**")
                st.write(blueprint.get("hypothesis"))
            if blueprint.get("novelty_rationale"):
                st.markdown(f"**{_ra_text('novelty')}**")
                st.write(blueprint.get("novelty_rationale"))
            if blueprint.get("self_critique"):
                st.markdown(f"**{_ra_text('self_critique')}**")
                for item in list(blueprint.get("self_critique") or [])[:5]:
                    st.markdown(f"- {item}")
            domain_notes = list(blueprint.get("domain_gate_notes") or [])
            if domain_notes:
                with st.expander(_ra_text("domain_gate_notes"), expanded=False):
                    for note in domain_notes[:12]:
                        st.markdown(f"- {note}")
        else:
            st.info(_ra_text("no_hypothesis_blueprint"))

    with st.container(border=True):
        st.markdown(f"**3. {_ra_text('analysis_plan')}**")
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if steps:
            if plan.get("rationale"):
                st.markdown(f"**{_ra_text('plan_rationale')}**")
                st.write(plan.get("rationale"))
            rows = []
            for idx, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    continue
                rows.append({
                    "#": idx,
                    _ra_text("step_id"): step.get("step_id"),
                    _ra_text("step_intent"): step.get("intent"),
                    _ra_text("method"): step.get("method"),
                    _ra_text("inputs"): ", ".join(map(str, step.get("inputs") or []))[:180],
                    _ra_text("outputs"): ", ".join(map(str, step.get("expected_outputs") or []))[:180],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(_ra_text("no_analysis_plan"))


def _render_step_records(run_dir: Path, manifest: Dict[str, Any], *, key_prefix: str) -> None:
    records = [r for r in manifest.get("per_step_records", []) or [] if isinstance(r, dict)]
    if not records:
        st.info(_ra_text("no_steps"))
        return

    status_counts: Dict[str, int] = {}
    for record in records:
        status = _safe_step_status(record)
        status_counts[status] = status_counts.get(status, 0) + 1
    cols = st.columns(4)
    cols[0].metric(_ra_text("steps_total"), len(records))
    cols[1].metric(_ra_text("steps_ok"), status_counts.get("ok", 0))
    cols[2].metric(_ra_text("steps_failed"), sum(v for k, v in status_counts.items() if k != "ok"))
    cols[3].metric(_ra_text("figures"), sum(1 for r in manifest.get("evidence", []) or [] if r.get("kind") == "figure"))

    for idx, record in enumerate(records, start=1):
        step_id = str(record.get("step_id") or f"step_{idx}")
        status = _safe_step_status(record)
        title = f"{idx}. {step_id} · {status}"
        with st.expander(title, expanded=idx == len(records)):
            intent = record.get("intent")
            if intent:
                st.markdown(f"**{_ra_text('step_intent')}**: {intent}")
            meta_cols = st.columns(4)
            meta_cols[0].metric(_ra_text("generation_mode"), str(record.get("generation_mode") or "-"))
            meta_cols[1].metric(_ra_text("return_code"), str(record.get("returncode", "-")))
            meta_cols[2].metric(_ra_text("repair_attempts"), int(record.get("code_repair_attempts") or 0))
            meta_cols[3].metric(_ra_text("evidence"), len(_evidence_for_step(record, manifest)))

            summary = record.get("step_summary")
            if isinstance(summary, dict) and summary:
                st.markdown(f"**{_ra_text('key_metrics')}**")
                items = _scalar_summary_items(summary)
                if items:
                    st.dataframe(
                        pd.DataFrame(items, columns=[_ra_text("metric"), _ra_text("value")]),
                        use_container_width=True,
                        hide_index=True,
                    )
                with st.expander(_ra_text("full_step_summary"), expanded=False):
                    st.markdown(f"**{_ra_text('full_step_summary')}**")
                    st.json(summary)

            finding_rows: List[Dict[str, Any]] = []
            for group_key in (
                "usage_findings",
                "stat_findings",
                "clinical_findings",
                "guard_findings",
                "contract_findings",
                "visual_findings",
            ):
                for finding in record.get(group_key) or []:
                    if isinstance(finding, dict):
                        finding_rows.append({
                            "source": group_key,
                            "severity": finding.get("severity"),
                            "validator": finding.get("validator"),
                            "message": finding.get("message"),
                        })
            if finding_rows:
                with st.container(border=True):
                    st.markdown(f"**{_ra_text('step_findings')}**")
                    st.dataframe(pd.DataFrame(finding_rows), use_container_width=True, hide_index=True)

            artefacts = _evidence_for_step(record, manifest)
            visible_artefacts = [rec for rec in artefacts if _is_user_facing_step_artifact(rec)]
            hidden_count = len(artefacts) - len(visible_artefacts)
            if visible_artefacts:
                st.markdown(f"**{_ra_text('step_artifacts')}**")
                for art_idx, rec in enumerate(visible_artefacts):
                    _render_artifact_preview(
                        run_dir,
                        rec,
                        key_prefix=f"{key_prefix}_{step_id}_{art_idx}",
                    )
            if hidden_count:
                st.caption(_ra_text("technical_artifacts_hidden", count=hidden_count))


def _render_artifact_gallery(run_dir: Path, manifest: Dict[str, Any], *, kind: Optional[str] = None) -> None:
    records = [r for r in manifest.get("evidence", []) or [] if isinstance(r, dict)]
    if kind is not None:
        records = [r for r in records if r.get("kind") == kind]
    if not records:
        st.info(_ra_text("no_evidence"))
        return
    for idx, rec in enumerate(records):
        _render_artifact_preview(run_dir, rec, key_prefix=f"gallery_{idx}")


def _result_like_from_manifest(run_dir: Path, manifest: Dict[str, Any]) -> SimpleNamespace:
    run_id = str(manifest.get("run_id") or run_dir.name)
    report_path = run_dir / str(manifest.get("report_path") or "results_report.md")
    manuscript_path = run_dir / str(manifest.get("manuscript_path") or "manuscript_scaffold_bound.md")
    evidence_count = len(manifest.get("evidence", []) or [])
    findings_count = len(manifest.get("findings", []) or [])
    return SimpleNamespace(
        run_id=run_id,
        workdir=str(run_dir),
        report_path=str(report_path),
        manuscript_path=str(manuscript_path),
        manifest_path=str(run_dir / "manifest.json"),
        evidence_count=evidence_count,
        findings_count=findings_count,
    )


def _render_run_manifest(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    result: Optional[Any] = None,
    manifest_path: Optional[Path] = None,
    key_prefix: str = "research_agent_run",
) -> None:
    result = result or _result_like_from_manifest(run_dir, manifest)
    partial = manifest_path is not None and manifest_path.name == "manifest_partial.json"
    paused_after_analysis = "paused_after_analysis" in str(manifest.get("notes") or "")
    summary = _run_summary_from_manifest(run_dir, manifest, partial=partial)

    if partial:
        st.info(_ra_text("partial_run_notice", run_id=result.run_id))
    elif summary["step_failed"]:
        st.warning(_ra_text(
            "run_complete_with_review",
            run_id=result.run_id,
            failed=summary["step_failed"],
            evidence=summary["evidence_count"],
            findings=len(manifest.get("findings", []) or []),
        ))
        if summary["figure_count"] == 0 and summary["table_count"] == 0:
            st.error(_ra_text("run_no_tables_figures"))
    else:
        st.success(_ra_text(
            "run_complete",
            run_id=result.run_id,
            evidence=summary["evidence_count"],
            findings=len(manifest.get("findings", []) or []),
        ))
    if paused_after_analysis:
        st.info(_ra_text("paused_notice"))

    tab_labels = [
        _ra_text("tab_report"),
        _ra_text("tab_steps"),
        _ra_text("tab_artifacts"),
        _ra_text("tab_evidence"),
        _ra_text("tab_manuscript"),
        _ra_text("tab_debug"),
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        report_path = Path(result.report_path)
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning(_ra_text("report_missing"))

    with tabs[1]:
        _render_literature_and_plan(run_dir, manifest)
        st.divider()
        _render_step_records(run_dir, manifest, key_prefix=f"{key_prefix}_steps")

    with tabs[2]:
        st.markdown(f"### {_ra_text('figures')}")
        figure_records = [r for r in manifest.get("evidence", []) or [] if r.get("kind") == "figure"]
        if figure_records:
            _render_artifact_gallery(run_dir, {"evidence": figure_records}, kind=None)
        else:
            st.info(_ra_text("no_figures"))
        st.markdown(f"### {_ra_text('tables')}")
        table_records = [r for r in manifest.get("evidence", []) or [] if r.get("kind") == "table"]
        if table_records:
            _render_artifact_gallery(run_dir, {"evidence": table_records}, kind=None)
        else:
            st.info(_ra_text("no_tables"))

    with tabs[3]:
        st.markdown(f"### {_ra_text('findings')}")
        _render_findings(manifest)
        st.markdown(f"### {_ra_text('tab_evidence')}")
        _render_evidence_table(run_dir, manifest)

    with tabs[4]:
        mp = Path(result.manuscript_path)
        if mp.exists():
            text = mp.read_text(encoding="utf-8")
            if paused_after_analysis:
                st.info(_ra_text("manuscript_skipped"))
                if st.button(
                    _ra_text("draft_from_analysis"),
                    key=f"research_agent_draft_from_{result.run_id}_{key_prefix}",
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
                    key=f"{key_prefix}_download_md",
                )
        else:
            st.warning(_ra_text("bound_missing"))

    with tabs[5]:
        debug_records = [
            r for r in manifest.get("evidence", []) or []
            if isinstance(r, dict) and _is_debug_artifact(r)
        ]
        if debug_records:
            with st.expander(
                f"{_ra_text('technical_artifacts')} ({len(debug_records)})",
                expanded=False,
            ):
                st.caption(_ra_text("technical_artifacts_help"))
                _render_artifact_gallery(run_dir, {"evidence": debug_records}, kind=None)
        st.markdown(f"### {_ra_text('latex')}")
        tex_path = run_dir / "manuscript_scaffold.tex"
        if paused_after_analysis:
            st.info(_ra_text("no_latex_paused"))
        elif tex_path.exists():
            tex = tex_path.read_text(encoding="utf-8")
            st.download_button(
                _ra_text("download_tex"), data=tex,
                file_name="manuscript_scaffold.tex", mime="text/x-tex",
                key=f"{key_prefix}_download_tex",
            )
            st.code(tex, language="latex")
        else:
            st.info(_ra_text("no_latex"))
        st.markdown(f"### {_ra_text('manifest')}")
        st.json(manifest)


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
    _render_run_manifest(
        run_dir=run_dir,
        manifest=manifest,
        result=result,
        manifest_path=manifest_path,
        key_prefix=f"research_agent_result_{result.run_id}",
    )


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
                    resolved = Path(p).expanduser().resolve()
                    extra_roots.append(resolved)
                    # Also add the parent so sibling dated-export dirs
                    # (e.g. miiv_20260428 next to miiv_20260427) are discovered
                    # even when export_path points to a specific run folder.
                    parent = resolved.parent
                    if parent not in extra_roots and parent != resolved:
                        extra_roots.append(parent)
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
        selected_folder_value = (
            str(dirs[dir_labels.index(picked_label)])
            if dir_labels and picked_label not in {"", _ra_text("manual_path")}
            else manual_default
        )
        _hide_prefilled_directory_text("research_agent_module_dir_text", selected_folder_value)
        folder_text = _directory_input(
            _ra_text("module_folder"),
            value=selected_folder_value,
            input_key="research_agent_module_dir_text",
            button_key="research_agent_module_dir_browse",
            placeholder=_placeholder_path("easyicu_export/miiv"),
            help=_ra_text("module_folder_help"),
            show_value=False,
        )
        folder_text = folder_text or selected_folder_value
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

        # Merge strategy is fixed to outer — keeps all patient IDs from
        # every selected file, which is the correct default for ICU cohorts.
        join_how = "outer"
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
        if st.session_state.get("data_path") and not st.session_state.get("research_agent_extract_data_path"):
            st.session_state.research_agent_extract_data_path = st.session_state.data_path
        default_output_dir = st.session_state.get("export_path", str(Path.home() / "easyicu_export" / f"{db}_research_agent"))
        if default_output_dir and not st.session_state.get("research_agent_extract_output_dir"):
            st.session_state.research_agent_extract_output_dir = default_output_dir
        data_path = _directory_input(
            _ra_text("raw_path"),
            value=st.session_state.get("data_path", ""),
            input_key="research_agent_extract_data_path",
            button_key="research_agent_extract_data_path_browse",
            placeholder=_placeholder_path(db),
        ) or st.session_state.get("data_path", "")
        output_dir = _directory_input(
            _ra_text("output_folder"),
            value=default_output_dir,
            input_key="research_agent_extract_output_dir",
            button_key="research_agent_extract_output_dir_browse",
            placeholder=_placeholder_path("easyicu_export"),
        )
        output_dir = output_dir or default_output_dir
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
            index=5,
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
            st.session_state["_scroll_to_tab"] = "export_progress"
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
                resolved = Path(p).resolve()
                extra_roots.append(resolved)
                parent = resolved.parent
                if parent not in extra_roots and parent != resolved:
                    extra_roots.append(parent)
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


def _request_examples() -> List[Dict[str, str]]:
    """Detailed request templates shown in the unified request box."""
    lang = st.session_state.get("language", "en")
    if lang == "zh":
        return [
            {
                "key": "prediction",
                "label": "预测模型",
                "summary": "风险预测、早期预警、死亡/并发症/LOS 预测",
                "outcome": "death",
                "prompt": (
                    "请基于年龄、性别、SOFA-2、乳酸、MAP、血管活性药物使用和肌酐构建 ICU 死亡预测模型；"
                    "使用训练/测试集划分，报告 AUROC、Brier score、校准情况和主要变量系数，"
                    "并输出可发表的多面板图。"
                ),
            },
            {
                "key": "clustering",
                "label": "表型/聚类分析",
                "summary": "患者分群、亚型发现、stay-level 聚类",
                "outcome": "death",
                "prompt": (
                    "请根据 SOFA-2、乳酸、MAP、心率、肌酐和血管活性药物使用对 ICU 患者做聚类分析，"
                    "总结每个簇的人数、主要生理特征和死亡率差异，并输出可发表的聚类结果图。"
                ),
            },
            {
                "key": "association",
                "label": "相关性分析",
                "summary": "危险因素、预后因素、暴露-结局关联",
                "outcome": "death",
                "prompt": (
                    "请分析入 ICU 时 SOFA-2 与 ICU 死亡的关系；先做队列描述、结局发生率和缺失值审计，"
                    "再做多变量 logistic 回归，报告 OR、95% CI 和可发表图。"
                ),
            },
            {
                "key": "survival",
                "label": "生存/时间到事件分析",
                "summary": "时间到死亡、时间到拔机、时间到出 ICU",
                "outcome": "death",
                "prompt": (
                    "请针对 ICU 患者开展时间到事件分析，研究入 ICU 时 SOFA-2、乳酸和血流动力学指标与 28 天死亡的关系；"
                    "明确 time zero、删失处理、Kaplan-Meier/累计发生率展示和 Cox 或其他合适模型。"
                ),
            },
            {
                "key": "dynamic_prediction",
                "label": "动态预测/早期预警",
                "summary": "基于时序数据的持续风险更新",
                "outcome": "death",
                "prompt": (
                    "请设计一个 ICU 动态预测分析，基于生命体征、实验室和支持治疗的时序变化，"
                    "持续预测未来 24 小时内病情恶化或死亡风险，并说明时间窗、更新频率和评估指标。"
                ),
            },
            {
                "key": "treatment",
                "label": "治疗反应/比较效果",
                "summary": "用药、通气、RRT、输血等干预效果研究",
                "outcome": "death",
                "prompt": (
                    "请评估血管活性药物使用与 ICU 死亡和乳酸清除的关系，明确时间对齐、潜在混杂、"
                    "亚组差异和稳健性分析方案，并给出推荐的分析流程。"
                ),
            },
            {
                "key": "causal",
                "label": "因果分析",
                "summary": "目标试验模拟、ATE/CATE、倾向评分/加权",
                "outcome": "death",
                "prompt": (
                    "请把早期使用血管活性药物对 ICU 死亡的影响表述成目标试验框架，定义纳入标准、time zero、"
                    "治疗策略、混杂因素、positivity 检查和主要效应估计方案。"
                ),
            },
            {
                "key": "rl",
                "label": "强化学习",
                "summary": "序贯治疗策略、液体/升压药/呼吸机决策优化",
                "outcome": "death",
                "prompt": (
                    "请设计一个用于 ICU 血流动力学管理的强化学习分析方案，明确 state、action、reward、"
                    "轨迹构建、离策略评估和安全约束，并输出推荐的研究流程。"
                ),
            },
            {
                "key": "multimodal",
                "label": "多模态建模",
                "summary": "结构化 EHR + 病历文本 + 波形/影像",
                "outcome": "death",
                "prompt": (
                    "请设计一个 ICU 多模态分析，融合结构化 EHR、病历文本以及可用波形或影像信息，"
                    "用于死亡或恶化风险预测，并明确各模态的数据准备、对齐、缺失处理和评估方案。"
                ),
            },
            {
                "key": "validation",
                "label": "外部验证/评分比较",
                "summary": "模型外部验证、校准、跨库可迁移性、评分比较",
                "outcome": "death",
                "prompt": (
                    "请比较 SOFA-2、qSOFA、SAPS-II 或自建模型在 ICU 死亡预测中的表现，"
                    "重点评估区分度、校准、亚组表现和跨数据库外部验证方案。"
                ),
            },
            {
                "key": "data_quality",
                "label": "数据质量/缺失值/映射审计",
                "summary": "概念覆盖、缺失、单位、时间对齐、跨库一致性",
                "outcome": "death",
                "prompt": (
                    "请先把这批 ICU 数据作为数据质量审计任务处理，系统评估缺失值、变量覆盖、单位范围、"
                    "时间戳一致性以及跨模块映射问题，并输出适合研究前复核的图表和结论。"
                ),
            },
        ]
    return [
        {
            "key": "prediction",
            "label": "Prediction model",
            "summary": "Risk prediction, early warning, mortality/LOS/complication prediction",
            "outcome": "death",
            "prompt": (
                "Build an ICU mortality prediction model using age, sex, SOFA-2, lactate, MAP, vasopressor use, "
                "and creatinine; use an explicit train/test split, report AUROC, Brier score, calibration, "
                "key coefficients, and generate a publication-ready multi-panel figure."
            ),
        },
        {
            "key": "clustering",
            "label": "Phenotyping / clustering",
            "summary": "Patient grouping, subphenotypes, stay-level clustering",
            "outcome": "death",
            "prompt": (
                "Cluster ICU patients using SOFA-2, lactate, MAP, heart rate, creatinine, and vasopressor use; "
                "summarize cluster size, physiologic profiles, and mortality differences, and generate a "
                "publication-ready clustering figure."
            ),
        },
        {
            "key": "association",
            "label": "Association",
            "summary": "Risk factors, prognosis, exposure-outcome association",
            "outcome": "death",
            "prompt": (
                "Analyze whether admission SOFA-2 is associated with ICU mortality; include cohort summary, "
                "outcome incidence, missingness audit, multivariable logistic regression, odds ratios with "
                "95% confidence intervals, and a publication-ready figure."
            ),
        },
        {
            "key": "survival",
            "label": "Survival / time-to-event",
            "summary": "Time to death, extubation, discharge, or other event",
            "outcome": "death",
            "prompt": (
                "Run a time-to-event analysis in ICU patients to study how admission SOFA-2, lactate, and hemodynamic variables "
                "relate to 28-day mortality, making time zero, censoring, Kaplan-Meier or cumulative-incidence plots, and Cox-style "
                "or other appropriate models explicit."
            ),
        },
        {
            "key": "dynamic_prediction",
            "label": "Dynamic prediction / early warning",
            "summary": "Time-updated risk estimation from longitudinal ICU data",
            "outcome": "death",
            "prompt": (
                "Design a dynamic ICU prediction analysis that updates the risk of deterioration or death over the next 24 hours "
                "using longitudinal vital signs, laboratory values, and support therapies, and specify the time window, update "
                "frequency, and evaluation metrics."
            ),
        },
        {
            "key": "treatment",
            "label": "Treatment response / comparative effectiveness",
            "summary": "Drug, ventilation, RRT, transfusion, or other intervention effects",
            "outcome": "death",
            "prompt": (
                "Evaluate the relationship between vasopressor use and ICU mortality plus lactate clearance, "
                "making timing alignment, confounding, subgroup heterogeneity, and robustness checks explicit, "
                "then recommend the analysis workflow."
            ),
        },
        {
            "key": "causal",
            "label": "Causal inference",
            "summary": "Target-trial emulation, ATE/CATE, weighting, and bias checks",
            "outcome": "death",
            "prompt": (
                "Frame early vasopressor use versus no early vasopressor use as a target-trial emulation for ICU mortality, "
                "defining eligibility, time zero, treatment strategies, confounders, positivity diagnostics, and the main "
                "effect-estimation strategy."
            ),
        },
        {
            "key": "rl",
            "label": "Reinforcement learning",
            "summary": "Sequential treatment policy optimization",
            "outcome": "death",
            "prompt": (
                "Design a reinforcement-learning analysis for ICU hemodynamic management, specifying state, action, reward, "
                "trajectory assembly, off-policy evaluation, and safety constraints, and return the recommended study workflow."
            ),
        },
        {
            "key": "multimodal",
            "label": "Multimodal modeling",
            "summary": "Structured EHR + notes + waveforms/imaging",
            "outcome": "death",
            "prompt": (
                "Design a multimodal ICU analysis that combines structured EHR data, clinical notes, and any available waveforms "
                "or imaging to predict mortality or deterioration, explicitly covering preprocessing, modality alignment, missingness, "
                "and evaluation."
            ),
        },
        {
            "key": "validation",
            "label": "External validation / score benchmarking",
            "summary": "Transportability, calibration, and score comparison across cohorts",
            "outcome": "death",
            "prompt": (
                "Compare SOFA-2, qSOFA, SAPS-II, or a custom model for ICU mortality prediction, focusing on discrimination, "
                "calibration, subgroup performance, and an external-validation plan across databases."
            ),
        },
        {
            "key": "data_quality",
            "label": "Data-quality / missingness / harmonization audit",
            "summary": "Coverage, missingness, units, timing, and cross-database consistency",
            "outcome": "death",
            "prompt": (
                "Treat this as an ICU data-quality audit first: assess missingness, concept coverage, unit/range issues, temporal "
                "consistency, and cross-module mapping problems, then generate review-ready tables and figures."
            ),
        },
    ]


def _section_request_picker() -> Tuple[Optional[str], Optional[str]]:
    """Render one unified request box with detailed example prompts."""
    examples = _request_examples()
    st.caption(_ra_text("request_intro"))
    st.markdown(_ra_text("request_capabilities"))

    choice_labels = [_ra_text("starter_none")] + [f"{ex['label']} — {ex['summary']}" for ex in examples]
    selected = st.selectbox(
        _ra_text("starter_template"),
        choice_labels,
        index=0,
        key="research_agent_template_pick",
        help=_ra_text("starter_template_help"),
    )
    selected_example = None
    if selected != choice_labels[0]:
        selected_example = examples[choice_labels.index(selected) - 1]
        st.session_state["research_agent_template_current"] = selected_example["key"]
        c1, c2 = st.columns([5, 1.4])
        with c1:
            st.caption(_ra_text("starter_selected", label=selected_example["label"], summary=selected_example["summary"]))
        with c2:
            if st.button(_ra_text("use_template"), key=f"research_agent_apply_{selected_example['key']}", use_container_width=True):
                st.session_state["research_agent_question"] = selected_example["prompt"]
                st.session_state["research_agent_target_outcome"] = selected_example.get("outcome", "")
                st.session_state["research_agent_example_active"] = selected_example["label"]
                st.session_state["research_agent_example_key"] = selected_example["key"]
                st.rerun()
    else:
        st.session_state["research_agent_template_current"] = None

    question = st.text_area(
        _ra_text("question"),
        value=st.session_state.get("research_agent_question", ""),
        help=_ra_text("question_help"),
        key="research_agent_question",
        height=180,
    )
    target_outcome = st.text_input(
        _ra_text("target_outcome_optional"),
        value=st.session_state.get("research_agent_target_outcome", ""),
        help=_ra_text("target_outcome_optional_help"),
        key="research_agent_target_outcome",
    )
    active = st.session_state.get("research_agent_example_active")
    if active:
        st.info(_ra_text("example_loaded", example=active))
    return question.strip() or None, target_outcome.strip() or None


def _infer_request_family(question: Optional[str]) -> str:
    text = (question or "").strip().lower()
    if not text:
        return "general"

    def _keyword_present(keyword: str) -> bool:
        kw = (keyword or "").strip().lower()
        if not kw:
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in kw):
            return kw in text
        flexible = re.escape(kw).replace(r"\ ", r"[\s_-]+")
        pattern = rf"(?<![a-z0-9]){flexible}(?![a-z0-9])"
        return re.search(pattern, text) is not None

    keyword_map = [
        ("reinforcement_learning", ["reinforcement learning", "rl", "policy", "state", "action", "reward", "强化学习", "策略学习"]),
        ("causal_inference", ["target trial", "causal", "ate", "cate", "iptw", "propensity", "因果", "目标试验", "倾向评分"]),
        ("trajectory_clustering", ["cluster", "clustering", "phenotype", "subphenotype", "latent class", "亚型", "分群", "聚类", "表型"]),
        ("dynamic_prediction", ["dynamic", "time-updated", "early warning", "deterioration", "未来", "动态预测", "早期预警", "恶化"]),
        ("survival", ["survival", "time-to-event", "cox", "kaplan", "competing risk", "生存", "时间到事件", "删失"]),
        ("validation", ["external validation", "externally validate", "validate score", "score comparison", "benchmark", "transportability", "外部验证", "评分比较", "跨库验证"]),
        ("prediction_model", ["predict", "prediction", "auc", "auroc", "calibration", "brier", "预测", "风险评分", "预警模型"]),
        ("treatment_response", ["treatment", "therapy", "vasopressor", "ventilation", "rrt", "drug", "response", "治疗", "药物", "通气", "升压药"]),
        ("multimodal", ["multimodal", "notes", "waveform", "imaging", "text", "多模态", "病历文本", "波形", "影像"]),
        ("data_quality", ["missingness", "harmonization", "mapping", "coverage", "unit", "quality", "缺失值", "映射", "覆盖", "数据质量"]),
        ("association", ["associate", "association", "odds ratio", "hazard ratio", "risk factor", "prognostic", "相关", "关联", "危险因素", "预后"]),
    ]
    for family, keywords in keyword_map:
        if any(_keyword_present(k) for k in keywords):
            return family
    return "general"


def _template_family_hint() -> Optional[str]:
    key = st.session_state.get("research_agent_template_current") or st.session_state.get("research_agent_example_key")
    if not key:
        return None
    mapping = {
        "prediction": "prediction_model",
        "clustering": "trajectory_clustering",
        "association": "association",
        "survival": "survival",
        "dynamic_prediction": "dynamic_prediction",
        "treatment_response": "treatment_response",
        "causal": "causal_inference",
        "rl": "reinforcement_learning",
        "multimodal": "multimodal",
        "validation": "validation",
        "data_quality": "data_quality",
    }
    return mapping.get(str(key))


def _section_method_preferences(
    question: Optional[str],
    target_outcome: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """Collect optional analysis preferences and render them as run notes."""
    lang = st.session_state.get("language", "en")
    family = _infer_request_family(question) if (question or "").strip() else (_template_family_hint() or "general")
    family_text = {
        "zh": {
            "general": ("当前问题的方法空间可能较宽。", "例如：限定主要结局、指定是否需要分层分析、敏感性分析、外部验证、或只先出 protocol。"),
            "prediction_model": ("当前更像预测模型问题。", "例如：train/test split 或 bootstrap、AUROC、Brier score、calibration、decision curve、SHAP。"),
            "trajectory_clustering": ("当前更像聚类/表型问题。", "例如：标准化方式、聚类算法、簇数选择、稳定性评估、簇间结局比较。"),
            "association": ("当前更像相关性/预后问题。", "例如：主要暴露、主要模型、协变量调整、亚组分析、稳健性分析。"),
            "survival": ("当前更像生存/时间到事件问题。", "例如：time zero、删失规则、Kaplan-Meier、Cox、竞争风险。"),
            "dynamic_prediction": ("当前更像动态预测/早期预警问题。", "例如：预测时间窗、更新频率、时间切片、时序特征工程、动态校准。"),
            "treatment_response": ("当前更像治疗反应/比较效果问题。", "例如：暴露定义、治疗时间对齐、混杂控制、异质性分析、稳健性检查。"),
            "causal_inference": ("当前更像因果分析问题。", "例如：target trial、纳入标准、time zero、倾向评分/加权、balance、敏感性分析。"),
            "reinforcement_learning": ("当前更像强化学习问题。", "例如：state/action/reward、轨迹构建、离策略评估、安全约束。"),
            "multimodal": ("当前更像多模态建模问题。", "例如：各模态对齐、缺失处理、融合策略、模态消融和外部验证。"),
            "validation": ("当前更像外部验证/评分比较问题。", "例如：判别度、校准、重分类、亚组表现、跨库迁移。"),
            "data_quality": ("当前更像数据质量/缺失值审计问题。", "例如：缺失模式、单位与范围检查、时间戳一致性、概念映射和跨库 harmonization。"),
        },
        "en": {
            "general": ("This request could map to several study families.", "For example: constrain the primary outcome, ask for subgroup analyses, sensitivity analyses, external validation, or protocol-first planning."),
            "prediction_model": ("This looks like a prediction-model request.", "For example: train/test split or bootstrap, AUROC, Brier score, calibration, decision curve, SHAP."),
            "trajectory_clustering": ("This looks like a clustering / phenotyping request.", "For example: scaling strategy, clustering algorithm, number-of-clusters rule, stability assessment, outcome comparison across clusters."),
            "association": ("This looks like an association / prognosis request.", "For example: primary exposure, main model, covariate adjustment, subgroup analysis, robustness checks."),
            "survival": ("This looks like a survival / time-to-event request.", "For example: time zero, censoring rules, Kaplan-Meier, Cox, competing risks."),
            "dynamic_prediction": ("This looks like a dynamic-prediction request.", "For example: prediction horizon, update frequency, temporal slicing, longitudinal feature engineering, dynamic calibration."),
            "treatment_response": ("This looks like a treatment-response / comparative-effectiveness request.", "For example: exposure definition, treatment timing alignment, confounding control, heterogeneity analysis, robustness checks."),
            "causal_inference": ("This looks like a causal-inference request.", "For example: target-trial framing, eligibility, time zero, propensity/weighting, balance, sensitivity analyses."),
            "reinforcement_learning": ("This looks like a reinforcement-learning request.", "For example: state/action/reward, trajectory assembly, off-policy evaluation, safety constraints."),
            "multimodal": ("This looks like a multimodal-modeling request.", "For example: modality alignment, missing-data handling, fusion strategy, modality ablation, external validation."),
            "validation": ("This looks like an external-validation / score-benchmarking request.", "For example: discrimination, calibration, reclassification, subgroup performance, transportability."),
            "data_quality": ("This looks like a data-quality / missingness audit request.", "For example: missingness patterns, unit/range checks, timestamp consistency, concept mapping, and harmonization."),
        },
    }
    headline, hint = family_text["zh" if lang == "zh" else "en"].get(
        family,
        family_text["zh" if lang == "zh" else "en"]["general"],
    )
    st.caption(headline)
    st.info(hint)

    method_pref = st.text_area(
        _ra_text("methods_freeform"),
        value=st.session_state.get("research_agent_method_preferences_text", ""),
        height=90,
        key="research_agent_method_preferences_text",
        help=_ra_text("methods_help"),
        placeholder=hint,
    )
    evaluation_focus = st.text_area(
        _ra_text("evaluation_focus"),
        value=st.session_state.get("research_agent_evaluation_focus", ""),
        height=80,
        key="research_agent_evaluation_focus",
        help=_ra_text("evaluation_focus_help"),
    )
    subgroup_sensitivity = st.text_area(
        _ra_text("subgroup_sensitivity"),
        value=st.session_state.get("research_agent_subgroup_sensitivity", ""),
        height=80,
        key="research_agent_subgroup_sensitivity",
        help=_ra_text("subgroup_sensitivity_help"),
    )
    timing_design = st.text_area(
        _ra_text("timing_design"),
        value=st.session_state.get("research_agent_timing_design", ""),
        height=80,
        key="research_agent_timing_design",
        help=_ra_text("timing_design_help"),
    )
    data_constraints = st.text_area(
        _ra_text("data_constraints"),
        value=st.session_state.get("research_agent_data_constraints", ""),
        height=70,
        key="research_agent_data_constraints",
        help=_ra_text("data_constraints_help"),
    )
    must_have_outputs = st.text_area(
        _ra_text("must_have_outputs"),
        value=st.session_state.get("research_agent_must_have_outputs", ""),
        height=70,
        key="research_agent_must_have_outputs",
        help=_ra_text("must_have_outputs_help"),
    )

    covariate_families = {
        "association",
        "survival",
        "prediction_model",
        "treatment_response",
        "causal_inference",
        "validation",
    }
    covariate_placeholders = {
        "association": "e.g. age, sex, baseline severity, comorbidity burden",
        "survival": "e.g. age, sex, baseline severity, treatment status",
        "prediction_model": "Optional if you want to force certain predictors or adjustment variables",
        "treatment_response": "e.g. age, sex, indication severity, prior organ support",
        "causal_inference": "e.g. confounders measured before treatment assignment",
        "validation": "Optional if score recalibration or subgroup adjustment is needed",
    }
    covariates = ""
    if family in covariate_families:
        covariates = st.text_input(
            _ra_text("covariates"),
            value=st.session_state.get("research_agent_covariates", ""),
            key="research_agent_covariates",
            placeholder=covariate_placeholders.get(family, ""),
        )
    extra = st.text_area(
        _ra_text("extra_notes"),
        value="",
        height=80,
        key="research_agent_extra_notes",
    )
    notes: List[str] = []
    if method_pref.strip():
        notes.append("User method preferences: " + method_pref.strip())
    if evaluation_focus.strip():
        notes.append("User evaluation focus: " + evaluation_focus.strip())
    if subgroup_sensitivity.strip():
        notes.append("User subgroup/sensitivity preferences: " + subgroup_sensitivity.strip())
    if timing_design.strip():
        notes.append("User timing/design constraints: " + timing_design.strip())
    if data_constraints.strip():
        notes.append("User data constraints: " + data_constraints.strip())
    if must_have_outputs.strip():
        notes.append("User must-have outputs: " + must_have_outputs.strip())
    if target_outcome and target_outcome.strip():
        notes.append("User target outcome override: " + target_outcome.strip())
    if covariates.strip():
        notes.append("User requested covariates: " + covariates.strip())
    if extra.strip():
        notes.append("User notes: " + extra.strip())
    prefs: Dict[str, Any] = {
        "inferred_analysis_family": family,
        "starter_template_key": st.session_state.get("research_agent_template_current") or st.session_state.get("research_agent_example_key"),
        "preferred_methods": method_pref.strip() or None,
        "evaluation_focus": evaluation_focus.strip() or None,
        "subgroup_sensitivity": subgroup_sensitivity.strip() or None,
        "timing_and_design": timing_design.strip() or None,
        "data_constraints": data_constraints.strip() or None,
        "must_have_outputs": must_have_outputs.strip() or None,
        "covariates": [c.strip() for c in covariates.split(",") if c.strip()],
        "extra_notes": extra.strip() or None,
    }
    return "\n".join(notes), prefs


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
    shared_provider = st.session_state.get("llm_provider", "")
    sidebar_hosted_blocked = shared_provider == "easyicu_hosted"
    options = [mock_choice]
    sdk_ok = handles["OpenAIClient"] is not None
    if sdk_ok:
        options = [mock_choice, override_choice] if sidebar_hosted_blocked else [sidebar_choice, mock_choice, override_choice]
    else:
        st.caption(_ra_text("sdk_missing"))
    if st.session_state.get("research_agent_llm_choice") not in (None, *options):
        st.session_state.pop("research_agent_llm_choice", None)
    default_index = (
        options.index(sidebar_choice)
        if sdk_ok and not sidebar_hosted_blocked and is_shared_llm_configured()
        else options.index(mock_choice)
    )
    choice = st.radio(
        _ra_text("llm_client"),
        options,
        index=default_index,
        key="research_agent_llm_choice",
    )

    if sidebar_hosted_blocked:
        st.info(_ra_text("hosted_blocked"))

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
            default="openai/gpt-oss-120b:free",
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
            "openai/gpt-oss-120b:free",
            "google/gemma-4-31b-it:free",
            "z-ai/glm-4.5-air:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "qwen/qwen3-coder:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "openrouter/free",
            "deepseek/deepseek-chat",
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
    default_workdir = str((Path.cwd() / "research_output" / "webapp").resolve())
    _hide_prefilled_directory_text("research_agent_workdir", default_workdir)
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
        workdir_text = _directory_input(
            _ra_text("workdir"),
            value=default_workdir,
            input_key="research_agent_workdir",
            button_key="research_agent_workdir_browse",
            placeholder=_placeholder_path("research_output/webapp"),
            show_value=False,
        )
    return disable_icu_context, (workdir_text or default_workdir), stop_choice == stop_options[0]


def _format_history_label(row: Dict[str, Any]) -> str:
    status = "partial" if row.get("manifest_partial") else "complete"
    started = str(row.get("started_at") or "")[:19].replace("T", " ")
    question = str(row.get("question") or "").strip()
    if len(question) > 64:
        question = question[:61] + "..."
    bits = [
        str(row.get("run_id") or ""),
        status,
        f"{row.get('step_ok', 0)}/{row.get('step_total', 0)} steps",
    ]
    if started:
        bits.append(started)
    if question:
        bits.append(question)
    return " · ".join(bits)


def _render_run_history(workdir: Path) -> None:
    selected_run: Dict[str, Any] | None = None
    with st.expander(_ra_text("history_title"), expanded=False):
        rows = _scan_research_agent_runs(workdir)
        if not rows:
            st.info(_ra_text("history_empty"))
            return
        table = pd.DataFrame([
            {
                _ra_text("history_run_id"): row["run_id"],
                _ra_text("history_status"): row["status"],
                _ra_text("history_started"): row["started_at"][:19].replace("T", " "),
                _ra_text("history_steps"): f"{row['step_ok']}/{row['step_total']}",
                _ra_text("history_figures"): row["figure_count"],
                _ra_text("history_tables"): row["table_count"],
                _ra_text("history_findings"): f"{row['finding_errors']}E / {row['finding_warnings']}W",
                _ra_text("history_question"): row["question"][:120],
            }
            for row in rows
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)
        labels = [_format_history_label(row) for row in rows]
        selected_label = st.selectbox(
            _ra_text("history_open"),
            labels,
            index=0,
            key="research_agent_history_pick",
        )
        selected_run = rows[labels.index(selected_label)]

    if selected_run:
        manifest, manifest_path, _partial = _load_run_manifest(selected_run["run_dir"])
        if manifest:
            st.markdown(f"### {_ra_text('history_selected')}: `{selected_run['run_id']}`")
            _render_run_manifest(
                run_dir=selected_run["run_dir"],
                manifest=manifest,
                manifest_path=manifest_path,
                key_prefix=f"research_agent_history_{selected_run['run_id']}",
            )


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
            "Family-specific tables, figures, diagnostics, and findings"
            if is_en else
            "按研究类型生成表格、图、诊断指标和结果发现",
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
        "In this prediction-style example, the analysis pack suggested clinically meaningful mortality risk separation. "
        "The draft is only generated after reviewing cohort balance, missingness, discrimination, and calibration."
        if is_en else
        "在这个“预测型”示例中，分析包显示死亡风险存在具有临床意义的分层。只有在复核队列构成、缺失、区分度和校准后，才进入文章生成。"
    )
    synthetic_banner = (
        "Synthetic illustration — these numbers, plots and finding cards are static demo content. "
        "They are not produced by a real EasyICU research-agent run and must not be cited as results."
        if is_en else
        "演示示意 — 下方的数字、图表和发现卡片均为静态示例，并非真实的 EasyICU 研究智能体运行结果，"
        "不可作为研究结论引用。"
    )
    synthetic_chip = "Synthetic" if is_en else "演示数据"
    finding_card_title = (
        "Example findings before manuscript (illustrative wording only)"
        if is_en else
        "文章前的结果复核示例（仅为示例文案）"
    )
    finding_card_note = (
        "In a real run the agent stops here so users can catch wrong cohorts, weak signal or bad calibration "
        "before spending writing tokens. The text below is sample wording, not a real finding."
        if is_en else
        "真实运行中智能体会停在这里，让用户在消耗写作 token 前发现队列错误、信号不足或校准较差。"
        "下面只是示例文案，不是真实发现。"
    )
    finding_text_1 = (
        "Example wording: SOFA-2 and lactate could carry most of the risk signal; calibration should be checked in the high-risk decile."
        if is_en else
        "示例文案：SOFA-2 和乳酸可能贡献主要风险信号；高风险分位的校准需要重点复核。"
    )
    finding_text_2 = (
        "Example wording: missingness acceptable for core predictors; manuscript drafting can be considered after sensitivity review."
        if is_en else
        "示例文案：核心预测变量缺失可接受；完成敏感性复核后可考虑生成文章。"
    )
    table_caption = (
        "Illustrative values only — these numbers are static demo content, not from a real run."
        if is_en else
        "仅为示意值 — 这些数字是静态演示内容，并非真实运行产生。"
    )
    disc_caption = (
        "Example AUROC / Brier values for layout only — not a real metric."
        if is_en else
        "AUROC / Brier 数值仅用于版面示意 — 并非真实指标。"
    )
    calib_caption = (
        "Example calibration curve — illustrative only."
        if is_en else
        "示例校准曲线 — 仅为示意。"
    )
    manuscript_caption = (
        "Example manuscript preview wording only."
        if is_en else
        "示例文章预览文案。"
    )
    st.markdown(
        f"""
        <div class="ra-demo-hero">
            <div class="ra-demo-flow">{flow_html}</div>
        </div>
        <div style="margin:0.6rem 0 0.4rem 0;padding:0.55rem 0.8rem;border:1px solid #f59e0b;
                    background:#fef3c7;color:#7c2d12;border-radius:8px;font-weight:600;
                    display:flex;gap:0.55rem;align-items:flex-start;">
            <span aria-hidden="true">⚠️</span>
            <span>{html.escape(synthetic_banner)}</span>
        </div>
        <div class="ra-output-grid">
            <div class="ra-output-card">
                <div class="ra-output-title">{"Table 1 preview" if is_en else "表 1 预览"}
                    <span style="margin-left:0.4rem;padding:0.05rem 0.4rem;border:1px solid #f59e0b;
                                 background:#fef3c7;color:#7c2d12;border-radius:999px;
                                 font-size:0.72rem;font-weight:700;">{html.escape(synthetic_chip)}</span>
                </div>
                <div class="ra-output-note">{html.escape(table_caption)}</div>
                <table class="ra-mini-table">
                    <thead><tr><th>{"Feature" if is_en else "变量"}</th><th>{"Alive" if is_en else "存活"}</th><th>{"Died" if is_en else "死亡"}</th><th>Δ</th></tr></thead>
                    <tbody>{table_html}</tbody>
                </table>
            </div>
            <div class="ra-output-card">
                <div class="ra-output-title">{"Discrimination" if is_en else "区分度"}
                    <span style="margin-left:0.4rem;padding:0.05rem 0.4rem;border:1px solid #f59e0b;
                                 background:#fef3c7;color:#7c2d12;border-radius:999px;
                                 font-size:0.72rem;font-weight:700;">{html.escape(synthetic_chip)}</span>
                </div>
                <div class="ra-output-note">AUROC 0.82 · Brier 0.14 · {html.escape(disc_caption)}</div>
                <svg viewBox="0 0 220 128" width="100%" height="128" role="img" aria-label="Synthetic ROC curve (illustrative only)">
                    <rect x="0" y="0" width="220" height="128" rx="10" fill="#f8fbff"/>
                    <line x1="28" y1="102" x2="196" y2="102" stroke="#cbd5e1" stroke-width="1.5"/>
                    <line x1="28" y1="102" x2="28" y2="18" stroke="#cbd5e1" stroke-width="1.5"/>
                    <line x1="28" y1="102" x2="196" y2="18" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4"/>
                    <polyline points="28,102 44,76 61,58 83,42 113,31 150,24 196,18" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round"/>
                    <text x="36" y="26" fill="#082957" font-size="12" font-weight="700">ROC</text>
                    <text x="196" y="120" fill="#b45309" font-size="10" font-weight="700"
                          text-anchor="end" opacity="0.85">{html.escape(synthetic_chip).upper()}</text>
                </svg>
            </div>
            <div class="ra-output-card">
                <div class="ra-output-title">{"Calibration" if is_en else "校准"}
                    <span style="margin-left:0.4rem;padding:0.05rem 0.4rem;border:1px solid #f59e0b;
                                 background:#fef3c7;color:#7c2d12;border-radius:999px;
                                 font-size:0.72rem;font-weight:700;">{html.escape(synthetic_chip)}</span>
                </div>
                <div class="ra-output-note">{html.escape(calib_caption)}</div>
                <svg viewBox="0 0 220 128" width="100%" height="128" role="img" aria-label="Synthetic calibration curve (illustrative only)">
                    <rect x="0" y="0" width="220" height="128" rx="10" fill="#f8fbff"/>
                    <line x1="28" y1="102" x2="196" y2="18" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4"/>
                    <polyline points="28,98 57,82 87,68 117,54 151,38 196,24" fill="none" stroke="#0f766e" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="57" cy="82" r="4" fill="#0f766e"/>
                    <circle cx="117" cy="54" r="4" fill="#0f766e"/>
                    <circle cx="196" cy="24" r="4" fill="#0f766e"/>
                    <text x="36" y="26" fill="#082957" font-size="12" font-weight="700">calibration</text>
                    <text x="196" y="120" fill="#b45309" font-size="10" font-weight="700"
                          text-anchor="end" opacity="0.85">{html.escape(synthetic_chip).upper()}</text>
                </svg>
            </div>
            <div class="ra-output-card wide">
                <div class="ra-output-title">{html.escape(finding_card_title)}
                    <span style="margin-left:0.4rem;padding:0.05rem 0.4rem;border:1px solid #f59e0b;
                                 background:#fef3c7;color:#7c2d12;border-radius:999px;
                                 font-size:0.72rem;font-weight:700;">{html.escape(synthetic_chip)}</span>
                </div>
                <div class="ra-output-note">{html.escape(finding_card_note)}</div>
                <div class="ra-finding">{html.escape(finding_text_1)}</div>
                <div class="ra-finding">{html.escape(finding_text_2)}</div>
            </div>
            <div class="ra-output-card">
                <div class="ra-output-title">{"Optional manuscript preview" if is_en else "可选文章预览"}
                    <span style="margin-left:0.4rem;padding:0.05rem 0.4rem;border:1px solid #f59e0b;
                                 background:#fef3c7;color:#7c2d12;border-radius:999px;
                                 font-size:0.72rem;font-weight:700;">{html.escape(synthetic_chip)}</span>
                </div>
                <div class="ra-output-note">{html.escape(manuscript_caption)}</div>
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
            "Question -> study family, cohort, design constraints, target variables, outputs, and analysis steps."
            if is_en else
            "把问题转成研究类型、队列、设计约束、目标变量、输出要求和分析步骤。"
        ),
        (
            "Data recipe" if is_en else "数据配方",
            "Stay-level parquet, module export folder, or guided EasyICU extraction."
            if is_en else
            "支持 stay-level parquet、模块导出文件夹，或引导 EasyICU 先提取。"
        ),
        (
            "Analysis pack" if is_en else "分析包",
            "Tables, figures, family-specific diagnostics, and findings."
            if is_en else
            "表格、图、与研究类型相匹配的诊断指标和结果摘要。"
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
        "Example workflow (prediction use case)"
        if is_en else
        "示例工作流（预测型用例）"
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
            "2. 定制方法、评估重点、时间设计、数据约束、队列筛选和停止点。",
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

    _lang = st.session_state.get("language", "en")
    _is_en = _lang == "en"
    _step_titles = [
        _ra_text("step1_title"),
        _ra_text("step2_title"),
        _ra_text("step3_title"),
        _ra_text("step4_title"),
        _ra_text("step5_title"),
    ]
    _step_required = [True, False, True, True, False]
    _required_label = "required" if _is_en else "必填"
    _optional_label = "optional" if _is_en else "可选"
    _stepper_legend = (
        "Suggested order. Steps 1, 3 and 4 are required to run the agent; steps 2 and 5 are optional."
        if _is_en else
        "推荐顺序。第 1、3、4 步为必填，第 2、5 步为可选。"
    )
    _stepper_items = "".join(
        f'<div style="display:flex;align-items:center;gap:0.45rem;padding:0.3rem 0.55rem;'
        f'border:1px solid {"#2563eb" if req else "#cbd5e1"};border-radius:999px;'
        f'background:{"#eff6ff" if req else "#f8fafc"};color:#0f172a;font-size:0.78rem;'
        f'font-weight:600;white-space:nowrap;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:1.35rem;height:1.35rem;border-radius:999px;background:{"#2563eb" if req else "#94a3b8"};'
        f'color:white;font-size:0.72rem;">{idx + 1}</span>'
        f'<span>{html.escape(title)}</span>'
        f'<span style="font-size:0.66rem;color:{"#2563eb" if req else "#64748b"};'
        f'text-transform:uppercase;letter-spacing:0.04em;">'
        f'{_required_label if req else _optional_label}</span>'
        f'</div>'
        for idx, (title, req) in enumerate(zip(_step_titles, _step_required))
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:0.45rem;margin:0.4rem 0 0.25rem 0;">{_stepper_items}</div>'
        f'<div style="font-size:0.78rem;color:#475569;margin-bottom:0.55rem;">{html.escape(_stepper_legend)}</div>',
        unsafe_allow_html=True,
    )

    with st.expander(f"1 · {_step_titles[0]}", expanded=True):
        free_question, target_outcome = _section_request_picker()
        skill_key = None
    with st.expander(f"2 · {_step_titles[1]} ({_optional_label})", expanded=False):
        method_notes, user_preferences = _section_method_preferences(free_question, target_outcome)
    question_hint = free_question
    with st.expander(f"3 · {_step_titles[2]}", expanded=True):
        cohort, cohort_label = _section_cohort_picker(research_question=question_hint)
    with st.expander(f"4 · {_step_titles[3]}", expanded=False):
        llm_choice, api_key, model, base_url, extra_headers = _section_llm_picker(handles)
    with st.expander(f"5 · {_step_titles[4]} ({_optional_label})", expanded=False):
        disable_icu_context, workdir_text, stop_after_analysis = _section_options()
    resume_run_id = st.session_state.get("research_agent_resume_run_id")
    force_manuscript = bool(st.session_state.get("research_agent_force_manuscript"))
    if force_manuscript:
        stop_after_analysis = False

    st.divider()
    history_workdir = Path(workdir_text).expanduser().resolve()
    _render_run_history(history_workdir)

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
    live_steps = st.empty()
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
        milestones = [
            e for e in progress_events
            if str(e.get("stage") or "") in _RA_MILESTONE_STAGES
            or e.get("status") in {"complete", "error", "paused"}
        ]
        latest = progress_events[-1:]
        milestone_lines = [_progress_event_line(e) for e in milestones[-8:]]
        latest_lines = [_progress_event_line(e) for e in latest]
        blocks = []
        if milestone_lines:
            blocks.append(
                f"**{_ra_text('run_milestones')}**\n"
                + "\n".join(f"- {line}" for line in milestone_lines)
            )
        if latest_lines and (not milestone_lines or latest_lines[-1] != milestone_lines[-1]):
            blocks.append(
                f"**{_ra_text('latest_activity')}**\n"
                + "\n".join(f"- {line}" for line in latest_lines)
            )
        progress_log.markdown("\n\n".join(blocks))
        run_id = event.get("run_id")
        if run_id:
            manifest, _manifest_path, _partial = _load_run_manifest(workdir / str(run_id))
            if manifest:
                with live_steps.container():
                    st.markdown(f"### {_ra_text('live_steps')}")
                    _render_literature_and_plan(workdir / str(run_id), manifest)
                    if manifest.get("per_step_records"):
                        st.divider()
                        _render_step_records(
                            workdir / str(run_id),
                            manifest,
                            key_prefix=f"research_agent_live_{run_id}",
                        )

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
                user_preferences=user_preferences,
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
    live_steps.empty()
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
