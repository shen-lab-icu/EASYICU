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
import hashlib
import json
import re
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from easyicu.webapp.ai_optin import AIOptInError, enforce_external_llm_opt_in
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


_LOCAL_LLM_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _assert_local_llm_endpoint_reachable(base_url: Optional[str], *, timeout: float = 1.0) -> None:
    """Fail fast when a local OpenAI-compatible endpoint is not listening."""
    if not base_url:
        return
    parsed = urlparse(str(base_url))
    host = parsed.hostname
    if not host or host.lower() not in _LOCAL_LLM_HOSTS:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    try:
        with socket.create_connection((connect_host, int(port)), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(
            f"Local LLM endpoint is unreachable at {base_url}. Start the local "
            "API service, update the Base URL, or choose MockLLMClient for an "
            "offline test run."
        ) from exc


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
        _assert_local_llm_endpoint_reachable(base_url)
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


def _bind_workbench_state(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    partial: Optional[bool] = None,
    progress_events: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Populate the Shell-A Workbench from a real run manifest."""
    try:
        from easyicu.webapp.agent_workbench import build_workbench_state_from_manifest

        st.session_state["_agent_workbench"] = build_workbench_state_from_manifest(
            run_dir,
            manifest,
            lang=st.session_state.get("language", "en"),
            partial=partial,
            progress_events=progress_events,
        )
        st.session_state["_agent_workbench_source_run_dir"] = str(run_dir)
        st.session_state["_agent_workbench_is_active_selection"] = True
    except Exception:
        # Workbench binding is an observation layer; the canonical report
        # renderer below must remain available even if the visual adapter fails.
        pass


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
    review = _load_review_decision(run_dir)
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
        "review_decision": str(review.get("decision") or ""),
        "review_updated_at": str(review.get("updated_at") or ""),
    }


def _review_decision_path(run_dir: Path) -> Path:
    return Path(run_dir) / "review_decision.json"


def _load_review_decision(run_dir: Path) -> Dict[str, Any]:
    path = _review_decision_path(run_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_review_decision(
    run_dir: Path,
    *,
    decision: str,
    note: str,
    manifest: Optional[Dict[str, Any]] = None,
) -> Path:
    path = _review_decision_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision": str(decision),
        "note": str(note or ""),
        "run_id": str((manifest or {}).get("run_id") or Path(run_dir).name),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "easyicu_web_research_agent",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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


def _gate_passed(value: Any) -> Optional[bool]:
    """Best-effort pass/fail read of a readiness-gate value.

    Readiness-gate values are loosely typed (bool, status dict, or string),
    so this normalises them; an unknown shape returns None (rendered neutral).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "ok", "ready", "complete", "satisfied"):
            if isinstance(value.get(key), bool):
                return value[key]
        status = str(value.get("status", "")).lower()
        if status in {"pass", "passed", "ok", "ready", "complete"}:
            return True
        if status in {"fail", "failed", "blocked", "incomplete"}:
            return False
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"pass", "passed", "ok", "ready", "complete", "true", "yes"}:
            return True
        if low in {"fail", "failed", "blocked", "incomplete", "false", "no"}:
            return False
    return None


def _render_reproducibility_panel(manifest: Dict[str, Any]) -> None:
    """Surface the evidence-enforcement / reproducibility story for a run.

    Everything here is already in the run manifest — this panel makes the
    fail-closed readiness gates, validator findings and LLM reproducibility
    envelope visible instead of leaving them buried in the raw JSON.
    """
    is_en = st.session_state.get("language", "en") == "en"
    findings = [f for f in manifest.get("findings", []) if isinstance(f, dict)]
    readiness = manifest.get("readiness")
    readiness = readiness if isinstance(readiness, dict) else {}
    repro = manifest.get("reproducibility")
    repro = repro if isinstance(repro, dict) else {}
    used_mock = bool(manifest.get("used_mock_llm"))

    errors = [f for f in findings if f.get("severity") == "error"]
    warnings = [f for f in findings if f.get("severity") == "warning"]
    infos = [f for f in findings if f.get("severity") == "info"]

    title = ("🔒 Reproducibility & Evidence Enforcement"
             if is_en else "🔒 可复现性与证据校验")
    with st.expander(title, expanded=True):
        if errors:
            st.error(
                f"{len(errors)} error-severity finding(s). Under STRICT evidence "
                f"enforcement these would block the bound manuscript."
                if is_en else
                f"{len(errors)} 个 error 级问题。在 STRICT 强制模式下会阻止生成绑定手稿。"
            )
        else:
            st.success(
                "No error-severity findings — this run satisfies STRICT evidence enforcement."
                if is_en else
                "无 error 级问题 —— 该 run 可通过 STRICT 强制校验。"
            )

        # Only the boolean entries are fail-closed gates; readiness also
        # carries count/list diagnostics (step counts, missing steps) that
        # would be noise in the gate grid.
        gate_items = [
            (name, ok)
            for name, value in readiness.items()
            if (ok := _gate_passed(value)) is not None
        ]
        if gate_items:
            st.markdown(
                f"**{'Fail-closed readiness gates' if is_en else '失败即拦截的就绪门控'}**"
            )
            grid = st.columns(min(3, len(gate_items)))
            for idx, (name, ok) in enumerate(gate_items):
                icon = "✅" if ok else "❌"
                grid[idx % len(grid)].markdown(f"{icon} {str(name).replace('_', ' ')}")
            done = readiness.get("completed_step_count")
            total = readiness.get("required_step_count")
            if isinstance(done, int) and isinstance(total, int) and total:
                st.caption(
                    f"Execution: {done}/{total} planned steps completed."
                    if is_en else
                    f"执行进度：{total} 个计划步骤中完成 {done} 个。"
                )

        st.markdown(f"**{'Validator findings' if is_en else '校验器发现'}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Errors" if is_en else "错误", len(errors))
        c2.metric("Warnings" if is_en else "警告", len(warnings))
        c3.metric("Info", len(infos))
        for finding in errors + warnings:
            sev = finding.get("severity", "info")
            st.markdown(
                f"{_FINDING_BADGE.get(sev, '⚪')} **`{finding.get('validator', '?')}`** — "
                f"{finding.get('message', '')}"
            )

        st.markdown(f"**{'LLM reproducibility' if is_en else 'LLM 可复现性'}**")
        if used_mock:
            st.caption(
                "Mock LLM — deterministic responses, no external model call."
                if is_en else
                "使用 Mock LLM —— 确定性响应，未调用外部模型。"
            )
        if repro:
            calls = repro.get("calls") or repro.get("llm_calls") or []
            meta_bits: List[str] = []
            for key in ("provider", "model", "requested_seed", "seed", "temperature"):
                if repro.get(key) not in (None, ""):
                    meta_bits.append(f"{key}={repro[key]}")
            if isinstance(calls, list) and calls:
                meta_bits.append(
                    f"{len(calls)} call(s) hashed" if is_en
                    else f"已记录 {len(calls)} 次调用哈希"
                )
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            with st.expander(
                "Raw reproducibility envelope" if is_en else "原始可复现性信封",
                expanded=False,
            ):
                st.json(repro)
        elif not used_mock:
            st.caption(
                "No reproducibility envelope recorded for this run."
                if is_en else
                "该 run 未记录可复现性信封。"
            )


def _render_review_decision_controls(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    key_prefix: str,
) -> None:
    """Persist a local human review decision beside a run manifest."""
    lang = st.session_state.get("language", "en")
    is_en = lang == "en"
    review = _load_review_decision(run_dir)
    options = [
        ("approved", "Mark reviewed" if is_en else "标记已复核"),
        ("repair_requested", "Request repair / rerun" if is_en else "请求修复 / 重跑"),
        ("locked", "Keep manuscript locked" if is_en else "保持手稿锁定"),
        ("blocked", "Mark blocked" if is_en else "标记阻塞"),
    ]
    current_decision = str(review.get("decision") or "locked")
    current_index = next((i for i, (value, _label) in enumerate(options) if value == current_decision), 2)
    safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", key_prefix)
    with st.expander(
        "Reviewer decision" if is_en else "人工审核决定",
        expanded=bool(review),
    ):
        if review:
            st.success(
                (
                    f"Current decision: {review.get('decision')} · {review.get('updated_at', '')}"
                    if is_en else
                    f"当前决定：{review.get('decision')} · {review.get('updated_at', '')}"
                )
            )
            if review.get("note"):
                st.caption(str(review.get("note")))
        decision_label = st.radio(
            "Decision" if is_en else "审核决定",
            [label for _value, label in options],
            index=current_index,
            horizontal=True,
            key=f"{safe_key}_review_decision_choice",
        )
        decision_value = options[[label for _value, label in options].index(decision_label)][0]
        note_text = st.text_area(
            "Review note" if is_en else "审核备注",
            value=str(review.get("note") or ""),
            height=76,
            key=f"{safe_key}_review_decision_note",
            help=(
                "This writes a local review_decision.json next to the manifest."
                if is_en else
                "会在 manifest 旁写入本地 review_decision.json。"
            ),
        )
        if st.button(
            "Save review decision" if is_en else "保存审核决定",
            type="primary",
            use_container_width=True,
            key=f"{safe_key}_review_decision_save",
        ):
            path = _write_review_decision(
                run_dir,
                decision=decision_value,
                note=note_text,
                manifest=manifest,
            )
            st.success(
                f"Saved to `{path.name}`."
                if is_en else
                f"已保存到 `{path.name}`。"
            )
            st.rerun()


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

    _render_reproducibility_panel(manifest)
    _render_review_decision_controls(
        run_dir=run_dir,
        manifest=manifest,
        key_prefix=f"{key_prefix}_{result.run_id}",
    )

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
    patient_ids: Optional[Sequence[Any]] = None,
) -> Optional["pd.DataFrame"]:
    """Convert ``st.session_state.loaded_concepts`` into a stay-level frame.

    Each concept dataframe is reduced to **one row per stay** by taking
    the most recent value (or the value itself if already stay-level),
    then horizontally merged on ``id_col``. Drops empty / unparseable
    concepts silently rather than blocking the handoff.
    """
    if not loaded_concepts:
        return None
    patient_id_set = set(patient_ids or [])
    base: Optional[pd.DataFrame] = None
    for concept, df in loaded_concepts.items():
        if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns:
            continue
        if patient_id_set:
            df = df[df[id_col].isin(patient_id_set)]
            if df.empty:
                continue
        # Pick the value column (first non-id, non-time-like column).
        time_cols = {"charttime", "starttime", "endtime", "time", "timestamp",
                     "stay_id_time", "_time", "time_to_event"}
        value_cols = [
            c for c in df.columns
            if c != id_col and str(c).lower() not in time_cols
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
    if base is not None and patient_id_set:
        base = base[base[id_col].isin(patient_id_set)].reset_index(drop=True)
    return base


_DB_TAG_CANDIDATES = ('miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic')


def _infer_db_tag_from_folder(folder: Path) -> str:
    """Guess the database tag from an EasyICU module-export folder name.

    Most prepared exports are named ``<dbtag>_<date>`` (e.g.
    ``miiv_20260420`` / ``eicu_20260512``) — we extract the leading
    token. Falls back to the parent folder name, then to the empty
    string so the caller can let the user override.
    """
    name = folder.name.lower()
    for tag in _DB_TAG_CANDIDATES:
        if name == tag or name.startswith(tag + '_') or name.startswith(tag + '-'):
            return tag
    parent = folder.parent.name.lower()
    for tag in _DB_TAG_CANDIDATES:
        if parent == tag or parent.startswith(tag + '_'):
            return tag
    return ''


def _discover_db_export_folders() -> List[Path]:
    """Return all detectable EasyICU module-export folders under known
    workspace roots plus the session's last-export folder."""
    extra_roots: List[Path] = []
    for key in ('export_path', 'last_export_dir'):
        p = st.session_state.get(key) or ''
        if not p:
            continue
        try:
            resolved = Path(p).expanduser().resolve()
            extra_roots.append(resolved)
            parent = resolved.parent
            if parent not in extra_roots and parent != resolved:
                extra_roots.append(parent)
        except Exception:
            pass
    return _scan_workspace_for_module_dirs(extra_roots + _candidate_cohort_roots())


def _render_db_exports_multipicker(
    *, key_prefix: str, min_selected: int = 1, intro: str = '',
) -> List[Tuple[str, Path]]:
    """Render a multi-select of detected EasyICU exports with per-row
    DB-tag override. Returns ``[(db_tag, folder), ...]`` for the chosen
    rows. Shared by the cross-DB cohort builder and the replication
    runner so both surfaces stay aligned.
    """
    dirs = _discover_db_export_folders()
    if not dirs:
        st.info(_ra_text(
            'multi_db_no_dirs' if min_selected >= 2 else 'replication_no_dirs'
        ))
        return []
    if intro:
        st.caption(intro)
    dir_labels = [_display_path(p) for p in dirs]
    picked_labels = st.multiselect(
        _ra_text('multi_db_pick' if min_selected >= 2 else 'replication_pick'),
        dir_labels,
        default=dir_labels[: max(min_selected, 2)] if len(dir_labels) >= min_selected else dir_labels,
        key=f'{key_prefix}_picks',
    )
    if not picked_labels:
        return []
    chosen: List[Tuple[str, Path]] = []
    for label in picked_labels:
        folder = dirs[dir_labels.index(label)]
        inferred = _infer_db_tag_from_folder(folder) or _DB_TAG_CANDIDATES[0]
        cols = st.columns([1.4, 4])
        with cols[0]:
            tag = st.selectbox(
                f'DB tag for {folder.name}',
                list(_DB_TAG_CANDIDATES),
                index=list(_DB_TAG_CANDIDATES).index(inferred),
                key=f'{key_prefix}_tag_{folder.name}',
                label_visibility='collapsed',
            )
        with cols[1]:
            st.caption(f'`{folder}`')
        chosen.append((tag, folder))
    return chosen


def _build_multi_db_cohort() -> Tuple[Optional[pd.DataFrame], str]:
    """Cross-DB cohort source: pick N exports → build per-DB stay-level
    cohort → concat with a ``database`` column."""
    st.caption(_ra_text('multi_db_intro'))
    chosen = _render_db_exports_multipicker(
        key_prefix='research_agent_multi_db', min_selected=2,
    )
    if not chosen:
        return None, ''
    if len(chosen) < 2:
        st.info(_ra_text('multi_db_need_two'))
        return None, ''

    frames: List[pd.DataFrame] = []
    per_db_summary: List[str] = []
    for db_tag, folder in chosen:
        module_files = _list_module_parquets(folder)
        if not module_files:
            st.warning(_ra_text('no_module_parquets', folder=folder))
            continue
        # Use first detected id column shared across files.
        id_counts: Dict[str, int] = {}
        for p in module_files:
            for c in _parquet_file_summary(p).get('id_columns') or []:
                id_counts[c] = id_counts.get(c, 0) + 1
        if not id_counts:
            st.warning(f'`{folder}` — no shared id column detected.')
            continue
        id_col = max(id_counts, key=lambda c: id_counts[c])
        try:
            sub = _build_stay_level_from_module_folder(
                folder=folder,
                selected_files=module_files,
                id_col=id_col,
                join_how='outer',
            )
        except Exception as exc:
            st.warning(f'`{folder}` — build failed: {exc}')
            continue
        if sub is None or sub.empty:
            continue
        sub = sub.copy()
        sub.insert(0, 'database', db_tag)
        frames.append(sub)
        per_db_summary.append(f'{db_tag}={len(sub):,}')

    if not frames:
        st.error('No databases could be loaded; check the warnings above.')
        return None, ''

    cohort = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    st.success(_ra_text(
        'multi_db_built', rows=len(cohort), dbs=len(frames),
        per_db=', '.join(per_db_summary),
    ))
    st.dataframe(cohort.head(8), use_container_width=True, hide_index=True)
    return cohort, f'multi_db:{",".join(d for d, _ in chosen)}'


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
    source_multi_db = _ra_text("source_multi_db")
    source_synthetic = _ra_text("source_synthetic")

    # 2026-05 Phase G: removed `source_workspace` (pick a parquet from
    # workspace) — it was a strict subset of `source_module` (pick a
    # folder, then pick a parquet inside it). Added `source_multi_db` to
    # surface the cross-database cohort builder that the experiment_spec
    # already supports via cross_database_validation.
    options: List[str] = []
    if has_inbound:
        options.append(source_handoff)
    if has_loaded_concepts and not has_inbound:
        options.append(source_loaded)
    if st.session_state.get("entry_mode") == "real":
        options += [
            source_no_data,
            source_module,
            source_multi_db,
            source_upload,
            source_synthetic,
        ]
    else:
        options += [
            source_synthetic,
            source_upload,
            source_module,
            source_multi_db,
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
            df = _stay_level_from_loaded_concepts(
                loaded_concepts,
                id_col=id_col,
                patient_ids=st.session_state.get("patient_ids") or None,
            )
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

    # Cross-DB cohort builder: pick 2+ module export folders, infer each
    # one's database tag from its folder name (or let the user override),
    # build per-DB stay-level cohorts via the existing single-folder
    # builder, then concatenate with a `database` column so downstream
    # cross_database_validation can stratify.
    if source == source_multi_db:
        return _build_multi_db_cohort()

    # Should not reach — every option above has a return path.
    return None, ""


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
    # 2026-05 Phase G: prefer a real API endpoint over MockLLMClient by
    # default. Most users don't know what MockLLMClient is, and seeing a
    # deterministic mock pipeline as "the default agent run" was misleading.
    # Priority: sidebar-configured shared LLM > override (Custom OpenAI/
    # OpenRouter, prompts for key) > Mock (only as last-resort offline path).
    if sdk_ok and not sidebar_hosted_blocked and is_shared_llm_configured():
        default_index = options.index(sidebar_choice)
    elif sdk_ok and override_choice in options:
        default_index = options.index(override_choice)
    else:
        default_index = options.index(mock_choice)
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
    if _env_or_secret("OPENAI_BASE_URL") or _env_or_secret("EASYICU_CUSTOM_MODEL"):
        default_override_client = "Custom OpenAI-compatible"
    elif _env_or_secret("OPENROUTER_API_KEY"):
        default_override_client = "OpenRouter"
    elif _env_or_secret("OPENAI_API_KEY"):
        default_override_client = "OpenAI"
    else:
        default_override_client = "OpenAI"
    override_client = st.selectbox(
        _ra_text("llm_client"),
        override_options,
        index=override_options.index(default_override_client),
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
    expand_history = bool(st.session_state.pop("_research_agent_expand_history", False))
    with st.expander(_ra_text("history_title"), expanded=expand_history):
        history_loaded = bool(st.session_state.get("_research_agent_history_loaded")) or expand_history
        if not history_loaded:
            st.caption(
                "History is loaded on demand so Setup stays responsive."
                if st.session_state.get("language", "en") == "en" else
                "历史记录按需加载，避免配置页初始渲染变慢。"
            )
            if st.button(
                "Load recent runs" if st.session_state.get("language", "en") == "en" else "加载最近 run",
                key="research_agent_history_load",
                use_container_width=True,
            ):
                st.session_state["_research_agent_history_loaded"] = True
                st.rerun()
            return
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
                "Review" if st.session_state.get("language", "en") == "en" else "审核": row.get("review_decision") or "not reviewed",
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
            cols = st.columns([1.4, 1.0, 1.0, 4.0])
            cols[0].metric(_ra_text("history_steps"), f"{selected_run['step_ok']}/{selected_run['step_total']}")
            cols[1].metric(_ra_text("history_findings"), f"{selected_run['finding_errors']}E / {selected_run['finding_warnings']}W")
            cols[2].metric(_ra_text("history_figures"), selected_run["figure_count"])
            safe_run_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(selected_run["run_id"]))
            if cols[3].button(
                "Open in Workbench" if st.session_state.get("language", "en") == "en" else "在工作台打开",
                key=f"research_agent_history_open_wb_{safe_run_id}",
                type="primary",
                use_container_width=True,
            ):
                _bind_workbench_state(
                    run_dir=selected_run["run_dir"],
                    manifest=manifest,
                    partial=_partial,
                )
                st.session_state["_ra_view"] = "workbench"
                st.rerun()
            with st.expander(
                "Detailed report and artefacts" if st.session_state.get("language", "en") == "en" else "详细报告与产物",
                expanded=False,
            ):
                _render_run_manifest(
                    run_dir=selected_run["run_dir"],
                    manifest=manifest,
                    manifest_path=manifest_path,
                    key_prefix=f"research_agent_history_{selected_run['run_id']}",
                )


def _render_research_agent_demo_visuals(*, is_en: bool) -> None:
    """Render a compact, non-token guide to the Research Agent workflow."""
    flow = [
        (
            "01",
            "Plan" if is_en else "规划",
            "Question -> study recipe"
            if is_en else
            "问题 -> 研究配方",
            "",
        ),
        (
            "02",
            "Build" if is_en else "组装",
            "EasyICU exports -> one row per ICU stay"
            if is_en else
            "EasyICU 导出 -> 每次 ICU 住院一行",
            "",
        ),
        (
            "03",
            "Analyze" if is_en else "分析",
            "Tables, figures, diagnostics, findings"
            if is_en else
            "表格、图、诊断、结果发现",
            "",
        ),
        (
            "04",
            "Gate" if is_en else "关口",
            "Evidence checks before drafting"
            if is_en else
            "写作前先过证据检查",
            " review",
        ),
    ]
    flow_html = ""
    for label, title, body, klass in flow:
        flow_html += (
            f'<div class="ra-demo-node{klass}">'
            f'<div class="ra-demo-node-label">{html.escape(label)}</div>'
            f'<div class="ra-demo-node-title">{html.escape(title)}</div>'
            f'<div class="ra-demo-node-body">{html.escape(body)}</div>'
            '</div>'
        )

    deliverables = [
        "Study plan" if is_en else "研究方案",
        "Cohort table" if is_en else "队列表",
        "Results report" if is_en else "结果报告",
        "Tables + figures" if is_en else "表格 + 图",
        "Evidence manifest" if is_en else "证据清单",
        "Optional draft" if is_en else "可选草稿",
    ]
    deliverables_html = "".join(
        f'<span class="ra-demo-chip">{html.escape(item)}</span>'
        for item in deliverables
    )
    value_cards = [
        (
            "Bring" if is_en else "输入",
            "Question + ICU data"
            if is_en else
            "问题 + ICU 数据",
        ),
        (
            "Agent adds" if is_en else "Agent 增加",
            "Plan, cohort, analysis, evidence checks"
            if is_en else
            "规划、队列、分析、证据检查",
        ),
        (
            "Get" if is_en else "产出",
            "Reviewable results first, draft later"
            if is_en else
            "先复核结果，后生成草稿",
        ),
    ]
    value_cards_html = "".join(
        '<div class="ra-value-card">'
        f'<div class="ra-value-card-title">{html.escape(title)}</div>'
        f'<div class="ra-value-card-body">{html.escape(body)}</div>'
        '</div>'
        for title, body in value_cards
    )
    demo_title = (
        "Question + EasyICU data -> evidence-bound research output"
        if is_en else
        "研究问题 + EasyICU 数据 -> 绑定证据的研究产出"
    )
    demo_body = (
        "The agent turns prepared ICU data into a reviewable analysis pack, then drafts only after the evidence gate."
        if is_en else
        "Agent 先把 ICU 数据变成可复核分析包，通过证据关口后才进入文章草稿。"
    )
    value_title = (
        "Real run outputs"
        if is_en else
        "真实运行会产出"
    )
    demo_note = (
        "Static demo. No fake metrics."
        if is_en else
        "静态 Demo：不编造指标。"
    )

    st.markdown(
        f"""
        <div class="ra-demo-hero">
            <div class="ra-demo-intro">
                <div>
                    <div class="ra-demo-kicker">{"Demo guide" if is_en else "Demo 导览"}</div>
                    <div class="ra-demo-heading">{html.escape(demo_title)}</div>
                    <div class="ra-demo-copy">{html.escape(demo_body)}</div>
                </div>
                <div class="ra-demo-note">{html.escape(demo_note)}</div>
            </div>
            <div class="ra-value-grid">{value_cards_html}</div>
            <div class="ra-demo-flow">{flow_html}</div>
        </div>
        <div class="ra-output-grid" style="grid-template-columns: 1fr;">
            <div class="ra-output-card">
                <div class="ra-output-title">{html.escape(value_title)}</div>
                <div class="ra-demo-chip-row">{deliverables_html}</div>
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
        icon="",
        kicker=_ra_text("kicker"),
    )

    st.info(
        "Use Real Data Mode when you are ready to connect a stay-level file, an EasyICU module export folder, "
        "or let EasyICU prepare data first."
        if is_en else
        "准备接入 stay-level 文件、EasyICU 模块导出文件夹，或让 EasyICU 先提取数据时，再进入真实数据模式。"
    )
    _render_research_agent_demo_visuals(is_en=is_en)

    # 2026-05 Phase F simplification: removed the 4-column "Study plan /
    # Data recipe / Analysis pack / Manuscript draft" row, the "Example
    # workflow" question-and-outputs block, and the 3-step "Real-data
    # workflow" list. All three duplicated what the 4-step flow at the
    # top and the section pickers in Real Data Mode already show. The
    # CTA below is enough to get users into the real flow.

    st.divider()
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


def _render_replication_section(*, default_workdir: Path) -> None:
    """Thin web entry point for the deterministic lactate-MAP-vaso
    replication package (``easyicu-research-replication`` without
    ``--paper``). Reuses the multi-DB exports multipicker so users can
    add multiple databases just by checking more rows.

    Kept deliberately minimal:
    * No paper-aware (LLM) mode — that path needs API keys, a paper
      file, and writes a manuscript draft; users who want it should
      use the CLI for now.
    * Window defaults to 0–24 h (the package's default).
    * Output dir defaults to ``<workdir>/replication``.
    """
    st.caption(_ra_text("replication_caption"))
    chosen = _render_db_exports_multipicker(
        key_prefix="research_agent_replication", min_selected=1,
    )

    col1, col2 = st.columns([1.0, 1.0])
    with col1:
        win_start = st.number_input(
            _ra_text("replication_window") + " — start",
            value=0.0, step=1.0,
            key="research_agent_replication_win_start",
        )
    with col2:
        win_end = st.number_input(
            _ra_text("replication_window") + " — end",
            value=24.0, step=1.0,
            key="research_agent_replication_win_end",
        )
    output_dir_default = str(default_workdir / "replication")
    output_dir = st.text_input(
        _ra_text("replication_output"),
        value=output_dir_default,
        key="research_agent_replication_output",
    )

    run_clicked = st.button(
        _ra_text("replication_run"),
        type="primary",
        disabled=not chosen,
        use_container_width=True,
        key="research_agent_replication_run",
    )
    if not chosen:
        st.info(_ra_text("replication_need_one"))
        return
    if not run_clicked:
        return

    try:
        from easyicu.research_agent.replication import (
            run_lactate_map_vaso_replication,
        )
    except Exception as exc:
        st.error(f"Could not import replication module: {exc}")
        return

    targets: Dict[str, Optional[Path]] = {tag: folder for tag, folder in chosen}
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    with st.spinner(_ra_text("replication_running")):
        try:
            paths = run_lactate_map_vaso_replication(
                targets,
                str(output_path),
                window=(float(win_start), float(win_end)),
            )
        except Exception as exc:
            st.error(f"Replication failed: {type(exc).__name__}: {exc}")
            st.code(traceback.format_exc())
            return

    st.success(_ra_text("replication_done"))
    rows = [{"output": name, "path": str(p)} for name, p in paths.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _template_contract(template_key: Optional[str], *, language: str = "en") -> Dict[str, Any]:
    """Return expected outputs/checkpoints for the selected clinical template."""
    is_en = language == "en"
    defaults = {
        "label": "Free-form research request" if is_en else "自由研究问题",
        "expected_outputs": [
            "analysis plan" if is_en else "分析计划",
            "evidence manifest" if is_en else "证据清单",
            "results report" if is_en else "结果报告",
        ],
        "checkpoints": [
            "cohort denominator reported" if is_en else "报告队列分母",
            "numeric claims have evidence references" if is_en else "数值主张带证据引用",
            "audit findings gate manuscript drafting" if is_en else "审计发现控制手稿生成",
        ],
    }
    templates: Dict[str, Dict[str, List[str] | str]] = {
        "prediction": {
            "label": "Prediction model" if is_en else "预测模型",
            "expected_outputs": [
                "cohort summary",
                "train/test performance table",
                "calibration and discrimination figure",
                "feature effects table",
            ] if is_en else ["队列摘要", "训练/测试性能表", "校准与区分度图", "变量效应表"],
            "checkpoints": [
                "explicit train/test or bootstrap design",
                "AUROC, Brier score, and calibration reported",
                "missingness and leakage checks completed",
            ] if is_en else ["明确训练/测试或 bootstrap 设计", "报告 AUROC、Brier 与校准", "完成缺失和泄漏检查"],
        },
        "association": {
            "label": "Association analysis" if is_en else "相关性分析",
            "expected_outputs": [
                "cohort and missingness table",
                "adjusted model table",
                "odds ratio or effect plot",
            ] if is_en else ["队列与缺失表", "调整模型表", "OR 或效应图"],
            "checkpoints": [
                "exposure, outcome, and covariates named",
                "denominator and complete-case loss reported",
                "causal overclaim warnings retained",
            ] if is_en else ["明确暴露、结局和协变量", "报告分母与完整病例损耗", "保留因果过度解释警告"],
        },
        "validation": {
            "label": "External validation / score benchmarking" if is_en else "外部验证 / 评分比较",
            "expected_outputs": [
                "per-database cohort table",
                "transportability summary",
                "calibration and discrimination panels",
            ] if is_en else ["逐数据库队列表", "迁移性摘要", "校准与区分度 panel"],
            "checkpoints": [
                "per-database concept availability checked",
                "same cohort rule applied across databases",
                "performance heterogeneity reported",
            ] if is_en else ["检查逐库概念可用性", "跨库应用同一队列规则", "报告性能异质性"],
        },
        "data_quality": {
            "label": "Data-quality / harmonization audit" if is_en else "数据质量 / 映射审计",
            "expected_outputs": [
                "coverage table",
                "missingness figure",
                "unit/range audit",
                "source mapping notes",
            ] if is_en else ["覆盖表", "缺失图", "单位/范围审计", "源映射说明"],
            "checkpoints": [
                "out-of-range values flagged",
                "time alignment checked",
                "cross-module mapping issues listed",
            ] if is_en else ["标记越界值", "检查时间对齐", "列出跨模块映射问题"],
        },
    }
    if template_key and template_key in templates:
        out = dict(defaults)
        out.update(templates[template_key])
        return out
    return defaults


def _build_execution_preflight_contract(
    *,
    free_question: str,
    target_outcome: str,
    cohort: Optional[pd.DataFrame],
    cohort_label: str,
    llm_choice: str,
    model: str,
    workdir_text: str,
    stop_after_analysis: bool,
    force_manuscript: bool,
    template_key: Optional[str],
    language: str = "en",
) -> Dict[str, Any]:
    question = (free_question or target_outcome or "").strip()
    try:
        cohort_rows = len(cohort) if cohort is not None else 0
    except Exception:
        cohort_rows = 0
    try:
        cohort_cols = [str(c) for c in list(cohort.columns)[:16]] if cohort is not None else []
    except Exception:
        cohort_cols = []
    external_llm = "MockLLMClient" not in str(llm_choice) and "offline" not in str(llm_choice).lower()
    output_dir = str(Path(workdir_text).expanduser())
    write_targets = [
        str(Path(output_dir) / "run_<timestamp>" / "manifest.json"),
        str(Path(output_dir) / "run_<timestamp>" / "results_report.md"),
    ]
    if force_manuscript or not stop_after_analysis:
        write_targets.append(str(Path(output_dir) / "run_<timestamp>" / "manuscript_scaffold_bound.md"))
    return {
        "question": question,
        "target_outcome": target_outcome or "",
        "cohort_label": cohort_label or "",
        "cohort_rows": cohort_rows,
        "cohort_columns": cohort_cols,
        "llm_choice": str(llm_choice or "mock"),
        "model": str(model or "default"),
        "external_llm": external_llm,
        "workdir": output_dir,
        "write_targets": write_targets,
        "mode": "draft_continuation" if force_manuscript else ("analysis_only" if stop_after_analysis else "analysis_plus_manuscript_gate"),
        "template_key": template_key or "",
        "template_contract": _template_contract(template_key, language=language),
    }


def _preflight_signature(contract: Dict[str, Any]) -> str:
    payload = {
        key: contract.get(key)
        for key in (
            "question",
            "target_outcome",
            "cohort_label",
            "cohort_rows",
            "cohort_columns",
            "llm_choice",
            "model",
            "external_llm",
            "workdir",
            "write_targets",
            "mode",
            "template_key",
        )
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _render_execution_preflight(
    *,
    free_question: str,
    target_outcome: str,
    cohort: Optional[pd.DataFrame],
    cohort_label: str,
    llm_choice: str,
    model: str,
    workdir_text: str,
    stop_after_analysis: bool,
    force_manuscript: bool,
) -> bool:
    """Show the human-confirmation contract before a real agent run."""
    lang = st.session_state.get("language", "en")
    is_en = lang == "en"
    template_key = st.session_state.get("research_agent_template_current") or st.session_state.get("research_agent_example_key")
    contract = _build_execution_preflight_contract(
        free_question=free_question,
        target_outcome=target_outcome,
        cohort=cohort,
        cohort_label=cohort_label,
        llm_choice=llm_choice,
        model=model,
        workdir_text=workdir_text,
        stop_after_analysis=stop_after_analysis,
        force_manuscript=force_manuscript,
        template_key=str(template_key or ""),
        language=lang,
    )
    mode = (
        "Draft continuation" if force_manuscript else
        ("Analysis only" if stop_after_analysis else "Analysis + optional manuscript gate")
    ) if is_en else (
        "续写草稿" if force_manuscript else
        ("仅分析" if stop_after_analysis else "分析 + 可选手稿关口")
    )
    template_contract = contract["template_contract"]
    rows = [
        ("Request" if is_en else "请求", contract["question"] or ("not set" if is_en else "未设置")),
        ("Cohort" if is_en else "队列", f"{contract['cohort_label']} · {contract['cohort_rows']:,} rows"),
        ("LLM" if is_en else "模型", f"{contract['llm_choice']} · {contract['model']}"),
        ("Write path" if is_en else "写入路径", contract["workdir"]),
        ("Mode" if is_en else "模式", mode),
        ("Template" if is_en else "模板", str(template_contract["label"])),
    ]
    row_html = "".join(
        '<div class="ra-preflight-row">'
        f'<span>{html.escape(label)}</span>'
        f'<b>{html.escape(value)}</b>'
        '</div>'
        for label, value in rows
    )
    checks = [
        "Evidence manifest and run status are written for every run."
        if is_en else "每次运行都会写入证据清单和 run status。",
        "Numeric claims remain gated by evidence and validator findings."
        if is_en else "数值主张继续受证据和校验发现约束。",
        "Manuscript drafting is second-stage unless explicitly requested."
        if is_en else "除非明确请求，手稿写作保持第二阶段。",
        "External provider will receive the question plus schema/summary context."
        if (is_en and contract["external_llm"]) else
        "外部模型会收到研究问题以及 schema/summary 上下文。"
        if contract["external_llm"] else
        "Offline mock mode does not send cohort context outside this machine."
        if is_en else
        "离线 mock 模式不会把队列上下文发出本机。",
    ]
    check_html = "".join(
        '<div class="ra-preflight-check"><span></span><p>'
        f'{html.escape(check)}</p></div>'
        for check in checks
    )
    st.markdown(
        f"""
        <div class="ra-preflight">
          <div class="ra-preflight-head">
            <div>
              <div class="ra-preflight-kicker">{"Execution preflight" if is_en else "执行前预览"}</div>
              <b>{"Confirm what the agent will read, write, and gate" if is_en else "确认 agent 将读取、写入和审计什么"}</b>
            </div>
            <span>{"human confirmation before run" if is_en else "运行前人工确认"}</span>
          </div>
          <div class="ra-preflight-grid">{row_html}</div>
          <div class="ra-preflight-checks">{check_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(
        "Step contract preview" if is_en else "步骤契约预览",
        expanded=False,
    ):
        st.markdown("**Expected outputs**" if is_en else "**预期产物**")
        st.write(", ".join(map(str, template_contract["expected_outputs"])))
        st.markdown("**Checkpoints**" if is_en else "**复核点**")
        st.write(", ".join(map(str, template_contract["checkpoints"])))
        st.markdown("**File impact**" if is_en else "**文件影响**")
        for target in contract["write_targets"]:
            st.caption(f"WRITE `{target}`")

    signature = _preflight_signature(contract)
    prior_signature = st.session_state.get("research_agent_preflight_signature")
    if prior_signature != signature:
        st.session_state["research_agent_preflight_confirmed"] = False
        st.session_state["research_agent_preflight_signature"] = signature
    confirmed = bool(st.session_state.get("research_agent_preflight_confirmed"))
    ack = st.checkbox(
        (
            "I reviewed the plan, context disclosure, and file impact."
            if is_en else
            "我已复核计划、上下文披露和文件影响。"
        ),
        value=bool(st.session_state.get("research_agent_preflight_ack", False)),
        key="research_agent_preflight_ack",
    )
    can_confirm = bool(contract["question"] or force_manuscript) and contract["cohort_rows"] > 0
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        if st.button(
            "Confirm plan" if is_en else "确认计划",
            type="primary",
            disabled=not ack or not can_confirm,
            use_container_width=True,
            key="research_agent_preflight_confirm",
        ):
            st.session_state["research_agent_preflight_confirmed"] = True
            st.session_state["research_agent_preflight_signature"] = signature
            confirmed = True
    with c2:
        if st.button(
            "Reset confirmation" if is_en else "重置确认",
            use_container_width=True,
            key="research_agent_preflight_reset",
        ):
            st.session_state["research_agent_preflight_confirmed"] = False
            confirmed = False
    if confirmed:
        st.success("Plan confirmed. The run button is enabled." if is_en else "计划已确认，可以启动运行。")
    else:
        st.warning("Confirm the preflight before launching the agent." if is_en else "启动 agent 前请先确认执行前预览。")
    return confirmed


def render_research_agent_page() -> None:
    """Top-level entry point used by the main webapp."""
    render_page_header(
        _ra_text("header"),
        _ra_text("subheader"),
        icon="",
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
    _optional_label = "optional" if _is_en else "可选"
    # 2026-05 Phase F simplification: removed the top stepper
    # visualization (5 pills with required/optional badges) and its
    # legend. Both duplicated the expander headers below — the user
    # learned nothing new from the pills they wouldn't pick up from
    # "1 · Research request", "2 · Study design (optional)", etc.

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

    # 2026-05 Phase G: paper replication as a sibling action below the
    # interactive flow. Deterministic, no LLM, no tokens — uses the
    # lactate × MAP × vasopressor 24-hour package from
    # easyicu.research_agent.replication.
    with st.expander(_ra_text("replication_title"), expanded=False):
        _render_replication_section(default_workdir=history_workdir)

    preflight_confirmed = _render_execution_preflight(
        free_question=free_question,
        target_outcome=target_outcome,
        cohort=cohort,
        cohort_label=cohort_label,
        llm_choice=llm_choice,
        model=model,
        workdir_text=workdir_text,
        stop_after_analysis=stop_after_analysis,
        force_manuscript=force_manuscript,
    )

    external_llm_selected = "MockLLMClient" not in str(llm_choice) and "offline" not in str(llm_choice).lower()
    if external_llm_selected and not st.session_state.get("llm_enabled", False):
        enable_for_run = st.checkbox(
            (
                "Enable external LLM calls for this run"
                if _is_en else
                "允许本次运行调用外部 LLM"
            ),
            key="research_agent_enable_external_llm_for_run",
            help=(
                "The research question, cohort schema/summary, generated prompts, and run logs may be sent to the selected provider."
                if _is_en else
                "研究问题、队列表结构/摘要、生成提示词和运行日志可能会发送给所选模型服务商。"
            ),
        )
        if enable_for_run:
            st.session_state["llm_enabled"] = True
            st.session_state["_llm_toggle"] = True
            st.info(
                "External LLM calls are enabled for this session."
                if _is_en else
                "本会话已允许外部 LLM 调用。"
            )

    request_ready = bool(str(free_question or "").strip()) or force_manuscript
    run_button_clicked = st.button(
        "▶  " + (_ra_text("draft_button") if force_manuscript else _ra_text("run_button")),
        type="primary",
        disabled=cohort is None or not request_ready or not preflight_confirmed,
        use_container_width=True,
    )
    run_clicked = run_button_clicked or (force_manuscript and preflight_confirmed)

    if cohort is None:
        st.info(_ra_text("select_cohort"))
        return
    if not request_ready:
        st.info(
            "Enter a research request above before launching the agent."
            if _is_en else
            "请先在上方填写研究请求，再启动 agent。"
        )
        return
    if not preflight_confirmed:
        st.info(
            "The run is locked until the preflight plan is confirmed."
            if _is_en else
            "执行前预览未确认前，运行保持锁定。"
        )
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
        # Central opt-in gate: external LLM calls require the sidebar
        # AI toggle to be on. MockLLMClient is bypassed (offline).
        enforce_external_llm_opt_in(
            llm_choice,
            language=st.session_state.get("language", "zh"),
        )
        llm = _resolve_llm(
            handles, llm_choice,
            api_key=api_key, model=model,
            base_url=base_url, extra_headers=extra_headers,
        )
    except AIOptInError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(str(exc))
        return

    st.session_state["_ra_view"] = "workbench"
    progress = st.empty()
    progress.info(
        _ra_text("running")
        + (
            " Live Workbench is updating below."
            if _is_en else
            " 下方实时工作台正在更新。"
        )
    )
    progress_bar = st.progress(0)
    progress_log = st.empty()
    live_workbench = st.empty()
    progress_events: List[Dict[str, Any]] = []

    def _render_live_workbench_snapshot() -> None:
        try:
            from easyicu.webapp.agent_workbench import render_agent_live_workbench

            with live_workbench.container():
                render_agent_live_workbench(st.session_state.get("language", "en"))
        except Exception:
            # The canonical run should never fail just because the visual
            # observation layer could not repaint during a callback.
            pass

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
                _bind_workbench_state(
                    run_dir=workdir / str(run_id),
                    manifest=manifest,
                    partial=_partial,
                    progress_events=progress_events,
                )
            else:
                _bind_workbench_state(
                    run_dir=workdir / str(run_id),
                    manifest={
                        "run_id": str(run_id),
                        "research_question": free_question or target_outcome or str(run_id),
                        "per_step_records": [],
                        "evidence": [],
                        "findings": [],
                    },
                    partial=True,
                    progress_events=progress_events,
                )
        else:
            _bind_workbench_state(
                run_dir=workdir / "run_pending_webapp",
                manifest={
                    "run_id": "run_pending_webapp",
                    "research_question": free_question or target_outcome or "Research Agent run",
                    "per_step_records": [],
                    "evidence": [],
                    "findings": [],
                },
                partial=True,
                progress_events=progress_events,
            )
        _render_live_workbench_snapshot()

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
    live_workbench.empty()
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
    final_manifest, _final_manifest_path, _final_partial = _load_run_manifest(Path(result.workdir))
    if final_manifest:
        _bind_workbench_state(
            run_dir=Path(result.workdir),
            manifest=final_manifest,
            partial=_final_partial,
            progress_events=progress_events,
        )
        if "entry_mode" in st.session_state:
            st.session_state["_ra_view"] = "workbench"
            st.rerun()
    _render_run_outputs(result, Path(result.workdir))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def _standalone_main() -> None:
    """Make this module runnable via ``streamlit run …/research_agent.py``."""
    st.set_page_config(
        page_title="EasyICU Research Agent",
        page_icon=None,
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
