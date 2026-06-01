"""Streamlit page for the EasyICU research agent (ROADMAP T1.7).

This module hosts a single, self-contained page that lets a reviewer
or end user click through the full ICU-aware research-agent pipeline:

1. Pick a cohort — upload a parquet/CSV, point at one of the user's
   prior ``extract_database`` outputs in the workspace, or fall back
   to the synthetic SOFA cohort baked into ``examples/``.
2. Pick a :class:`ClinicalSkill` (an ICU research-question analysis
   method — e.g. SOFA-2 vs mortality association, vasopressor target-trial
   emulation; *not* a generic data-science skill) from the registry, or
   choose the canonical "free-form question" mode.
3. Run the ICU-aware context by default. The historical naive ablation
   remains a CLI-only benchmark path, not a web UI option.
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
import os
import re
import socket
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from easyicu.webapp import cohort_charts as cc
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
from easyicu.webapp.session_state import clear_agent_continuation_state, clear_run_state
from easyicu.webapp.ui_helpers import icon as _shell_icon


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
        safe_name = name.replace("/", "\\")
        return f"D:\\path\\to\\{safe_name}"
    return f"/path/to/{name}"


def _is_agent_run_artifact_dir(folder: Path) -> bool:
    """Exclude Research Agent run history from module-export discovery."""
    try:
        resolved = folder.expanduser().resolve()
    except Exception:
        resolved = folder
    name = resolved.name.lower()
    parent_name = resolved.parent.name.lower() if resolved.parent else ""
    if name == "webapp" and parent_name == "research_output":
        return True
    if name.startswith("run_") and any(
        (resolved / marker).exists()
        for marker in ("run_status.json", "manifest.json", "cohort.parquet")
    ):
        return True
    return False


def _sync_extract_db_with_active_data_source(
    state: MutableMapping[str, Any],
    options: Sequence[str],
) -> str:
    """Keep raw-extract defaults aligned with the active validated source."""
    fallback = "miiv" if "miiv" in options else (str(options[0]) if options else "")
    active_db = str(state.get("database") or "").strip()
    if active_db not in options:
        selected = str(state.get("research_agent_extract_db") or "").strip()
        return selected if selected in options else fallback
    selected_db = str(state.get("research_agent_extract_db") or "").strip()
    synced_from = str(state.get("_research_agent_extract_db_source") or "").strip()
    should_follow = (
        not selected_db
        or selected_db == synced_from
        or (not synced_from and selected_db == fallback)
    )
    if should_follow:
        if selected_db and selected_db != active_db:
            state.pop("research_agent_extract_db", None)
        state["_research_agent_extract_db_source"] = active_db
        return active_db
    return selected_db if selected_db in options else fallback


def _hide_prefilled_directory_text(input_key: str, mirrored_value: str) -> None:
    pending_key = f"{input_key}__pending_value"
    current = str(st.session_state.get(input_key, "") or "")
    if pending_key in st.session_state:
        return
    if mirrored_value and current == str(mirrored_value):
        st.session_state[input_key] = ""


def _clear_module_folder_handoff_focus() -> None:
    """Let an explicit detected-folder pick take over from post-export handoff."""
    st.session_state["_eu_ra_focus_module_folder"] = False
    st.session_state.pop("_eu_ra_apply_export_file_selection", None)


def _detect_id_columns(columns: Sequence[str]) -> List[str]:
    by_lower = {str(c).lower(): str(c) for c in columns}
    found: List[str] = []
    for c in _ID_COLUMN_CANDIDATES:
        hit = by_lower.get(c.lower())
        if hit and hit not in found:
            found.append(hit)
    return found


_MODULE_TABLE_EXTS = {".parquet", ".pq", ".csv", ".tsv", ".xlsx", ".xls", ".feather"}


def _is_module_table_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _MODULE_TABLE_EXTS


def _module_file_sort_key(path: Path, folder: Path) -> str:
    try:
        return str(path.relative_to(folder))
    except Exception:
        return str(path)


def _scan_workspace_for_module_dirs(roots: List[Path]) -> List[Path]:
    """Return candidate EasyICU export folders containing module table files."""
    out: List[Path] = []
    seen: Set[Path] = set()

    def _has_module_files(folder: Path) -> bool:
        try:
            if any(_is_module_table_file(p) for p in folder.iterdir()):
                return True
            return any(_is_module_table_file(p) for p in folder.glob("*/*"))
        except Exception:
            return False

    def _add(folder: Path) -> None:
        try:
            resolved = folder.resolve()
        except Exception:
            return
        if (
            resolved in seen
            or not resolved.is_dir()
            or _is_agent_run_artifact_dir(resolved)
            or not _has_module_files(resolved)
        ):
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
                if (
                    child.is_dir()
                    and child.name not in _KNOWN_MODULE_DIR_NAMES
                    and not _is_agent_run_artifact_dir(child)
                    and _has_module_files(child)
                ):
                    child_export_dirs.append(child)
        except Exception:
            continue
        if child_export_dirs:
            for child in child_export_dirs:
                _add(child)
            if any(_is_module_table_file(p) for p in root.iterdir()):
                _add(root)
        else:
            _add(root)
    return out


def _list_module_parquets(folder: Path) -> List[Path]:
    try:
        direct = [p.resolve() for p in folder.iterdir() if _is_module_table_file(p)]
        if direct:
            return sorted(direct, key=lambda p: _module_file_sort_key(p, folder))
        return sorted(
            (p.resolve() for p in folder.rglob("*") if _is_module_table_file(p)),
            key=lambda p: _module_file_sort_key(p, folder),
        )
    except Exception:
        return []


def _export_result_file_labels_for_folder(
    state: Mapping[str, Any],
    folder: Path,
) -> List[str]:
    """Return file labels from the latest export result that belong to ``folder``."""
    result = state.get("_export_success_result")
    if not isinstance(result, Mapping):
        return []
    raw_files = result.get("files")
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
        return []
    try:
        folder_resolved = folder.expanduser().resolve()
    except Exception:
        folder_resolved = folder
    labels: List[str] = []
    for raw_path in raw_files:
        if not raw_path:
            continue
        try:
            path = Path(str(raw_path)).expanduser().resolve()
        except Exception:
            path = Path(str(raw_path))
        if path.suffix.lower() not in _MODULE_TABLE_EXTS:
            continue
        try:
            label = str(path.relative_to(folder_resolved))
        except Exception:
            try:
                if path.parent.resolve() != folder_resolved:
                    continue
            except Exception:
                continue
            label = path.name
        if label not in labels:
            labels.append(label)
    return labels


def _module_dir_parquet_count(folder: Path) -> int:
    """Count immediate module tables for ranking detected export folders."""
    try:
        direct = [p for p in folder.iterdir() if _is_module_table_file(p)]
        if direct:
            return len(direct)
        return len([p for p in folder.rglob("*") if _is_module_table_file(p)])
    except Exception:
        return 0


def _has_child_module_export_dirs(folder: Path) -> bool:
    try:
        return any(
            child.is_dir() and _module_dir_parquet_count(child) > 0
            for child in folder.iterdir()
        )
    except Exception:
        return False


_GENERIC_MODULE_EXPORT_CONTAINER_NAMES = {"easyicu_export", "exports", "output", "outputs"}


def _is_generic_module_export_container(folder: Path) -> bool:
    """Return true for broad export buckets that contain child export runs."""
    try:
        resolved = folder.expanduser().resolve()
    except Exception:
        resolved = folder
    return (
        resolved.name.lower() in _GENERIC_MODULE_EXPORT_CONTAINER_NAMES
        and _has_child_module_export_dirs(resolved)
    )


def _module_folder_manual_handoff_dir(state: Mapping[str, Any]) -> str:
    """Return a concrete export folder worth anchoring as a manual handoff.

    ``last_export_dir`` may represent a freshly completed export, so preserve
    it when we can tie it to the latest export result. ``export_path`` is often
    only the user's broad export bucket (for example ``~/easyicu_export``);
    when that bucket contains child folders, let detected-folder ranking choose
    the best child instead of forcing manual mode.
    """
    candidates = (
        ("last_export_dir", str(state.get("last_export_dir") or "").strip()),
        ("export_path", str(state.get("export_path") or "").strip()),
    )
    for key, raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if _is_generic_module_export_container(path):
            has_current_export_files = (
                key == "last_export_dir"
                and bool(_export_result_file_labels_for_folder(state, path))
            )
            if not has_current_export_files:
                continue
        return str(path)
    return ""


def _clear_generic_module_folder_manual_default(state: MutableMapping[str, Any]) -> None:
    """Release a stale manual default when it only mirrors a generic export root."""
    raw_current = str(state.get("research_agent_module_dir_text") or "").strip()
    raw_export_path = str(state.get("export_path") or "").strip()
    if not raw_current or not raw_export_path:
        return
    try:
        current = Path(raw_current).expanduser().resolve()
        export_path = Path(raw_export_path).expanduser().resolve()
    except Exception:
        return
    if current != export_path or not _is_generic_module_export_container(current):
        return
    state.pop("research_agent_module_dir_text", None)
    state.pop("research_agent_module_dir_pick", None)


def _default_module_dir_pick_index(options: Sequence[str], dirs: Sequence[Path]) -> int:
    """Return the selectbox index for the most complete detected export.

    ``options`` includes the manual-path sentinel at index 0. The Research
    Agent needs broad context, so the first automatic choice should be the
    detected folder with the most module files, not whichever sibling sorts
    first alphabetically.
    """
    if not dirs or len(options) <= 1:
        return 0
    best_dir_idx = max(
        range(len(dirs)),
        key=lambda idx: (
            not _is_generic_module_export_container(dirs[idx]),
            _module_dir_parquet_count(dirs[idx]),
            dirs[idx].stat().st_mtime if dirs[idx].exists() else 0.0,
            str(dirs[idx]),
        ),
    )
    return best_dir_idx + 1


def _read_module_table(
    path: Path,
    columns: Optional[Sequence[str]] = None,
    *,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    cols = list(dict.fromkeys(str(c) for c in columns if c)) if columns else None
    if suffix in {".parquet", ".pq"}:
        if cols:
            try:
                df = pd.read_parquet(path, columns=cols)
            except Exception:
                df = pd.read_parquet(path)
                keep = [c for c in cols if c in df.columns]
                df = df[keep].copy() if keep else df
        else:
            df = pd.read_parquet(path)
        return df.head(nrows) if nrows is not None else df
    if suffix == ".feather":
        df = pd.read_feather(path)
    elif suffix == ".tsv":
        read_kwargs: Dict[str, Any] = {"sep": "\t"}
        if nrows is not None:
            read_kwargs["nrows"] = nrows
        if cols:
            read_kwargs["usecols"] = lambda c: str(c) in set(cols)
        try:
            return pd.read_csv(path, **read_kwargs)
        except Exception:
            df = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=nrows)
    else:
        read_kwargs = {}
        if nrows is not None:
            read_kwargs["nrows"] = nrows
        if cols:
            read_kwargs["usecols"] = lambda c: str(c) in set(cols)
        try:
            return pd.read_csv(path, **read_kwargs)
        except Exception:
            df = pd.read_csv(path)
    if cols:
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy() if keep else df
    return df.head(nrows) if nrows is not None else df


def _parquet_file_summary(path: Path) -> Dict[str, Any]:
    """Small metadata summary for supported module table files."""
    rows: Optional[int] = None
    columns: List[str] = []
    error: Optional[str] = None
    if path.suffix.lower() in {".parquet", ".pq"}:
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
                error = f"{type(read_exc).__name__}: {read_exc}" or f"{type(exc).__name__}: {exc}"
    else:
        try:
            df = _read_module_table(path, nrows=0)
            rows = None
            columns = [str(c) for c in df.columns]
        except Exception as read_exc:
            error = f"{type(read_exc).__name__}: {read_exc}"
    return {
        "path": path,
        "rows": rows,
        "columns": columns,
        "id_columns": _detect_id_columns(columns),
        "error": error,
    }


def _read_parquet_columns(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    return _read_module_table(path, columns=columns)


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


def _normalize_module_merge_id(series: pd.Series) -> pd.Series:
    """Keep numeric ICU ids on one dtype before pandas merge operations."""
    if not pd.api.types.is_numeric_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    present = numeric.dropna()
    if present.empty:
        return numeric.astype("Int64")
    try:
        integral = ((present % 1) == 0).all()
    except Exception:
        integral = False
    return numeric.astype("Int64" if bool(integral) else "Float64")


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
    """Read one module table file and return ``(df, is_temporal)``.

    * **Temporal** files have at least one time column.  The first detected
      time column is renamed to *canonical_time* so all temporal files share
      the same key name when merged.
    * **Static** files (demographics, outcomes, …) have no time column and
      return a single row per patient.

    Returns ``None`` when the file cannot yield usable columns.
    """
    df = _read_module_table(path)
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
        sub[id_col] = _normalize_module_merge_id(sub[id_col])
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
        sub[id_col] = _normalize_module_merge_id(sub[id_col])

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
    """Merge selected module table files into a single cohort dataframe.

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


def _sync_module_file_multiselect_defaults(
    state: MutableMapping[str, Any],
    *,
    key: str,
    signature_key: str,
    folder: Path,
    labels: Sequence[str],
) -> None:
    """Keep module-folder defaults broad without clobbering user edits.

    Older sessions may carry a one-file multiselect state from the previous
    UI. When a folder is first selected, or when the folder changes, reset to
    all files. Once the folder signature matches, preserve deliberate user
    narrowing and only drop labels that no longer exist.
    """
    label_list = list(labels)
    if not label_list:
        state.pop(key, None)
        state[signature_key] = str(folder)
        return

    folder_signature = str(folder)
    if state.get(signature_key) != folder_signature:
        state[key] = _default_module_selection(label_list)
        state[signature_key] = folder_signature
        return

    current = state.get(key)
    if not isinstance(current, (list, tuple)):
        return
    valid = [str(label) for label in current if str(label) in label_list]
    if valid != list(current):
        state[key] = valid or _default_module_selection(label_list)


def _restore_module_file_selection_after_build_rerun(
    state: MutableMapping[str, Any],
    *,
    key: str,
    signature_key: str,
    folder: Path,
    labels: Sequence[str],
) -> None:
    """Recover a just-built module selection if a later rerun drops it.

    Streamlit's multiselect can transiently return an empty value when another
    control reruns the page while the dropdown is still active. After a
    successful build, restore that exact file set once so the built cohort stays
    bound to the launch gate. A deliberate later clear is left alone.
    """
    if not state.pop("_research_agent_module_restore_built_selection", False):
        return
    current = state.get(key)
    if current:
        return
    cached_build = state.get("research_agent_module_built")
    if not isinstance(cached_build, dict):
        return
    signature = cached_build.get("signature")
    if not isinstance(signature, dict) or signature.get("folder") != str(folder):
        return
    label_set = set(labels)
    restored: list[str] = []
    for file_name in signature.get("files") or []:
        try:
            label = str(Path(str(file_name)).relative_to(folder))
        except ValueError:
            label = Path(str(file_name)).name
        if label in label_set:
            restored.append(label)
    if restored:
        state[key] = restored
        state[signature_key] = str(folder)


def _preserve_module_file_selection_for_next_rerun(
    state: MutableMapping[str, Any],
    *,
    key: str = "research_agent_module_files",
    signature_key: str = "research_agent_module_files_folder",
) -> None:
    """Remember module-file choices before an early manual ``st.rerun``."""
    current = state.get(key)
    if not isinstance(current, (list, tuple)) or not current:
        return
    folder_signature = str(state.get(signature_key) or "")
    if not folder_signature:
        return
    state["_research_agent_module_pending_selection_restore"] = {
        "folder": folder_signature,
        "labels": [str(label) for label in current],
        "source": str(state.get("research_agent_cohort_source") or ""),
    }


def _restore_pending_module_source(
    state: MutableMapping[str, Any],
    *,
    options: Sequence[str],
) -> None:
    """Restore module-folder source before the cohort radio renders."""
    pending = state.get("_research_agent_module_pending_selection_restore")
    if not isinstance(pending, dict):
        return
    source = pending.get("source")
    if isinstance(source, str) and source in options:
        state["research_agent_cohort_source"] = source


def _restore_pending_module_folder_path(
    state: MutableMapping[str, Any],
    *,
    manual_path_label: str,
) -> None:
    """Restore the module folder path saved before an early setup rerun."""
    pending = state.get("_research_agent_module_pending_selection_restore")
    if not isinstance(pending, dict):
        return
    folder = str(pending.get("folder") or "").strip()
    if not folder:
        return
    state["research_agent_module_dir_text"] = folder
    state["_research_agent_module_dir_restore_folder"] = folder
    state.pop("research_agent_module_dir_pick", None)


def _restore_pending_module_file_selection(
    state: MutableMapping[str, Any],
    *,
    key: str,
    signature_key: str,
    folder: Path,
    labels: Sequence[str],
) -> None:
    """Restore choices saved before Apply question/template reruns."""
    current = state.get(key)
    if current:
        state.pop("_research_agent_module_pending_selection_restore", None)
        return
    pending = state.pop("_research_agent_module_pending_selection_restore", None)
    if not isinstance(pending, dict) or pending.get("folder") != str(folder):
        return
    label_set = set(labels)
    restored = [
        str(label)
        for label in pending.get("labels") or []
        if str(label) in label_set
    ]
    if restored:
        state[key] = restored
        state[signature_key] = str(folder)


_RAW_EXTRACT_MODULES_KEY = "research_agent_extract_modules"
_RAW_EXTRACT_MODULE_PRESET_KEY = "research_agent_extract_module_preset"
_LEGACY_RAW_EXTRACT_DEFAULT_MODULES = (
    "demographics",
    "outcome",
    "sofa2_score",
    "sepsis3_sofa2",
    "vitals",
    "blood_gas",
)


def _default_extract_module_selection(modules: Dict[str, List[str]]) -> List[str]:
    # Research-agent runs are context hungry: default to a complete export and
    # let advanced users narrow the module list deliberately.
    return list(modules.keys())


def _migrate_legacy_extract_module_selection(
    state: MutableMapping[str, Any],
    modules: Dict[str, List[str]],
) -> None:
    current = state.get(_RAW_EXTRACT_MODULES_KEY)
    legacy_default = [module for module in _LEGACY_RAW_EXTRACT_DEFAULT_MODULES if module in modules]
    if legacy_default and isinstance(current, (list, tuple)) and list(current) == legacy_default:
        state[_RAW_EXTRACT_MODULES_KEY] = _default_extract_module_selection(modules)


def _raw_extract_module_selection_for_preset(
    modules: Dict[str, List[str]],
    preset: str,
    custom_modules: Sequence[str] | None = None,
) -> List[str]:
    """Return the no-data extraction module list for a compact preset.

    The default remains the full module export because Research Agent runs are
    context hungry. The preset only keeps the UI calm: the long multiselect is
    shown when the user deliberately chooses a custom subset.
    """
    if preset == "core":
        core = [module for module in _LEGACY_RAW_EXTRACT_DEFAULT_MODULES if module in modules]
        return core or _default_extract_module_selection(modules)
    if preset == "custom":
        valid = set(modules)
        return [str(module) for module in custom_modules or [] if str(module) in valid]
    return _default_extract_module_selection(modules)


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
    audit_relax_probe: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Invoke the pipeline; return the :class:`PipelineResult`.

    ``audit_relax_probe`` is a per-run override of
    ``EASYICU_AUDIT_RELAX_PROBE`` — when True, the probe-stage of the
    concept-usage auditor downgrades reporting-practice violations from
    block to warning. Used by the webapp resume editor as a documented
    ablation; the strict default is preserved across the process by
    restoring the prior env var value after the run.
    """
    prior_relax = os.environ.get("EASYICU_AUDIT_RELAX_PROBE")
    if audit_relax_probe:
        os.environ["EASYICU_AUDIT_RELAX_PROBE"] = "1"
    try:
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
    finally:
        if audit_relax_probe:
            if prior_relax is None:
                os.environ.pop("EASYICU_AUDIT_RELAX_PROBE", None)
            else:
                os.environ["EASYICU_AUDIT_RELAX_PROBE"] = prior_relax


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
        s = path.stat()
        fingerprint = (str(path), s.st_mtime_ns, s.st_size)
    except OSError:
        return {}
    return _cached_read_json_file(fingerprint)


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_read_json_file(fingerprint: Tuple[str, int, int]) -> Dict[str, Any]:
    """Cache JSON manifest reads keyed by (path, mtime_ns, size).

    Run history + workbench resolve manifest data on every Streamlit
    rerun. Without this cache, every click in the Research Agent page
    re-parses every ``manifest.json`` in the workdir — perceptibly slow
    once a workdir holds more than a couple of runs.
    """
    path_str, _mtime, _size = fingerprint
    try:
        data = json.loads(Path(path_str).read_text(encoding="utf-8"))
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


def _resume_run_dir_from_state(state: Dict[str, Any], run_id: str) -> Optional[Path]:
    """Resolve a resume run directory without guessing outside the local workdir."""
    for key in ("research_agent_resume_run_dir", "_agent_workbench_source_run_dir"):
        raw = str(state.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.exists() or not path.is_dir():
            continue
        if not run_id or path.name == run_id:
            return path.resolve()
        manifest, _manifest_path, _partial = _load_run_manifest(path)
        if str(manifest.get("run_id") or "").strip() == run_id:
            return path.resolve()
    if run_id:
        candidate = Path(_default_research_agent_workdir()).expanduser() / run_id
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def _store_resume_run_dir_context(
    state: Dict[str, Any],
    run_dir: Path | str,
    *,
    defer_workdir: bool = False,
) -> None:
    """Remember a selected run and keep follow-up writes beside it."""
    run_dir_text = str(run_dir or "").strip()
    if not run_dir_text:
        return
    state["research_agent_resume_run_dir"] = run_dir_text
    try:
        workdir_text = str(Path(run_dir_text).expanduser().resolve().parent)
    except Exception:
        workdir_text = str(Path(run_dir_text).expanduser().parent)
    if defer_workdir:
        state["_research_agent_workdir_pending"] = workdir_text
    else:
        state["research_agent_workdir"] = workdir_text


def _apply_pending_research_agent_workdir(state: Dict[str, Any]) -> None:
    """Apply a deferred workdir before Streamlit creates the matching widget."""
    pending = str(state.pop("_research_agent_workdir_pending", "") or "").strip()
    if pending:
        state["research_agent_workdir"] = pending


def _cohort_path_from_resume_run(run_dir: Path) -> Optional[Path]:
    """Find the stay-level cohort parquet recorded for a prior web run."""
    candidates: List[Path] = []
    for rel in (
        "data_extraction_result.json",
        "evidence/data_extraction_result__data_extraction_result.json",
    ):
        payload = _read_json_file(run_dir / rel)
        raw = str(payload.get("cohort_path") or "").strip()
        if raw:
            path = Path(raw).expanduser()
            candidates.append(path if path.is_absolute() else run_dir / path)
    manifest, _manifest_path, _partial = _load_run_manifest(run_dir)
    raw_manifest = str(manifest.get("cohort_path") or "").strip()
    if raw_manifest:
        path = Path(raw_manifest).expanduser()
        candidates.append(path if path.is_absolute() else run_dir / path)
    candidates.append(run_dir / "cohort.parquet")
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved.exists() and resolved.is_file() and resolved.suffix.lower() in {".parquet", ".pq", ".csv", ".tsv"}:
            return resolved
    return None


def _restore_resume_cohort_handoff(state: Dict[str, Any]) -> bool:
    """Seed the cohort picker from the active resume run's local cohort artifact."""
    run_id = str(state.get("research_agent_resume_run_id") or "").strip()
    resume_mode = str(state.get("research_agent_resume_mode") or "")
    if not run_id or resume_mode not in {"continue", "force_manuscript"}:
        return False
    inbound = state.get("research_agent_inbound_cohort")
    existing_signature = str(state.get("research_agent_resume_cohort_signature") or "")
    current_signature = str(state.get("research_agent_inbound_signature") or "")
    if (
        isinstance(inbound, pd.DataFrame)
        and not inbound.empty
        and existing_signature.startswith(f"resume:{run_id}:")
        and current_signature == existing_signature
    ):
        return True
    run_dir = _resume_run_dir_from_state(state, run_id)
    if run_dir is None:
        return False
    cohort_path = _cohort_path_from_resume_run(run_dir)
    if cohort_path is None:
        return False
    try:
        if cohort_path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(cohort_path)
        else:
            sep = "\t" if cohort_path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(cohort_path, sep=sep)
    except Exception:
        return False
    if df.empty:
        return False
    try:
        stat = cohort_path.stat()
        signature = f"resume:{run_id}:{cohort_path}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        signature = f"resume:{run_id}:{cohort_path}"
    state["research_agent_inbound_cohort"] = df
    state["research_agent_inbound_cohort_label"] = f"resume:{run_id}:{cohort_path.name}"
    state["research_agent_inbound_signature"] = signature
    state["research_agent_resume_cohort_signature"] = signature
    state["research_agent_cohort_source"] = _ra_text("source_handoff")
    return True


def _clear_resume_cohort_handoff(state: Dict[str, Any]) -> None:
    """Remove only the cohort handoff that was loaded from a resume run."""
    signature = state.pop("research_agent_resume_cohort_signature", None)
    if signature and state.get("research_agent_inbound_signature") == signature:
        for key in (
            "research_agent_inbound_cohort",
            "research_agent_inbound_cohort_label",
            "research_agent_inbound_signature",
        ):
            state.pop(key, None)


def _bind_workbench_state(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    partial: Optional[bool] = None,
    progress_events: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Populate the Shell-A Workbench from a real run manifest."""
    run_id = str(manifest.get("run_id") or run_dir.name or "").strip()
    if run_id:
        st.session_state["research_agent_last_run_id"] = run_id
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
    step_failed = sum(1 for s in statuses if "fail" in s or "error" in s or "blocked" in s)
    run_status = _read_json_file(run_dir / "run_status.json")
    backend_status = str(run_status.get("status") or "").strip().lower()
    if partial:
        status = "partial"
    elif step_failed:
        status = "blocked"
    else:
        status = backend_status or "complete"
    review = _load_review_decision(run_dir)
    return {
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "run_dir": run_dir,
        "status": status,
        "started_at": str(manifest.get("started_at") or ""),
        "finished_at": str(manifest.get("finished_at") or ""),
        "question": str(manifest.get("research_question") or ""),
        "step_total": len(records),
        "step_ok": sum(1 for s in statuses if s == "ok"),
        "step_failed": step_failed,
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


def _normalized_db_tag(tag: str) -> str:
    return str(tag or '').strip().lower()


def _duplicate_db_tags(chosen: Sequence[Tuple[str, Path]]) -> List[str]:
    counts: Dict[str, int] = {}
    for tag, _folder in chosen:
        norm = _normalized_db_tag(tag)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
    return sorted(tag for tag, count in counts.items() if count > 1)


def _has_min_distinct_db_tags(chosen: Sequence[Tuple[str, Path]], min_count: int = 2) -> bool:
    tags = {_normalized_db_tag(tag) for tag, _folder in chosen if _normalized_db_tag(tag)}
    return len(tags) >= min_count


def _render_cohort_source_quick_actions(
    *,
    source_no_data: str,
    source_module: str,
    source_synthetic: str,
    is_en: bool,
) -> None:
    """Render low-friction cohort source shortcuts before the full source radio."""
    title = "Quick cohort choices" if is_en else "快速选择队列"
    title_copy = (
        "Pick the most common path first; the full source list stays below."
        if is_en
        else "先选最常用途径；完整来源列表仍保留在下方。"
    )
    safety_title = "Safe to change" if is_en else "可随时切换"
    safety_copy = (
        "Changing the cohort only resets launch review, not your question."
        if is_en
        else "切换队列只会重置启动复核，不会清空研究问题。"
    )
    st.markdown(
        textwrap.dedent(f"""
        <div class="ra-request-brief">
          <div>
            <b>{title}</b>
            <span>{title_copy}</span>
          </div>
          <div>
            <b>{safety_title}</b>
            <span>{safety_copy}</span>
          </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )
    shortcut_cols = st.columns(3, gap="small")
    shortcuts = [
        (
            "research_agent_quick_synthetic",
            "Use test cohort" if is_en else "使用测试队列",
            "Built-in 800-row SOFA cohort" if is_en else "内置 800 行 SOFA 队列",
            source_synthetic,
            False,
            False,
        ),
        (
            "research_agent_quick_module_folder",
            "Pick export folder" if is_en else "选择导出文件夹",
            "Use an existing EasyICU export" if is_en else "使用已有 EasyICU 导出",
            source_module,
            True,
            False,
        ),
        (
            "research_agent_quick_no_data",
            "Prepare data export" if is_en else "准备数据导出",
            "Start from raw ICU tables" if is_en else "从原始 ICU 表开始",
            source_no_data,
            False,
            True,
        ),
    ]
    for col, (
        key,
        label,
        caption,
        source,
        focus_module,
        focus_no_data,
    ) in zip(shortcut_cols, shortcuts):
        with col:
            if st.button(label, key=key, use_container_width=True):
                _activate_research_agent_cohort_source(
                    st.session_state,
                    source,
                    focus_module=focus_module,
                    focus_no_data=focus_no_data,
                )
                st.rerun()
            st.caption(caption)


def _multi_db_label_tags(cohort_label: str) -> List[str]:
    label = str(cohort_label or "")
    if not label.startswith("multi_db:"):
        return []
    raw_tags = label.split(":", 1)[1]
    return [_normalized_db_tag(tag) for tag in raw_tags.split(",") if _normalized_db_tag(tag)]


def _multi_db_label_is_distinct(cohort_label: str) -> bool:
    tags = _multi_db_label_tags(cohort_label)
    return not tags or len(set(tags)) >= 2 and len(set(tags)) == len(tags)


def _clear_research_agent_preflight_confirmation() -> None:
    st.session_state["research_agent_preflight_confirmed"] = False
    st.session_state["research_agent_preflight_ack"] = False


def _activate_research_agent_cohort_source(
    state: MutableMapping[str, object],
    source: str,
    *,
    focus_module: bool = False,
    focus_no_data: bool = False,
) -> None:
    """Select a cohort source from a shortcut and invalidate stale launch review."""
    state["research_agent_cohort_source"] = source
    state["research_agent_preflight_confirmed"] = False
    state["research_agent_preflight_ack"] = False
    if focus_module:
        state["_eu_ra_focus_module_folder"] = True
    if focus_no_data:
        state["_eu_ra_focus_no_data"] = True


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
    if min_selected >= 2:
        duplicate_tags = _duplicate_db_tags(chosen)
        if duplicate_tags or (len(chosen) >= 2 and not _has_min_distinct_db_tags(chosen, min_count=2)):
            st.error(_ra_text(
                'multi_db_duplicate_tags',
                tags=', '.join(duplicate_tags) if duplicate_tags else ', '.join(d for d, _ in chosen),
            ))
            _clear_research_agent_preflight_confirmation()
            return []
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
    duplicate_tags = _duplicate_db_tags(chosen)
    if duplicate_tags or not _has_min_distinct_db_tags(chosen, min_count=2):
        st.error(_ra_text(
            'multi_db_duplicate_tags',
            tags=', '.join(duplicate_tags) if duplicate_tags else ', '.join(d for d, _ in chosen),
        ))
        return None, ''

    frames: List[pd.DataFrame] = []
    per_db_summary: List[str] = []
    loaded_db_tags: List[str] = []
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
        loaded_db_tags.append(_normalized_db_tag(db_tag))
        per_db_summary.append(f'{db_tag}={len(sub):,}')

    if not frames:
        st.error('No databases could be loaded; check the warnings above.')
        return None, ''
    loaded_unique_tags = sorted({tag for tag in loaded_db_tags if tag})
    if len(loaded_unique_tags) < 2:
        st.error(_ra_text('multi_db_loaded_need_distinct'))
        return None, ''

    cohort = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    st.success(_ra_text(
        'multi_db_built', rows=len(cohort), dbs=len(loaded_unique_tags),
        per_db=', '.join(per_db_summary),
    ))
    st.dataframe(cohort.head(8), use_container_width=True, hide_index=True)
    return cohort, f'multi_db:{",".join(loaded_unique_tags)}'


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

    # 2026-05 Phase G: removed `source_workspace` (pick a single file from
    # workspace) — it was a strict subset of `source_module` (pick a
    # folder, then pick module files inside it). Added `source_multi_db` to
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
    if st.session_state.pop("_eu_ra_no_data_entry", False):
        st.session_state["research_agent_cohort_source"] = source_no_data
        _clear_research_agent_preflight_confirmation()
    _restore_pending_module_source(st.session_state, options=options)
    if st.session_state.get("research_agent_cohort_source") not in (None, *options):
        st.session_state.pop("research_agent_cohort_source", None)
    if st.session_state.get("entry_mode") == "real":
        _render_cohort_source_quick_actions(
            source_no_data=source_no_data,
            source_module=source_module,
            source_synthetic=source_synthetic,
            is_en=st.session_state.get("language", "en") == "en",
        )
    source = st.radio(
        _ra_text("cohort_source"),
        options=options,
        horizontal=True,
        key="research_agent_cohort_source",
    )
    previous_source = st.session_state.get("_research_agent_previous_cohort_source")
    if previous_source != source:
        st.session_state["_research_agent_previous_cohort_source"] = source
        _clear_research_agent_preflight_confirmation()
        if source == source_module:
            export_dir = _module_folder_manual_handoff_dir(st.session_state)
            if export_dir:
                st.session_state["_eu_ra_focus_module_folder"] = True
                st.session_state["_eu_ra_module_pick_force_manual"] = True
                st.session_state["_eu_ra_apply_export_file_selection"] = True
                st.session_state["research_agent_module_dir_text"] = export_dir
                st.session_state.pop("research_agent_module_dir_pick", None)
            else:
                _clear_generic_module_folder_manual_default(st.session_state)

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
        manual_path_label = _ra_text("manual_path")
        if dir_labels:
            _restore_pending_module_folder_path(
                st.session_state,
                manual_path_label=manual_path_label,
            )
        restore_folder = str(st.session_state.pop("_research_agent_module_dir_restore_folder", "") or "")
        manual_default = restore_folder or (str(extra_roots[0]) if extra_roots else "")
        force_manual_pick = bool(
            st.session_state.pop("_eu_ra_module_pick_force_manual", False)
            and manual_default
        )
        handoff_manual_active = bool(
            st.session_state.get("_eu_ra_focus_module_folder") and manual_default
        )
        picked_label = ""
        if dir_labels:
            restore_pick_index: Optional[int] = None
            if restore_folder:
                try:
                    restore_resolved = Path(restore_folder).expanduser().resolve()
                    for idx, folder_candidate in enumerate(dirs):
                        if folder_candidate.expanduser().resolve() == restore_resolved:
                            restore_pick_index = idx + 1
                            break
                except Exception:
                    restore_pick_index = None
            current_pick = str(st.session_state.get("research_agent_module_dir_pick") or "")
            current_manual_text = str(st.session_state.get("research_agent_module_dir_text", "") or "")
            if (
                handoff_manual_active
                and current_pick not in {"", manual_path_label}
                and current_manual_text in {"", manual_default}
            ):
                st.session_state.pop("research_agent_module_dir_pick", None)
            folder_options = [manual_path_label] + dir_labels
            if force_manual_pick or handoff_manual_active:
                folder_pick_index = 0
            elif restore_pick_index is not None:
                folder_pick_index = restore_pick_index
            elif restore_folder:
                folder_pick_index = 0
            else:
                folder_pick_index = _default_module_dir_pick_index(folder_options, dirs)
            picked_label = st.selectbox(
                _ra_text("detected_folders"),
                folder_options,
                index=folder_pick_index,
                key="research_agent_module_dir_pick",
                on_change=_clear_module_folder_handoff_focus,
            )
        selected_folder_value = (
            str(dirs[dir_labels.index(picked_label)])
            if dir_labels and picked_label not in {"", manual_path_label}
            else manual_default
        )
        picked_manual_path = picked_label in {"", manual_path_label}
        show_handoff_path = bool(
            st.session_state.get("_eu_ra_focus_module_folder")
            and picked_manual_path
            and manual_default
        )
        if (
            not handoff_manual_active
            and not picked_manual_path
            and str(st.session_state.get("research_agent_module_dir_text", "") or "") == manual_default
        ):
            st.session_state["research_agent_module_dir_text"] = ""
        if not show_handoff_path:
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
        apply_export_files = bool(st.session_state.pop("_eu_ra_apply_export_file_selection", False))
        if apply_export_files:
            export_labels = _export_result_file_labels_for_folder(st.session_state, folder)
            selected_export_labels = [label for label in labels if label in export_labels]
            if selected_export_labels:
                st.session_state["research_agent_module_files"] = selected_export_labels
                st.session_state["research_agent_module_files_folder"] = str(folder)
        _sync_module_file_multiselect_defaults(
            st.session_state,
            key="research_agent_module_files",
            signature_key="research_agent_module_files_folder",
            folder=folder,
            labels=labels,
        )
        _restore_pending_module_file_selection(
            st.session_state,
            key="research_agent_module_files",
            signature_key="research_agent_module_files_folder",
            folder=folder,
            labels=labels,
        )
        _restore_module_file_selection_after_build_rerun(
            st.session_state,
            key="research_agent_module_files",
            signature_key="research_agent_module_files_folder",
            folder=folder,
            labels=labels,
        )
        selected_labels = st.multiselect(
            _ra_text("module_files"),
            labels,
            key="research_agent_module_files",
        )
        selected_files = [module_files[labels.index(label)] for label in selected_labels]
        if not selected_files:
            st.info(_ra_text("select_module_file"))
            return None, ""
        selected_summaries = [
            summaries[labels.index(label)]
            for label in selected_labels
            if label in labels
        ]
        if len(selected_files) < len(module_files):
            st.caption(
                _ra_text(
                    "module_files_subset",
                    selected=len(selected_files),
                    total=len(module_files),
                )
            )

        default_filter_path, default_filter_col = _infer_filter_defaults(
            selected_summaries,
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
            filter_labels = selected_labels
            default_idx = 0
            if default_filter_path is not None and default_filter_path in selected_files:
                default_idx = selected_files.index(default_filter_path)
            filter_label = st.selectbox(
                _ra_text("filter_file"),
                filter_labels,
                index=default_idx,
                key="research_agent_module_filter_file",
            )
            filter_path = selected_files[filter_labels.index(filter_label)]
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

        selected_size_mb = sum(
            (p.stat().st_size for p in selected_files if p.exists()),
            start=0,
        ) / (1024 * 1024)
        large_merge = selected_size_mb >= 200
        if large_merge:
            st.warning(
                _ra_text(
                    "large_merge_warning",
                    files=len(selected_files),
                    size=selected_size_mb,
                )
            )
            if not st.checkbox(
                _ra_text("large_merge_confirm"),
                key="research_agent_module_large_merge_confirmed",
            ):
                return None, ""

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
        st.session_state["_research_agent_module_restore_built_selection"] = True
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        return df, f"module_folder:{folder}"

    if source == source_no_data:
        modules = _available_extract_modules()
        st.info(_ra_text("no_data_info"))
        db_options = ["miiv", "mimic", "eicu", "aumc", "hirid", "sic", "mock"]
        extract_db_value = _sync_extract_db_with_active_data_source(st.session_state, db_options)
        if extract_db_value not in db_options:
            extract_db_value = "miiv"
        db = st.selectbox(
            _ra_text("database"),
            db_options,
            index=db_options.index(extract_db_value),
            key="research_agent_extract_db",
        )
        if db == str(st.session_state.get("database") or ""):
            st.session_state["_research_agent_extract_db_source"] = db
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
        default_modules = _default_extract_module_selection(modules)
        _migrate_legacy_extract_module_selection(st.session_state, modules)
        if _RAW_EXTRACT_MODULE_PRESET_KEY not in st.session_state:
            st.session_state[_RAW_EXTRACT_MODULE_PRESET_KEY] = "all"
        module_preset = st.radio(
            _ra_text("module_preset"),
            ["all", "core", "custom"],
            horizontal=True,
            format_func=lambda value: _ra_text(f"module_preset_{value}"),
            key=_RAW_EXTRACT_MODULE_PRESET_KEY,
        )
        custom_modules: Sequence[str] | None = st.session_state.get(_RAW_EXTRACT_MODULES_KEY)
        if module_preset == "custom":
            custom_modules = st.multiselect(
                _ra_text("modules_extract"),
                list(modules.keys()),
                default=custom_modules or default_modules,
                key=_RAW_EXTRACT_MODULES_KEY,
            )
            st.caption(_ra_text("module_preset_custom_help"))
        picked_modules = _raw_extract_module_selection_for_preset(
            modules,
            str(module_preset),
            custom_modules,
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
        if module_preset != "custom":
            st.caption(
                _ra_text(
                    f"module_preset_{module_preset}_help",
                    modules=len(picked_modules),
                    concepts=len(concepts),
                )
            )
        raw_path_exists = bool(data_path and Path(data_path).expanduser().exists())
        start_export_disabled = not picked_modules or (db != "mock" and not raw_path_exists)
        if st.button(
            _ra_text("start_export"),
            type="primary",
            use_container_width=True,
            key="research_agent_start_export",
            disabled=start_export_disabled,
        ):
            Path(output_dir).expanduser().mkdir(parents=True, exist_ok=True)
            _queue_raw_extract_handoff(
                st.session_state,
                database=db,
                data_path=data_path,
                output_dir=output_dir,
                concepts=concepts,
                modules=picked_modules,
                patient_limit=max_patients,
            )
            st.success(_ra_text("export_queued"))
            st.rerun()
        if start_export_disabled:
            if not picked_modules:
                st.caption(_ra_text("start_export_needs_modules"))
            elif db != "mock" and not raw_path_exists:
                st.caption(_ra_text("start_export_needs_path"))
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
                "Treat this as an ICU clinical-research data-quality audit first: assess missingness for SOFA/SOFA-2 components, "
                "lactate and other lab concepts, unit/range issues on vitals, temporal consistency of measurement timestamps, "
                "and cross-database mapping problems (MIMIC-IV vs eICU vs HiRID etc.), then generate review-ready tables and "
                "figures for an ICU manuscript."
            ),
        },
    ]


def _section_request_picker() -> Tuple[Optional[str], Optional[str]]:
    """Render one unified request box with detailed example prompts."""
    examples = _request_examples()
    is_en = st.session_state.get("language", "en") == "en"
    st.markdown(
        textwrap.dedent(f"""
        <div class="ra-request-brief">
          <div>
            <b>{"Question first" if is_en else "先写问题"}</b>
            <span>{html.escape(_ra_text("request_intro"))}</span>
          </div>
          <div>
            <b>{"Templates optional" if is_en else "模板可选"}</b>
            <span>{html.escape(_ra_text("request_capabilities"))}</span>
          </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

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
                _preserve_module_file_selection_for_next_rerun(st.session_state)
                st.session_state["research_agent_question"] = selected_example["prompt"]
                st.session_state["research_agent_target_outcome"] = selected_example.get("outcome", "")
                st.session_state["research_agent_example_active"] = selected_example["label"]
                st.session_state["research_agent_example_key"] = selected_example["key"]
                st.rerun()
    else:
        st.session_state["research_agent_template_current"] = None

    st.session_state.setdefault("research_agent_question", "")
    st.session_state.setdefault("research_agent_target_outcome", "")
    question = st.text_area(
        _ra_text("question"),
        help=_ra_text("question_help"),
        key="research_agent_question",
        height=112,
    )
    apply_cols = st.columns([1.15, 3.85])
    with apply_cols[0]:
        apply_question = st.button(
            _ra_text("apply_question"),
            key="research_agent_apply_question",
            help=_ra_text("apply_question_help"),
            use_container_width=True,
        )
    with apply_cols[1]:
        st.caption(_ra_text("apply_question_help"))
    if apply_question:
        _preserve_module_file_selection_for_next_rerun(st.session_state)
        if str(st.session_state.get("research_agent_question", "")).strip():
            st.session_state["_research_agent_question_applied_notice"] = True
        else:
            st.session_state["_research_agent_question_empty_notice"] = True
        st.rerun()
    if st.session_state.pop("_research_agent_question_applied_notice", False):
        st.success(_ra_text("question_applied"))
    if st.session_state.pop("_research_agent_question_handoff_notice", False):
        st.success(_ra_text("question_handoff"))
    if st.session_state.pop("_research_agent_question_empty_notice", False):
        st.warning(_ra_text("question_empty"))
    target_outcome = st.text_input(
        _ra_text("target_outcome_optional"),
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

    preference_widget_keys = [
        "research_agent_method_preferences_text",
        "research_agent_evaluation_focus",
        "research_agent_subgroup_sensitivity",
        "research_agent_timing_design",
        "research_agent_data_constraints",
        "research_agent_must_have_outputs",
        "research_agent_covariates",
        "research_agent_extra_notes",
    ]
    for widget_key in preference_widget_keys:
        st.session_state.setdefault(widget_key, "")

    method_pref = st.text_area(
        _ra_text("methods_freeform"),
        height=90,
        key="research_agent_method_preferences_text",
        help=_ra_text("methods_help"),
        placeholder=hint,
    )
    evaluation_focus = st.text_area(
        _ra_text("evaluation_focus"),
        height=80,
        key="research_agent_evaluation_focus",
        help=_ra_text("evaluation_focus_help"),
    )
    subgroup_sensitivity = st.text_area(
        _ra_text("subgroup_sensitivity"),
        height=80,
        key="research_agent_subgroup_sensitivity",
        help=_ra_text("subgroup_sensitivity_help"),
    )
    timing_design = st.text_area(
        _ra_text("timing_design"),
        height=80,
        key="research_agent_timing_design",
        help=_ra_text("timing_design_help"),
    )
    data_constraints = st.text_area(
        _ra_text("data_constraints"),
        height=70,
        key="research_agent_data_constraints",
        help=_ra_text("data_constraints_help"),
    )
    must_have_outputs = st.text_area(
        _ra_text("must_have_outputs"),
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
            key="research_agent_covariates",
            placeholder=covariate_placeholders.get(family, ""),
        )
    extra = st.text_area(
        _ra_text("extra_notes"),
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
    prior_choice = st.session_state.get("research_agent_llm_choice")
    # Preserve the user's explicit choice first. Without this, Streamlit reruns
    # from unrelated controls (preflight checkbox, build cohort) can snap the
    # widget back to the preferred real-provider default.
    if prior_choice in options:
        default_index = options.index(prior_choice)
    # Priority: sidebar-configured shared LLM > Mock for Hosted-only shared
    # settings > override (Custom/OpenRouter prompts for key) > Mock. Hosted
    # is intentionally blocked for Research Agent runs, so defaulting from
    # Settings/Hosted straight into a Custom external endpoint is surprising
    # and makes the launch gate look less local-first than it really is.
    elif sdk_ok and not sidebar_hosted_blocked and is_shared_llm_configured():
        default_index = options.index(sidebar_choice)
    elif sidebar_hosted_blocked:
        default_index = options.index(mock_choice)
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


def _llm_run_readiness(llm_choice: str, api_key: str, model: str) -> Tuple[bool, str]:
    """Return whether the selected LLM can be launched without later key errors."""
    choice = str(llm_choice or "").strip()
    if not choice:
        return False, "provider_missing"
    if "MockLLMClient" in choice or "offline" in choice.lower():
        return True, ""
    if not str(api_key or "").strip():
        return False, "api_key_missing"
    if not str(model or "").strip():
        return False, "model_missing"
    return True, ""


def _llm_readiness_message(issue: str, *, is_en: bool) -> str:
    """Localize compact LLM readiness issues for the launch gate."""
    if issue == "provider_missing":
        return "Model provider is missing." if is_en else "缺少模型服务。"
    if issue == "model_missing":
        return "Model is missing." if is_en else "缺少模型名称。"
    if issue == "api_key_missing":
        return "API key is missing for the selected external provider." if is_en else "当前外部模型缺少 API Key。"
    return str(issue or "")


def _default_research_agent_workdir() -> str:
    return str((Path.cwd() / "research_output" / "webapp").resolve())


def _section_options() -> Tuple[bool, str, bool]:
    cols = st.columns([1.05, 1, 1.35])
    default_workdir = _default_research_agent_workdir()
    _hide_prefilled_directory_text("research_agent_workdir", default_workdir)
    with cols[0]:
        disable_icu_context = False
        st.markdown(
            f"""
            <div class="ra-context-policy">
              <b>{html.escape(_ra_text("context_policy"))}</b>
              <span>{html.escape(_ra_text("context_policy_help"))}</span>
            </div>
            """,
            unsafe_allow_html=True,
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
    status = str(row.get("status") or ("partial" if row.get("manifest_partial") else "complete"))
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


def _format_history_duration(row: Dict[str, Any]) -> str:
    """Return a compact real duration for a history row, or an em dash."""
    started = str(row.get("started_at") or "").strip()
    finished = str(row.get("finished_at") or "").strip()
    if not started or not finished:
        return "—"
    try:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        finish_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return "—"
    seconds = max(0, int((finish_dt - start_dt).total_seconds()))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def _history_status_pill_html(row: Dict[str, Any], *, is_en: bool) -> str:
    status = str(row.get("status") or ("partial" if row.get("manifest_partial") else "complete")).lower()
    failed = int(row.get("step_failed") or 0)
    if row.get("manifest_partial"):
        cls = "warn"
        label = "partial" if is_en else "部分完成"
    elif failed:
        cls = "bad"
        label = "blocked" if is_en else "已阻断"
    elif status in {"complete", "completed", "ok"}:
        cls = "ok"
        label = "complete" if is_en else "完成"
    else:
        cls = "warn"
        label = html.escape(status or ("unknown" if is_en else "未知"))
    return f'<span class="ra-history-pill {cls}"><span></span>{label}</span>'


def _history_rows_table_html(
    rows: Sequence[Dict[str, Any]],
    *,
    is_en: bool,
    selected_run_id: str,
) -> str:
    if not rows:
        empty = "No local manifests found in this workdir." if is_en else "当前工作目录未找到本机 manifest。"
        return f'<div class="ra-history-empty">{html.escape(empty)}</div>'
    headers = (
        ("Run", "Scope", "Status", "Duration", "When", "Evidence")
        if is_en else
        ("Run", "范围", "状态", "耗时", "时间", "证据")
    )
    body_rows: list[str] = []
    for row in rows[:10]:
        run_id = str(row.get("run_id") or "")
        question = _short_card_text(
            row.get("question"),
            "No research question captured." if is_en else "未记录研究问题。",
            limit=72,
        )
        when = str(row.get("started_at") or "")[:16].replace("T", " ") or "—"
        evidence = f"{int(row.get('figure_count') or 0)}F · {int(row.get('table_count') or 0)}T"
        row_cls = " active" if run_id == selected_run_id else ""
        body_rows.append(
            f'<tr class="{row_cls.strip()}">'
            f'<td class="key" data-label="{html.escape(headers[0])}">{html.escape(run_id)}</td>'
            f'<td data-label="{html.escape(headers[1])}">{html.escape(question)}</td>'
            f'<td data-label="{html.escape(headers[2])}">{_history_status_pill_html(row, is_en=is_en)}</td>'
            f'<td class="num" data-label="{html.escape(headers[3])}">{html.escape(_format_history_duration(row))}</td>'
            f'<td class="num muted" data-label="{html.escape(headers[4])}">{html.escape(when)}</td>'
            f'<td class="num muted" data-label="{html.escape(headers[5])}">{html.escape(evidence)}</td>'
            "</tr>"
        )
    return (
        '<div class="ra-history-table-scroll">'
        '<table class="ra-history-table">'
        "<thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers[:3])
        + "".join(f'<th class="num">{html.escape(h)}</th>' for h in headers[3:])
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def _history_export_payload(rows: Sequence[Dict[str, Any]], *, workdir: Path) -> str:
    payload = {
        "source": "easyicu_web_research_agent_history",
        "workdir": str(workdir),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
                "question": row.get("question"),
                "steps": {
                    "ok": row.get("step_ok"),
                    "total": row.get("step_total"),
                    "failed": row.get("step_failed"),
                },
                "evidence": {
                    "figures": row.get("figure_count"),
                    "tables": row.get("table_count"),
                    "records": row.get("evidence_count"),
                },
                "findings": {
                    "errors": row.get("finding_errors"),
                    "warnings": row.get("finding_warnings"),
                },
                "review_decision": row.get("review_decision"),
                "manifest_partial": row.get("manifest_partial"),
                "run_dir": str(row.get("run_dir") or ""),
            }
            for row in rows
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _history_selected_summary_html(
    *,
    run_id: str,
    row: Dict[str, Any],
    status: str,
    question: str,
    finding_summary: str,
    is_en: bool,
) -> str:
    """Compact selected-run summary for the History picker."""
    labels = {
        "selected": "Selected manifest" if is_en else "已选 manifest",
        "status": "Status" if is_en else "状态",
        "steps": "Steps" if is_en else "步骤",
        "duration": "Duration" if is_en else "耗时",
        "evidence": "Evidence" if is_en else "证据",
        "findings": "Findings" if is_en else "发现",
    }
    return f"""
    <div class="ra-history-selected compact">
      <div class="ra-history-selected-title">
        <span>{labels["selected"]}</span>
        <b>{html.escape(run_id)}</b>
      </div>
      <p>{html.escape(question)}</p>
      <div class="ra-history-metrics compact">
        <div><span>{labels["status"]}</span><b>{html.escape(status)}</b></div>
        <div><span>{labels["steps"]}</span><b>{row.get("step_ok", 0)}/{row.get("step_total", 0)}</b></div>
        <div><span>{labels["duration"]}</span><b>{html.escape(_format_history_duration(row))}</b></div>
        <div><span>{labels["evidence"]}</span><b>{row.get("evidence_count", 0)}</b></div>
        <div><span>{labels["findings"]}</span><b>{html.escape(finding_summary)}</b></div>
      </div>
    </div>
    """


def _format_history_findings(errors: Any, warnings: Any, *, is_en: bool) -> str:
    """Human-facing audit count label for history cards and tables."""
    try:
        error_count = int(errors or 0)
    except Exception:
        error_count = 0
    try:
        warning_count = int(warnings or 0)
    except Exception:
        warning_count = 0
    if is_en:
        return f"{error_count} error(s) · {warning_count} warning(s)"
    return f"{error_count} 个错误 · {warning_count} 个警告"


def _render_run_history(workdir: Path) -> None:
    selected_run: Dict[str, Any] | None = None
    expand_history = bool(st.session_state.pop("_research_agent_expand_history", False))
    with st.expander(_ra_text("history_title"), expanded=expand_history):
        history_loaded = bool(st.session_state.get("_research_agent_history_loaded")) or expand_history
        if not history_loaded:
            st.caption(
                "Local run history is loaded on demand from this workdir only; it is not uploaded to GitHub."
                if st.session_state.get("language", "en") == "en" else
                "本机运行历史只会按需从当前工作目录读取；不会上传到 GitHub。"
            )
            if st.button(
                "Load local recent runs" if st.session_state.get("language", "en") == "en" else "加载本机最近 run",
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
        is_en = st.session_state.get("language", "en") == "en"
        table = pd.DataFrame([
            {
                _ra_text("history_run_id"): row["run_id"],
                _ra_text("history_status"): row["status"],
                _ra_text("history_started"): row["started_at"][:19].replace("T", " "),
                _ra_text("history_steps"): f"{row['step_ok']}/{row['step_total']}",
                _ra_text("history_figures"): row["figure_count"],
                _ra_text("history_tables"): row["table_count"],
                _ra_text("history_findings"): _format_history_findings(
                    row["finding_errors"],
                    row["finding_warnings"],
                    is_en=is_en,
                ),
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
            cols[1].metric(
                _ra_text("history_findings"),
                _format_history_findings(
                    selected_run["finding_errors"],
                    selected_run["finding_warnings"],
                    is_en=is_en,
                ),
            )
            cols[2].metric(_ra_text("history_figures"), selected_run["figure_count"])
            safe_run_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(selected_run["run_id"]))
            if cols[3].button(
                "Open in Workbench" if st.session_state.get("language", "en") == "en" else "在工作台打开",
                key=f"research_agent_history_open_wb_{safe_run_id}",
                type="primary",
                use_container_width=True,
            ):
                clear_agent_continuation_state(st.session_state)
                _bind_workbench_state(
                    run_dir=selected_run["run_dir"],
                    manifest=manifest,
                    partial=_partial,
                )
                st.session_state["_active_main_page"] = "research_agent"
                st.session_state["_ra_view"] = "workbench"
                st.rerun()
            _render_resume_panel(
                run_dir=selected_run["run_dir"],
                manifest=manifest,
                row=selected_run,
                key_prefix=f"research_agent_history_resume_{safe_run_id}",
            )
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


def render_research_agent_history_page(lang: Optional[str] = None, *, show_header: bool = True) -> None:
    """Render saved local Research Agent runs as a project picker."""
    _apply_pending_research_agent_workdir(st.session_state)
    lang = lang or st.session_state.get("language", "en")
    is_en = lang == "en"
    title = "EasyICU Research Agent" if is_en else "EasyICU 研究智能体"
    subtitle = (
        "An auditable, evidence-bound workflow — plan, run, review, then draft."
        if is_en else
        "可审计、证据绑定的研究流程：先计划、运行、审阅，再进入起草。"
    )
    if show_header:
        render_page_header(
            title,
            subtitle,
            icon="",
            kicker=_ra_text("kicker"),
        )
    default_workdir = _default_research_agent_workdir()
    workdir_text = str(st.session_state.get("research_agent_workdir") or default_workdir)
    workdir = Path(workdir_text or default_workdir).expanduser().resolve()
    rows = _scan_research_agent_runs(workdir)

    labels = [_format_history_label(row) for row in rows]
    selected_run: Dict[str, Any] | None = None
    selected_label = ""
    if rows:
        selected_label = st.session_state.get("research_agent_history_page_pick") or labels[0]
        if selected_label not in labels:
            selected_label = labels[0]
        selected_run = rows[labels.index(selected_label)]
    selected_run_id = str((selected_run or {}).get("run_id") or "")

    with st.container(key="ra_history_card"):
        head_l, head_r = st.columns([5.0, 1.15], vertical_alignment="center")
        with head_l:
            st.markdown(
                f"""
                <div class="ra-history-card-head">
                  <div>
                    <div class="ra-history-kicker">{"Local manifests only" if is_en else "仅本机 manifest"}</div>
                    <h3>{"Run history" if is_en else "运行历史"}</h3>
                    <p>{"Nothing leaves your machine. Pick a local manifest only when you want to inspect or resume it." if is_en else "所有记录都只来自本机；只有明确选择某个 manifest 时才进入检查或续跑。"}</p>
                  </div>
                  <span>{len(rows)} {"runs found" if is_en else "个 run"}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with head_r:
            st.download_button(
                "Export ledger" if is_en else "导出记录",
                data=_history_export_payload(rows, workdir=workdir),
                file_name="easyicu_research_agent_run_history.json",
                mime="application/json",
                key="research_agent_history_export_ledger",
                use_container_width=True,
                disabled=not rows,
            )
        st.markdown(
            _history_rows_table_html(rows, is_en=is_en, selected_run_id=selected_run_id),
            unsafe_allow_html=True,
        )

        if not rows:
            c1, _c2 = st.columns([1.1, 5.0])
            if c1.button("Back to setup" if is_en else "返回配置", key="research_agent_history_back_empty"):
                st.session_state["_active_main_page"] = "research_agent"
                st.session_state["_ra_view"] = "setup"
                st.rerun()
            return

        st.markdown(
            f'<div class="ra-history-workdir"><span>{"Workdir" if is_en else "工作目录"}</span><b>{html.escape(str(workdir))}</b></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Change local workdir" if is_en else "切换本机工作目录", expanded=False):
            _hide_prefilled_directory_text("research_agent_workdir", default_workdir)
            _directory_input(
                "Local run workdir" if is_en else "本机 run 工作目录",
                value=default_workdir,
                input_key="research_agent_workdir",
                button_key="research_agent_history_workdir_browse",
                placeholder=_placeholder_path("research_output/webapp"),
                show_value=False,
            )
            st.caption(
                "History is scanned from this folder only; changing it refreshes the table on the next render."
                if is_en else
                "历史记录只从这个目录读取；切换目录后下次渲染会刷新表格。"
            )
        picker_cols = st.columns([4.8, 1.25, 1.05], vertical_alignment="bottom")
        with picker_cols[0]:
            selected_label = st.selectbox(
                "Open local run" if is_en else "打开本机 run",
                labels,
                index=labels.index(selected_label),
                key="research_agent_history_page_pick",
            )
        selected_run = rows[labels.index(selected_label)]
        manifest, manifest_path, partial = _load_run_manifest(selected_run["run_dir"])
        run_id = str(selected_run.get("run_id") or selected_run["run_dir"].name)
        safe_run_id = re.sub(r"[^A-Za-z0-9_]+", "_", run_id)
        status = str(selected_run.get("status") or ("partial" if partial else "complete"))
        question = _short_card_text(
            selected_run.get("question"),
            "No research question captured." if is_en else "未记录研究问题。",
            limit=220,
        )
        finding_summary = _format_history_findings(
            selected_run.get("finding_errors", 0),
            selected_run.get("finding_warnings", 0),
            is_en=is_en,
        )

        with picker_cols[1]:
            if st.button(
                "Open in Workbench" if is_en else "在工作台打开",
                key=f"research_agent_history_page_open_{safe_run_id}",
                type="primary",
                use_container_width=True,
                disabled=not bool(manifest),
            ):
                if manifest:
                    clear_agent_continuation_state(st.session_state)
                    _bind_workbench_state(
                        run_dir=selected_run["run_dir"],
                        manifest=manifest,
                        partial=partial,
                    )
                    st.session_state["_active_main_page"] = "research_agent"
                    st.session_state["_ra_view"] = "workbench"
                    st.rerun()
        with picker_cols[2]:
            if st.button(
                "Back to setup" if is_en else "返回配置",
                key=f"research_agent_history_page_setup_{safe_run_id}",
                use_container_width=True,
            ):
                st.session_state["_active_main_page"] = "research_agent"
                st.session_state["_ra_view"] = "setup"
                st.rerun()

        if manifest:
            st.markdown(
                _history_selected_summary_html(
                    run_id=run_id,
                    row=selected_run,
                    status=status,
                    question=question,
                    finding_summary=finding_summary,
                    is_en=is_en,
                ),
                unsafe_allow_html=True,
            )

    if manifest:
        with st.container(key=f"ra_history_utilities_{safe_run_id}"):
            st.markdown(
                f"""
                <div class="ra-history-utilities-head">
                  <div>
                    <span>{"Selected run utilities" if is_en else "已选 run 工具"}</span>
                    <b>{"Resume or inspect only when needed" if is_en else "仅在需要时继续运行或检查详情"}</b>
                  </div>
                  <em>{"optional" if is_en else "可选"}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
            util_cols = st.columns([1, 1], gap="small")
            with util_cols[0]:
                show_resume = st.checkbox(
                    "Resume controls" if is_en else "继续运行控制",
                    key=f"research_agent_history_page_show_resume_{safe_run_id}",
                )
            with util_cols[1]:
                show_details = st.checkbox(
                    "Detailed report and artefacts" if is_en else "详细报告与产物",
                    key=f"research_agent_history_page_show_details_{safe_run_id}",
                )
        if show_resume:
            _render_resume_panel(
                run_dir=selected_run["run_dir"],
                manifest=manifest,
                row=selected_run,
                key_prefix=f"research_agent_history_page_resume_{safe_run_id}",
            )
        if show_details:
            _render_run_manifest(
                run_dir=selected_run["run_dir"],
                manifest=manifest,
                manifest_path=manifest_path,
                key_prefix=f"research_agent_history_page_{safe_run_id}",
            )

    with st.expander(_ra_text("replication_title"), expanded=False):
        st.caption(
            "Deterministic replication is kept here as a local utility so Setup stays focused on launching one agent run."
            if is_en else
            "确定性复现实用工具放在这里，配置页只负责启动一次 agent 运行。"
        )
        _render_replication_section(default_workdir=workdir)


def _render_resume_panel(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    row: Dict[str, Any],
    key_prefix: str,
) -> None:
    """Editor letting the user resume a non-complete run with new guidance.

    Surfaces:
      * a summary of completed vs failed/blocked steps (so the user knows
        what will be reused vs replanned);
      * a free-form notes field that is appended to ``notes`` for the
        planner / coder agents on the re-run;
      * a toggle for ``EASYICU_AUDIT_RELAX_PROBE`` so the user can opt
        into the documented ablation (probe-stage block → warning) when
        the prior run was halted by ``blocked_by_concept_audit``;
      * a "Force manuscript" shortcut (reuses the existing
        ``force_manuscript`` flag) — useful when analysis is complete but
        the writer step needs to re-run after a manual evidence fix.

    The panel only stashes session_state and switches to the Setup view;
    the existing kickoff path picks up ``research_agent_resume_run_id``
    and invokes ``_run_pipeline`` with the resume kwargs.
    """
    is_en = st.session_state.get("language", "en") == "en"
    records = [r for r in manifest.get("per_step_records", []) if isinstance(r, dict)]
    findings = [f for f in manifest.get("findings", []) if isinstance(f, dict)]

    def _step_status(rec: Dict[str, Any]) -> str:
        return str(rec.get("status") or rec.get("step_status") or "unknown")

    completed = [
        r for r in records
        if _step_status(r) in {"ok", "complete", "success", "completed"}
    ]
    failed = [
        r for r in records
        if any(tok in _step_status(r) for tok in ("fail", "error", "blocked", "skipped"))
    ]
    status_label = str(row.get("status") or ("analysis_only" if row.get("manifest_partial") else "complete"))
    can_resume = status_label != "manuscript_ready"
    has_concept_audit_block = any(
        "blocked_by_concept_audit" in _step_status(r) for r in failed
    )

    audit_warnings = [
        f for f in findings
        if str(f.get("severity") or "") in {"warning", "error"}
    ]

    resume_expanded = can_resume and (status_label != "complete" or bool(failed))
    with st.expander(_ra_text("resume_section"), expanded=resume_expanded):
        st.caption(_ra_text("resume_intro"))
        info_cols = st.columns([1, 1, 1])
        info_cols[0].metric(_ra_text("resume_status"), status_label)
        info_cols[1].metric(_ra_text("resume_completed_steps"), len(completed))
        info_cols[2].metric(_ra_text("resume_failed_steps"), len(failed))

        if failed:
            st.markdown(f"**{_ra_text('resume_failed_steps')}**")
            for rec in failed[:20]:
                sid = rec.get("step_id") or rec.get("id") or "?"
                status = _step_status(rec)
                msg = (
                    rec.get("error_message")
                    or rec.get("message")
                    or (rec.get("error") or {}).get("message", "")
                    if isinstance(rec.get("error"), dict)
                    else rec.get("error", "")
                )
                line = f"- `{sid}` · **{status}**"
                if msg:
                    line += f" — {str(msg)[:240]}"
                st.markdown(line)

        if audit_warnings:
            st.markdown(f"**{_ra_text('resume_findings_summary')}**")
            for f in audit_warnings[:15]:
                sev = str(f.get("severity") or "info")
                code = f.get("code") or f.get("validator") or "audit"
                msg = f.get("message") or ""
                badge = _FINDING_BADGE.get(sev, "🔵")
                st.markdown(f"{badge} `{code}` — {str(msg)[:280]}")

        relax_default = bool(has_concept_audit_block)
        relax_probe = st.checkbox(
            _ra_text("resume_relax_probe_label"),
            value=relax_default,
            key=f"{key_prefix}_relax_probe",
            help=_ra_text("resume_relax_probe_help"),
        )
        if relax_probe:
            st.warning(_ra_text("resume_relaxed_active"))

        # Pre-fill helpful guidance based on the failure pattern. Users
        # see this and can edit before submitting.
        if has_concept_audit_block:
            default_notes = (
                "The previous run was blocked by the concept-usage auditor. "
                "Avoid reporting mean/std on SOFA-family component or total scores; "
                "use median + IQR for distribution descriptions and use the "
                "SOFA score as an ordinal/categorical covariate in any model."
                if is_en else
                "上次运行被 concept-usage auditor 拦截。不要对 SOFA 系列（总分或分项）"
                "汇报 mean/std；分布请改用中位数 + IQR；在任何建模里把 SOFA 当作"
                "有序变量或分类协变量处理。"
            )
        else:
            default_notes = ""

        notes_state_key = f"{key_prefix}_notes"
        if notes_state_key not in st.session_state:
            st.session_state[notes_state_key] = default_notes
        extra_notes = st.text_area(
            _ra_text("resume_notes_label"),
            value=st.session_state[notes_state_key],
            placeholder=_ra_text("resume_notes_hint"),
            key=notes_state_key,
            height=140,
        )

        run_id = str(manifest.get("run_id") or row.get("run_id") or run_dir.name)
        # Both continuation paths return to Setup. Seed the prior question so
        # users can review or edit it there instead of reconstructing it.
        prior_question = str(manifest.get("research_question") or row.get("question") or "").strip()
        action_cols = st.columns([2, 2])
        if action_cols[0].button(
            _ra_text("resume_button"),
            key=f"{key_prefix}_btn_resume",
            type="primary",
            use_container_width=True,
            disabled=not can_resume,
        ):
            st.session_state["research_agent_resume_run_id"] = run_id
            _store_resume_run_dir_context(st.session_state, run_dir, defer_workdir=True)
            st.session_state["research_agent_force_manuscript"] = False
            st.session_state["research_agent_resume_mode"] = "continue"
            st.session_state["research_agent_resume_notes"] = extra_notes
            st.session_state["research_agent_resume_relax_probe"] = bool(relax_probe)
            if prior_question:
                st.session_state["research_agent_question"] = prior_question
            st.session_state["_active_main_page"] = "research_agent"
            st.session_state["_ra_view"] = "setup"
            st.session_state["_research_agent_expand_history"] = False
            st.rerun()

        if action_cols[1].button(
            _ra_text("resume_force_manuscript"),
            key=f"{key_prefix}_btn_force_ms",
            use_container_width=True,
            help=_ra_text("resume_force_manuscript_help"),
            disabled=not can_resume,
        ):
            st.session_state["research_agent_resume_run_id"] = run_id
            _store_resume_run_dir_context(st.session_state, run_dir, defer_workdir=True)
            st.session_state["research_agent_force_manuscript"] = True
            st.session_state["research_agent_resume_mode"] = "force_manuscript"
            st.session_state["research_agent_resume_notes"] = ""
            st.session_state["research_agent_resume_relax_probe"] = False
            if prior_question:
                st.session_state["research_agent_question"] = prior_question
            st.session_state["_active_main_page"] = "research_agent"
            st.session_state["_ra_view"] = "setup"
            st.session_state["_research_agent_expand_history"] = False
            st.rerun()


def _render_research_agent_demo_visuals(*, is_en: bool) -> None:
    """Render the Claude-reference setup overview without launching a run."""
    stages = [
        ("01", "play", "Plan" if is_en else "规划", "question -> recipe" if is_en else "问题 -> 配方"),
        ("02", "layers", "Build" if is_en else "组装", "exports -> one row / stay" if is_en else "导出 -> 每次住院一行"),
        ("03", "bars", "Analyze" if is_en else "分析", "tables, figures, checks" if is_en else "表格、图、检查"),
        ("04", "agent", "Gate" if is_en else "关口", "evidence before drafting" if is_en else "写作前证据检查"),
        ("05", "check", "Review" if is_en else "复核", "approve, rerun, export" if is_en else "批准、重跑、导出"),
    ]
    stage_html = "".join(
        '<div class="ra-setup-stage">'
        f'<span title="{html.escape(idx)}">{_shell_icon(icon_name) or html.escape(idx)}</span>'
        f'<b>{html.escape(title)}</b>'
        f'<em>{html.escape(body)}</em>'
        '</div>'
        for idx, icon_name, title, body in stages
    )
    context_items = [
        ("Dataset" if is_en else "数据集", "Demo · 10 stays"),
        ("Mode" if is_en else "模式", "static preview" if is_en else "静态预览"),
        ("Modules" if is_en else "模块", "19 feature groups" if is_en else "19 个特征组"),
        ("Privacy" if is_en else "隐私", "local only" if is_en else "仅本机"),
    ]
    context_html = "".join(
        '<div class="eu-ref-context-item">'
        f'<span>{html.escape(label)}</span>'
        f'<b>{html.escape(value)}</b>'
        '</div>'
        for label, value in context_items
    )
    concepts = ["vitals", "labs", "sofa", "demographics", "outcomes", "vent", "lactate", "renal"]
    concepts_html = "".join(
        f'<span class="eu-ref-chip">{html.escape(item)}</span>'
        for item in concepts
    )
    plan = [
        ("Cohort summary", "n, demographics, outcome rates", "ready"),
        ("Table 1", "baseline characteristics by group", "ready"),
        ("Missingness audit", "per-concept coverage + denominators", "ready"),
        ("Model: LR + SOFA + lactate", "first-24h predictors", "ready"),
        ("ROC · Calibration", "discrimination + calibration", "ready"),
        ("Manuscript draft", "methods + results", "gated"),
    ]
    plan_html = "".join(
        '<div class="eu-ref-plan-item {cls}">'.format(cls="gated" if status == "gated" else "ready")
        + f'<div class="eu-ref-pi-n mono">{idx:02d}</div>'
        + '<div class="eu-ref-pi-node"></div>'
        + '<div class="eu-ref-pi-body">'
        + f'<div class="eu-ref-pi-t">{html.escape(title)}</div>'
        + f'<div class="eu-ref-pi-d">{html.escape(desc)}</div>'
        + '</div>'
        + '<div class="eu-ref-pi-tag">'
        + (
            '<span class="eu-ref-pill gated">requires review</span>'
            if status == "gated" else
            '<span class="eu-ref-pill ok"><span class="dot"></span>planned</span>'
        )
        + '</div></div>'
        for idx, (title, desc, status) in enumerate(plan, start=1)
    )

    st.markdown(
        textwrap.dedent(f"""
        <div class="eu-ref-workbench eu-ref-agent-setup">
          <div class="ra-setup-overview eu-ref-setup-operating ra-pipeline-overview">
            <div class="ra-setup-head">
              <div>
                <div class="ra-setup-kicker">{"Operating model" if is_en else "运行模型"}</div>
                <h3>{"An auditable workflow, not a black-box chat" if is_en else "可审计工作流，而不是黑箱聊天"}</h3>
                <p>{"Each stage produces a reviewable artifact. Drafting stays locked until evidence checks pass and you confirm." if is_en else "每个阶段都会生成可复核产物；证据检查和人工确认前不会进入写作。"}</p>
              </div>
            </div>
            <div class="ra-setup-stage-list">{stage_html}</div>
          </div>
          <div class="eu-ref-split eu-ref-setup-split">
          <div class="eu-ref-card eu-ref-pad ra-context-pack-card">
              <div class="eu-ref-card-head">
                <div class="eu-ref-eyebrow">{"Context pack" if is_en else "上下文包"}</div>
                <span>{"handed off" if is_en else "已交接"}</span>
              </div>
              <div class="ra-context-pack-title">sepsis_mortality_demo</div>
              <div class="ra-context-pack-sub">{"demo · 10 stays · 19 modules" if is_en else "演示 · 10 次住院 · 19 个模块"}</div>
              <div class="ra-context-pack-list">{context_html}</div>
              <div class="eu-ref-eyebrow eu-ref-chip-title">{"Concept tray · 8 selected" if is_en else "概念托盘 · 8 个已选"}</div>
              <div class="eu-ref-chip-row">{concepts_html}</div>
              <div class="ra-context-pack-action">{"Switch cohort" if is_en else "切换队列"}</div>
          </div>
          <div class="eu-ref-setup-stack">
          <div class="eu-ref-card eu-ref-pad ra-question-card">
                <div class="eu-ref-card-head">
                  <div>
                    <div class="eu-ref-eyebrow">{"Research question" if is_en else "研究问题"}</div>
                    <span class="ra-question-helper">{"One sentence. The agent drafts a plan first; you confirm before any model call." if is_en else "一句话问题。智能体先草拟计划，确认后才调用模型。"}</span>
                  </div>
                  <span class="eu-ref-pill">{"Demo · no LLM call" if is_en else "Demo · 不调用模型"}</span>
                </div>
                <div class="eu-ref-question-box">{"Which bedside features within the first 24 hours best predict in-hospital mortality among Sepsis-3 patients, and how does adding lactate change calibration?" if is_en else "入 ICU 后 24 小时内哪些床旁特征最能预测 Sepsis-3 患者院内死亡？加入乳酸后校准表现如何变化？"}</div>
                <div class="eu-ref-chip-row">
                  <span class="eu-ref-chip">@sepsis_demo</span>
                  <span class="eu-ref-chip">@first_24h</span>
                  <span class="eu-ref-chip">@lactate</span>
                </div>
          </div>
          <div class="eu-ref-card eu-ref-pad ra-plan-preview-card">
                <div class="eu-ref-card-head">
                  <div class="eu-ref-eyebrow">{"Plan preview · 6 steps" if is_en else "计划预览 · 6 步"}</div>
                  <span>{"5 ready · 1 gated" if is_en else "5 就绪 · 1 受控"}</span>
                </div>
                <div class="eu-ref-planlist">{plan_html}</div>
          </div>
          <div class="eu-ref-note warn">
            <div class="eu-ref-note-ico">!</div>
            <div class="eu-ref-note-body">
              <div class="eu-ref-note-head">
                <b>{"Preflight gate" if is_en else "执行前关口"}</b>
                <span class="eu-ref-pill queued">{"real data required" if is_en else "需要真实数据"}</span>
              </div>
              <p>{"Demo Mode explains the workflow only. Switch to Real Data Mode to bind a cohort, confirm the plan, and launch the real backend pipeline." if is_en else "演示模式只解释工作流；切换到真实数据模式后，才会绑定队列、确认计划并启动真实后端 pipeline。"}</p>
            </div>
          </div>
          </div>
          </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def _activate_real_data_mode_from_agent(state: MutableMapping[str, Any]) -> None:
    """Move the shared workspace from demo preview into real-data setup."""
    previous_database = state.get("database")
    state["entry_mode"] = "real"
    state["use_mock_data"] = False
    if previous_database not in {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}:
        state["database"] = "miiv"
        state["path_validated"] = False
        state.pop("last_validated_path", None)
    for key in ("step1_confirmed", "step2_confirmed", "step3_confirmed", "export_completed"):
        state[key] = False
    state["trigger_export"] = False
    state["_exporting_in_progress"] = False
    state["loaded_concepts"] = {}
    state["loaded_data_origin"] = "none"
    state["patient_ids"] = []
    state["all_patient_count"] = 0
    state["selected_patient"] = None
    state["selected_concepts"] = []
    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "setup"


def _queue_raw_extract_handoff(
    state: MutableMapping[str, Any],
    *,
    database: str,
    data_path: str,
    output_dir: str,
    concepts: list[str],
    modules: list[str],
    patient_limit: int,
) -> None:
    """Copy raw-extraction settings into Data Extraction for review.

    The no-data Agent path prepares the same source, cohort, and concept
    choices the extraction workflow would collect manually, then opens the
    Step 4 review screen. Export still requires the user to press
    ``Confirm & Export`` in Data Extraction.
    """
    is_mock = database == "mock"
    state["entry_mode"] = "real"
    state["database"] = database
    state["use_mock_data"] = is_mock
    state["data_path"] = "" if is_mock else data_path
    state["path_validated"] = is_mock or bool(data_path and Path(data_path).expanduser().exists())
    if is_mock:
        state.pop("last_validated_path", None)
    elif state["path_validated"]:
        state["last_validated_path"] = data_path
    state["export_path"] = output_dir
    state["selected_concepts"] = list(concepts)
    state["selected_groups"] = list(modules)
    state["step1_confirmed"] = True
    state["step2_confirmed"] = True
    state["step3_confirmed"] = True
    state["export_format"] = "Parquet"
    state["patient_limit"] = int(patient_limit or 0)
    state["export_completed"] = False
    state["loaded_concepts"] = {}
    state["loaded_data_origin"] = "none"
    state["patient_ids"] = []
    state["all_patient_count"] = 0
    state["selected_patient"] = None
    for key in (
        "_skipped_modules",
        "_overwrite_modules",
        "_existing_modules_list",
        "_export_conflict_pending",
        "_export_cancel_notice",
        "_post_export_navigation_pending",
        "_post_export_target_panel",
        "_post_export_guidance_dismissed",
        "_export_success_result",
    ):
        state.pop(key, None)
    state["trigger_export"] = False
    state["_exporting_in_progress"] = False
    state["_active_main_page"] = "extract"
    state.pop("_scroll_to_tab", None)
    state["_scroll_to_top"] = True


def render_research_agent_demo_page(*, show_header: bool = True) -> None:
    """Render a guide-only Research Agent page for Demo Mode.

    Demo mode should explain the research-agent value proposition without
    asking for API keys, work directories, or a real pipeline run. The full
    execution workflow remains available in Real Data Mode.
    """
    lang = st.session_state.get("language", "en")
    is_en = lang == "en"

    caption = (
        "Demo Mode is a lightweight preview. It shows what the agent can produce. "
        "No LLM call, no token use, no fabricated analysis pack."
        if is_en else
        "演示模式只做轻量导览：展示智能体能产出什么，不真正运行 LLM pipeline，也不消耗 token。"
    )
    if show_header:
        st.markdown(
            cc.render_design_page_header(
                kicker=_ra_text("kicker"),
                title_en="EasyICU Research Agent",
                title_zh="EasyICU Research Agent",
                desc=(
                    "An auditable, evidence-bound workflow — plan, run, review, then draft."
                    if is_en else
                    "一个可审计、证据绑定的工作流：先计划、运行、复核，再进入草稿。"
                ),
                right_html=(
                    '<span class="eu-pill">Runs · preview</span>'
                    '<span class="eu-pill">Static guide</span>'
                    if is_en else
                    '<span class="eu-pill">运行 · 预览</span>'
                    '<span class="eu-pill">静态导览</span>'
                ),
                lang=lang,
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        textwrap.dedent(f"""
        <div class="eu-ref-note demo">
          <div class="eu-ref-note-ico">i</div>
          <div class="eu-ref-note-body">
            <div class="eu-ref-note-head"><b>{"Demo Mode — lightweight preview." if is_en else "Demo 模式 — 轻量预览。"}</b></div>
            <p>{html.escape(caption)} {"Switch to Real Data to connect a stay-level file or module export." if is_en else "切换到真实数据模式后再接入 stay-level 文件或模块导出。"}</p>
          </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
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
        _activate_real_data_mode_from_agent(st.session_state)
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
    llm_ready: bool = True,
    llm_issue: str = "",
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
        "llm_ready": bool(llm_ready),
        "llm_issue": str(llm_issue or ""),
        "workdir": output_dir,
        "write_targets": write_targets,
        "mode": "draft_continuation" if force_manuscript else ("analysis_only" if stop_after_analysis else "analysis_plus_manuscript_gate"),
        "template_key": template_key or "",
        "template_contract": _template_contract(template_key, language=language),
    }


def _short_card_text(value: Any, fallback: str, *, limit: int = 170) -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _render_research_agent_setup_overview(
    *,
    free_question: Optional[str],
    target_outcome: Optional[str],
    cohort: Optional[pd.DataFrame],
    cohort_label: str,
    llm_choice: str,
    model: str,
    workdir_text: str,
    stop_after_analysis: bool,
    force_manuscript: bool,
    llm_ready: bool = True,
    llm_issue: str = "",
) -> None:
    """Render a Claude-reference overview above the detailed setup controls."""
    lang = st.session_state.get("language", "en")
    is_en = lang == "en"
    template_key = st.session_state.get("research_agent_template_current") or st.session_state.get("research_agent_example_key")
    contract = _build_execution_preflight_contract(
        free_question=free_question or "",
        target_outcome=target_outcome or "",
        cohort=cohort,
        cohort_label=cohort_label or "",
        llm_choice=llm_choice or "",
        model=model or "",
        workdir_text=workdir_text or _default_research_agent_workdir(),
        stop_after_analysis=stop_after_analysis,
        force_manuscript=force_manuscript,
        template_key=str(template_key or ""),
        language=lang,
        llm_ready=llm_ready,
        llm_issue=llm_issue,
    )
    template_contract = contract["template_contract"]
    rows = int(contract.get("cohort_rows") or 0)
    question_ready = bool(str(contract.get("question") or "").strip()) or force_manuscript
    cohort_ready = cohort is not None and rows > 0
    preflight_confirmed = bool(st.session_state.get("research_agent_preflight_confirmed"))
    llm_label = str(contract.get("llm_choice") or ("provider selected below" if is_en else "在下方选择模型"))
    model_label = str(contract.get("model") or ("default" if is_en else "默认"))
    llm_ready = bool(contract.get("llm_ready", True))
    llm_issue_msg = _llm_readiness_message(str(contract.get("llm_issue") or ""), is_en=is_en)
    mode_label = {
        "draft_continuation": "Draft continuation" if is_en else "续写草稿",
        "analysis_only": "Analysis only" if is_en else "仅分析",
        "analysis_plus_manuscript_gate": "Analysis + manuscript gate" if is_en else "分析 + 手稿关口",
    }.get(str(contract.get("mode")), str(contract.get("mode") or ""))
    stages = [
        ("01", "play", "Plan" if is_en else "规划", "question -> recipe" if is_en else "问题 -> 配方", question_ready),
        ("02", "layers", "Build" if is_en else "组装", "cohort -> one row / stay" if is_en else "队列 -> 每次住院一行", cohort_ready),
        ("03", "bars", "Analyze" if is_en else "分析", "tables, figures, checks" if is_en else "表格、图、检查", cohort_ready and question_ready and llm_ready),
        ("04", "agent", "Gate" if is_en else "关口", "evidence before drafting" if is_en else "写作前证据检查", preflight_confirmed),
        ("05", "check", "Review" if is_en else "复核", "approve, rerun, export" if is_en else "批准、重跑、导出", preflight_confirmed),
    ]
    stage_html = "".join(
        f'<div class="ra-setup-stage {"done" if ok else ""}">'
        f'<span title="{html.escape(idx)}">{_shell_icon(icon_name) or html.escape(idx)}</span>'
        f'<b>{html.escape(title)}</b>'
        f'<em>{html.escape(body)}</em>'
        '</div>'
        for idx, icon_name, title, body, ok in stages
    )

    def _plan_item(idx: int, title: str, detail: str, state: str) -> str:
        state_label = {
            "ready": "ready" if is_en else "就绪",
            "missing": "missing" if is_en else "缺失",
            "gated": "gated" if is_en else "待确认",
            "planned": "planned" if is_en else "已计划",
        }.get(state, state)
        return (
            f'<div class="ra-setup-plan-item {html.escape(state)}">'
            f'<div class="ra-setup-plan-index">{idx:02d}</div>'
            '<div class="ra-setup-plan-node"></div>'
            '<div class="ra-setup-plan-body">'
            f'<b>{html.escape(title)}</b>'
            f'<span>{html.escape(detail)}</span>'
            '</div>'
            f'<em>{html.escape(state_label)}</em>'
            '</div>'
        )

    expected_outputs = [str(item) for item in list(template_contract.get("expected_outputs", []))[:3]]
    checkpoints = [str(item) for item in list(template_contract.get("checkpoints", []))[:3]]
    plan_items = [
        (
            "Research request" if is_en else "研究请求",
            "question captured" if question_ready and is_en else ("问题已填写" if question_ready else ("enter a one-sentence request" if is_en else "填写一句研究问题")),
            "ready" if question_ready else "missing",
        ),
        (
            "Cohort context" if is_en else "队列上下文",
            f"{rows:,} stay-level rows" if cohort_ready else ("select or upload a cohort" if is_en else "选择或上传队列"),
            "ready" if cohort_ready else "missing",
        ),
        (
            "Analysis contract" if is_en else "分析契约",
            str(template_contract.get("label") or ("free-form template" if is_en else "自由模板")),
            "planned",
        ),
        (
            "Expected outputs" if is_en else "预期产物",
            ", ".join(expected_outputs) if expected_outputs else ("results report" if is_en else "结果报告"),
            "planned",
        ),
        (
            "Evidence checkpoints" if is_en else "证据检查点",
            ", ".join(checkpoints) if checkpoints else ("numeric claims gated" if is_en else "数值主张受控"),
            "planned",
        ),
        (
            "Human preflight" if is_en else "人工预检查",
            "confirmed below" if preflight_confirmed and is_en else ("已在下方确认" if preflight_confirmed else ("confirm below before launch" if is_en else "启动前在下方确认")),
            "ready" if preflight_confirmed else "gated",
        ),
    ]
    plan_html = "".join(
        _plan_item(i + 1, title, detail, state)
        for i, (title, detail, state) in enumerate(plan_items)
    )

    def _gate_row(label: str, value: str, ok: bool) -> str:
        klass = "ok" if ok else "warn"
        return (
            f'<div class="ra-setup-gate-row {klass}">'
            f'<span>{html.escape(label)}</span>'
            f'<b>{html.escape(value)}</b>'
            '</div>'
        )

    gate_rows = "".join([
        _gate_row(
            "Request" if is_en else "请求",
            "ready" if question_ready and is_en else ("已就绪" if question_ready else ("missing" if is_en else "未填写")),
            question_ready,
        ),
        _gate_row(
            "Cohort" if is_en else "队列",
            f"{rows:,} rows" if cohort_ready else ("not selected" if is_en else "未选择"),
            cohort_ready,
        ),
        _gate_row(
            "LLM" if is_en else "模型",
            llm_issue_msg if not llm_ready else (
                "external provider" if contract.get("external_llm") and is_en else (
                    "外部模型" if contract.get("external_llm") else ("offline/mock" if is_en else "离线/模拟")
                )
            ),
            llm_ready,
        ),
        _gate_row(
            "Preflight" if is_en else "预检查",
            "confirmed" if preflight_confirmed and is_en else ("已确认" if preflight_confirmed else ("locked below" if is_en else "下方确认")),
            preflight_confirmed,
        ),
    ])
    context_values = [
        (("Cohort" if is_en else "队列"), contract.get("cohort_label") or ("not selected" if is_en else "未选择")),
        (("Rows" if is_en else "行数"), f"{rows:,}" if rows else "0"),
        (("Model" if is_en else "模型"), f"{llm_label} · {model_label}"),
        (("Mode" if is_en else "模式"), mode_label),
    ]
    context_html = "".join(
        '<div>'
        f'<span>{html.escape(label)}</span>'
        f'<b>{html.escape(_short_card_text(value, "-", limit=90))}</b>'
        '</div>'
        for label, value in context_values
    )
    concept_tray = [
        "stay-level cohort" if is_en else "住院级队列",
        "local manifest" if is_en else "本机 manifest",
        "evidence gate" if is_en else "证据关口",
        "human review" if is_en else "人工复核",
    ]
    tray_html = "".join(
        f'<span>{html.escape(item)}</span>'
        for item in concept_tray
    )
    workdir_html = html.escape(_short_card_text(str(contract.get("workdir") or ""), "-", limit=120))
    question_copy = _short_card_text(
        contract.get("question"),
        "No request yet. Start with step 1 below." if is_en else "还没有研究请求，请从下方第 1 步开始。",
        limit=230,
    )
    preflight_count = int(question_ready) + int(cohort_ready) + int(llm_ready) + int(preflight_confirmed)
    gate_state_label = f"{preflight_count} / 4 ready" if is_en else f"{preflight_count} / 4 就绪"
    gate_message = (
        "No model call happens until this page confirms the plan, context disclosure, and file impact."
        if is_en else
        "在本页确认计划、上下文披露和文件影响之前，不会调用模型。"
    )
    gate_action = (
        ("Ready to run" if is_en else "可以运行")
        if preflight_confirmed else
        ("Confirm below before launch" if is_en else "启动前在下方确认")
    )
    context_badge = (
        ("ready" if is_en else "已就绪")
        if cohort_ready else
        ("awaiting cohort" if is_en else "等待队列")
    )
    st.markdown(
        f"""
        <div class="ra-setup-overview ra-pipeline-overview">
          <div class="ra-setup-operating">
            <div>
              <div class="ra-setup-kicker">{"Operating model" if is_en else "运行模型"}</div>
              <h3>{"An auditable workflow, not a black-box chat" if is_en else "可审计流程，而不是黑箱聊天"}</h3>
              <p>{"Each stage produces a reviewable artifact. Drafting stays locked until evidence checks pass and you confirm." if is_en else "每个阶段都会产生可复核产物；证据检查通过并由你确认前，手稿起草保持锁定。"}</p>
            </div>
          </div>
          <div class="ra-setup-stage-list">{stage_html}</div>
          <div class="ra-setup-split">
            <div class="ra-setup-card context">
              <div class="ra-setup-card-title">{"Context pack" if is_en else "上下文包"} <span>{html.escape(context_badge)}</span></div>
              <div class="ra-setup-context-grid">{context_html}</div>
              <div class="ra-setup-card-title tray">{"Concept tray" if is_en else "概念托盘"}</div>
              <div class="ra-setup-tray">{tray_html}</div>
              <div class="ra-setup-workdir"><span>{"Output" if is_en else "输出"}</span><b>{workdir_html}</b></div>
            </div>
            <div class="ra-setup-main">
              <div class="ra-setup-card question">
                <div class="ra-setup-question-head">
                  <div>
                    <b>{"Research question" if is_en else "研究问题"}</b>
                    <span>{"One sentence. The agent drafts a plan first; you confirm before any model call." if is_en else "一句话描述。智能体先生成计划，任何模型调用前都需要你确认。"}</span>
                  </div>
                  <em>{html.escape("ready" if question_ready and is_en else ("已就绪" if question_ready else ("missing" if is_en else "未填写")))}</em>
                </div>
                <div class="ra-setup-qbox">{html.escape(question_copy)}</div>
              </div>
              <div class="ra-setup-card plan">
                <div class="ra-setup-card-title">{"Plan preview · 6 steps" if is_en else "计划预览 · 6 步"}</div>
                <div class="ra-setup-plan-list">{plan_html}</div>
              </div>
              <div class="ra-setup-gate-strip">
                <span>{html.escape(gate_state_label)}</span>
                <div><b>{"Preflight gate" if is_en else "执行前关口"}</b><p>{html.escape(gate_message)}</p></div>
                <em>{html.escape(gate_action)}</em>
              </div>
            </div>
          </div>
          <div class="ra-setup-gates compact">{gate_rows}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            "llm_ready",
            "llm_issue",
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
    llm_ready: bool = True,
    llm_issue: str = "",
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
        llm_ready=llm_ready,
        llm_issue=llm_issue,
    )
    mode = (
        "Draft continuation" if force_manuscript else
        ("Analysis only" if stop_after_analysis else "Analysis + optional manuscript gate")
    ) if is_en else (
        "续写草稿" if force_manuscript else
        ("仅分析" if stop_after_analysis else "分析 + 可选手稿关口")
    )
    template_contract = contract["template_contract"]
    signature = _preflight_signature(contract)
    prior_signature = st.session_state.get("research_agent_preflight_signature")
    if prior_signature != signature:
        st.session_state["research_agent_preflight_confirmed"] = False
        st.session_state["research_agent_preflight_signature"] = signature
    confirmed = bool(st.session_state.get("research_agent_preflight_confirmed"))

    request_ready = bool(contract["question"] or force_manuscript)
    cohort_ready = int(contract["cohort_rows"] or 0) > 0
    llm_ready = bool(contract.get("llm_ready", True))
    llm_issue_msg = _llm_readiness_message(str(contract.get("llm_issue") or ""), is_en=is_en)
    if contract["external_llm"] and not llm_ready:
        model_state = llm_issue_msg or ("missing settings" if is_en else "配置缺失")
    else:
        model_state = "external" if contract["external_llm"] else "offline"
    gate_state = "confirmed" if confirmed else "locked"
    ready_label = "ready" if is_en else "已就绪"
    missing_question_label = "missing" if is_en else "未填写"
    missing_cohort_label = "missing" if is_en else "未选择"
    state_steps = [
        (
            "01",
            "Question" if is_en else "问题",
            ready_label if request_ready else missing_question_label,
            "required input" if is_en else "必填输入",
            "ok" if request_ready else "warn",
        ),
        (
            "02",
            "Cohort" if is_en else "队列",
            ready_label if cohort_ready else missing_cohort_label,
            "rows + schema" if is_en else "行数和 schema",
            "ok" if cohort_ready else "warn",
        ),
        (
            "03",
            "Model" if is_en else "模型",
            model_state if is_en else (llm_issue_msg if contract["external_llm"] and not llm_ready else ("外部模型" if contract["external_llm"] else "离线")),
            "key/model required" if (is_en and contract["external_llm"] and not llm_ready) else ("密钥/模型必填" if contract["external_llm"] and not llm_ready else ("context disclosure" if is_en else "上下文披露")),
            "warn" if contract["external_llm"] and not llm_ready else ("warn" if contract["external_llm"] else "ok"),
        ),
        (
            "04",
            "Gate" if is_en else "关口",
            gate_state if is_en else ("已确认" if confirmed else "锁定"),
            "human confirmation" if is_en else "人工确认",
            "ok" if confirmed else "locked",
        ),
    ]
    state_html = "".join(
        f'<div class="ra-preflight-step {state_class}">'
        f'<span>{html.escape(index)}</span>'
        f'<b>{html.escape(label)}</b>'
        f'<em>{html.escape(state)}</em>'
        f'<p>{html.escape(detail)}</p>'
        '</div>'
        for index, label, state, detail, state_class in state_steps
    )
    cohort_value = (
        f"{contract['cohort_label'] or ('not selected' if is_en else '未选择')} · "
        f"{contract['cohort_rows']:,} rows"
    )
    llm_value = f"{contract['llm_choice']} · {contract['model']}"
    if not llm_ready and llm_issue_msg:
        llm_value = f"{llm_value} · {llm_issue_msg}"
    rows = [
        ("Request" if is_en else "请求", contract["question"] or ("not set" if is_en else "未设置")),
        ("Cohort" if is_en else "队列", cohort_value),
        ("LLM" if is_en else "模型", llm_value),
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
        "Manifest and run status will be written."
        if is_en else "将写入 manifest 与 run status。",
        "Numeric claims stay blocked until evidence checks pass."
        if is_en else "数值主张需通过证据校验才放行。",
        "Drafting stays second-stage unless requested."
        if is_en else "除非明确请求，写作仍在第二阶段。",
    ]
    if contract["external_llm"] and not llm_ready:
        checks.append(
            "Complete LLM settings above or choose MockLLMClient for an offline test run."
            if is_en else
            "请先补齐上方 LLM 设置，或选择 MockLLMClient 进行离线测试运行。"
        )
    else:
        checks.append(
            "External provider receives the question plus schema/summary context."
            if (is_en and contract["external_llm"]) else
            "外部模型会收到研究问题以及 schema/summary 上下文。"
            if contract["external_llm"] else
            "Offline mode keeps cohort context on this machine."
            if is_en else
            "离线模式不会把队列上下文发出本机。"
        )
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
              <div class="ra-preflight-kicker">{"Launch review" if is_en else "启动复核"}</div>
              <b>{"Review the current run contract" if is_en else "复核当前运行契约"}</b>
            </div>
            <span>{"confirmed" if confirmed and is_en else ("已确认" if confirmed else ("locked until confirmed" if is_en else "确认前锁定"))}</span>
          </div>
          <div class="ra-preflight-steps">{state_html}</div>
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

    ack = st.checkbox(
        (
            "I reviewed the current run contract, context disclosure, and file impact."
            if is_en else
            "我已复核当前运行契约、上下文披露和文件影响。"
        ),
        value=bool(st.session_state.get("research_agent_preflight_ack", False)),
        key="research_agent_preflight_ack",
    )
    can_confirm = request_ready and cohort_ready and llm_ready
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        if st.button(
            "Confirm launch review" if is_en else "确认启动复核",
            type="primary",
            disabled=not ack or not can_confirm,
            use_container_width=True,
            key="research_agent_preflight_confirm",
        ):
            st.session_state["research_agent_preflight_confirmed"] = True
            st.session_state["research_agent_preflight_signature"] = signature
            st.rerun()
    with c2:
        if st.button(
            "Reset review" if is_en else "重置复核",
            use_container_width=True,
            key="research_agent_preflight_reset",
        ):
            st.session_state["research_agent_preflight_confirmed"] = False
            st.rerun()
    external_consent_needed = bool(contract["external_llm"] and not st.session_state.get("llm_enabled", False))
    if confirmed and external_consent_needed:
        st.info(
            "Plan confirmed. Enable external LLM calls below to unlock Run."
            if is_en else
            "计划已确认。请在下方启用外部模型调用后再解锁运行。"
        )
    elif confirmed:
        st.success("Plan confirmed. The run button is enabled." if is_en else "计划已确认，可以启动运行。")
    elif not llm_ready:
        st.info(
            "Complete LLM settings above or choose MockLLMClient for an offline test run."
            if is_en else
            "请先补齐上方 LLM 设置，或选择 MockLLMClient 进行离线测试运行。"
        )
    else:
        st.warning("Review and confirm the launch gate before running the agent." if is_en else "运行 agent 前请先复核并确认启动关口。")
    return confirmed


def _render_setup_controls_intro(*, is_en: bool) -> None:
    st.markdown(
        textwrap.dedent(f"""
        <div class="ra-setup-controls-intro">
          <div>
            <div class="ra-setup-kicker">{"Run setup" if is_en else "运行配置"}</div>
            <h3>{"Complete the missing fields" if is_en else "补齐缺失字段"}</h3>
            <p>{"The first open panel is the required input. Method, data, model, and output details stay collapsed until needed." if is_en else "第一个展开面板是必填输入；方法、数据、模型和输出细节在需要时再展开。"}</p>
          </div>
          <span>{"local setup" if is_en else "本机配置"}</span>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def _render_preflight_controls_intro(*, is_en: bool) -> None:
    st.markdown(
        textwrap.dedent(f"""
        <div class="ra-preflight-controls-intro">
          <div>
            <div class="ra-setup-kicker">{"Launch gate" if is_en else "启动关口"}</div>
            <h3>{"Confirm inputs, files, and evidence gates" if is_en else "确认输入、文件和证据关口"}</h3>
            <p>{"One human check separates setup from execution. The run button stays locked until this contract matches the current question, cohort, model, and write path." if is_en else "一次人工复核把配置和执行分开；只有当前问题、队列、模型和写入路径与契约一致后，运行按钮才会放行。"}</p>
          </div>
          <span>{"human-controlled run" if is_en else "人工控制运行"}</span>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_research_agent_page(*, show_header: bool = True) -> None:
    """Top-level entry point used by the main webapp."""
    _apply_pending_research_agent_workdir(st.session_state)
    if show_header:
        render_page_header(
            _ra_text("header"),
            _ra_text("subheader"),
            icon="",
            kicker=_ra_text("kicker"),
        )
    if st.session_state.pop("_eu_ra_launch_requested", False):
        st.info(
            "Run controls are below. Confirm the research request, cohort, model provider, and preflight plan before launching."
            if st.session_state.get("language", "en") == "en" else
            "运行控制在下方。启动前请先确认研究请求、队列、模型提供方和执行前预览。"
        )

    try:
        handles = _import_agent_layer()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not import easyicu.research_agent: {exc}")
        st.code(traceback.format_exc())
        return

    _lang = st.session_state.get("language", "en")
    _is_en = _lang == "en"
    resume_run_id = st.session_state.get("research_agent_resume_run_id")
    resume_mode = str(st.session_state.get("research_agent_resume_mode") or "")
    if resume_run_id and resume_mode in {"continue", "force_manuscript"}:
        _restore_resume_cohort_handoff(st.session_state)
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
    overview_slot = st.container()

    with st.container(key="eu_ra_setup_controls"):
        _render_setup_controls_intro(is_en=_is_en)
        with st.expander(_step_titles[0], expanded=True):
            free_question, target_outcome = _section_request_picker()
            skill_key = None
        with st.expander(f"{_step_titles[1]} ({_optional_label})", expanded=False):
            method_notes, user_preferences = _section_method_preferences(free_question, target_outcome)
        question_hint = free_question
        focus_module_folder = bool(st.session_state.get("_eu_ra_focus_module_folder", False))
        focus_no_data = bool(st.session_state.get("_eu_ra_focus_no_data", False))
        with st.expander(_step_titles[2], expanded=bool(question_hint) or focus_module_folder or focus_no_data):
            cohort, cohort_label = _section_cohort_picker(research_question=question_hint)
        if cohort is not None and not _multi_db_label_is_distinct(cohort_label):
            tags = _multi_db_label_tags(cohort_label)
            duplicate_tags = sorted({tag for tag in tags if tags.count(tag) > 1})
            st.error(_ra_text(
                "multi_db_duplicate_tags",
                tags=', '.join(duplicate_tags) if duplicate_tags else ', '.join(tags),
            ))
            _clear_research_agent_preflight_confirmation()
            cohort = None
            cohort_label = ""
        with st.expander(_step_titles[3], expanded=False):
            llm_choice, api_key, model, base_url, extra_headers = _section_llm_picker(handles)
        focus_options = bool(st.session_state.pop("_eu_ra_focus_options", False))
        with st.expander(f"{_step_titles[4]} ({_optional_label})", expanded=focus_options):
            disable_icu_context, workdir_text, stop_after_analysis = _section_options()
    llm_ready, llm_issue = _llm_run_readiness(llm_choice, api_key, model)
    force_manuscript = bool(st.session_state.get("research_agent_force_manuscript"))
    resume_notes = str(st.session_state.get("research_agent_resume_notes") or "")
    resume_relax_probe = bool(st.session_state.get("research_agent_resume_relax_probe"))
    if force_manuscript:
        stop_after_analysis = False

    template_key = st.session_state.get("research_agent_template_current") or st.session_state.get("research_agent_example_key")
    preview_contract = _build_execution_preflight_contract(
        free_question=free_question or "",
        target_outcome=target_outcome or "",
        cohort=cohort,
        cohort_label=cohort_label or "",
        llm_choice=llm_choice or "",
        model=model or "",
        workdir_text=workdir_text or _default_research_agent_workdir(),
        stop_after_analysis=stop_after_analysis,
        force_manuscript=force_manuscript,
        template_key=str(template_key or ""),
        language=_lang,
        llm_ready=llm_ready,
        llm_issue=llm_issue,
    )
    preview_signature = _preflight_signature(preview_contract)
    if st.session_state.get("research_agent_preflight_signature") != preview_signature:
        st.session_state["research_agent_preflight_confirmed"] = False
        st.session_state["research_agent_preflight_signature"] = preview_signature

    with overview_slot:
        _render_research_agent_setup_overview(
            free_question=free_question,
            target_outcome=target_outcome,
            cohort=cohort,
            cohort_label=cohort_label,
            llm_choice=llm_choice,
            model=model,
            workdir_text=workdir_text,
            stop_after_analysis=stop_after_analysis,
            force_manuscript=force_manuscript,
            llm_ready=llm_ready,
            llm_issue=llm_issue,
        )

    # Surface a banner so the user knows the next "Run" click is a
    # resume, not a fresh run, and remembers the toggles they picked
    # in the history panel.
    if resume_run_id and resume_mode == "continue":
        banner_msg = (
            f"Resuming run `{resume_run_id}` from checkpoint. "
            "Completed steps will be reused; the planner replans the rest."
        ) if _is_en else (
            f"将从 checkpoint 继续运行 `{resume_run_id}`。已完成步骤会复用，"
            "planner 会重新规划剩余步骤。"
        )
        st.info(banner_msg)
        if resume_relax_probe:
            st.warning(_ra_text("resume_relaxed_active"))
        clear_cols = st.columns([1, 4])
        if clear_cols[0].button(
            "Cancel resume" if _is_en else "取消继续",
            key="research_agent_cancel_resume",
        ):
            st.session_state.pop("research_agent_resume_run_id", None)
            st.session_state.pop("research_agent_resume_run_dir", None)
            st.session_state.pop("research_agent_force_manuscript", None)
            st.session_state.pop("research_agent_resume_mode", None)
            st.session_state.pop("research_agent_resume_notes", None)
            st.session_state.pop("research_agent_resume_relax_probe", None)
            _clear_resume_cohort_handoff(st.session_state)
            st.rerun()

    with st.container(key="eu_ra_preflight_panel"):
        _render_preflight_controls_intro(is_en=_is_en)
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
            llm_ready=llm_ready,
            llm_issue=llm_issue,
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
                st.session_state["_llm_toggle_sync_pending"] = True
                st.session_state["_eu_ra_external_llm_enabled_notice"] = True
                st.rerun()
        if external_llm_selected and st.session_state.pop("_eu_ra_external_llm_enabled_notice", False):
            st.info(
                "External LLM calls are enabled for this session."
                if _is_en else
                "本会话已允许外部 LLM 调用。"
            )

        request_ready = bool(str(free_question or "").strip()) or force_manuscript
        consent_ready = not external_llm_selected or bool(st.session_state.get("llm_enabled", False))
        run_button_clicked = st.button(
            "▶  " + (_ra_text("draft_button") if force_manuscript else _ra_text("run_button")),
            type="primary",
            disabled=cohort is None or not request_ready or not preflight_confirmed or not llm_ready or not consent_ready,
            use_container_width=True,
        )
    run_clicked = run_button_clicked

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
    if not llm_ready:
        st.info(
            "Complete LLM settings above or choose MockLLMClient for an offline test run."
            if _is_en else
            "请先补齐上方 LLM 设置，或选择 MockLLMClient 进行离线测试运行。"
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

    st.session_state["_active_main_page"] = "research_agent"
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

    # Build the effective notes by appending the resume guidance the
    # user wrote in the history-panel editor, so the planner / coder see
    # it on the next iteration without overwriting any methods notes the
    # user already wrote.
    effective_notes = method_notes or ""
    if resume_notes and resume_mode == "continue":
        marker = (
            "\n\n[resume guidance — added during resume from "
            f"{resume_run_id}]\n"
        )
        effective_notes = (effective_notes + marker + resume_notes).strip()

    # The resume_run_id should be honoured in BOTH force_manuscript mode
    # (existing behaviour: skip analysis, rebuild manuscript) AND
    # continue mode (new behaviour: pick up the checkpoint and replan
    # the rest). The pipeline itself decides what to reuse via
    # ``per_step_records`` in the prior partial manifest.
    effective_resume = resume_run_id if (force_manuscript or resume_mode == "continue") else None
    effective_relax_probe = bool(resume_relax_probe) and resume_mode == "continue"

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
                notes=effective_notes,
                stop_after_analysis=stop_after_analysis,
                resume_run_id=effective_resume,
                audit_relax_probe=effective_relax_probe,
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
    if force_manuscript or resume_mode == "continue":
        st.session_state.pop("research_agent_resume_run_id", None)
        st.session_state.pop("research_agent_resume_run_dir", None)
        st.session_state.pop("research_agent_force_manuscript", None)
        st.session_state.pop("research_agent_resume_mode", None)
        st.session_state.pop("research_agent_resume_notes", None)
        st.session_state.pop("research_agent_resume_relax_probe", None)
        _clear_resume_cohort_handoff(st.session_state)

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
            st.session_state["_active_main_page"] = "research_agent"
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


__all__ = [
    "render_research_agent_demo_page",
    "render_research_agent_history_page",
    "render_research_agent_page",
]
