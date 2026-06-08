"""Copilot ↔ classic-engine wiring (the single execution收口).

This module exists to kill the divergence described in
``easyicu美化/copilot_接线施工计划.md``: the Research Copilot and the classic
Data Extraction views must run the **same** execution functions so that a
cohort filtered in chat is byte-for-byte identical to the same cohort filtered
in the classic view.

Two responsibilities:

1. **Depth axis** — how far a study runs (``extract`` / ``review`` / ``full``),
   orthogonal to *what* it studies (the branch / template). Pure functions,
   trivially unit-testable.
2. **Step dispatcher** — ``run_copilot_step`` routes a study step to the
   classic engine function for that step. The classic functions are injected
   via :class:`CopilotEngines` (default = lazy import) so the routing logic can
   be unit-tested with fakes and *without* importing Streamlit.

Iron rule (from the对接图): parsing / wording lives in ``llm_chat``; every
write that changes execution ``state`` goes through the classic functions
reached here. ``copilot_engine`` itself must stay import-light — **no
top-level ``streamlit`` import** — so these helpers remain testable.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Callable

# Mirror of ``COPILOT_STUDY_STEPS`` in ``llm_chat`` (kept here to avoid importing
# the 12k-line module — which pulls Streamlit — into this light module).
# ``llm_chat`` asserts the two stay in sync at import time.
COPILOT_STEP_SEQUENCE: tuple[str, ...] = (
    "question",
    "data",
    "cohort",
    "concepts",
    "extract",
    "review",
    "analysis",
    "draft",
)
_STEP_INDEX: dict[str, int] = {step: idx for idx, step in enumerate(COPILOT_STEP_SEQUENCE)}


# ---------------------------------------------------------------------------
# Depth axis
# ---------------------------------------------------------------------------

DEPTH_ORDER: tuple[str, ...] = ("extract", "review", "full")
DEFAULT_DEPTH: str = "full"

# depth -> the step where the study stops (its "finish line").
_DEPTH_GOAL: dict[str, str] = {
    "extract": "extract",  # just extract + filter a cohort
    "review": "review",    # extract + visual review
    "full": "draft",       # full agent workflow to the draft gate
}


def normalize_depth(depth: object) -> str:
    """Coerce any value to a known depth, defaulting to ``full``."""
    text = str(depth or "").strip().lower()
    return text if text in _DEPTH_GOAL else DEFAULT_DEPTH


def copilot_goal_step(depth: object) -> str:
    """Return the step id where ``depth`` finishes."""
    return _DEPTH_GOAL[normalize_depth(depth)]


def step_index(step: str) -> int:
    return _STEP_INDEX.get(str(step), 0)


def copilot_goal_index(depth: object) -> int:
    return step_index(copilot_goal_step(depth))


def copilot_bump_depth(depth: object) -> str:
    """"Take it further": move one rung up the depth ladder (capped at full)."""
    current = normalize_depth(depth)
    idx = DEPTH_ORDER.index(current)
    return DEPTH_ORDER[min(idx + 1, len(DEPTH_ORDER) - 1)]


def is_step_beyond_goal(depth: object, step: str) -> bool:
    """True when ``step`` is past the depth goal (rendered greyed/``beyond``)."""
    return step_index(step) > copilot_goal_index(depth)


def clamp_step_to_goal(depth: object, step: str) -> str:
    """Never let the active step run past the depth goal."""
    return COPILOT_STEP_SEQUENCE[min(step_index(step), copilot_goal_index(depth))]


def next_step_capped(depth: object, step: str) -> str:
    """Advance one step, but stop at the depth goal (gated auto-advance)."""
    nxt_idx = min(step_index(step) + 1, len(COPILOT_STEP_SEQUENCE) - 1)
    nxt = COPILOT_STEP_SEQUENCE[nxt_idx]
    if is_step_beyond_goal(depth, nxt):
        return copilot_goal_step(depth)
    return nxt


def at_goal(depth: object, step: str) -> bool:
    return step_index(step) >= copilot_goal_index(depth)


# ---------------------------------------------------------------------------
# Engine injection
# ---------------------------------------------------------------------------

@dataclass
class CopilotEngines:
    """The classic execution functions the Copilot must reuse.

    Defaults are wired by :func:`default_engines` via lazy import. Tests pass
    fakes to exercise routing without touching real data or Streamlit.
    """

    apply_cohort_filter: Callable[..., Any]
    post_filter_cohort_data: Callable[..., Any]
    positive_patient_ids: Callable[..., Any]
    check_data_status: Callable[..., Any]
    load_data: Callable[..., Any]
    load_data_for_preview: Callable[..., Any]
    execute_export: Callable[..., Any]
    load_from_exported: Callable[..., Any]
    cohort_feature_counts: Callable[..., Any]


def default_engines() -> CopilotEngines:
    """Bind the real classic functions (lazy import keeps首屏 fast)."""
    from easyicu.webapp import (  # local import: pulls Streamlit only at run time
        cohort_filters,
        data_workflows,
        export_workflow,
        services,
    )

    return CopilotEngines(
        apply_cohort_filter=data_workflows.apply_cohort_filter,
        post_filter_cohort_data=cohort_filters._post_filter_cohort_data,
        positive_patient_ids=cohort_filters._get_positive_patient_ids_from_data,
        check_data_status=data_workflows.check_data_status,
        load_data=data_workflows.load_data,
        load_data_for_preview=data_workflows.load_data_for_preview,
        execute_export=export_workflow.execute_sidebar_export,
        load_from_exported=data_workflows.load_from_exported,
        cohort_feature_counts=services.cohort_feature_counts,
    )


# ---------------------------------------------------------------------------
# Step handlers — each routes to a classic function, no re-implementation.
# ---------------------------------------------------------------------------

def _real_data_target(state: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return (data_path, database) only when a real local source is configured."""
    data_path = state.get("data_path") or state.get("prepared_data_path")
    database = state.get("database")
    if not data_path or not database or database == "mock":
        return None, None
    if state.get("use_mock_data"):
        return None, None
    return str(data_path), str(database)


def _step_data(study, state, app_context, engines, **_kwargs) -> dict[str, Any]:
    data_path, database = _real_data_target(state)
    if not data_path:
        return {"step": "data", "status": "no_real_data"}
    status = engines.check_data_status(data_path, database, app_context)
    state["_data_status"] = status
    return {"step": "data", "status": "ok", "data_status": status}


def _step_cohort(study, state, app_context, engines, **_kwargs) -> dict[str, Any]:
    """Run the SAME cohort filter the classic view runs.

    Pre-condition: ``state['cohort_filter']`` / ``state['cohort_enabled']`` were
    already written (by ``_copilot_submit_cohort_filter`` — same canonical
    schema as classic Step 2). The previously-missing piece was actually
    *executing* the filter; this does it through the classic function.
    """
    data_path, database = _real_data_target(state)
    if not data_path:
        return {"step": "cohort", "status": "no_real_data"}
    result = engines.apply_cohort_filter(data_path, database, app_context=app_context)
    if result is None:
        # No active filter -> mirror classic "no filter" behaviour exactly.
        state["_cohort_stats"] = None
        state["_cohort_filtered_ids"] = None
        return {"step": "cohort", "status": "no_active_filter"}
    state["_cohort_stats"] = result
    state["_cohort_filtered_ids"] = result.get("filtered_ids")
    state["filtered_patient_count"] = result.get("total_after")
    return {
        "step": "cohort",
        "status": "ok",
        "total_before": result.get("total_before"),
        "total_after": result.get("total_after"),
    }


def _step_concepts(study, state, app_context, engines, **_kwargs) -> dict[str, Any]:
    counts = engines.cohort_feature_counts(state)
    state["_copilot_feature_counts"] = counts
    return {"step": "concepts", "status": "ok", "counts": counts}


def _step_extract(study, state, app_context, engines, *, preview: bool = True,
                  max_patients: int | None = None, **_kwargs) -> dict[str, Any]:
    data_path, _database = _real_data_target(state)
    if not data_path:
        return {"step": "extract", "status": "no_real_data"}
    if preview:
        n = int(max_patients or study.get("patient_n") or state.get("demo_mode_patients") or 10)
        data = engines.load_data_for_preview(n, app_context)
    else:
        data = engines.load_data(app_context)
    state["loaded_concepts"] = data
    state["_extraction_done"] = True
    return {"step": "extract", "status": "ok", "preview": preview}


def _step_review(study, state, app_context, engines, **_kwargs) -> dict[str, Any]:
    # Actual UI embed (render_quick_visualization_page) happens during page
    # render in llm_chat; here we only mark the workspace ready.
    state["_review_workspace_ready"] = True
    return {"step": "review", "status": "ready"}


def _step_export(study, state, app_context, engines, **_kwargs) -> dict[str, Any]:
    engines.execute_export(app_context)
    return {"step": "export", "status": "ok"}


# question / analysis / draft stay on their existing paths (LLM framing and the
# research_agent evidence gate), so they intentionally have no engine handler.
_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "data": _step_data,
    "cohort": _step_cohort,
    "concepts": _step_concepts,
    "extract": _step_extract,
    "review": _step_review,
    "export": _step_export,
}


def run_copilot_step(
    step_id: str,
    study: MutableMapping[str, Any],
    state: MutableMapping[str, Any],
    *,
    app_context: Mapping[str, Any] | None = None,
    engines: CopilotEngines | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Route one study step to its classic execution function.

    Returns a small result dict (always includes ``step`` and ``status``).
    Steps with no engine handler return ``status='noop'``.
    """
    handler = _HANDLERS.get(str(step_id))
    if handler is None:
        return {"step": step_id, "status": "noop"}
    engines = engines or default_engines()
    return handler(study, state, app_context, engines, **kwargs)


def run_study_up_to_goal(
    study: MutableMapping[str, Any],
    state: MutableMapping[str, Any],
    *,
    app_context: Mapping[str, Any] | None = None,
    engines: CopilotEngines | None = None,
    only_steps: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute each engine-backed step from the start up to the depth goal.

    Honours the depth axis: nothing past ``copilot_goal_step(study['depth'])``
    is run. ``only_steps`` optionally restricts which steps execute.
    """
    engines = engines or default_engines()
    goal_idx = copilot_goal_index(study.get("depth"))
    results: list[dict[str, Any]] = []
    for step in COPILOT_STEP_SEQUENCE[: goal_idx + 1]:
        if step not in _HANDLERS:
            continue
        if only_steps is not None and step not in only_steps:
            continue
        results.append(
            run_copilot_step(step, study, state, app_context=app_context, engines=engines)
        )
    return results
