"""Streamlit compatibility helpers for the EasyICU web app."""

from __future__ import annotations

import inspect
from typing import Any


def _normalize_width_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate deprecated Streamlit width flags to the newer width API."""
    if "use_container_width" in kwargs and kwargs.get("use_container_width") is not None and "width" not in kwargs:
        use_container_width = kwargs.pop("use_container_width")
        kwargs["width"] = "stretch" if use_container_width else "content"
    else:
        kwargs.pop("use_container_width", None)
    return kwargs


def _legacy_width_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fallback for Streamlit builds that do not accept width='stretch'."""
    width = kwargs.pop("width", None)
    if width == "stretch":
        kwargs["use_container_width"] = True
    elif width == "content":
        kwargs["use_container_width"] = False
    return kwargs


def _needs_legacy_width_api(fn: Any) -> bool:
    """Return True for Streamlit builds where width only accepts pixel ints."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "use_container_width" not in params:
        return False
    width_param = params.get("width")
    if width_param is None:
        return True
    annotation = str(width_param.annotation).lower()
    return "int" in annotation and "stretch" not in annotation and "content" not in annotation


def _coerce_dataframe_for_arrow(data):
    """Avoid noisy Streamlit/Arrow tracebacks for mixed object preview columns."""
    try:
        import pandas as pd
    except Exception:
        return data
    if not isinstance(data, pd.DataFrame) or data.empty:
        return data

    object_cols = [col for col in data.columns if data[col].dtype == object]
    if not object_cols:
        return data

    converted = None
    for col in object_cols:
        non_null = data[col].dropna()
        if non_null.empty:
            continue
        type_names = {type(value).__name__ for value in non_null.head(100)}
        if len(type_names) > 1:
            if converted is None:
                converted = data.copy()
            converted[col] = converted[col].astype(str)
    return converted if converted is not None else data


def _dataframe_compat(st_obj: Any, data, **kwargs):
    """Render a dataframe across Streamlit width API variants."""
    dataframe_fn = getattr(st_obj, "_easyicu_original_dataframe", st_obj.dataframe)
    kwargs = _normalize_width_kwargs(dict(kwargs))
    data = _coerce_dataframe_for_arrow(data)
    if kwargs.get("width") in {"stretch", "content"} and _needs_legacy_width_api(dataframe_fn):
        return dataframe_fn(data, **_legacy_width_kwargs(dict(kwargs)))
    try:
        return dataframe_fn(data, **kwargs)
    except TypeError:
        if kwargs.get("width") != "stretch":
            raise
        return dataframe_fn(data, **_legacy_width_kwargs(dict(kwargs)))


def _button_compat(st_obj: Any, label, *args, **kwargs):
    """Render a button across Streamlit width API variants."""
    button_fn = getattr(st_obj, "_easyicu_original_button", st_obj.button)
    kwargs = _normalize_width_kwargs(dict(kwargs))
    if kwargs.get("width") in {"stretch", "content"} and _needs_legacy_width_api(button_fn):
        return button_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))
    try:
        return button_fn(label, *args, **kwargs)
    except TypeError:
        return button_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))


def _download_button_compat(st_obj: Any, label, data, *args, **kwargs):
    """Render a download button across Streamlit width API variants."""
    download_button_fn = getattr(
        st_obj,
        "_easyicu_original_download_button",
        st_obj.download_button,
    )
    kwargs = _normalize_width_kwargs(dict(kwargs))
    if kwargs.get("width") in {"stretch", "content"} and _needs_legacy_width_api(download_button_fn):
        return download_button_fn(label, data, *args, **_legacy_width_kwargs(dict(kwargs)))
    try:
        return download_button_fn(label, data, *args, **kwargs)
    except TypeError:
        return download_button_fn(
            label,
            data,
            *args,
            **_legacy_width_kwargs(dict(kwargs)),
        )


def _form_submit_button_compat(st_obj: Any, label="Submit", *args, **kwargs):
    """Render a form submit button across Streamlit width API variants."""
    submit_fn = getattr(
        st_obj,
        "_easyicu_original_form_submit_button",
        st_obj.form_submit_button,
    )
    kwargs = _normalize_width_kwargs(dict(kwargs))
    if kwargs.get("width") in {"stretch", "content"} and _needs_legacy_width_api(submit_fn):
        return submit_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))
    try:
        return submit_fn(label, *args, **kwargs)
    except TypeError:
        return submit_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))


def _plotly_chart_compat(st_obj: Any, figure_or_data, *args, **kwargs):
    """Keep Plotly-specific width kwargs untouched."""
    plotly_chart_fn = getattr(
        st_obj,
        "_easyicu_original_plotly_chart",
        st_obj.plotly_chart,
    )
    return plotly_chart_fn(figure_or_data, *args, **kwargs)


def apply_streamlit_compat(st: Any) -> None:
    """Apply idempotent Streamlit API shims used by legacy EasyICU pages."""

    def dataframe_compat(data, **kwargs):
        return _dataframe_compat(st, data, **kwargs)

    def button_compat(label, *args, **kwargs):
        return _button_compat(st, label, *args, **kwargs)

    def download_button_compat(label, data, *args, **kwargs):
        return _download_button_compat(st, label, data, *args, **kwargs)

    def form_submit_button_compat(label="Submit", *args, **kwargs):
        return _form_submit_button_compat(st, label, *args, **kwargs)

    def plotly_chart_compat(figure_or_data, *args, **kwargs):
        return _plotly_chart_compat(st, figure_or_data, *args, **kwargs)

    if not hasattr(st, "_easyicu_original_dataframe"):
        st._easyicu_original_dataframe = st.dataframe
        st.dataframe = dataframe_compat
    if not hasattr(st, "_easyicu_original_button"):
        st._easyicu_original_button = st.button
        st.button = button_compat
    if not hasattr(st, "_easyicu_original_download_button"):
        st._easyicu_original_download_button = st.download_button
        st.download_button = download_button_compat
    if not hasattr(st, "_easyicu_original_form_submit_button"):
        st._easyicu_original_form_submit_button = st.form_submit_button
        st.form_submit_button = form_submit_button_compat
    if not hasattr(st, "_easyicu_original_plotly_chart"):
        st._easyicu_original_plotly_chart = st.plotly_chart
        st.plotly_chart = plotly_chart_compat


def query_param_exists(st: Any, key: str) -> bool:
    """Return whether a query parameter is present across Streamlit versions."""
    try:
        params = getattr(st, "query_params", {})
        return key in params
    except Exception:
        return False


def query_param_value(st: Any, key: str, default: str = "") -> str:
    """Read a Streamlit query parameter without depending on a specific API version."""
    try:
        params = getattr(st, "query_params", {})
        value = params.get(key, default)
    except Exception:
        value = default
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value).strip()


def query_flag_enabled(st: Any, key: str) -> bool:
    """Read a truthy/present Streamlit query flag across API versions."""
    if not query_param_exists(st, key):
        return False
    value = query_param_value(st, key)
    return value.lower() not in {"0", "false", "no", "off", "none"}
