"""Streamlit compatibility helpers for the EasyICU web app."""

from __future__ import annotations

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


def apply_streamlit_compat(st: Any) -> None:
    """Apply idempotent Streamlit API shims used by legacy EasyICU pages."""

    def dataframe_compat(data, **kwargs):
        dataframe_fn = getattr(st, "_easyicu_original_dataframe", st.dataframe)
        kwargs = _normalize_width_kwargs(dict(kwargs))
        try:
            return dataframe_fn(data, **kwargs)
        except TypeError:
            if kwargs.get("width") != "stretch":
                raise
            return dataframe_fn(data, **_legacy_width_kwargs(dict(kwargs)))

    def button_compat(label, *args, **kwargs):
        button_fn = getattr(st, "_easyicu_original_button", st.button)
        kwargs = _normalize_width_kwargs(dict(kwargs))
        try:
            return button_fn(label, *args, **kwargs)
        except TypeError:
            return button_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))

    def download_button_compat(label, data, *args, **kwargs):
        download_button_fn = getattr(st, "_easyicu_original_download_button", st.download_button)
        kwargs = _normalize_width_kwargs(dict(kwargs))
        try:
            return download_button_fn(label, data, *args, **kwargs)
        except TypeError:
            return download_button_fn(label, data, *args, **_legacy_width_kwargs(dict(kwargs)))

    def form_submit_button_compat(label="Submit", *args, **kwargs):
        submit_fn = getattr(st, "_easyicu_original_form_submit_button", st.form_submit_button)
        kwargs = _normalize_width_kwargs(dict(kwargs))
        try:
            return submit_fn(label, *args, **kwargs)
        except TypeError:
            return submit_fn(label, *args, **_legacy_width_kwargs(dict(kwargs)))

    def plotly_chart_compat(figure_or_data, *args, **kwargs):
        plotly_chart_fn = getattr(st, "_easyicu_original_plotly_chart", st.plotly_chart)
        return plotly_chart_fn(figure_or_data, *args, **kwargs)

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
