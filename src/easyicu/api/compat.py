"""Fail-closed compatibility endpoints retained for EasyICU 1.x."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd


def align_to_icu_admission(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    aggregate_hourly: bool = True,
    agg_func: str = "median",
    filter_icu_window: bool = True,
    before_icu_hours: int = 0,
    after_icu_hours: int = 0,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Reject a historical stub that never performed time alignment."""
    del (
        data,
        database,
        data_path,
        aggregate_hourly,
        agg_func,
        filter_icu_window,
        before_icu_hours,
        after_icu_hours,
        verbose,
    )
    raise NotImplementedError(
        "align_to_icu_admission() is not implemented and no longer returns "
        "unaligned data as if alignment succeeded. Use load_concepts() for "
        "canonical relative-time output."
    )


__all__ = ["align_to_icu_admission"]
