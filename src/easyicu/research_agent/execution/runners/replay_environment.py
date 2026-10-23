"""Provenance and environment policy for a nested primary-model replay.

Desktop replays inherit the run's persistent font cache and can execute only
host-rendered primary scripts. Sandboxed replays keep their per-replay cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def primary_replay_refusal(source: Mapping[str, Any]) -> str | None:
    """Return the typed refusal before writing or running a replay script."""
    if (
        os.environ.get("EASYICU_HOST_RENDERED_ONLY") == "1"
        and source.get("record", {}).get("code_origin") != "host_rendered_executor"
    ):
        return "host_runner_refuses_model_code"
    return None


def replay_environment(
    cohort_path: Path, replay_outputs: Path, replay_root: Path,
) -> dict[str, str]:
    """Bind replay inputs and outputs while preserving the desktop font cache."""
    env = os.environ.copy()
    env["COHORT_PARQUET"] = str(cohort_path)
    env["STEP_OUT_DIR"] = str(replay_outputs)
    if env.get("EASYICU_HOST_RENDERED_ONLY") != "1":
        env["MPLCONFIGDIR"] = str(replay_root / "matplotlib")
    return env
