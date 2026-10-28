"""Run the host Cox owner in a child with an isolated output environment."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ...reference_io import LoadedCohort
from ..spec import SurvivalReferenceSpec

_WORKER = """
import json
import sys
from easyicu.research_agent.execution.runners.survival_primary_executor import run_survival_primary
run_survival_primary(**json.load(sys.stdin))
"""


def run_analysis(loaded: LoadedCohort, spec: SurvivalReferenceSpec, out_dir: str | Path) -> dict:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Fit exactly the loaded rows even when the source was a DataFrame or CSV.
    # The materialized local input is what the host receipt binds.
    input_path = out / "prepared_cohort.parquet"
    loaded.frame.to_parquet(input_path, index=False)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    primary = spec.primary
    payload = {
        "input_path": str(input_path), "input_product": primary.input_product,
        "input_evidence_id": f"sha256:{digest}", "input_sha256": digest,
        "result_product": primary.expected_result_product,
        "exposure": primary.exposure_source, "outcome": primary.outcome,
        "effect_scale": primary.effect_scale, "population": primary.population,
        "time_origin": primary.time_origin, "time_column": primary.time_column, "time_unit": primary.time_unit,
        "event_column": primary.event_column, "event_value": primary.event_value,
        "event_definition": primary.event_definition, "censoring_strategy": primary.censoring_strategy,
        "competing_risk_strategy": primary.competing_risk_strategy,
        "competing_event_value": primary.competing_event_value,
        "competing_event_definition": primary.competing_event_definition,
        "time_horizon": primary.time_horizon, "time_horizon_value": primary.time_horizon_value,
        "estimator": primary.estimator, "effect_measure": primary.effect_measure,
        "covariates": list(primary.covariates or []),
        "model_terms": [term.model_dump(mode="json") for term in primary.model_terms or []],
        "ph_diagnostic": primary.proportional_hazards_diagnostic,
        "ph_alpha": primary.proportional_hazards_alpha, "ph_policy": primary.proportional_hazards_policy,
    }
    env = dict(os.environ, STEP_OUT_DIR=str(out), MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER], input=json.dumps(payload, allow_nan=False),
        env=env, capture_output=True, text=True, timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(f"Host survival executor failed:\n{completed.stderr}")
    return json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
