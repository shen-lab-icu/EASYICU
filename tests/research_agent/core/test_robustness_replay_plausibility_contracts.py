"""A robustness replay hands the primary script its own sealed contracts.

A standard executor's primary script carries a plausibility receipt: before it
fits anything, it checks the raw-input contracts in its resolved-input manifest
against the digest its step was sealed with.  A complete-case robustness
variant replays that exact script on a smaller cohort, so the replay manifest
must carry the contracts of the primary step -- read from the manifest the
step record registered -- and never the contracts of the robustness step that
happens to be running the replay.  Each step's contracts have their own digest,
so borrowing them made every such replay fail its receipt, and the variant
emitted no estimate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.authority.plausibility_receipt_code import (
    render_standard_plausibility_receipt_code,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _find_structured_primary_model_source,
    _replay_primary_model_for_complete_case,
)
from easyicu.research_agent.robustness.panel import RobustnessSpec

from tests.support.robustness_sources import write_structured_source_authority


def _contracts(maximum: float) -> dict:
    payload = {
        "contracts": {
            "age": {
                "analysis_plausibility_range": {"minimum": 0.0, "maximum": maximum},
                "plausibility_policy": {
                    "range_policy": "flag_only",
                    "out_of_range_action": "retain_and_flag",
                },
            }
        }
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "contracts_sha256": digest}


def _register_manifest(run_dir: Path, record: dict, contracts: dict) -> Path:
    manifest = run_dir / "resolved_inputs" / f"{record['step_id']}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"step_id": record["step_id"], "raw_input_contracts": contracts}),
        encoding="utf-8",
    )
    record["resolved_inputs_path"] = str(manifest.relative_to(run_dir))
    record["resolved_inputs_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return manifest


def test_the_source_carries_the_primary_steps_registered_contracts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    record, evidence, _script = write_structured_source_authority(run_dir)
    primary_contracts = _contracts(120.0)
    _register_manifest(run_dir, record, primary_contracts)

    source = _find_structured_primary_model_source(
        records=[record], run_dir=run_dir, evidence_records=evidence
    )

    assert source is not None
    assert source["raw_input_contracts"] == primary_contracts


def test_an_unverified_manifest_supplies_no_contracts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    record, evidence, _script = write_structured_source_authority(run_dir)
    manifest = _register_manifest(run_dir, record, _contracts(120.0))
    manifest.write_text(
        json.dumps({"raw_input_contracts": _contracts(999.0)}), encoding="utf-8"
    )

    source = _find_structured_primary_model_source(
        records=[record], run_dir=run_dir, evidence_records=evidence
    )

    # The source is still found; only its contracts are withheld, so a replay
    # that needs them fails closed inside the script.
    assert source is not None
    assert source["raw_input_contracts"] is None

    del record["resolved_inputs_path"]
    unregistered = _find_structured_primary_model_source(
        records=[record], run_dir=run_dir, evidence_records=evidence
    )
    assert unregistered is not None
    assert unregistered["raw_input_contracts"] is None


def _receipt_bound_primary(tmp_path: Path, contracts: dict) -> Path:
    scope = FlagOnlyPlausibilityScope(
        step_id="primary",
        expected_columns=("age",),
        source_contracts_sha256=contracts["contracts_sha256"],
        authority_kind="resolved_raw_input_contracts",
    )
    script_path = tmp_path / "primary" / "analysis.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "\n\n".join(
            [
                "from easyicu.research_agent.execution.runners.typed_input_binding "
                "import load_step_cohort_frame\n"
                "frame, _cohort_path = load_step_cohort_frame(\n"
                "    typed_cohort_input='artifact:analysis_cohort'\n"
                ")",
                render_standard_plausibility_receipt_code(scope, frame_name="frame"),
                """
out = Path(os.environ["STEP_OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{
    "model_id": "primary",
    "term": "exposure",
    "term_role": "exposure",
    "source_variable": "exposure",
    "odds_ratio": 1.5,
    "ci_low": 1.1,
    "ci_high": 2.0,
    "std_error": 0.1,
}]).to_csv(out / "coefficients.csv", index=False)
summary = {
    "primary_model_id": "primary",
    "coefficient_table": "coefficients.csv",
    "plausibility_audit": plausibility_audit,
    "model_contracts": [{
        "model_id": "primary",
        "analysis_role": "primary",
        "analysis_set": "source_aware",
        "exposure_role": "primary",
        "exposure_source": "exposure",
        "exposure_expression": "exposure",
        "n": len(frame),
        "event_n": int(frame["outcome"].sum()),
        "fit_status": "fitted",
        "converged": True,
        "fit_method": "registered_test_model",
    }],
}
(out / "step_summary.json").write_text(json.dumps(summary))
""".strip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return script_path


def _source(script_path: Path, contracts: dict | None) -> dict:
    return {
        "step_id": "primary",
        "script_path": script_path,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "summary": {"input_bindings": [{"input_key": "artifact:analysis_cohort"}]},
        "primary_contract": {
            "exposure_source": "exposure",
            "exposure_expression": "exposure",
        },
        "raw_input_contracts": contracts,
    }


_SPEC = RobustnessSpec(
    spec_id="complete_case",
    axis="missing",
    description="Replay the locked complete-case membership.",
    missing_override={
        "strategy": "complete_case",
        "variables": ["exposure", "outcome", "age"],
    },
)


def _primary_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 1, 0, 1],
            "age": [50.0, None, 70.0, 80.0],
        }
    )


def _run_as_robustness_step(tmp_path: Path, monkeypatch, contracts: dict) -> None:
    # The replay runs inside the robustness step, whose own manifest carries
    # the robustness step's contracts.
    current = tmp_path / "robustness_inputs.json"
    current.write_text(
        json.dumps({"step_id": "robustness_replay", "raw_input_contracts": contracts}),
        encoding="utf-8",
    )
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(current))


def test_complete_case_replay_passes_the_primary_receipt_with_its_own_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    primary_contracts = _contracts(120.0)
    _run_as_robustness_step(tmp_path, monkeypatch, _contracts(110.0))
    script_path = _receipt_bound_primary(tmp_path, primary_contracts)

    replay = _replay_primary_model_for_complete_case(
        spec=_SPEC,
        source=_source(script_path, primary_contracts),
        primary_data=_primary_data(),
        out_dir=tmp_path / "robustness",
    )

    assert replay["error"] is None
    assert replay["row"].n == 3
    assert replay["index"]["input_n"] == replay["index"]["modeled_n"] == 3


@pytest.mark.parametrize("current_matches_primary", [False, True])
def test_a_replay_never_borrows_the_running_steps_contracts(
    tmp_path: Path, monkeypatch, current_matches_primary: bool
) -> None:
    primary_contracts = _contracts(120.0)
    _run_as_robustness_step(
        tmp_path,
        monkeypatch,
        primary_contracts if current_matches_primary else _contracts(110.0),
    )
    script_path = _receipt_bound_primary(tmp_path, primary_contracts)

    replay = _replay_primary_model_for_complete_case(
        spec=_SPEC,
        source=_source(script_path, None),
        primary_data=_primary_data(),
        out_dir=tmp_path / "robustness",
    )

    # Without the primary step's verified contracts the receipt cannot pass,
    # even when the running step's contracts happen to carry the same digest.
    assert replay["error"] is not None
    assert "exit code" in replay["error"]
