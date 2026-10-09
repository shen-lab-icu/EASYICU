"""What a robustness step carries of the primary model's contracts.

Owner
-----
A robustness replay refits the primary model and reports beside it, so it
must carry that model's own contracts rather than re-derive them:

- the raw-input contracts the primary step was sealed against, which a
  replay of its script is handed (``registered_raw_input_contracts``);
- a replay-local typed cohort binding with those contracts
  (``variant_typed_manifest_path``);
- copies of the primary's coefficient and contract artifacts in this step's
  outputs (``copy_structured_primary_contract_artifacts``);
- which fitted-model contract supplied each scalar row
  (``matrix_model_trace``).

``execution.runners.deterministic_robustness`` runs the replay and calls
these.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...authority.step_capsule import (
    StepAuthorityCapsuleRef,
    load_verified_step_authority_capsule,
    read_verified_content,
)
from ...contracts.cohort_product_keys import is_closed_cohort_product_key
from ...robustness.panel import PRIMARY_SPEC_ID, RobustnessSpec

#: The name this step gives the primary coefficients it copies into its own
#: outputs.  One spelling, used by the copy and by the matrix row that points
#: at it, because those two disagreeing is what made the row unreadable.
PRIMARY_COEFFICIENT_COPY_NAME = "coefficients.csv"


def registered_raw_input_contracts(
    *,
    record: Dict[str, Any],
    run_root: Path,
) -> Optional[Dict[str, Any]]:
    """Return the raw-input contracts the primary step was sealed against.

    A standard executor's script checks its plausibility receipt against the
    digest of the contracts in its own resolved-input manifest.  A replay of
    that script must be handed those contracts and never another step's.

    They are read from the manifest sealed in the step's own executed
    authority capsule: content-addressed, bound to the executed code and to
    the digest the record keeps.  A resumed run drops the record's mutable
    manifest path, and a later attempt may overwrite that file, so neither
    is consulted.  ``None`` when the contracts cannot be verified; a replay
    that needs them then fails closed inside the script.
    """

    step_id = str(record.get("step_id") or "").strip()
    raw_ref = record.get("step_authority_capsule_ref")
    expected_sha256 = str(record.get("resolved_inputs_sha256") or "").strip()
    if not step_id or not isinstance(raw_ref, Mapping) or not expected_sha256:
        return None
    try:
        capsule = load_verified_step_authority_capsule(
            run_root,
            ref=StepAuthorityCapsuleRef.model_validate(dict(raw_ref)),
            expected_step_id=step_id,
        ).capsule
        execution = capsule.execution
        if (
            execution is None
            or execution.returncode != 0
            or capsule.candidate_code.sha256
            != str(record.get("executed_code_sha256") or "").strip()
            or capsule.resolved_inputs.sha256 != expected_sha256
        ):
            return None
        manifest = json.loads(
            read_verified_content(run_root, capsule.resolved_inputs).decode("utf-8")
        )
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None
    contracts = manifest.get("raw_input_contracts")
    return dict(contracts) if isinstance(contracts, dict) else None


def variant_typed_manifest_path(
    *,
    source: Dict[str, Any],
    spec: RobustnessSpec,
    variant_cohort: Any,
    cohort_path: Path,
    cohort_sha256: str,
    replay_root: Path,
) -> Optional[Path]:
    """Issue a replay-local typed cohort binding when the source used one."""

    source_bindings = (source.get("summary") or {}).get("input_bindings") or []
    cohort_keys = [
        str(item.get("input_key") or "")
        for item in source_bindings
        if isinstance(item, Mapping)
        and is_closed_cohort_product_key(item.get("input_key"))
    ]
    if not cohort_keys:
        return None
    if len(cohort_keys) != 1:
        raise ValueError("registered primary model has no unique typed cohort binding")
    cohort_key = cohort_keys[0]
    declared_kind, _, product = cohort_key.partition(":")
    replay_evidence_id = f"robustness_replay_{spec.spec_id}_cohort"
    identity_row = {
        "input_key": cohort_key,
        "declared_kind": declared_kind,
        "product": product,
        "evidence_id": replay_evidence_id,
        "sha256": cohort_sha256,
        "produced_by_step": spec.spec_id,
    }
    binding = {
        "absolute_path": str(cohort_path),
        "relative_path": cohort_path.name,
        "sha256": cohort_sha256,
        "declared_kind": declared_kind,
        "evidence_kind": "table",
        "product": product,
        "evidence_id": replay_evidence_id,
        "produced_by_step": spec.spec_id,
        "identity_row": identity_row,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "parquet",
            "columns": [str(column) for column in variant_cohort.columns],
            "row_count": int(len(variant_cohort)),
            "identity_row": identity_row,
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": cohort_key,
            "mode": "all_rows",
            "artifact_sha256": cohort_sha256,
            "verified_row_count": int(len(variant_cohort)),
        },
    }
    replay_manifest = {
        "step_id": source["step_id"],
        "inputs": {cohort_key: binding},
    }
    # The replayed script verifies its plausibility receipt against the
    # contracts its own step was sealed with.  The robustness step's contracts
    # (the current environment's manifest) belong to another step, so they
    # are never substituted.
    raw_input_contracts = source.get("raw_input_contracts")
    if isinstance(raw_input_contracts, dict):
        replay_manifest["raw_input_contracts"] = raw_input_contracts
    replay_manifest_path = replay_root / "resolved_inputs.json"
    replay_manifest_path.write_text(
        json.dumps(replay_manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return replay_manifest_path


def copy_structured_primary_contract_artifacts(
    *,
    source: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, str]:
    import pandas as pd  # type: ignore

    copied: Dict[str, str] = {}
    coefficient_path = source.get("coefficient_path")
    if isinstance(coefficient_path, Path) and coefficient_path.is_file():
        shutil.copy2(coefficient_path, out_dir / PRIMARY_COEFFICIENT_COPY_NAME)
        copied["coefficients"] = PRIMARY_COEFFICIENT_COPY_NAME

    # Never copy an unregistered sibling model_summaries.csv.  Re-materialize
    # it from the digest-verified step_summary evidence that authorized the
    # structured source instead.
    raw_contracts = source.get("summary", {}).get("model_contracts")
    if (
        isinstance(raw_contracts, list)
        and raw_contracts
        and all(isinstance(contract, dict) for contract in raw_contracts)
    ):
        pd.DataFrame(raw_contracts).to_csv(
            out_dir / "model_summaries.csv",
            index=False,
        )
        copied["model_summaries"] = "model_summaries.csv"
    return copied


def matrix_model_trace(
    *,
    spec_id: str,
    spec: Optional[RobustnessSpec],
    structured_source: Optional[Dict[str, Any]],
    structured_replay: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind one scalar sensitivity row to the exact fitted-model contract.

    The full ``spec_id x model_id`` evidence remains in
    ``robustness_model_contracts`` and the coefficient tables.  This helper
    records which one of those models supplied the scalar row plotted in the
    manuscript-facing sensitivity figure, so renderers never have to infer a
    model identity from prose in ``notes``.
    """

    empty = {
        "model_contract_n": None,
        "event_n": None,
        "model_id": None,
        "source_model_id": None,
        "exposure_source": None,
        "exposure_expression": None,
        "exposure_role": None,
        "analysis_role": None,
        "analysis_set": None,
        "baseline_missing_policy": None,
        "fit_status": None,
        "fit_method": None,
        "replay_mode": None,
        "coefficient_source_table": None,
        "coefficient_term": None,
        "model_contract_source": None,
        "source_script_sha256": None,
    }
    if structured_source is None:
        return empty

    primary_contract = structured_source.get("primary_contract")
    if not isinstance(primary_contract, dict):
        return empty
    primary_model_id = str(primary_contract.get("model_id") or "").strip()
    primary_source = str(primary_contract.get("exposure_source") or "").strip()

    contract: Optional[Dict[str, Any]] = None
    # Still resolved, because the rows are READ from the upstream file.
    coefficient_path = structured_source.get("coefficient_path")
    # Name the copy THIS step owns, not the upstream file it was copied from.
    #
    # ``copy_structured_primary_contract_artifacts`` copies the primary
    # coefficients into this step's outputs under
    # ``PRIMARY_COEFFICIENT_COPY_NAME``.  Naming the source path instead left
    # the row pointing at a file that exists only in the parent step, and the
    # figure lineage check resolves ``coefficient_source_table`` against the
    # outputs of the step that owns the row -- so it read nothing and reported
    # ``coefficient_source_unreadable``.  Measured over every recorded run: 11
    # matrix rows name a file their own step does not own (all of them the
    # primary row, all naming the parent's filename) against 4 that name one it
    # does.
    coefficient_source = PRIMARY_COEFFICIENT_COPY_NAME
    contract_source = "step_summary.json:model_contracts"
    replay_mode = "completed_primary_step_output"
    coefficient_rows: List[Dict[str, Any]] = []
    if spec_id == PRIMARY_SPEC_ID:
        contract = dict(primary_contract)
        try:
            import pandas as pd  # type: ignore

            coefficient_rows = pd.read_csv(coefficient_path).to_dict(orient="records")
        except Exception:
            coefficient_rows = []
    else:
        raw_contracts = structured_replay.get("variant_contracts") or []
        candidates = [
            dict(item)
            for item in raw_contracts
            if isinstance(item, dict)
            and str(item.get("spec_id") or "") == spec_id
            and str(item.get("exposure_source") or "") == primary_source
            and str(item.get("exposure_role") or "primary").lower() == "primary"
        ]
        desired_analysis_set = str(primary_contract.get("analysis_set") or "").lower()
        if spec is not None and spec.axis == "missing":
            strategy = str((spec.missing_override or {}).get("strategy") or "").lower()
            desired_analysis_set = (
                "complete_case" if strategy == "complete_case" else "source_aware"
            )
        preferred = [
            item
            for item in candidates
            if str(item.get("analysis_set") or "").lower() == desired_analysis_set
        ]
        if spec is not None and spec.axis == "cohort" and primary_model_id:
            same_model = [
                item
                for item in preferred
                if str(item.get("source_model_id") or item.get("model_id") or "")
                == primary_model_id
            ]
            if same_model:
                preferred = same_model
        if len(preferred) == 1:
            contract = preferred[0]
        elif len(candidates) == 1:
            contract = candidates[0]
        coefficient_source = "robustness_variant_coefficients.csv"
        contract_source = "step_summary.json:robustness_model_contracts"
        replay_mode = str((contract or {}).get("replay_mode") or "") or None
        coefficient_rows = [
            dict(item)
            for item in (structured_replay.get("variant_coefficients") or [])
            if isinstance(item, dict) and str(item.get("spec_id") or "") == spec_id
        ]

    if not isinstance(contract, dict):
        return empty
    model_id = str(contract.get("model_id") or "")
    exposure_terms = [
        item
        for item in coefficient_rows
        if str(item.get("model_id") or "") == model_id
        and str(item.get("term_role") or "").lower() == "exposure"
        and str(item.get("source_variable") or "") == primary_source
    ]
    # Same shape as the headline selection in ``_structured_model_row``: with a
    # declared gradient several exposure terms are fitted, and "exactly one or
    # give up" leaves this trace field empty. The trace check then refuses a row
    # whose coefficient IS identified -- the contract names it right here.
    exposure_expression = str(contract.get("exposure_expression") or "").strip()
    if len(exposure_terms) > 1 and exposure_expression:
        exposure_terms = [
            item
            for item in exposure_terms
            if str(item.get("term") or "") == exposure_expression
        ]
    coefficient_term = (
        exposure_terms[0].get("term") if len(exposure_terms) == 1 else None
    )
    return {
        "model_contract_n": contract.get("n"),
        "event_n": contract.get("event_n"),
        "model_id": contract.get("model_id"),
        "source_model_id": contract.get("source_model_id") or contract.get("model_id"),
        "exposure_source": contract.get("exposure_source"),
        "exposure_expression": contract.get("exposure_expression"),
        "exposure_role": contract.get("exposure_role"),
        "analysis_role": contract.get("analysis_role"),
        "analysis_set": contract.get("analysis_set"),
        "baseline_missing_policy": contract.get("baseline_missing_policy"),
        "fit_status": contract.get("fit_status"),
        "fit_method": contract.get("fit_method"),
        "replay_mode": replay_mode,
        "coefficient_source_table": coefficient_source,
        "coefficient_term": coefficient_term,
        "model_contract_source": contract_source,
        "source_script_sha256": structured_source.get("script_sha256"),
    }


__all__ = [
    "PRIMARY_COEFFICIENT_COPY_NAME",
    "copy_structured_primary_contract_artifacts",
    "matrix_model_trace",
    "registered_raw_input_contracts",
    "variant_typed_manifest_path",
]
