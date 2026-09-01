"""Fail-closed validation for the Figure 2 v2.1 review candidate.

This module is evaluator-side design infrastructure. It has no Provider client,
does not load patient data, and cannot launch an experiment.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, NoReturn


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "experiment_protocol_v2_1.json"
TASKBANK_PATH = PACKAGE_ROOT / "heldout27_taskbank_v1.jsonl"
SAFETY_TASKBANK_PATH = PACKAGE_ROOT / "formal_safety12_taskbank_v2.jsonl"
RUBRIC_PATH = PACKAGE_ROOT / "heldout27_evaluation_rubric_v2.json"
SAP_PATH = PACKAGE_ROOT / "statistical_analysis_plan_v2.json"
GENERIC_SPEC_PATH = PACKAGE_ROOT / "generic_code_agent_spec_v1.json"
REVIEW_CONTRACT_PATH = PACKAGE_ROOT / "review_bundle_contract_v1.json"
WP1_PATH = PACKAGE_ROOT / "data_platform_validation_protocol_v2.json"
SAFETY_RATIONALE_PATH = PACKAGE_ROOT / "safety12_external_rationale_v1.json"
LAUNCH_CONTRACT_PATH = PACKAGE_ROOT / "formal_launch_contract_v1.json"
PREREGISTRATION_PLAN_PATH = PACKAGE_ROOT / "preregistration_plan_v1.json"
IDEA_TO_EVIDENCE_PROTOCOL_PATH = PACKAGE_ROOT / "idea_to_evidence_protocol_v1.json"
IDEA_TO_EVIDENCE_RUBRIC_PATH = (
    PACKAGE_ROOT / "idea_to_evidence_evaluation_rubric_v1.json"
)


class DesignContractError(ValueError):
    """Stable design-contract failure."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> NoReturn:
    raise DesignContractError(reason_code, detail)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        _fail("DESIGN_JSON_SHAPE_INVALID", f"expected object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            _fail(
                "DESIGN_JSONL_SHAPE_INVALID",
                f"expected object at {path}:{line_number}",
            )
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_heldout_schedule(
    tasks: list[dict[str, Any]], seed: int
) -> tuple[list[str], dict[str, str]]:
    """Reproduce the frozen stratified order and arm-first assignment."""
    difficulties = ("basic", "intermediate", "advanced")
    groups = {
        difficulty: [
            task["task_id"]
            for task in tasks
            if task["difficulty"] == difficulty
        ]
        for difficulty in difficulties
    }
    if any(len(group) != 9 for group in groups.values()):
        _fail("HELDOUT_DIFFICULTY_BALANCE_INVALID", repr(groups))
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    order = [groups[difficulty][index] for index in range(9) for difficulty in difficulties]
    arm_first: dict[str, str] = {}
    easyicu_first_counts = {"basic": 5, "intermediate": 5, "advanced": 4}
    for difficulty in difficulties:
        candidates = list(groups[difficulty])
        rng.shuffle(candidates)
        easyicu_first = set(candidates[: easyicu_first_counts[difficulty]])
        for task_id in groups[difficulty]:
            arm_first[task_id] = (
                "easyicu_full" if task_id in easyicu_first else "generic_code_agent"
            )
    return order, arm_first


def _binomial_probability(n: int, successes: int, probability: float) -> float:
    return (
        math.comb(n, successes)
        * probability**successes
        * (1.0 - probability) ** (n - successes)
    )


def _exact_binomial_two_sided_p(n: int, successes: int) -> float:
    observed = _binomial_probability(n, successes, 0.5)
    return min(
        1.0,
        sum(
            _binomial_probability(n, candidate, 0.5)
            for candidate in range(n + 1)
            if _binomial_probability(n, candidate, 0.5) <= observed + 1e-15
        ),
    )


def exact_mcnemar_power(n_pairs: int, p10: float, p01: float) -> float:
    """Enumerate exact McNemar rejection probability for paired multinomial cells."""
    discordance = p10 + p01
    conditional_p10 = p10 / discordance
    power = 0.0
    for discordant_count in range(n_pairs + 1):
        p_discordant_count = _binomial_probability(
            n_pairs, discordant_count, discordance
        )
        for p10_count in range(discordant_count + 1):
            if _exact_binomial_two_sided_p(discordant_count, p10_count) <= 0.05:
                power += p_discordant_count * _binomial_probability(
                    discordant_count, p10_count, conditional_p10
                )
    return power


def _validate_assets(protocol: dict[str, Any]) -> None:
    roles: set[str] = set()
    for asset in protocol["frozen_assets"]:
        role = asset["role"]
        if role in roles:
            _fail("FROZEN_ASSET_ROLE_DUPLICATE", role)
        roles.add(role)
        path = REPO_ROOT / asset["path"]
        if not path.is_file() or path.is_symlink():
            _fail("FROZEN_ASSET_PATH_INVALID", str(path))
        actual = _sha256(path)
        if actual != asset["sha256"]:
            _fail(
                "FROZEN_ASSET_DIGEST_MISMATCH",
                f"{role}: expected {asset['sha256']}, got {actual}",
            )
    required_roles = {
        "action_space",
        "qualification12_taskbank",
        "heldout27_taskbank",
        "heldout27_evaluation_rubric",
        "statistical_analysis_plan",
        "formal_safety12_taskbank",
        "formal_safety12_rubric",
        "safety12_external_rationale",
        "data_platform_validation_protocol",
        "generic_code_agent_spec",
        "review_bundle_contract",
        "preregistration_plan",
        "formal_launch_contract",
        "idea_to_evidence_protocol",
        "idea_to_evidence_evaluation_rubric",
    }
    missing = sorted(required_roles - roles)
    if missing:
        _fail("FROZEN_ASSET_ROLE_MISSING", ", ".join(missing))


def validate_review_candidate_bundle() -> dict[str, Any]:
    """Validate v2.1 design integrity without granting run authority."""
    protocol = _load_json(PROTOCOL_PATH)
    if protocol["freeze_status"] != "review_candidate_no_formal_run_authority":
        _fail("REVIEW_STATUS_INVALID", protocol["freeze_status"])
    authority = protocol["current_authority"]
    if any(
        authority[key]
        for key in (
            "provider_calls_authorized",
            "planner_calls_authorized",
            "formal_batch_authorized",
            "paper_result_authority",
        )
    ):
        _fail("REVIEW_CANDIDATE_AUTHORITY_ESCALATED", repr(authority))
    _validate_assets(protocol)

    tasks = _load_jsonl(TASKBANK_PATH)
    task_ids = [task["task_id"] for task in tasks]
    if task_ids != [f"icu27_t{index:02d}" for index in range(1, 28)]:
        _fail("HELDOUT_TASK_IDENTITY_INVALID", repr(task_ids))
    if {task["expected_behavior"] for task in tasks} != {"bound_result"}:
        _fail("HELDOUT_EXPECTED_BEHAVIOR_INVALID", "all 27 must be bound_result")
    expected_counts = {
        "difficulty": {"basic": 9, "intermediate": 9, "advanced": 9},
        "database": {"miiv": 5, "mimic": 4, "eicu": 6, "aumc": 5, "hirid": 3, "sic": 4},
        "analysis_family": {"descriptive": 5, "association": 6, "prediction": 4, "time_to_event": 4, "causal_emulation": 4, "phenotyping": 4},
    }
    for field, expected in expected_counts.items():
        actual = dict(Counter(task[field] for task in tasks))
        if actual != expected:
            _fail("HELDOUT_COVERAGE_INVALID", f"{field}: {actual}")
    schedule = protocol["formal_schedule"]
    order, arm_first = derive_heldout_schedule(tasks, schedule["randomization_seed"])
    if order != schedule["heldout27_execution_order"]:
        _fail("HELDOUT_ORDER_DRIFT", repr(order))
    if arm_first != schedule["heldout27_arm_first"]:
        _fail("HELDOUT_ARM_BALANCE_DRIFT", repr(arm_first))
    if schedule.get("wp5_showcase_id") != "ite_showcase_01":
        _fail("WP5_SHOWCASE_IDENTITY_INVALID", repr(schedule.get("wp5_showcase_id")))

    rubric = _load_json(RUBRIC_PATH)
    if rubric["primary_endpoint"]["name"] != "reportable_as_specified_without_postrun_repair":
        _fail("PRIMARY_ENDPOINT_DRIFT", rubric["primary_endpoint"]["name"])
    derivation = rubric["task_specific_derivation"]
    if "required_stages" in derivation["fields"]:
        _fail("CIRCULAR_RUBRIC_STAGE_LEAK", "required_stages entered A/B criteria")
    gates = {gate["gate_id"] for gate in rubric["hard_gates"]}
    if "HG09_TASK_QUESTION_ANSWERED" not in gates:
        _fail("PRIMARY_GATE_MISSING", "HG09_TASK_QUESTION_ANSWERED")

    safety_tasks = _load_jsonl(SAFETY_TASKBANK_PATH)
    if len(safety_tasks) != 12 or len({task["challenge_category"] for task in safety_tasks}) != 12:
        _fail("SAFETY12_IDENTITY_INVALID", "requires 12 distinct challenges")
    allowed_dispositions = {"safe_block", "scope_down", "restricted_report", "request_clarification"}
    actual_dispositions = {task["expected_disposition"] for task in safety_tasks}
    if not actual_dispositions <= allowed_dispositions:
        _fail("SAFETY12_DISPOSITION_NOT_NEUTRAL", repr(actual_dispositions))
    rationale = _load_json(SAFETY_RATIONALE_PATH)
    if set(rationale["task_to_rationale"]) != {task["task_id"] for task in safety_tasks}:
        _fail("SAFETY12_EXTERNAL_RATIONALE_INCOMPLETE", "task mapping mismatch")

    generic_spec = _load_json(GENERIC_SPEC_PATH)
    floor = generic_spec["qualification_floor"]
    if "At least 5 of the 7" not in floor["positive_task_floor"]:
        _fail("GENERIC_QUALIFICATION_FLOOR_DRIFT", repr(floor))
    if "maximum_model_turns" in generic_spec["agent_loop"]:
        _fail("GENERIC_ARM_SPECIFIC_TURN_CAP_FORBIDDEN", "maximum_model_turns")
    iteration_fairness = generic_spec["model_and_budget"]["iteration_fairness"]
    if "No arm-specific" not in iteration_fairness or "same numeric value" not in iteration_fairness:
        _fail("GENERIC_ITERATION_FAIRNESS_MISSING", iteration_fairness)
    reserve_policy = floor["reserve_set_policy"]
    if "Qualification12-B and Qualification12-C" not in reserve_policy:
        _fail("QUALIFICATION_RESERVE_SET_POLICY_MISSING", reserve_policy)
    symmetric_consumption = protocol["splits"]["qualification12"]["set_consumption_policy"]
    required_consumption_terms = ("either arm", "shared normalizer", "consumes that set", "next unopened")
    if not all(term in symmetric_consumption for term in required_consumption_terms):
        _fail("QUALIFICATION_SET_CONSUMPTION_ASYMMETRIC", symmetric_consumption)
    if not all(term in floor["failed_floor_policy"] for term in ("either arm", "shared normalizer")):
        _fail("GENERIC_QUALIFICATION_CONSUMPTION_DRIFT", floor["failed_floor_policy"])

    sap = _load_json(SAP_PATH)
    for scenario in sap["a_priori_sensitivity_and_power"]["scenarios"]:
        actual_power = exact_mcnemar_power(27, scenario["p10"], scenario["p01"])
        if round(actual_power, 3) != scenario["power"]:
            _fail("MCNEMAR_POWER_DRIFT", f"{scenario}: recomputed {actual_power:.6f}")

    wp5_protocol = _load_json(IDEA_TO_EVIDENCE_PROTOCOL_PATH)
    if wp5_protocol["case_count"] != 1 or wp5_protocol["comparison_arm"] is not None:
        _fail("WP5_CASE_DESIGN_INVALID", repr(wp5_protocol))
    if "does not autonomously select" not in wp5_protocol["selection_authority"]:
        _fail("WP5_HUMAN_SELECTION_AUTHORITY_MISSING", wp5_protocol["selection_authority"])
    expected_wp5_stages = [f"I{index:02d}_{suffix}" for index, suffix in (
        (1, "DIRECTION_INGESTION"),
        (2, "BOUNDED_EVIDENCE_MAPPING"),
        (3, "ITERATIVE_CANDIDATE_REGISTRY"),
        (4, "HUMAN_SCIENTIFIC_DELIBERATION"),
        (5, "DATA_FEASIBILITY_AND_INPUT_DRAFT"),
        (6, "FINAL_QUESTION_PROTOCOL_AND_INPUT_SEAL"),
        (7, "GOVERNED_EXECUTION"),
        (8, "EVIDENCE_AND_REPORTING"),
    )]
    actual_wp5_stages = [stage["stage_id"] for stage in wp5_protocol["workflow_stages"]]
    if actual_wp5_stages != expected_wp5_stages:
        _fail("WP5_STAGE_ORDER_INVALID", repr(actual_wp5_stages))
    direction_contract = wp5_protocol["initial_direction_contract"]
    if "purposively selected" not in direction_contract["showcase_selection_disclosure"]:
        _fail("WP5_PURPOSIVE_SHOWCASE_DISCLOSURE_MISSING", repr(direction_contract))
    run_policy = wp5_protocol["run_policy"]
    if not run_policy["iterative_phase_a_allowed"] or not run_policy["append_only_candidate_history"]:
        _fail("WP5_ITERATIVE_DISCOVERY_DISABLED", repr(run_policy))
    if not run_policy["preoutcome_candidate_revision_or_replacement"]:
        _fail("WP5_PREOUTCOME_ITERATION_FORBIDDEN", repr(run_policy))
    if run_policy["postoutcome_candidate_replacement"] or run_policy["postresult_product_repair"]:
        _fail("WP5_POSTOUTCOME_SWITCH_GUARD_INVALID", repr(run_policy))
    if "without a Provider call or patient-level outcome result" not in wp5_protocol["workflow_stages"][4]["gate"]:
        _fail("WP5_INTERPHASE_DATA_GATE_MISSING", repr(wp5_protocol["workflow_stages"][4]))
    if "before any patient-level outcome analysis" not in wp5_protocol["workflow_stages"][5]["gate"]:
        _fail("WP5_FINAL_LOCK_TIMING_MISSING", repr(wp5_protocol["workflow_stages"][5]))
    demonstration = wp5_protocol["demonstration_package"]
    if "immutable receipts" not in demonstration["required_for_showcase"]:
        _fail("WP5_DEMONSTRATION_RECEIPT_BINDING_MISSING", repr(demonstration))
    if not any("postoutcome switching" in item for item in demonstration["prohibited"]):
        _fail("WP5_POSTOUTCOME_SELECTION_GUARD_MISSING", repr(demonstration))

    wp5_rubric = _load_json(IDEA_TO_EVIDENCE_RUBRIC_PATH)
    if wp5_rubric["confirmatory_status"] != "descriptive_only_no_hypothesis_test":
        _fail("WP5_CONFIRMATORY_SCOPE_ESCALATED", wp5_rubric["confirmatory_status"])
    wp5_analysis_rules = wp5_rubric["analysis_rules"]
    if not wp5_analysis_rules["flagship_success_showcase_allowed"]:
        _fail("WP5_SUCCESS_SHOWCASE_FORBIDDEN", repr(wp5_analysis_rules))
    if not wp5_analysis_rules["purposive_selection_must_be_disclosed"]:
        _fail("WP5_PURPOSIVE_SELECTION_HIDDEN", repr(wp5_analysis_rules))
    if not wp5_analysis_rules["complete_append_only_discovery_trace_required"]:
        _fail("WP5_DISCOVERY_TRACE_OPTIONAL", repr(wp5_analysis_rules))
    if not wp5_analysis_rules["postoutcome_candidate_switch_forbidden"]:
        _fail("WP5_POSTOUTCOME_SWITCH_ALLOWED", repr(wp5_analysis_rules))
    if wp5_analysis_rules["hypothesis_tests"] != "none":
        _fail("WP5_HYPOTHESIS_TEST_FORBIDDEN", wp5_analysis_rules["hypothesis_tests"])
    wp5_sap = sap["idea_to_evidence_showcase_analysis"]
    if "No hypothesis test" not in wp5_sap["inferential_policy"]:
        _fail("WP5_SAP_INFERENCE_INVALID", wp5_sap["inferential_policy"])

    work_packages = {item["work_package"]: item for item in protocol["work_packages"]}
    wp5_package = work_packages.get("WP5_IDEA_TO_EVIDENCE_SHOWCASE")
    if wp5_package is None or wp5_package["case_count"] != 1:
        _fail("WP5_WORK_PACKAGE_MISSING", repr(wp5_package))
    if wp5_package["confirmatory_denominator_effect"] != 0:
        _fail("WP5_CONFIRMATORY_DENOMINATOR_CHANGED", repr(wp5_package))
    formal_policy = protocol["formal_run_policy"]
    if formal_policy["wp5_descriptive_workflow_runs"] != 1 or formal_policy["core_plus_wp5_workflows"] != 79:
        _fail("WP5_RUN_COUNT_DRIFT", repr(formal_policy))

    wp1 = _load_json(WP1_PATH)
    gate_text = wp1["all_or_none_formal_input_gate"]["policy"]
    if "Every patient-level database-concept cell required by Heldout27" not in gate_text:
        _fail("WP1_ALL_OR_NONE_GATE_MISSING", gate_text)
    safety_boundary = wp1["all_or_none_formal_input_gate"]["formal_safety12_boundary"]
    if "sealed neutral evaluator fixtures" not in safety_boundary or "has no WP1 database-concept cells" not in safety_boundary:
        _fail("WP1_SAFETY12_SCOPE_CONTRADICTION", safety_boundary)
    if "separately sealed neutral fixtures" not in protocol["input_fairness"]["all_or_none_gate"]:
        _fail("PROTOCOL_SAFETY12_INPUT_BOUNDARY_MISSING", protocol["input_fairness"]["all_or_none_gate"])
    if "separately sealed neutral fixtures" not in sap["analysis_sets"]["wp1_all_or_none_gate"]:
        _fail("SAP_SAFETY12_INPUT_BOUNDARY_MISSING", sap["analysis_sets"]["wp1_all_or_none_gate"])
    manual_audit = wp1["reference_standard"]["manual"]
    if "at least 35" not in manual_audit or "at least 50" not in manual_audit:
        _fail("WP1_RISK_TIER_SAMPLE_SIZE_MISSING", manual_audit)
    audit_rule = wp1["all_or_none_formal_input_gate"]["sampled_audit_rule"]
    if "new seed and fresh sample" not in audit_rule or "zero critical discrepancies" not in audit_rule:
        _fail("WP1_FRESH_REAUDIT_RULE_MISSING", audit_rule)
    thresholds = wp1["release_thresholds"]
    if "Zero residual unexplained" not in thresholds["independent_implementation"]:
        _fail("WP1_COMPARATOR_RELEASE_THRESHOLD_MISSING", repr(thresholds))
    comparator = wp1["reference_standard"]["independent_implementation_comparator"]
    if "sic" in comparator["overlapping_databases"] or "does not include SICdb" not in comparator["sicdb_exception"]:
        _fail("WP1_COMPARATOR_SCOPE_INVALID", repr(comparator))

    review_contract = _load_json(REVIEW_CONTRACT_PATH)
    preservation = review_contract["normalization"]["content_preservation"]
    if "may not repair, add, delete, reinterpret, or recompute" not in preservation:
        _fail("BLINDING_NORMALIZER_CONTENT_MUTATION", preservation)
    forbidden_markers = review_contract["normalization"]["forbidden_markers"]
    if not any("arm-diagnostic resource profiles" in marker for marker in forbidden_markers):
        _fail("BLINDING_RESOURCE_FINGERPRINT_GUARD_MISSING", repr(forbidden_markers))
    receipt_projection = review_contract["normalization"]["blinded_run_receipt_projection"]
    if "model-turn and provider-call counts" not in receipt_projection["reviewer_hidden_until_scores_lock"]:
        _fail("BLINDING_RAW_RESOURCE_PROJECTION_INVALID", repr(receipt_projection))

    independence = rubric["human_review"]["independence_eligibility"]
    if "At least one" not in independence or "every adjudicator" not in independence:
        _fail("PRIMARY_REVIEWER_INDEPENDENCE_MISSING", independence)
    aliases = derivation["implementation_neutrality_check"]["neutralization_aliases"]
    if "evidence and reportability bundle" not in aliases:
        _fail("REPORTABILITY_NEUTRALIZATION_MISSING", repr(aliases))

    concordance_boundary = sap["a_priori_sensitivity_and_power"]["concordant_scientific_noncompletion"]
    if "not excluded or invalid observations" not in concordance_boundary:
        _fail("CONCORDANT_FAILURE_ESTIMAND_BOUNDARY_MISSING", concordance_boundary)
    if "no additional hypothesis test" not in sap["secondary_analyses"]["failed_hard_gate_count"]:
        _fail("FAILED_GATE_COUNT_MULTIPLICITY_DRIFT", sap["secondary_analyses"]["failed_hard_gate_count"])

    preregistration = _load_json(PREREGISTRATION_PLAN_PATH)
    receipt_fields = set(preregistration["required_receipt_fields"])
    required_digest_fields = {"protocol_sha256", "validator_sha256", "validator_test_sha256", "design_commit", "annotated_tag"}
    if not required_digest_fields <= receipt_fields:
        _fail("PREREGISTRATION_DIGEST_BINDING_MISSING", repr(sorted(required_digest_fields - receipt_fields)))

    launch = _load_json(LAUNCH_CONTRACT_PATH)
    if launch["provider_call_default"] != "deny" or launch["current_authority"]["provider_calls_authorized"]:
        _fail("FORMAL_LAUNCH_FAIL_CLOSED_INVALID", repr(launch["current_authority"]))
    expected_scopes = {"qualification12", "core_wp2_wp3", "wp5_phase_a", "wp5_phase_b_showcase"}
    if set(launch["authorization_scopes"]) != expected_scopes:
        _fail("FORMAL_LAUNCH_SCOPE_INVALID", repr(launch["authorization_scopes"]))
    if "external registration receipt" not in launch["required_receipts"]["qualification_preconditions"][0]:
        _fail("QUALIFICATION_REGISTRATION_PRECONDITION_MISSING", repr(launch["required_receipts"]))
    qualification_scope = launch["authorization_scopes"]["qualification12"]
    if "either arm" not in qualification_scope or "shared normalizer" not in qualification_scope:
        _fail("LAUNCH_QUALIFICATION_CONSUMPTION_ASYMMETRIC", qualification_scope)
    data_receipts = launch["required_receipts"]["data"]
    if not any("Heldout27 input manifests and sealed Safety12 fixture manifests" in item for item in data_receipts):
        _fail("LAUNCH_DATA_RECEIPT_SCOPE_AMBIGUOUS", repr(data_receipts))
    if "WP5 flagship showcase" not in protocol["formal_run_policy"]["first_provider_call_lock"]:
        _fail("WP5_RELEASE_LOCK_INCOMPLETE", protocol["formal_run_policy"]["first_provider_call_lock"])

    return {
        "protocol_ref": protocol["protocol_ref"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "heldout_task_count": len(tasks),
        "safety_task_count": len(safety_tasks),
        "idea_to_evidence_case_count": wp5_protocol["case_count"],
        "frozen_asset_count": len(protocol["frozen_assets"]),
        "provider_calls_authorized": False,
        "formal_batch_authorized": False,
    }


def authorize_formal_provider_call(receipts: dict[str, Any]) -> NoReturn:
    """Deny launch until a future sealed contract explicitly enables it."""
    del receipts
    launch = _load_json(LAUNCH_CONTRACT_PATH)
    reason = launch["current_authority"]["reason"]
    _fail("FORMAL_PROVIDER_CALL_NOT_AUTHORIZED", reason)


__all__ = [
    "DesignContractError",
    "authorize_formal_provider_call",
    "derive_heldout_schedule",
    "exact_mcnemar_power",
    "validate_review_candidate_bundle",
]
