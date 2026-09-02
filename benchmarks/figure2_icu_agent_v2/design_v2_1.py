"""Fail-closed validation for the Figure 2 v2.1 review candidate.

This module is evaluator-side design infrastructure. It has no Provider client,
does not load patient data, and cannot launch an experiment.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, NoReturn

from .design_errors import DesignContractError
from .formal_authority import authorize_formal_provider_call


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PROTOCOL_PATH = PACKAGE_ROOT / "experiment_protocol_v2_1.json"
TASKBANK_PATH = PACKAGE_ROOT / "heldout27_taskbank_v1.jsonl"
SAFETY_TASKBANK_PATH = PACKAGE_ROOT / "formal_safety12_taskbank_v2.jsonl"
SAFETY_RUBRIC_PATH = PACKAGE_ROOT / "formal_safety12_rubric_v2.json"
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
EXECUTION_ACCEPTANCE_PATH = PACKAGE_ROOT / "execution_acceptance_contract_v1.json"
GENERIC_HARNESS_PATH = PACKAGE_ROOT / "generic_code_agent_harness.py"
FORMAL_GENERIC_RUNNER_PATH = PACKAGE_ROOT / "formal_generic_runner.py"
FORMAL_EASYICU_RUNNER_PATH = PACKAGE_ROOT / "formal_easyicu_runner.py"
FORMAL_AUTHORITY_PATH = PACKAGE_ROOT / "formal_authority.py"
FORMAL_PROVIDER_GATE_PATH = PACKAGE_ROOT / "formal_provider_gate.py"
FORMAL_SCHEDULER_PATH = PACKAGE_ROOT / "formal_scheduler.py"
EASYICU_REVIEW_ADAPTER_PATH = PACKAGE_ROOT / "easyicu_review_bundle_adapter.py"
BLINDED_EVALUATOR_PATH = PACKAGE_ROOT / "blinded_evaluator.py"
MULTI_HOST_ACCEPTANCE_PATH = PACKAGE_ROOT / "multi_host_acceptance.py"
REVIEW_SEMANTICS_PATH = PACKAGE_ROOT / "review_bundle_semantics.py"


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
        "execution_acceptance_contract",
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
    heldout_sites = schedule.get("heldout27_execution_site", {})
    if set(heldout_sites) != set(order):
        _fail("HELDOUT_EXECUTION_SITE_IDENTITY_INVALID", repr(heldout_sites))
    heldout_site_counts = Counter(heldout_sites.values())
    if heldout_site_counts != {"server": 14, "laptop": 13}:
        _fail("HELDOUT_EXECUTION_SITE_BALANCE_INVALID", repr(heldout_site_counts))
    tasks_by_id = {task["task_id"]: task for task in tasks}
    expected_site_difficulty = {
        "server": {"basic": 5, "intermediate": 5, "advanced": 4},
        "laptop": {"basic": 4, "intermediate": 4, "advanced": 5},
    }
    expected_site_arm_first = {
        "server": {"easyicu_full": 7, "generic_code_agent": 7},
        "laptop": {"easyicu_full": 7, "generic_code_agent": 6},
    }
    expected_site_database = {
        "server": {
            "mimic": 2,
            "eicu": 3,
            "aumc": 3,
            "hirid": 1,
            "miiv": 3,
            "sic": 2,
        },
        "laptop": {
            "miiv": 2,
            "sic": 2,
            "mimic": 2,
            "aumc": 2,
            "eicu": 3,
            "hirid": 2,
        },
    }
    expected_site_analysis_family = {
        "server": {
            "descriptive": 3,
            "association": 3,
            "prediction": 2,
            "time_to_event": 2,
            "causal_emulation": 2,
            "phenotyping": 2,
        },
        "laptop": {
            "descriptive": 2,
            "association": 3,
            "prediction": 2,
            "time_to_event": 2,
            "causal_emulation": 2,
            "phenotyping": 2,
        },
    }
    for site in ("server", "laptop"):
        site_tasks = [task_id for task_id in order if heldout_sites[task_id] == site]
        difficulty_counts = dict(
            Counter(tasks_by_id[task_id]["difficulty"] for task_id in site_tasks)
        )
        first_counts = dict(Counter(arm_first[task_id] for task_id in site_tasks))
        database_counts = dict(
            Counter(tasks_by_id[task_id]["database"] for task_id in site_tasks)
        )
        analysis_family_counts = dict(
            Counter(tasks_by_id[task_id]["analysis_family"] for task_id in site_tasks)
        )
        if difficulty_counts != expected_site_difficulty[site]:
            _fail("HELDOUT_SITE_DIFFICULTY_BALANCE_INVALID", repr(difficulty_counts))
        if first_counts != expected_site_arm_first[site]:
            _fail("HELDOUT_SITE_ARM_ORDER_BALANCE_INVALID", repr(first_counts))
        if database_counts != expected_site_database[site]:
            _fail("HELDOUT_SITE_DATABASE_BALANCE_INVALID", repr(database_counts))
        if analysis_family_counts != expected_site_analysis_family[site]:
            _fail(
                "HELDOUT_SITE_ANALYSIS_FAMILY_BALANCE_INVALID",
                repr(analysis_family_counts),
            )
    if schedule.get("wp5_showcase_id") != "ite_showcase_01":
        _fail("WP5_SHOWCASE_IDENTITY_INVALID", repr(schedule.get("wp5_showcase_id")))
    multi_host = protocol.get("multi_host_execution", {})
    if multi_host.get("contract") != str(EXECUTION_ACCEPTANCE_PATH.relative_to(REPO_ROOT)):
        _fail("MULTI_HOST_CONTRACT_PATH_INVALID", repr(multi_host))
    if multi_host.get("logical_sites") != ["server", "laptop"]:
        _fail("MULTI_HOST_SITE_IDENTITY_INVALID", repr(multi_host))
    if not all(
        term in multi_host.get("assignment_rule", "")
        for term in ("server=20", "laptop=19", "Both arms", "sequentially")
    ):
        _fail("MULTI_HOST_PAIR_ASSIGNMENT_INVALID", repr(multi_host))

    execution_acceptance = _load_json(EXECUTION_ACCEPTANCE_PATH)
    if execution_acceptance.get("status") != "review_candidate_no_launch_authority":
        _fail("EXECUTION_ACCEPTANCE_STATUS_INVALID", repr(execution_acceptance))
    parallelization = execution_acceptance.get("parallelization", {})
    if (
        parallelization.get("unit") != "task_pair"
        or parallelization.get("maximum_active_trajectories_per_site") != 1
        or parallelization.get("maximum_active_trajectories_total") != 2
    ):
        _fail("MULTI_HOST_CONCURRENCY_INVALID", repr(parallelization))
    if execution_acceptance.get("site_preflight_boolean_expectations", {}).get(
        "provider_accessed"
    ) is not False:
        _fail("MULTI_HOST_PREFLIGHT_PROVIDER_BOUNDARY_INVALID", repr(execution_acceptance))
    if "Scientific A/B outcomes are never" not in multi_host.get("go_no_go_rule", ""):
        _fail("MULTI_HOST_OUTCOME_ADAPTATION_GUARD_MISSING", repr(multi_host))

    rubric = _load_json(RUBRIC_PATH)
    if rubric["primary_endpoint"]["name"] != "reportable_as_specified_without_postrun_repair":
        _fail("PRIMARY_ENDPOINT_DRIFT", rubric["primary_endpoint"]["name"])
    derivation = rubric["task_specific_derivation"]
    if "required_stages" in derivation["fields"]:
        _fail("CIRCULAR_RUBRIC_STAGE_LEAK", "required_stages entered A/B criteria")
    gates = {gate["gate_id"] for gate in rubric["hard_gates"]}
    if "HG09_TASK_QUESTION_ANSWERED" not in gates:
        _fail("PRIMARY_GATE_MISSING", "HG09_TASK_QUESTION_ANSWERED")
    manifest_warning = rubric["arm_neutral_review_bundle"].get(
        "manifest_interpretation_warning", ""
    )
    if not all(
        term in manifest_warning
        for term in (
            "Harness-computed file digests and substantive-output-file flags",
            "unverified assertions",
            "HG04 and HG06",
        )
    ):
        _fail("REVIEW_MANIFEST_ASSERTION_BOUNDARY_MISSING", manifest_warning)

    safety_tasks = _load_jsonl(SAFETY_TASKBANK_PATH)
    if len(safety_tasks) != 12 or len({task["challenge_category"] for task in safety_tasks}) != 12:
        _fail("SAFETY12_IDENTITY_INVALID", "requires 12 distinct challenges")
    allowed_dispositions = {"safe_block", "scope_down", "restricted_report", "request_clarification"}
    actual_dispositions = {task["expected_disposition"] for task in safety_tasks}
    if not actual_dispositions <= allowed_dispositions:
        _fail("SAFETY12_DISPOSITION_NOT_NEUTRAL", repr(actual_dispositions))
    safety_sites = schedule.get("safety12_execution_site")
    safety_first = schedule.get("safety12_arm_first")
    if not isinstance(safety_sites, list) or len(safety_sites) != 12:
        _fail("SAFETY12_EXECUTION_SITE_IDENTITY_INVALID", repr(safety_sites))
    if Counter(safety_sites) != {"server": 6, "laptop": 6}:
        _fail("SAFETY12_EXECUTION_SITE_BALANCE_INVALID", repr(safety_sites))
    for site in ("server", "laptop"):
        first_counts = Counter(
            first
            for first, assigned_site in zip(safety_first, safety_sites, strict=True)
            if assigned_site == site
        )
        if first_counts != {"easyicu_full": 3, "generic_code_agent": 3}:
            _fail("SAFETY12_SITE_ARM_ORDER_BALANCE_INVALID", repr(first_counts))
    rationale = _load_json(SAFETY_RATIONALE_PATH)
    if set(rationale["task_to_rationale"]) != {task["task_id"] for task in safety_tasks}:
        _fail("SAFETY12_EXTERNAL_RATIONALE_INCOMPLETE", "task mapping mismatch")
    safety_rubric = _load_json(SAFETY_RUBRIC_PATH)
    safety_fixture_boundary = safety_rubric["shared_response_contract"][
        "fixture_boundary"
    ]
    if not all(
        term in safety_fixture_boundary
        for term in (
            "no patient-level rows",
            "proposed, prespecified, and justified rather than executed",
        )
    ):
        _fail("SAFETY12_PATIENT_ROW_BOUNDARY_MISSING", safety_fixture_boundary)

    generic_spec = _load_json(GENERIC_SPEC_PATH)
    if generic_spec["status"] != (
        "review_candidate_harness_and_authority_implemented_no_registered_signer"
    ):
        _fail("GENERIC_HARNESS_STATUS_INVALID", generic_spec["status"])
    implementation = generic_spec["implementation"]
    if implementation["harness_owner"] != str(
        GENERIC_HARNESS_PATH.relative_to(REPO_ROOT)
    ) or implementation["formal_provider_owner"] != str(
        FORMAL_GENERIC_RUNNER_PATH.relative_to(REPO_ROOT)
    ) or implementation["formal_authority_owner"] != str(
        FORMAL_AUTHORITY_PATH.relative_to(REPO_ROOT)
    ):
        _fail("GENERIC_HARNESS_OWNER_DRIFT", repr(implementation))
    if not all(
        path.is_file()
        for path in (
            GENERIC_HARNESS_PATH,
            FORMAL_GENERIC_RUNNER_PATH,
            FORMAL_EASYICU_RUNNER_PATH,
            FORMAL_AUTHORITY_PATH,
            FORMAL_SCHEDULER_PATH,
            EASYICU_REVIEW_ADAPTER_PATH,
            BLINDED_EVALUATOR_PATH,
            MULTI_HOST_ACCEPTANCE_PATH,
            REVIEW_SEMANTICS_PATH,
        )
    ):
        _fail("GENERIC_HARNESS_IMPLEMENTATION_MISSING", repr(implementation))
    formal_paths = tuple(
        sorted(
            path
            for path in PACKAGE_ROOT.glob("formal_*.py")
            if path.name != "formal_provider_gate.py"
        )
    ) + (FORMAL_PROVIDER_GATE_PATH,)
    protected_paths = (*formal_paths, GENERIC_HARNESS_PATH, MULTI_HOST_ACCEPTANCE_PATH)
    protected_trees = {
        path.name: ast.parse(path.read_text(encoding="utf-8"))
        for path in protected_paths
    }
    imports_by_file: dict[str, set[str]] = {}
    calls_by_file: dict[str, list[ast.Call]] = {}
    for name, tree in protected_trees.items():
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        imports_by_file[name] = imports
        calls_by_file[name] = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
        ]
        if "importlib" in imports or any(
            isinstance(call.func, ast.Name) and call.func.id == "__import__"
            for call in calls_by_file[name]
        ):
            _fail("FORMAL_GENERIC_DYNAMIC_IMPORT_FORBIDDEN", name)

    for name in tuple(
        path.name for path in protected_paths if path != FORMAL_PROVIDER_GATE_PATH
    ):
        forbidden_provider_imports = {
            module
            for module in imports_by_file[name]
            if module.startswith("easyicu.research_agent.providers.")
            and module != "easyicu.research_agent.providers.protocol"
        }
        if forbidden_provider_imports:
            _fail(
                "FORMAL_GENERIC_PROVIDER_GATE_BYPASS",
                f"{name}: " + ", ".join(sorted(forbidden_provider_imports)),
            )
        if any(
            isinstance(call.func, ast.Name) and call.func.id == "authorized_complete"
            for call in calls_by_file[name]
        ):
            _fail("FORMAL_GENERIC_PROVIDER_GATE_BYPASS", name)
        if any(
            isinstance(call.func, ast.Name)
            and call.func.id == "getattr"
            and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant)
            and call.args[1].value
            in {
                "authorized_complete",
                "complete",
                "complete_with_usage",
                "complete_with_images",
            }
            for call in calls_by_file[name]
        ):
            _fail("FORMAL_PROVIDER_DYNAMIC_ATTRIBUTE_BYPASS", name)
        if any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "__getattribute__"
            for call in calls_by_file[name]
        ):
            _fail("FORMAL_PROVIDER_DYNAMIC_ATTRIBUTE_BYPASS", name)

    formal_runner_calls = calls_by_file[FORMAL_GENERIC_RUNNER_PATH.name]
    runner_named_calls = {
        call.func.id for call in formal_runner_calls if isinstance(call.func, ast.Name)
    }
    if any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "complete"
        for call in formal_runner_calls
    ):
        _fail("FORMAL_GENERIC_PROVIDER_GATE_BYPASS", "direct .complete call")
    if "complete_formal_provider_call" not in runner_named_calls:
        _fail("FORMAL_GENERIC_PROVIDER_GATE_MISSING", repr(runner_named_calls))

    easyicu_runner_calls = calls_by_file[FORMAL_EASYICU_RUNNER_PATH.name]
    easyicu_runner_named_calls = {
        call.func.id
        for call in easyicu_runner_calls
        if isinstance(call.func, ast.Name)
    }
    required_easyicu_runner_calls = {
        "FormalAuthorizedHardStopClient",
        "write_easyicu_review_bundle",
    }
    if not required_easyicu_runner_calls <= easyicu_runner_named_calls:
        _fail(
            "FORMAL_EASYICU_PROVIDER_GATE_MISSING",
            repr(sorted(required_easyicu_runner_calls - easyicu_runner_named_calls)),
        )

    gate_calls = calls_by_file[FORMAL_PROVIDER_GATE_PATH.name]
    gate_named_calls = {
        call.func.id for call in gate_calls if isinstance(call.func, ast.Name)
    }
    if not {"authorize_formal_provider_call", "authorized_complete"} <= gate_named_calls:
        _fail("FORMAL_PROVIDER_GATE_SEQUENCE_MISSING", repr(gate_named_calls))
    authority_lines = [
        call.lineno
        for call in gate_calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "authorize_formal_provider_call"
    ]
    transport_lines = [
        call.lineno
        for call in gate_calls
        if isinstance(call.func, ast.Name) and call.func.id == "authorized_complete"
    ]
    if not authority_lines or not transport_lines or max(authority_lines) >= min(
        transport_lines
    ):
        _fail("FORMAL_PROVIDER_GATE_SEQUENCE_INVALID", FORMAL_PROVIDER_GATE_PATH.name)
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
    terminal_reporting_rule = run_policy["terminal_reporting_rule"]
    if not all(
        term in terminal_reporting_rule
        for term in (
            "safe_nonlanding or workflow_failure",
            "may not be withdrawn from the manuscript",
        )
    ):
        _fail("WP5_TERMINAL_REPORTING_RULE_MISSING", terminal_reporting_rule)
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
    terminal_evaluation = wp5_rubric["terminal_showcase_evaluation"]
    evaluator_text = " ".join(terminal_evaluation["evaluators"])
    if not all(
        term in evaluator_text
        for term in (
            "independent of EasyICU implementation",
            "not a manuscript author",
        )
    ):
        _fail("WP5_TERMINAL_EVALUATOR_INDEPENDENCE_MISSING", evaluator_text)
    if "all six showcase domains" not in terminal_evaluation["scope"]:
        _fail("WP5_TERMINAL_EVALUATION_SCOPE_MISSING", repr(terminal_evaluation))
    if not any(
        "signed independent terminal-evaluation receipt" in artifact
        and "final terminal disposition" in artifact
        for artifact in wp5_rubric["mandatory_showcase_artifacts"]
    ):
        _fail(
            "WP5_TERMINAL_EVALUATION_RECEIPT_MISSING",
            repr(wp5_rubric["mandatory_showcase_artifacts"]),
        )
    interpretation_domain = next(
        domain
        for domain in wp5_rubric["showcase_domains"]
        if domain["domain"] == "interpretation_ceiling"
    )
    if "internally authored" not in interpretation_domain["pass_rule"]:
        _fail("WP5_INTERNAL_AUTHORSHIP_DISCLOSURE_MISSING", repr(interpretation_domain))
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
    if not any("independently signed" in item for item in wp5_sap["reporting"]):
        _fail("WP5_SAP_INDEPENDENT_EVALUATION_MISSING", repr(wp5_sap["reporting"]))
    if not all(
        term in wp5_sap["iteration_and_failure_policy"]
        for term in (
            "registered flagship's terminal disposition is the WP5 result",
            "may not be withdrawn from the manuscript",
        )
    ):
        _fail(
            "WP5_SAP_TERMINAL_REPORTING_MISSING",
            wp5_sap["iteration_and_failure_policy"],
        )

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
    if review_contract["status"] != (
        "review_candidate_normalizer_implemented_no_formal_authority"
    ):
        _fail("BLINDING_NORMALIZER_STATUS_INVALID", review_contract["status"])
    expected_review_owners = {
        "easyicu_producer": str(EASYICU_REVIEW_ADAPTER_PATH.relative_to(REPO_ROOT)),
        "generic_producer": str(GENERIC_HARNESS_PATH.relative_to(REPO_ROOT)),
        "shared_structural_semantics": str(REVIEW_SEMANTICS_PATH.relative_to(REPO_ROOT)),
        "arm_neutral_normalizer": str(
            (PACKAGE_ROOT / "review_bundle_normalizer.py").relative_to(REPO_ROOT)
        ),
    }
    if review_contract.get("implementation_owners") != expected_review_owners:
        _fail(
            "REVIEW_BUNDLE_IMPLEMENTATION_OWNER_DRIFT",
            repr(review_contract.get("implementation_owners")),
        )
    preservation = review_contract["normalization"]["content_preservation"]
    if "may not repair, add, delete, reinterpret, or recompute" not in preservation:
        _fail("BLINDING_NORMALIZER_CONTENT_MUTATION", preservation)
    forbidden_markers = review_contract["normalization"]["forbidden_markers"]
    if not any("arm-diagnostic resource profiles" in marker for marker in forbidden_markers):
        _fail("BLINDING_RESOURCE_FINGERPRINT_GUARD_MISSING", repr(forbidden_markers))
    if not any("execution-site labels" in marker for marker in forbidden_markers):
        _fail("BLINDING_EXECUTION_SITE_GUARD_MISSING", repr(forbidden_markers))
    blinding_context = review_contract["normalization"].get(
        "required_runtime_blinding_context",
        "",
    )
    if not all(
        term in blinding_context
        for term in ("two exact formal host markers", "two absolute", "/Volumes", "Windows")
    ):
        _fail("BLINDING_RUNTIME_CONTEXT_MISSING", blinding_context)
    receipt_projection = review_contract["normalization"]["blinded_run_receipt_projection"]
    visible_fields = receipt_projection["reviewer_visible_fields"]
    if not any(
        "always present and null when not applicable" in field
        for field in visible_fields
    ):
        _fail("BLINDING_FAILURE_CATEGORY_SHAPE_AMBIGUOUS", repr(visible_fields))
    if not any(
        "explicitly labeled as assertions" in field for field in visible_fields
    ) or not any(
        "harness-computed substantive-output-file flags" in field
        for field in visible_fields
    ):
        _fail("BLINDING_ARTIFACT_ASSERTION_BOUNDARY_MISSING", repr(visible_fields))
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
    required_digest_fields = {
        "protocol_sha256",
        "validator_sha256",
        "validator_test_sha256",
        "formal_authority_sha256",
        "formal_authority_test_sha256",
        "formal_provider_gate_sha256",
        "formal_easyicu_runner_sha256",
        "formal_generic_runner_sha256",
        "generic_harness_sha256",
        "easyicu_review_adapter_sha256",
        "review_bundle_normalizer_sha256",
        "review_bundle_semantics_sha256",
        "formal_scheduler_sha256",
        "multi_host_acceptance_sha256",
        "blinded_evaluator_sha256",
        "formal_implementation_owner_test_sha256",
        "design_commit",
        "annotated_tag",
        "trusted_authority_signer_identity",
        "trusted_authority_ed25519_public_key_base64",
    }
    if not required_digest_fields <= receipt_fields:
        _fail("PREREGISTRATION_DIGEST_BINDING_MISSING", repr(sorted(required_digest_fields - receipt_fields)))
    qualification_consumption_step = preregistration["post_registration_sequence"][2]
    if not all(
        term in qualification_consumption_step
        for term in (
            "either arm's harness",
            "shared normalizer",
            "evaluator",
            "model route",
            "numeric budgets",
        )
    ):
        _fail(
            "PREREGISTRATION_QUALIFICATION_CONSUMPTION_INCOMPLETE",
            qualification_consumption_step,
        )

    launch = _load_json(LAUNCH_CONTRACT_PATH)
    if launch["provider_call_default"] != "deny" or launch["current_authority"]["provider_calls_authorized"]:
        _fail("FORMAL_LAUNCH_FAIL_CLOSED_INVALID", repr(launch["current_authority"]))
    if launch.get("authority_owner") != str(FORMAL_AUTHORITY_PATH.relative_to(REPO_ROOT)):
        _fail("FORMAL_AUTHORITY_OWNER_INVALID", repr(launch.get("authority_owner")))
    signed_payload_schema = launch.get("signature_verification", {}).get(
        "signed_payload_schema", ""
    )
    if not all(
        term in signed_payload_schema
        for term in (
            "easyicu.figure2_atomic_declaration/2",
            "site_assignment_sha256",
            "every qualification/core coordinate",
        )
    ):
        _fail("FORMAL_AUTHORITY_SITE_ASSIGNMENT_BINDING_MISSING", signed_payload_schema)
    site_receipt_transport = execution_acceptance.get("site_receipt_transport", {})
    if not all(
        term in " ".join(str(value) for value in site_receipt_transport.values())
        for term in (
            "unparsed UTF-8 JSON bytes",
            "duplicate-key-rejecting",
            "NaN",
            "already-parsed mappings",
        )
    ):
        _fail(
            "MULTI_HOST_RAW_RECEIPT_TRANSPORT_MISSING",
            repr(site_receipt_transport),
        )
    launch_receipt_ids = {
        f"{group}:{index:02d}"
        for group, descriptions in launch["required_receipts"].items()
        for index, _description in enumerate(descriptions, start=1)
    }
    gate_specs = {
        "prequalification_go_no_go": {
            receipt_id
            for receipt_id in launch_receipt_ids
            if receipt_id.startswith("qualification_preconditions:")
        },
        "core_go_no_go": {
            receipt_id
            for receipt_id in launch_receipt_ids
            if receipt_id.split(":", 1)[0]
            in {"design", "data", "evaluation", "qualification", "runtime", "batch"}
        },
    }
    for gate_name, expected_ids in gate_specs.items():
        requirements = execution_acceptance.get(gate_name, {}).get("required")
        if not isinstance(requirements, list) or not requirements:
            _fail("EXECUTION_GATE_REQUIREMENTS_INVALID", gate_name)
        gate_ids: list[str] = []
        mapped_ids: set[str] = set()
        for requirement in requirements:
            if not isinstance(requirement, dict) or set(requirement) != {
                "gate_id",
                "requirement",
                "launch_receipt_ids",
            }:
                _fail("EXECUTION_GATE_REQUIREMENT_SCHEMA_INVALID", repr(requirement))
            gate_ids.append(requirement["gate_id"])
            receipt_ids = requirement["launch_receipt_ids"]
            if (
                not isinstance(requirement["gate_id"], str)
                or not requirement["gate_id"].strip()
                or not isinstance(requirement["requirement"], str)
                or not requirement["requirement"].strip()
                or not isinstance(receipt_ids, list)
                or not receipt_ids
                or any(receipt_id not in expected_ids for receipt_id in receipt_ids)
            ):
                _fail("EXECUTION_GATE_RECEIPT_MAPPING_INVALID", repr(requirement))
            mapped_ids.update(receipt_ids)
        if len(gate_ids) != len(set(gate_ids)) or mapped_ids != expected_ids:
            _fail(
                "EXECUTION_GATE_RECEIPT_COVERAGE_INVALID",
                f"{gate_name}: missing={sorted(expected_ids - mapped_ids)!r}",
            )
    expected_launch_owners = {
        "provider_gate": str(FORMAL_PROVIDER_GATE_PATH.relative_to(REPO_ROOT)),
        "easyicu_formal_runner": str(
            FORMAL_EASYICU_RUNNER_PATH.relative_to(REPO_ROOT)
        ),
        "generic_formal_runner": str(
            FORMAL_GENERIC_RUNNER_PATH.relative_to(REPO_ROOT)
        ),
        "generic_harness": str(GENERIC_HARNESS_PATH.relative_to(REPO_ROOT)),
        "easyicu_review_adapter": str(
            EASYICU_REVIEW_ADAPTER_PATH.relative_to(REPO_ROOT)
        ),
        "shared_review_semantics": str(REVIEW_SEMANTICS_PATH.relative_to(REPO_ROOT)),
        "review_normalizer": str(
            (PACKAGE_ROOT / "review_bundle_normalizer.py").relative_to(REPO_ROOT)
        ),
        "formal_scheduler": str(FORMAL_SCHEDULER_PATH.relative_to(REPO_ROOT)),
        "multi_host_acceptance": str(
            MULTI_HOST_ACCEPTANCE_PATH.relative_to(REPO_ROOT)
        ),
        "blinded_evaluator": str(BLINDED_EVALUATOR_PATH.relative_to(REPO_ROOT)),
    }
    if launch.get("implementation_owners") != expected_launch_owners:
        _fail(
            "FORMAL_IMPLEMENTATION_OWNER_DRIFT",
            repr(launch.get("implementation_owners")),
        )
    if set(launch.get("registration_receipt_ids", ())) != {
        "qualification_preconditions:01",
        "design:02",
    }:
        _fail(
            "FORMAL_AUTHORITY_REGISTRATION_RECEIPT_IDS_INVALID",
            repr(launch.get("registration_receipt_ids")),
        )
    signature_verification = launch.get("signature_verification", {})
    if signature_verification.get("algorithm") != "Ed25519":
        _fail("FORMAL_AUTHORITY_ALGORITHM_INVALID", repr(signature_verification))
    receipt_schema = signature_verification.get("receipt_payload_schema", "")
    if not all(
        term in receipt_schema
        for term in (
            "easyicu.figure2_launch_receipt/1",
            "status=passed",
            "same signer key",
        )
    ):
        _fail("FORMAL_AUTHORITY_RECEIPT_SCHEMA_MISSING", receipt_schema)
    if any(
        signature_verification.get(field) is not None
        for field in ("trusted_signer_id", "trusted_public_key_base64")
    ):
        _fail(
            "REVIEW_CANDIDATE_SIGNER_PREMATURELY_REGISTERED",
            repr(signature_verification),
        )
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
    if not any("every Safety12 fixture contains no patient-level rows" in item for item in data_receipts):
        _fail("LAUNCH_SAFETY12_PATIENT_ROW_CERTIFICATION_MISSING", repr(data_receipts))
    wp5_phase_b_receipts = launch["required_receipts"]["wp5_phase_b_showcase"]
    if not any(
        "all six showcase domains" in item and "neither evaluator" in item
        for item in wp5_phase_b_receipts
    ):
        _fail("LAUNCH_WP5_EVALUATOR_INDEPENDENCE_MISSING", repr(wp5_phase_b_receipts))
    if "WP5 flagship showcase" not in protocol["formal_run_policy"]["first_provider_call_lock"]:
        _fail("WP5_RELEASE_LOCK_INCOMPLETE", protocol["formal_run_policy"]["first_provider_call_lock"])
    site_sensitivity = sap["secondary_analyses"].get("execution_site_sensitivity", "")
    if not all(
        term in site_sensitivity
        for term in ("Both arms", "colocated", "not an endpoint", "scores lock")
    ):
        _fail("MULTI_HOST_ANALYSIS_BOUNDARY_MISSING", site_sensitivity)

    return {
        "protocol_ref": protocol["protocol_ref"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "heldout_task_count": len(tasks),
        "safety_task_count": len(safety_tasks),
        "idea_to_evidence_case_count": wp5_protocol["case_count"],
        "frozen_asset_count": len(protocol["frozen_assets"]),
        "generic_harness_implemented": True,
        "formal_authority_owner_implemented": True,
        "trusted_signer_registered": False,
        "review_bundle_normalizer_implemented": True,
        "provider_calls_authorized": False,
        "formal_batch_authorized": False,
    }


__all__ = [
    "DesignContractError",
    "authorize_formal_provider_call",
    "derive_heldout_schedule",
    "exact_mcnemar_power",
    "validate_review_candidate_bundle",
]
