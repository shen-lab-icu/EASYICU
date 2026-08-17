"""Strict owner for the Figure 2 Dev9/Held-out27 experiment contract.

The module is evaluator-side only.  It validates experiment structure and
content identity; it is not imported by Planner, Coder, or any production
agent prompt.  Formal run authority is layered on top of this immutable bundle
after code, data, model, provider, image, and human-review coordinates exist.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
ACTION_SPACE_PATH = PACKAGE_ROOT / "action_space_v1.json"
EXPERIMENT_PROTOCOL_PATH = PACKAGE_ROOT / "experiment_protocol_v1.json"
HELDOUT_TASKBANK_PATH = PACKAGE_ROOT / "heldout27_taskbank_v1.jsonl"
QUALIFICATION_TASKBANK_PATH = (
    REPO_ROOT / "benchmarks/meta_generalization/meta_benchmark.jsonl"
)

ACTION_SPACE_RELATIVE_PATH = "benchmarks/figure2_icu_agent_v2/action_space_v1.json"
HELDOUT_TASKBANK_RELATIVE_PATH = (
    "benchmarks/figure2_icu_agent_v2/heldout27_taskbank_v1.jsonl"
)
QUALIFICATION_TASKBANK_RELATIVE_PATH = (
    "benchmarks/meta_generalization/meta_benchmark.jsonl"
)

DEV9_TASK_IDS = (
    "e1_sepsis3_prevalence_mortality",
    "e2_lactate_mortality",
    "e3_kdigo_gradient",
    "m1_hepatobiliary_missingness",
    "m2_mortality_prediction",
    "m3_sepsis_subphenotype",
    "h1_ventilation_survival",
    "h2_vasopressor_causal",
    "h3_trajectory_clustering",
)
QUALIFICATION12_TASK_IDS = tuple(f"MG{index:02d}" for index in range(1, 13))
HELDOUT27_TASK_IDS = tuple(f"icu27_t{index:02d}" for index in range(1, 28))
SCORING_DIMENSIONS = (
    "problem_formulation",
    "literature_grounding",
    "data_concept_cohort_authority",
    "estimand_method_selection",
    "execution_validity",
    "diagnostics_sensitivity",
    "evidence_artifact_binding",
    "interpretation_safety",
    "reproducibility_efficiency",
)
EXPECTED_DIFFICULTY_COUNTS = {"basic": 9, "intermediate": 9, "advanced": 9}
EXPECTED_DATABASE_COUNTS = {
    "miiv": 5,
    "mimic": 4,
    "eicu": 6,
    "aumc": 5,
    "hirid": 3,
    "sic": 4,
}
EXPECTED_FAMILY_COUNTS = {
    "descriptive": 5,
    "association": 6,
    "prediction": 4,
    "time_to_event": 4,
    "causal_emulation": 4,
    "phenotyping": 4,
}

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ReasonCode = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")]
TaskId = Annotated[str, Field(pattern=r"^icu27_t(?:0[1-9]|1[0-9]|2[0-7])$")]


class BenchmarkContractError(ValueError):
    """Typed, stable failure at the experiment-authority boundary."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ActionStage(_StrictFrozenModel):
    stage_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    order: int = Field(ge=1, le=32)
    owner: str = Field(min_length=3)
    public_contract: str = Field(min_length=3)
    success_artifact: str = Field(min_length=3)
    failure_reason_codes: tuple[ReasonCode, ...] = Field(min_length=1)


class ActionSpaceManifest(_StrictFrozenModel):
    schema_version: Literal["easyicu.icu_action_space/1"]
    action_space_ref: Literal["easyicu.figure2_icu_agent_v2/action-space-20260815"]
    stages: tuple[ActionStage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_stages(self) -> "ActionSpaceManifest":
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("action-space stage ids must be unique")
        if tuple(stage.order for stage in self.stages) != tuple(
            range(1, len(self.stages) + 1)
        ):
            raise ValueError("action-space order must be contiguous from one")
        return self


AnalysisFamily = Literal[
    "descriptive",
    "association",
    "prediction",
    "time_to_event",
    "causal_emulation",
    "phenotyping",
]
AnalysisType = Literal[
    "data_quality_audit",
    "measurement_bias_audit",
    "association_study",
    "prediction_model",
    "dynamic_prediction",
    "survival",
    "causal_inference",
    "trajectory_clustering",
]

_FAMILY_BY_ANALYSIS_TYPE: dict[str, str] = {
    "data_quality_audit": "descriptive",
    "measurement_bias_audit": "descriptive",
    "association_study": "association",
    "prediction_model": "prediction",
    "dynamic_prediction": "prediction",
    "survival": "time_to_event",
    "causal_inference": "causal_emulation",
    "trajectory_clustering": "phenotyping",
}


class HeldoutTask(_StrictFrozenModel):
    schema_version: Literal["easyicu.icu_heldout_task/1"]
    task_id: TaskId
    split: Literal["heldout27"]
    title: str = Field(min_length=8, max_length=180)
    question: str = Field(min_length=24, max_length=800)
    database: Literal["miiv", "mimic", "eicu", "aumc", "hirid", "sic"]
    analysis_type: AnalysisType
    analysis_family: AnalysisFamily
    difficulty: Literal["basic", "intermediate", "advanced"]
    expected_behavior: Literal["bound_result"]
    target_outcome: str = Field(min_length=2, max_length=160)
    exposure_or_index: str = Field(min_length=2, max_length=220)
    time_origin: str = Field(min_length=2, max_length=160)
    measurement_policy: str = Field(min_length=12, max_length=600)
    input_modules: tuple[str, ...] = Field(min_length=1)
    required_concepts: tuple[str, ...] = Field(min_length=2)
    expected_outputs: tuple[str, ...] = Field(min_length=3)
    semantic_guardrails: tuple[str, ...] = Field(min_length=2)
    required_stages: tuple[str, ...] = Field(min_length=1)
    agent_visibility: Literal["item_only_at_run"]
    human_review: Literal["clinical_and_methods"]
    paper_authority_before_freeze: Literal[False]

    @model_validator(mode="after")
    def _validate_family(self) -> "HeldoutTask":
        expected = _FAMILY_BY_ANALYSIS_TYPE[self.analysis_type]
        if self.analysis_family != expected:
            raise ValueError(
                f"analysis family mismatch for {self.task_id}: "
                f"{self.analysis_type} requires {expected}"
            )
        for field_name, values in (
            ("input_modules", self.input_modules),
            ("required_concepts", self.required_concepts),
            ("expected_outputs", self.expected_outputs),
            ("semantic_guardrails", self.semantic_guardrails),
            ("required_stages", self.required_stages),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} contains duplicates for {self.task_id}")
        return self


class SplitContract(_StrictFrozenModel):
    name: Literal["dev9", "qualification12", "heldout27"]
    purpose: Literal[
        "architecture_development",
        "nonpaper_generalization_qualification",
        "primary_formal_evaluation",
    ]
    task_count: int = Field(ge=1)
    task_ids: tuple[str, ...] = ()
    taskbank_path: str | None = None
    taskbank_sha256: Sha256 | None = None
    paper_authority: Literal["forbidden", "eligible_only_after_freeze"]


class FormalRunPolicy(_StrictFrozenModel):
    arms: tuple[Literal["aware"], ...]
    primary_runs_per_task: Literal[1]
    reuse_existing: Literal[False]
    resume: Literal[False]
    cross_run_memory: Literal[False]
    development_sample: Literal[False]
    posthoc_retry: Literal[False]
    failures_remain_in_denominator: Literal[True]
    execution_order: tuple[TaskId, ...]
    baseline_or_ablation_in_primary_batch: Literal[False]


class ExperimentProtocol(_StrictFrozenModel):
    schema_version: Literal["easyicu.icu_agent_experiment_protocol/1"]
    protocol_ref: Literal["easyicu.figure2_icu_agent_v2/20260815-v1"]
    action_space_path: str
    action_space_sha256: Sha256
    splits: tuple[SplitContract, SplitContract, SplitContract]
    scoring_dimensions: tuple[str, ...] = Field(min_length=5)
    contamination_firewall: tuple[str, ...] = Field(min_length=4)
    formal_run_policy: FormalRunPolicy

    @model_validator(mode="after")
    def _validate_protocol_shape(self) -> "ExperimentProtocol":
        if tuple(split.name for split in self.splits) != (
            "dev9",
            "qualification12",
            "heldout27",
        ):
            raise ValueError("experiment splits or their order drifted")
        dev, qualification, heldout = self.splits
        if dev.task_count != 9 or tuple(dev.task_ids) != DEV9_TASK_IDS:
            raise ValueError("dev9 task identity drifted")
        if qualification.task_count != 12:
            raise ValueError("qualification task count drifted")
        if tuple(qualification.task_ids) != QUALIFICATION12_TASK_IDS:
            raise ValueError("qualification task identity drifted")
        if heldout.task_count != 27 or tuple(heldout.task_ids) != HELDOUT27_TASK_IDS:
            raise ValueError("heldout27 task identity drifted")
        if set(dev.task_ids).intersection(heldout.task_ids):
            raise ValueError("development and held-out tasks overlap")
        if tuple(self.scoring_dimensions) != SCORING_DIMENSIONS:
            raise ValueError("scoring dimensions or their order drifted")
        if tuple(self.formal_run_policy.arms) != ("aware",):
            raise ValueError("formal primary batch requires exactly the aware arm")
        if tuple(self.formal_run_policy.execution_order) != HELDOUT27_TASK_IDS:
            raise ValueError("formal execution order drifted")
        if (
            dev.paper_authority != "forbidden"
            or qualification.paper_authority != "forbidden"
            or heldout.paper_authority != "eligible_only_after_freeze"
        ):
            raise ValueError("split paper-authority policy drifted")
        if heldout.taskbank_sha256 is None:
            raise ValueError("heldout27 taskbank digest is required")
        return self


@dataclass(frozen=True)
class HeldoutTaskbank:
    tasks: tuple[HeldoutTask, ...]
    sha256: str


@dataclass(frozen=True)
class ExperimentBundleReceipt:
    protocol_ref: str
    protocol_sha256: str
    action_space_ref: str
    action_space_sha256: str
    qualification_taskbank_sha256: str
    heldout_taskbank_sha256: str
    dev_task_count: int
    qualification_task_count: int
    heldout_task_count: int


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_loads(payload: bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _read_small_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BenchmarkContractError(
            "BENCHMARK_AUTHORITY_PATH_INVALID",
            f"authority path must not be a symlink: {path}",
        )
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise BenchmarkContractError(
            "BENCHMARK_AUTHORITY_PATH_INVALID",
            f"authority path is not a regular file: {path}",
        )
    size = resolved.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise BenchmarkContractError(
            "BENCHMARK_AUTHORITY_SIZE_INVALID",
            f"authority file size {size} is outside 1..{maximum_bytes}: {path}",
        )
    return resolved.read_bytes()


def load_action_space(path: Path = ACTION_SPACE_PATH) -> ActionSpaceManifest:
    try:
        raw = _read_small_regular_file(path, maximum_bytes=256_000)
        _strict_json_loads(raw)
        return ActionSpaceManifest.model_validate_json(raw, strict=True)
    except BenchmarkContractError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkContractError(
            "ACTION_SPACE_INVALID", f"unable to validate {path}: {exc}"
        ) from exc


def load_experiment_protocol(
    path: Path = EXPERIMENT_PROTOCOL_PATH,
) -> ExperimentProtocol:
    try:
        raw = _read_small_regular_file(path, maximum_bytes=512_000)
        _strict_json_loads(raw)
        return ExperimentProtocol.model_validate_json(raw, strict=True)
    except BenchmarkContractError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkContractError(
            "EXPERIMENT_PROTOCOL_INVALID", f"unable to validate {path}: {exc}"
        ) from exc


def load_heldout_taskbank(
    path: Path = HELDOUT_TASKBANK_PATH,
) -> HeldoutTaskbank:
    raw = _read_small_regular_file(path, maximum_bytes=2_000_000)
    tasks: list[HeldoutTask] = []
    line_number = 0
    try:
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith(b"#"):
                continue
            _strict_json_loads(line)
            tasks.append(HeldoutTask.model_validate_json(line, strict=True))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkContractError(
            "HELDOUT_TASKBANK_INVALID",
            f"taskbank row {line_number} is invalid: {exc}",
        ) from exc
    return HeldoutTaskbank(tasks=tuple(tasks), sha256=_sha256(raw))


def _assert_counter(
    *,
    field: str,
    actual: Counter[str],
    expected: dict[str, int],
    reason_code: str,
) -> None:
    if dict(actual) != expected:
        raise BenchmarkContractError(
            reason_code,
            f"{field} coverage drifted: actual={dict(actual)!r}, expected={expected!r}",
        )


def validate_experiment_bundle(
    *,
    protocol_path: Path = EXPERIMENT_PROTOCOL_PATH,
    action_space_path: Path = ACTION_SPACE_PATH,
    taskbank_path: Path = HELDOUT_TASKBANK_PATH,
) -> ExperimentBundleReceipt:
    protocol_raw = _read_small_regular_file(protocol_path, maximum_bytes=512_000)
    protocol = load_experiment_protocol(protocol_path)
    action_raw = _read_small_regular_file(action_space_path, maximum_bytes=256_000)
    action_space = load_action_space(action_space_path)
    taskbank = load_heldout_taskbank(taskbank_path)

    if protocol.action_space_path != ACTION_SPACE_RELATIVE_PATH:
        raise BenchmarkContractError(
            "ACTION_SPACE_PATH_DRIFT",
            f"protocol action-space path drifted: {protocol.action_space_path!r}",
        )
    qualification_split = protocol.splits[1]
    heldout_split = protocol.splits[2]
    if qualification_split.taskbank_path != QUALIFICATION_TASKBANK_RELATIVE_PATH:
        raise BenchmarkContractError(
            "QUALIFICATION_TASKBANK_PATH_DRIFT",
            "qualification taskbank path drifted",
        )
    if heldout_split.taskbank_path != HELDOUT_TASKBANK_RELATIVE_PATH:
        raise BenchmarkContractError(
            "HELDOUT_TASKBANK_PATH_DRIFT",
            "held-out taskbank path drifted",
        )

    if _sha256(action_raw) != protocol.action_space_sha256:
        raise BenchmarkContractError(
            "ACTION_SPACE_DIGEST_MISMATCH",
            "protocol action-space digest does not match the current bytes",
        )
    qualification_raw = _read_small_regular_file(
        QUALIFICATION_TASKBANK_PATH,
        maximum_bytes=1_000_000,
    )
    qualification_sha256 = _sha256(qualification_raw)
    if qualification_sha256 != qualification_split.taskbank_sha256:
        raise BenchmarkContractError(
            "QUALIFICATION_TASKBANK_DIGEST_MISMATCH",
            "protocol qualification taskbank digest does not match current bytes",
        )
    if taskbank.sha256 != heldout_split.taskbank_sha256:
        raise BenchmarkContractError(
            "HELDOUT_TASKBANK_DIGEST_MISMATCH",
            "protocol held-out taskbank digest does not match the current bytes",
        )

    task_ids = tuple(task.task_id for task in taskbank.tasks)
    if task_ids != HELDOUT27_TASK_IDS:
        raise BenchmarkContractError(
            "HELDOUT_TASK_SET_DRIFT",
            f"task ids/order drifted: {task_ids!r}",
        )
    if len({task.title for task in taskbank.tasks}) != len(taskbank.tasks):
        raise BenchmarkContractError(
            "HELDOUT_TASK_TITLE_DUPLICATED",
            "held-out task titles must be unique",
        )
    if len({task.question for task in taskbank.tasks}) != len(taskbank.tasks):
        raise BenchmarkContractError(
            "HELDOUT_TASK_QUESTION_DUPLICATED",
            "held-out task questions must be unique",
        )
    required_stages = tuple(stage.stage_id for stage in action_space.stages)
    for task in taskbank.tasks:
        if tuple(task.required_stages) != required_stages:
            raise BenchmarkContractError(
                "TASK_ACTION_SPACE_INCOMPLETE",
                f"{task.task_id} does not require the exact action-space sequence",
            )

    _assert_counter(
        field="difficulty",
        actual=Counter(task.difficulty for task in taskbank.tasks),
        expected=EXPECTED_DIFFICULTY_COUNTS,
        reason_code="HELDOUT_DIFFICULTY_COVERAGE_DRIFT",
    )
    _assert_counter(
        field="database",
        actual=Counter(task.database for task in taskbank.tasks),
        expected=EXPECTED_DATABASE_COUNTS,
        reason_code="HELDOUT_DATABASE_COVERAGE_DRIFT",
    )
    _assert_counter(
        field="analysis_family",
        actual=Counter(task.analysis_family for task in taskbank.tasks),
        expected=EXPECTED_FAMILY_COUNTS,
        reason_code="HELDOUT_FAMILY_COVERAGE_DRIFT",
    )
    if any(
        re.search(r"canonical9|figure\s*2", task.question, re.I)
        for task in taskbank.tasks
    ):
        raise BenchmarkContractError(
            "TASK_PROMPT_LEAKAGE_DETECTED",
            "held-out question contains benchmark or manuscript coordinates",
        )

    return ExperimentBundleReceipt(
        protocol_ref=protocol.protocol_ref,
        protocol_sha256=_sha256(protocol_raw),
        action_space_ref=action_space.action_space_ref,
        action_space_sha256=_sha256(action_raw),
        qualification_taskbank_sha256=qualification_sha256,
        heldout_taskbank_sha256=taskbank.sha256,
        dev_task_count=protocol.splits[0].task_count,
        qualification_task_count=protocol.splits[1].task_count,
        heldout_task_count=len(taskbank.tasks),
    )


__all__ = [
    "ACTION_SPACE_PATH",
    "DEV9_TASK_IDS",
    "EXPERIMENT_PROTOCOL_PATH",
    "HELDOUT27_TASK_IDS",
    "HELDOUT_TASKBANK_PATH",
    "QUALIFICATION_TASKBANK_PATH",
    "QUALIFICATION12_TASK_IDS",
    "SCORING_DIMENSIONS",
    "ActionSpaceManifest",
    "BenchmarkContractError",
    "ExperimentBundleReceipt",
    "ExperimentProtocol",
    "HeldoutTask",
    "HeldoutTaskbank",
    "load_action_space",
    "load_experiment_protocol",
    "load_heldout_taskbank",
    "validate_experiment_bundle",
]
