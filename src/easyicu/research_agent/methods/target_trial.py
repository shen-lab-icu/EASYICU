"""Target-trial emulation scaffolding: eligibility, time-zero, clone-censor-weight checklist.

A target trial emulation (Hernan and Robins, 2016) first writes the protocol of
the hypothetical randomised trial it wishes it could have run — who is eligible,
when follow-up starts (time zero), which strategies are compared — and only then
asks whether the observational data can emulate each protocol element. This
module provides deterministic *mechanical* checks for the first two elements
plus a review scaffold for the third. It is deliberately a skeleton, not a
causal inference engine:

1. :func:`reconcile_eligibility` — sequential denominator chain. Each
   inclusion/exclusion criterion is applied in declaration order; every step
   records its arrival denominator, marginal count, removals, and remainder so
   that ``n_start - n_excluded_total == n_analysis`` holds by construction
   (CONSORT-style flow, machine-checkable).
2. :func:`check_time_zero` — time-zero alignment. The comparison group must be
   knowable at time zero, no outcome may precede time zero (immortal-time
   guard), and any recorded treatment initiation must fall inside the declared
   grace period. Any violation raises :class:`TargetTrialError` (fail closed).
3. :func:`clone_censor_weight_checklist` — clone-censor-weight *review
   scaffold*. It outputs the mechanical bookkeeping a machine can compute
   (clone multiplicity, censoring-rule statement, weight-model specification
   slot) together with an explicitly labelled list of causal assumptions that
   require human confirmation. It never estimates a causal effect; the
   ``causal_effect_estimate`` field is always ``None``.

What this skeleton does NOT provide (scope boundary): the actual
clone-censor-weight estimator — per-interval cloning expansion, inverse
probability of censoring weights over the grace period, and their variance —
nor a marginal structural model (MSM) for fully time-varying regimes, nor a
dynamic treatment regime (DTR) optimiser. Those need a longitudinal g-formula /
sequential-IPCW machinery that is out of scope here; see
:mod:`easyicu.research_agent.methods.doubly_robust` for the documented gap.

Evidence ceiling: ``analysis_only``. Outputs from this module are descriptive
protocol checks and review aids. They cannot substitute for human confirmation
of the causal identification assumptions (consistency, conditional
exchangeability, positivity — see the checklist title
"需人工确认的因果假设清单"), and nothing here is reportable as a causal
finding.

References
----------
Hernan MA, Robins JM. "Using big data to emulate a target trial when a
randomized trial is not available." *Am J Epidemiol* 2016;183:758-764.
Hernan MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC, 2020,
Part III (target trial emulation, grace periods, cloning).

Pure pandas/numpy — no optional dependencies. Fully deterministic: no random
number use anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

#: Evidence ceiling for every product of this module.
EVIDENCE_CEILING = "analysis_only"

#: Exact label required for the manual causal-assumption list.
MANUAL_ASSUMPTIONS_TITLE = "需人工确认的因果假设清单"


class TargetTrialError(ValueError):
    """The target-trial scaffold refuses to proceed (fail closed)."""

    def __init__(self, message: str, *, code: str = "target_trial_invalid") -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Shared validation helpers.
# ---------------------------------------------------------------------------


def _as_bool_mask(values: Any, *, label: str, n: int) -> np.ndarray:
    """Coerce ``values`` to a length-``n`` boolean mask, fail closed.

    Only real booleans and exact 0/1 integers/floats are accepted. Anything
    else — strings, ``None``/``NaN``, other magnitudes, wrong length — raises
    :class:`TargetTrialError` instead of being silently coerced.
    """

    arr = np.asarray(values)
    if arr.shape != (n,):
        raise TargetTrialError(
            f"{label}: mask has shape {arr.shape}, expected ({n},)",
            code="target_trial_mask_length_mismatch",
        )
    if arr.dtype == bool:
        return arr.copy()
    if arr.dtype.kind in "iu":
        unique = set(np.unique(arr).tolist())
        if unique <= {0, 1}:
            return arr.astype(bool)
        raise TargetTrialError(
            f"{label}: integer mask must be 0/1, got values {sorted(unique)[:5]}",
            code="target_trial_mask_not_binary",
        )
    if arr.dtype.kind == "f":
        if not bool(np.isfinite(arr).all()):
            raise TargetTrialError(
                f"{label}: mask contains NaN or non-finite values",
                code="target_trial_mask_not_finite",
            )
        unique = set(np.unique(arr).tolist())
        if unique <= {0.0, 1.0}:
            return arr.astype(bool)
        raise TargetTrialError(
            f"{label}: float mask must be 0.0/1.0, got values {sorted(unique)[:5]}",
            code="target_trial_mask_not_binary",
        )
    raise TargetTrialError(
        f"{label}: mask must be boolean or 0/1, got dtype {arr.dtype}",
        code="target_trial_mask_not_binary",
    )


def _require_criteria(
    criteria: Mapping[str, Any] | None, *, label: str
) -> dict[str, Any]:
    if criteria is None:
        return {}
    if not isinstance(criteria, Mapping):
        raise TargetTrialError(
            f"{label} criteria must be a mapping of name -> boolean mask",
            code="target_trial_criteria_not_mapping",
        )
    names = [str(name) for name in criteria]
    if any(not name for name in names):
        raise TargetTrialError(
            f"{label} criterion names must be non-empty strings",
            code="target_trial_criterion_name_invalid",
        )
    if len(set(names)) != len(names):
        raise TargetTrialError(
            f"{label} criterion names must be unique",
            code="target_trial_criterion_name_invalid",
        )
    return {str(name): criteria[name] for name in criteria}


# ---------------------------------------------------------------------------
# 1. Eligibility reconciliation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibilityStep:
    """One row of the denominator chain."""

    name: str
    kind: str  # "inclusion" or "exclusion"
    n_before: int
    n_marginal: int  # mask count on the full starting cohort
    n_meeting: int  # arrivals meeting an inclusion / triggering an exclusion
    n_removed: int
    n_remaining: int


@dataclass(frozen=True)
class EligibilityReport:
    """Sequential eligibility accounting with a conservation invariant."""

    n_start: int
    steps: tuple[EligibilityStep, ...]
    n_excluded_total: int
    n_analysis: int
    analysis_positions: tuple[int, ...]
    evidence_ceiling: str = EVIDENCE_CEILING


def reconcile_eligibility(
    frame: pd.DataFrame,
    *,
    inclusions: Mapping[str, Any] | None = None,
    exclusions: Mapping[str, Any] | None = None,
) -> EligibilityReport:
    """Apply eligibility criteria sequentially and reconcile the denominator chain.

    Parameters
    ----------
    frame:
        Source cohort; only its row count and positional index are used.
    inclusions:
        Mapping of criterion name -> boolean mask (``True`` = satisfies the
        inclusion). Applied first, in declaration order; rows failing any
        inclusion are removed at that step.
    exclusions:
        Mapping of criterion name -> boolean mask (``True`` = triggers the
        exclusion). Applied after inclusions, in declaration order.

    Masks are positional (row ``i`` of each mask refers to row ``i`` of
    ``frame``). The chain invariant ``n_start - n_excluded_total ==
    n_analysis`` is verified before returning; an empty analysis set, unknown
    lengths, or non-binary masks raise :class:`TargetTrialError`.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TargetTrialError(
            "eligibility input must be a pandas DataFrame",
            code="target_trial_frame_invalid",
        )
    n = int(len(frame))
    if n == 0:
        raise TargetTrialError(
            "eligibility input has no rows",
            code="target_trial_frame_empty",
        )
    inclusion_map = _require_criteria(inclusions, label="inclusion")
    exclusion_map = _require_criteria(exclusions, label="exclusion")
    overlap = set(inclusion_map) & set(exclusion_map)
    if overlap:
        raise TargetTrialError(
            "criterion names overlap between inclusions and exclusions: "
            + ", ".join(sorted(overlap)),
            code="target_trial_criterion_name_invalid",
        )

    remaining = np.ones(n, dtype=bool)
    steps: list[EligibilityStep] = []

    def _apply(name: str, kind: str, mask: np.ndarray, marginal: int) -> None:
        nonlocal remaining
        n_before = int(remaining.sum())
        if kind == "inclusion":
            meeting = mask & remaining
            removed = remaining & ~mask
        else:
            meeting = mask & remaining
            removed = mask & remaining
        next_remaining = remaining & ~removed
        steps.append(
            EligibilityStep(
                name=name,
                kind=kind,
                n_before=n_before,
                n_marginal=marginal,
                n_meeting=int(meeting.sum()),
                n_removed=int(removed.sum()),
                n_remaining=int(next_remaining.sum()),
            )
        )
        remaining = next_remaining

    for name, raw in inclusion_map.items():
        mask = _as_bool_mask(raw, label=f"inclusion {name!r}", n=n)
        _apply(name, "inclusion", mask, int(mask.sum()))
    for name, raw in exclusion_map.items():
        mask = _as_bool_mask(raw, label=f"exclusion {name!r}", n=n)
        _apply(name, "exclusion", mask, int(mask.sum()))

    n_analysis = int(remaining.sum())
    n_excluded = int(n - n_analysis)
    if n_analysis == 0:
        raise TargetTrialError(
            "eligibility leaves an empty analysis set",
            code="target_trial_analysis_set_empty",
        )
    if n - n_excluded != n_analysis:  # structural guard, fail closed
        raise TargetTrialError(
            "eligibility denominator chain does not conserve",
            code="target_trial_chain_not_conserved",
        )
    positions = tuple(int(i) for i in np.flatnonzero(remaining).tolist())
    return EligibilityReport(
        n_start=n,
        steps=tuple(steps),
        n_excluded_total=n_excluded,
        n_analysis=n_analysis,
        analysis_positions=positions,
    )


# ---------------------------------------------------------------------------
# 2. Time-zero alignment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeZeroReport:
    """Clean time-zero alignment receipt (only returned when checks pass)."""

    n: int
    grace_period_days: float
    n_group_unknown_at_time_zero: int
    n_event_before_time_zero: int
    n_treatment_outside_grace: int
    n_analysis: int
    evidence_ceiling: str = EVIDENCE_CEILING


def check_time_zero(
    frame: pd.DataFrame,
    *,
    time_zero_col: str,
    event_time_col: str,
    group_col: str,
    group_known_at_time_zero: Any,
    treatment_time_col: str | None = None,
    grace_period_days: float = 0.0,
) -> TimeZeroReport:
    """Verify time-zero alignment, fail closed on any violation.

    Parameters
    ----------
    frame:
        One row per subject.
    time_zero_col / event_time_col:
        Numeric time columns; every ``event_time`` must be ``>= time_zero``
        (an earlier outcome implies immortal-time leakage).
    group_col:
        Comparison-group column; must be non-missing for every row.
    group_known_at_time_zero:
        Required boolean mask (``True`` = the row's group value was already
        knowable at its time zero). Any ``False`` raises — a strategy assigned
        on the basis of post-baseline information cannot emulate randomisation
        at time zero.
    treatment_time_col:
        Optional numeric column with the observed treatment-initiation time
        (``NaN`` = never initiated). Non-missing values must lie within
        ``[time_zero, time_zero + grace_period_days]``.
    grace_period_days:
        Allowed initiation window length; must be finite and ``>= 0``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TargetTrialError(
            "time-zero input must be a pandas DataFrame",
            code="target_trial_frame_invalid",
        )
    n = int(len(frame))
    if n == 0:
        raise TargetTrialError(
            "time-zero input has no rows",
            code="target_trial_frame_empty",
        )
    for col in (time_zero_col, event_time_col, group_col):
        if col not in frame.columns:
            raise TargetTrialError(
                f"time-zero input lacks column {col!r}",
                code="target_trial_column_missing",
            )
    if treatment_time_col is not None and treatment_time_col not in frame.columns:
        raise TargetTrialError(
            f"time-zero input lacks column {treatment_time_col!r}",
            code="target_trial_column_missing",
        )
    grace = float(grace_period_days)
    if not np.isfinite(grace) or grace < 0:
        raise TargetTrialError(
            "grace_period_days must be finite and >= 0",
            code="target_trial_grace_invalid",
        )

    time_zero = pd.to_numeric(frame[time_zero_col], errors="coerce").to_numpy(
        dtype=float
    )
    event_time = pd.to_numeric(frame[event_time_col], errors="coerce").to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(time_zero).all()):
        raise TargetTrialError(
            "time-zero values must be finite and complete",
            code="target_trial_time_zero_invalid",
        )
    if not bool(np.isfinite(event_time).all()):
        raise TargetTrialError(
            "event-time values must be finite and complete",
            code="target_trial_event_time_invalid",
        )
    if bool(frame[group_col].isna().any()):
        n_missing = int(frame[group_col].isna().sum())
        raise TargetTrialError(
            f"group column has {n_missing}/{n} missing values at time zero",
            code="target_trial_group_missing",
        )
    known = _as_bool_mask(
        group_known_at_time_zero, label="group_known_at_time_zero", n=n
    )
    n_unknown = int((~known).sum())
    if n_unknown:
        raise TargetTrialError(
            f"group not knowable at time zero for {n_unknown}/{n} rows",
            code="target_trial_group_not_known_at_time_zero",
        )
    n_early = int((event_time < time_zero).sum())
    if n_early:
        raise TargetTrialError(
            f"outcome precedes time zero for {n_early}/{n} rows (immortal-time risk)",
            code="target_trial_event_before_time_zero",
        )
    n_outside = 0
    if treatment_time_col is not None:
        treatment = pd.to_numeric(
            frame[treatment_time_col], errors="coerce"
        ).to_numpy(dtype=float)
        observed = np.isfinite(treatment)
        late = observed & (
            (treatment < time_zero) | (treatment > time_zero + grace)
        )
        n_outside = int(late.sum())
        if n_outside:
            raise TargetTrialError(
                f"treatment initiation outside the {grace}-day grace period "
                f"for {n_outside}/{n} rows",
                code="target_trial_treatment_outside_grace",
            )
    return TimeZeroReport(
        n=n,
        grace_period_days=grace,
        n_group_unknown_at_time_zero=0,
        n_event_before_time_zero=0,
        n_treatment_outside_grace=0,
        n_analysis=n,
    )


# ---------------------------------------------------------------------------
# 3. Clone-censor-weight review scaffold (no estimation).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloneCensorWeightChecklist:
    """Human-review sheet for a clone-censor-weight emulation.

    This is a scaffold, not an estimator: it records the mechanical facts a
    program can state deterministically and the causal assumptions only a human
    can confirm. ``causal_effect_estimate`` is always ``None`` — this module
    never issues a causal conclusion.
    """

    strategies: tuple[str, ...]
    grace_period_days: float
    n_analysis: int
    n_clones_mechanical: int
    censoring_rule_summary: str
    weight_model_spec: str
    manual_assumptions_title: str
    manual_assumptions: tuple[str, ...]
    machine_computable: tuple[str, ...]
    machine_gaps: tuple[str, ...]
    causal_effect_estimate: None = None
    evidence_ceiling: str = EVIDENCE_CEILING


_MANUAL_ASSUMPTIONS: tuple[str, ...] = (
    "需人工确认：一致性 — 观测结局等于所遵循策略下潜在结局，策略定义无多版本治疗含糊。",
    "需人工确认：基线条件可交换性 — 给定已测基线协变量，策略分配与潜在结局独立；无未测混杂需领域知识判断，机器无法验证。",
    "需人工确认：宽限期内的序贯可交换性 — 在 grace period 内依从/删失与潜在结局的关系无未测时变混杂。",
    "需人工确认：正性 — 每个符合条件的受试者在每个策略下都有非零的遵循概率；机器仅能报告可观测支持度，不能证明反事实支持。",
    "需人工确认：删失非信息性/IPCW 模型正确 — 人工删失（偏离策略即删失）的逆概率权重模型设定正确，权重分布经人工检查无极端值主导。",
    "需人工确认：无干扰 — 一人的策略不影响他人的潜在结局。",
    "需人工确认：time-zero/宽限期/克隆-删失规则与目标试验方案逐字对齐（eligibility 与 time-zero 检查的回执已附，人仍须确认方案本身正确）。",
    "需人工确认：结局在各策略下以相同方式、相同随访节律采集，无差异性测量或随访偏倚。",
    "需人工确认：缺失数据机制 — 基线与随访缺失的处理方式经人工确认，不默认随机缺失。",
)

_MACHINE_COMPUTABLE: tuple[str, ...] = (
    "克隆倍数 = 分析集人数 × 策略数（机械计数，无需估计）。",
    "删失规则文本与其触发条件的逐行计数（机械对账）。",
    "权重模型设定槽位记录（用何协变量、何模型、种子、截断规则），供人工复核。",
    "与 eligibility 分母链 / time-zero 回执的编号链接（机械可追溯性）。",
)

_MACHINE_GAPS: tuple[str, ...] = (
    "本次骨架不提供：逐区间克隆展开、grace period 内删失的逆概率权重估计与稳定化/截断诊断。",
    "本次骨架不提供：克隆数据的聚类/稳健方差（同一受试者的克隆行相关，朴素 SE 无效）。",
    "本次骨架不提供：时变 MSM 全估计器与纵向 g-formula（见 doubly_robust 模块注释的缺件清单）。",
    "本次骨架不提供：动态治疗方案（DTR）优化；仅覆盖预先声明的静态策略清单。",
)


def clone_censor_weight_checklist(
    *,
    strategies: Any,
    grace_period_days: float,
    n_analysis: int,
    censoring_rule_summary: str,
    weight_model_spec: str,
) -> CloneCensorWeightChecklist:
    """Build the human-review sheet for a clone-censor-weight emulation.

    All arguments describe the *intended* emulation; nothing is estimated.
    Invalid specifications (fewer than two strategies, negative grace,
    non-positive analysis size, empty rule text) raise
    :class:`TargetTrialError`.
    """

    names = [str(value) for value in (strategies if strategies is not None else [])]
    if len(names) < 2 or len(set(names)) != len(names) or any(not v for v in names):
        raise TargetTrialError(
            "clone-censor-weight strategies must list at least two unique "
            "non-empty strategy names",
            code="target_trial_strategies_invalid",
        )
    grace = float(grace_period_days)
    if not np.isfinite(grace) or grace < 0:
        raise TargetTrialError(
            "grace_period_days must be finite and >= 0",
            code="target_trial_grace_invalid",
        )
    size = int(n_analysis)
    if size <= 0:
        raise TargetTrialError(
            "n_analysis must be a positive integer",
            code="target_trial_analysis_size_invalid",
        )
    for label, text in (
        ("censoring_rule_summary", censoring_rule_summary),
        ("weight_model_spec", weight_model_spec),
    ):
        if not isinstance(text, str) or not text.strip():
            raise TargetTrialError(
                f"{label} must be a non-empty string",
                code="target_trial_review_text_invalid",
            )
    return CloneCensorWeightChecklist(
        strategies=tuple(names),
        grace_period_days=grace,
        n_analysis=size,
        n_clones_mechanical=size * len(names),
        censoring_rule_summary=censoring_rule_summary.strip(),
        weight_model_spec=weight_model_spec.strip(),
        manual_assumptions_title=MANUAL_ASSUMPTIONS_TITLE,
        manual_assumptions=_MANUAL_ASSUMPTIONS,
        machine_computable=_MACHINE_COMPUTABLE,
        machine_gaps=_MACHINE_GAPS,
    )


def render_review_sheet(checklist: CloneCensorWeightChecklist) -> str:
    """Render the checklist as plain review text for a human reader."""

    if not isinstance(checklist, CloneCensorWeightChecklist):
        raise TargetTrialError(
            "render_review_sheet expects a CloneCensorWeightChecklist",
            code="target_trial_review_input_invalid",
        )
    lines = [
        "目标试验模拟 clone-censor-weight 人工审查单",
        f"证据上限: {checklist.evidence_ceiling}（描述性审查辅助，不构成因果结论）",
        f"策略: {' / '.join(checklist.strategies)}",
        f"宽限期: {checklist.grace_period_days} 天；分析集 n={checklist.n_analysis}；"
        f"机械克隆行数={checklist.n_clones_mechanical}",
        f"删失规则: {checklist.censoring_rule_summary}",
        f"权重模型设定: {checklist.weight_model_spec}",
        "",
        checklist.manual_assumptions_title,
    ]
    lines.extend(f"{i + 1}. {item}" for i, item in enumerate(checklist.manual_assumptions))
    lines.append("")
    lines.append("机器可计算部分（已由本模块给出）")
    lines.extend(f"- {item}" for item in checklist.machine_computable)
    lines.append("")
    lines.append("机器不能代替部分（本次骨架缺件）")
    lines.extend(f"- {item}" for item in checklist.machine_gaps)
    lines.append("")
    lines.append("因果效应估计: 无（本模块不下因果结论，需人工确认上述假设后另行估计）。")
    return "\n".join(lines)


__all__ = [
    "EVIDENCE_CEILING",
    "MANUAL_ASSUMPTIONS_TITLE",
    "CloneCensorWeightChecklist",
    "EligibilityReport",
    "EligibilityStep",
    "TargetTrialError",
    "TimeZeroReport",
    "check_time_zero",
    "clone_censor_weight_checklist",
    "reconcile_eligibility",
    "render_review_sheet",
]


# ---------------------------------------------------------------------------
# 4. Given-regime valuation (Hajek IPW; NO regime optimisation).
# ---------------------------------------------------------------------------
#
# 多阶段优化明确不做：本节只对调用方传入的、固定的确定性策略做给定策略估值，
# 不学习、不比较、更不优化任何治疗规则（无 Q-learning / A-learning / OWL，
# 无多阶段回溯、无随机策略）。需要最优规则请另起 DTR 优化模块，本函数保持
# "给定策略估值"的诚实 scope。
#
# 识别假设（不可由数据验证，仅声明）：一致性；给定已测基线协变量的条件可交换
# 性（无未测混杂）；正性（每个受试者在观测治疗下有非零概率，本函数做机械筛查，
# 见下）。不支持纵向 g-formula、多阶段 DTR 优化、时变混杂的 pooled 拟合。
# 天花板 analysis_only：策略价值估计不是可报告的因果结论。


#: Hard propensity bounds for the regime-valuation PS fit. Fail closed.
#: Mirrors the doubly-robust kernel convention.
_REGIME_PS_HARD_EPS = 1e-6

#: Known limitations of the given-regime valuator, repeated on every result.
REGIME_LIMITATIONS: tuple[str, ...] = (
    "给定策略估值：只评估调用方传入的固定确定性策略；多阶段优化明确不做。",
    "不学习治疗规则：无 Q-learning / A-learning / OWL，无多阶段回溯。",
    "不支持纵向 g-formula、随机策略与时变混杂的 pooled 拟合。",
    "Hajek 归一化 IPW 依赖倾向模型设定正确；误设时价值估计有偏。",
)


@dataclass(frozen=True)
class DynamicRegimeResult:
    """Hajek IPW value of one fixed deterministic regime with an IF-based CI."""

    value: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float
    n: int
    n_adherent: int
    adherence_proportion: float
    ess: float
    weight_sum: float
    random_state: int
    ps_trim_low: float
    ps_trim_high: float
    trim_proportion: float
    ps_min: float
    ps_max: float
    ps_mean: float
    limitations: tuple[str, ...] = REGIME_LIMITATIONS
    evidence_ceiling: str = EVIDENCE_CEILING

    def to_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "n": self.n,
            "n_adherent": self.n_adherent,
            "adherence_proportion": self.adherence_proportion,
            "ess": self.ess,
            "weight_sum": self.weight_sum,
            "random_state": self.random_state,
            "ps_trim_low": self.ps_trim_low,
            "ps_trim_high": self.ps_trim_high,
            "trim_proportion": self.trim_proportion,
            "ps_min": self.ps_min,
            "ps_max": self.ps_max,
            "ps_mean": self.ps_mean,
            "limitations": list(self.limitations),
            "evidence_ceiling": self.evidence_ceiling,
        }


def _regime_binary_vector(values: Any, *, label: str) -> np.ndarray:
    """Strict 0/1 validation for observed and recommended treatments."""

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise TargetTrialError(
            f"{label} must be one-dimensional, got shape {arr.shape}",
            code="target_trial_regime_shape_mismatch",
        )
    flat = arr.ravel()
    if flat.shape[0] == 0:
        raise TargetTrialError(
            f"{label} is empty", code="target_trial_regime_empty_input"
        )
    try:
        numeric = flat.astype(float)
    except (TypeError, ValueError) as exc:
        raise TargetTrialError(
            f"{label} must be binary 0/1",
            code="target_trial_regime_not_binary",
        ) from exc
    if not bool(np.isfinite(numeric).all()):
        raise TargetTrialError(
            f"{label} must be finite and complete (no NaN/inf)",
            code="target_trial_regime_not_finite",
        )
    unique = set(np.unique(numeric).tolist())
    if unique - {0.0, 1.0}:
        raise TargetTrialError(
            f"{label} must be binary 0/1, got values {sorted(unique)[:5]}",
            code="target_trial_regime_not_binary",
        )
    return numeric.astype(float)


def evaluate_dynamic_regime(
    X: Any,
    A: Any,
    Y: Any,
    recommended: Any,
    *,
    random_state: int = 0,
    ps_trim: tuple[float, float] = (0.025, 0.975),
    min_adherence: float = 0.05,
    ci_level: float = 0.95,
) -> DynamicRegimeResult:
    """Value of one fixed deterministic regime by Hajek IPW (fail closed).

    Parameters
    ----------
    X:
        Baseline covariates, shape ``(n, k)`` (1-D input is treated as a
        single covariate). Must be finite and complete.
    A:
        Observed binary treatment vector (0/1).
    Y:
        Outcome vector (binary or continuous; must be finite and complete).
    recommended:
        Caller-provided per-subject recommended treatment vector (0/1, length
        ``n``). This is data, not a rule function: the kernel never calls
        back into caller code and never optimises a rule. 多阶段优化明确不做。
    random_state:
        Fixed seed for the ``sklearn`` logistic propensity fit.
    ps_trim:
        Acceptable propensity window. Any fitted score outside it (or at the
        hard bounds ``[1e-6, 1 - 1e-6]``) raises :class:`TargetTrialError`.
    min_adherence:
        Minimum required adherence proportion ``mean(A == recommended)`` in
        ``(0, 1]`` (default 0.05). Below it the value is unidentified from
        the adherent sliver and the kernel raises instead of reporting.
    ci_level:
        Wald interval level in ``(0, 1)``.

    The Hajek (self-normalised) estimator over adherent subjects with
    ``w_i = 1{A_i == rec_i} / pi_i`` (``pi_i`` the fitted treatment
    probability of the received treatment) is
    ``V_hat = sum(w_i Y_i) / sum(w_i)``. Its influence function is
    ``phi_i = w_i (Y_i - V_hat) / mean(w)``, giving
    ``SE = sd(phi, ddof=1) / sqrt(n)`` with a normal Wald interval — pure
    numpy from the fitted values, hence deterministic given ``random_state``.
    """

    from scipy import stats
    from sklearn.linear_model import LogisticRegression

    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TargetTrialError(
            "random_state must be an integer",
            code="target_trial_regime_seed_invalid",
        )
    seed = int(random_state)
    try:
        trim_low, trim_high = (float(ps_trim[0]), float(ps_trim[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise TargetTrialError(
            "ps_trim must be a (low, high) pair",
            code="target_trial_regime_trim_bounds_invalid",
        ) from exc
    if not (0.0 < trim_low < trim_high < 1.0):
        raise TargetTrialError(
            f"ps_trim must satisfy 0 < low < high < 1, got ({trim_low}, {trim_high})",
            code="target_trial_regime_trim_bounds_invalid",
        )
    adherence_floor = float(min_adherence)
    if not 0.0 < adherence_floor <= 1.0:
        raise TargetTrialError(
            f"min_adherence must be in (0, 1], got {min_adherence!r}",
            code="target_trial_regime_adherence_floor_invalid",
        )
    if isinstance(ci_level, bool) or not isinstance(
        ci_level, (int, float, np.floating)
    ):
        raise TargetTrialError(
            "ci_level must be a number in (0, 1)",
            code="target_trial_regime_ci_level_invalid",
        )
    level = float(ci_level)
    if not 0.0 < level < 1.0:
        raise TargetTrialError(
            f"ci_level must be in (0, 1), got {ci_level!r}",
            code="target_trial_regime_ci_level_invalid",
        )

    a_vec = _regime_binary_vector(A, label="observed treatment")
    rec_vec = _regime_binary_vector(recommended, label="recommended treatment")
    try:
        y_vec = np.asarray(Y, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise TargetTrialError(
            "outcome must be numeric", code="target_trial_regime_outcome_invalid"
        ) from exc
    n = int(a_vec.shape[0])
    if rec_vec.shape[0] != n or y_vec.shape[0] != n:
        raise TargetTrialError(
            "observed treatment, recommended treatment and outcome must have "
            "equal length",
            code="target_trial_regime_shape_mismatch",
        )
    if not bool(np.isfinite(y_vec).all()):
        raise TargetTrialError(
            "outcome must be finite and complete (no NaN/inf)",
            code="target_trial_regime_not_finite",
        )
    x_mat = np.asarray(X, dtype=float)
    if x_mat.ndim == 1:
        x_mat = x_mat.reshape(-1, 1)
    if x_mat.ndim != 2 or x_mat.shape[0] != n or x_mat.shape[1] < 1:
        raise TargetTrialError(
            f"covariates must have shape ({n}, k>=1), got shape {x_mat.shape}",
            code="target_trial_regime_shape_mismatch",
        )
    if not bool(np.isfinite(x_mat).all()):
        raise TargetTrialError(
            "covariates must be finite and complete (no NaN/inf)",
            code="target_trial_regime_not_finite",
        )
    if int((a_vec == 1.0).sum()) == 0 or int((a_vec == 0.0).sum()) == 0:
        raise TargetTrialError(
            "observed treatment must vary (one arm is empty)",
            code="target_trial_regime_single_arm",
        )

    adherent = a_vec == rec_vec
    n_adherent = int(adherent.sum())
    adherence = float(n_adherent / n)
    if adherence < adherence_floor:
        raise TargetTrialError(
            f"regime adherence {adherence:.4f} ({n_adherent}/{n}) is below the "
            f"floor {adherence_floor:.4f}; the regime value is unidentified "
            "from this sliver — refusing to report",
            code="target_trial_regime_low_adherence",
        )

    ps_model = LogisticRegression(max_iter=5000, random_state=seed)
    try:
        ps_model.fit(x_mat, a_vec)
    except Exception as exc:
        raise TargetTrialError(
            f"regime propensity fit failed: {exc}",
            code="target_trial_regime_propensity_fit_failed",
        ) from exc
    if int(np.max(np.atleast_1d(ps_model.n_iter_))) >= 5000:
        raise TargetTrialError(
            "regime propensity fit did not converge; refusing to report",
            code="target_trial_regime_propensity_not_converged",
        )
    ps = np.asarray(ps_model.predict_proba(x_mat)[:, 1], dtype=float)
    if not bool(np.isfinite(ps).all()):
        raise TargetTrialError(
            "regime propensity fit produced non-finite scores",
            code="target_trial_regime_propensity_not_finite",
        )
    hard = (ps <= _REGIME_PS_HARD_EPS) | (ps >= 1.0 - _REGIME_PS_HARD_EPS)
    outside = (ps < trim_low) | (ps > trim_high)
    trim_proportion = float(outside.mean())
    if bool(hard.any()):
        raise TargetTrialError(
            f"positivity violated: {int(hard.sum())}/{n} propensity scores at "
            f"or beyond [{_REGIME_PS_HARD_EPS}, {1.0 - _REGIME_PS_HARD_EPS}] "
            f"(trim-window proportion {trim_proportion:.4f}); refusing to estimate",
            code="target_trial_regime_propensity_out_of_bounds",
        )
    if bool(outside.any()):
        raise TargetTrialError(
            f"positivity violated: {int(outside.sum())}/{n} propensity scores "
            f"outside trim window [{trim_low}, {trim_high}] "
            f"(trim proportion {trim_proportion:.4f}); refusing to trim silently",
            code="target_trial_regime_positivity_trim_triggered",
        )

    pi_received = np.where(a_vec == 1.0, ps, 1.0 - ps)
    weights = np.where(adherent, 1.0 / pi_received, 0.0)
    if not bool(np.isfinite(weights).all()):
        raise TargetTrialError(
            "regime weights are non-finite",
            code="target_trial_regime_weights_not_finite",
        )
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise TargetTrialError(
            "regime weights sum to zero; value is undefined",
            code="target_trial_regime_weights_degenerate",
        )
    value = float(weights @ y_vec / weight_sum)
    mean_w = float(np.mean(weights))
    influence = weights * (y_vec - value) / mean_w
    se = float(np.std(influence, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    weight_sq = float(weights @ weights)
    ess = (weight_sum**2) / weight_sq if weight_sq > 0.0 else 0.0
    return DynamicRegimeResult(
        value=value,
        se=se,
        ci_low=value - z * se,
        ci_high=value + z * se,
        ci_level=level,
        n=n,
        n_adherent=n_adherent,
        adherence_proportion=adherence,
        ess=float(ess),
        weight_sum=weight_sum,
        random_state=seed,
        ps_trim_low=trim_low,
        ps_trim_high=trim_high,
        trim_proportion=0.0,
        ps_min=float(np.min(ps)),
        ps_max=float(np.max(ps)),
        ps_mean=float(np.mean(ps)),
    )
