"""Patient Review eligibility-flow metadata helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from easyicu.webserver import dataio


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _eligibility_flow_payload(
    path: Path, desc: Dict[str, Any], summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Return cohort attrition metadata without reading patient rows."""
    manifest = dataio._read_export_manifest(path)
    report = manifest.get("cohort_report") if isinstance(manifest, dict) else None
    contract = manifest.get("cohort_contract") if isinstance(manifest, dict) else None
    report = report if isinstance(report, dict) else {}
    contract = contract if isinstance(contract, dict) else {}
    final_count = _first_int(
        report.get("selected"),
        report.get("cohort_size"),
        summary.get("entities"),
        (desc.get("summary") or {}).get("stays"),
        manifest.get("patient_count") if isinstance(manifest, dict) else None,
    )
    source_total = _first_int(
        report.get("source_total"),
        report.get("initial"),
        manifest.get("source_total") if isinstance(manifest, dict) else None,
    )
    if source_total is None:
        source_total = final_count

    steps: List[Dict[str, Any]] = []

    def add_step(
        step_id: str,
        label_en: str,
        label_zh: str,
        count: Any,
        *,
        note_en: str = "",
        note_zh: str = "",
        basis: str = "manifest",
        final: bool = False,
    ) -> None:
        parsed = _int_or_none(count)
        if parsed is None:
            return
        if (
            steps
            and steps[-1].get("id") == step_id
            and steps[-1].get("count") == parsed
        ):
            return
        previous = _int_or_none(steps[-1].get("count")) if steps else None
        excluded = (
            max(0, previous - parsed)
            if previous is not None and parsed is not None
            else None
        )
        denominator = _int_or_none(source_total) or parsed
        pct = round(parsed / denominator * 100, 1) if denominator else None
        excluded_pct = (
            round(excluded / previous * 100, 1)
            if excluded is not None and previous
            else None
        )
        steps.append(
            {
                "id": step_id,
                "label": label_en,
                "label_i18n": {"en": label_en, "zh": label_zh},
                "count": parsed,
                "denominator": denominator,
                "pct_of_initial": pct,
                "excluded": excluded,
                "excluded_pct_of_previous": excluded_pct,
                "note": note_en,
                "note_i18n": {"en": note_en, "zh": note_zh},
                "basis": basis,
                "final": bool(final),
            }
        )

    has_report = bool(report)
    add_step(
        "source_total",
        "All ICU stays" if has_report else "All exported ICU stays",
        "全部 ICU 住院" if has_report else "全部已导出 ICU 住院",
        source_total,
        note_en=(
            "from source cohort before EasyICU filters"
            if has_report
            else "no stepwise filter log in this legacy export"
        ),
        note_zh=(
            "EasyICU 筛选前的来源队列" if has_report else "这个旧导出没有逐步筛选日志"
        ),
        basis="cohort_report" if has_report else "export_summary",
    )

    demo_count = _first_int(report.get("selected_before_concept_prefilter"))
    if demo_count is not None:
        add_step(
            "demographic_stay_filters",
            _demographic_flow_label(contract, "en"),
            _demographic_flow_label(contract, "zh"),
            demo_count,
            note_en=_demographic_flow_note(contract, "en"),
            note_zh=_demographic_flow_note(contract, "zh"),
            basis="cohort_report",
        )

    concept_count = _first_int(report.get("concept_matches"))
    if concept_count is not None:
        add_step(
            "concept_prefilter",
            _target_clinical_flow_label(contract, report, "en"),
            _target_clinical_flow_label(contract, report, "zh"),
            concept_count,
            note_en=_target_clinical_flow_note(contract, report, "en"),
            note_zh=_target_clinical_flow_note(contract, report, "zh"),
            basis="cohort_report",
        )

    icd = report.get("icd") if isinstance(report.get("icd"), dict) else {}
    if icd.get("enabled"):
        icd_count = _first_int(
            report.get("selected_before_cap"), report.get("selected")
        )
        include_count = len(icd.get("include_tokens") or [])
        exclude_count = len(icd.get("exclude_tokens") or [])
        add_step(
            "icd_filters",
            "ICD include / exclude",
            "ICD 纳入 / 排除",
            icd_count,
            note_en=f"{include_count} include · {exclude_count} exclude tokens",
            note_zh=f"{include_count} 个纳入 · {exclude_count} 个排除条件",
            basis="cohort_report",
        )

    selected_before_cap = _first_int(report.get("selected_before_cap"))
    if (
        selected_before_cap is not None
        and final_count is not None
        and selected_before_cap != final_count
    ):
        add_step(
            "max_patient_cap",
            "Runtime sample cap",
            "运行样本上限",
            final_count,
            note_en="max_patients applied",
            note_zh="已应用 max_patients 上限",
            basis="cohort_report",
        )

    if has_report or not steps or _int_or_none(steps[-1].get("count")) != final_count:
        add_step(
            "final_cohort",
            "Final cohort",
            "最终队列",
            final_count,
            note_en=(
                "analysis-ready exported denominator"
                if has_report
                else "current Patient Review denominator"
            ),
            note_zh=("可分析导出分母" if has_report else "当前患者审阅分母"),
            basis="cohort_report" if has_report else "export_summary",
            final=True,
        )
    elif steps:
        steps[-1]["id"] = "final_cohort"
        steps[-1]["label"] = "Final cohort"
        steps[-1]["label_i18n"] = {"en": "Final cohort", "zh": "最终队列"}
        steps[-1]["note"] = "current Patient Review denominator"
        steps[-1]["note_i18n"] = {
            "en": "current Patient Review denominator",
            "zh": "当前患者审阅分母",
        }
        steps[-1]["final"] = True
    if len(steps) == 1:
        steps[0]["final"] = True

    return {
        "title": "Eligibility flow (ICU stays)",
        "title_i18n": {
            "en": "Eligibility flow (ICU stays)",
            "zh": "入组筛选流程（ICU 住院）",
        },
        "steps": steps,
        "initial_count": steps[0]["count"] if steps else None,
        "final_count": final_count,
        "has_stepwise_report": has_report,
        "payload_scope": "cohort_attrition_metadata_only",
        "privacy": {
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
        },
    }


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _demographic_flow_label(contract: Dict[str, Any], lang: str) -> str:
    preset = str(contract.get("preset") or "").strip()
    age_min = _int_or_none(contract.get("age_min"))
    age_max = _int_or_none(contract.get("age_max"))
    min_los = _int_or_none(contract.get("min_icu_los_hours"))
    first_stay = bool(contract.get("exclude_readmissions"))
    if lang == "zh":
        if age_min is not None and age_max is not None and min_los:
            return f"年龄 {age_min}-{age_max} 岁 + ICU ≥ {min_los} 小时"
        if age_min is not None and age_max is not None:
            return f"年龄 {age_min}-{age_max} 岁"
        if min_los:
            return f"ICU ≥ {min_los} 小时"
        return "人口学 / 住院筛选" if first_stay or preset else "来源队列确认"
    if age_min is not None and age_max is not None and min_los:
        return f"Age {age_min}-{age_max} + ICU stay >= {min_los}h"
    if age_min is not None and age_max is not None:
        return f"Age {age_min}-{age_max} years"
    if min_los:
        return f"ICU stay >= {min_los}h"
    return (
        "Demographic / stay filters"
        if first_stay or preset
        else "Source cohort confirmed"
    )


def _demographic_flow_note(contract: Dict[str, Any], lang: str) -> str:
    pieces: List[str] = []
    if contract.get("exclude_readmissions"):
        pieces.append("first ICU stay" if lang == "en" else "首次 ICU")
    min_los = _int_or_none(contract.get("min_icu_los_hours"))
    if min_los:
        pieces.append(
            f"LOS >= {min_los}h" if lang == "en" else f"住院时长 >= {min_los} 小时"
        )
    return " · ".join(pieces)


def _target_clinical_flow_preset(
    contract: Dict[str, Any], report: Dict[str, Any]
) -> str:
    return str(contract.get("preset") or report.get("mode") or "").strip().lower()


def _target_clinical_flow_label(
    contract: Dict[str, Any], report: Dict[str, Any], lang: str
) -> str:
    preset = _target_clinical_flow_preset(contract, report)
    if lang == "zh":
        labels = {
            "sepsis3": "Sepsis-3 脓毒症队列",
            "aki": "AKI 目标队列",
            "ventilation": "机械通气目标队列",
            "vasopressor": "血管活性药物暴露队列",
            "respiratory": "呼吸支持目标队列",
            "icd": "ICD 定义目标队列",
        }
        return labels.get(preset, "目标临床队列")
    labels = {
        "sepsis3": "Sepsis-3 cohort",
        "aki": "AKI target cohort",
        "ventilation": "Mechanical ventilation cohort",
        "vasopressor": "Vasopressor exposure cohort",
        "respiratory": "Respiratory support cohort",
        "icd": "ICD-defined target cohort",
    }
    return labels.get(preset, "Target clinical cohort")


def _target_clinical_flow_note(
    contract: Dict[str, Any], report: Dict[str, Any], lang: str
) -> str:
    preset = _target_clinical_flow_preset(contract, report)
    if lang == "zh":
        notes = {
            "sepsis3": "疑似感染 + SOFA 信号",
            "aki": "AKI 规则阳性",
            "ventilation": "机械通气概念阳性",
            "vasopressor": "血管活性药物暴露阳性",
            "respiratory": "呼吸支持或氧合异常信号",
            "icd": "诊断编码规则",
        }
        note = notes.get(preset, "概念规则阳性")
        window = _int_or_none(contract.get("observation_window_hours"))
        return f"{note} · 前 {window} 小时窗口" if window else note
    notes = {
        "sepsis3": "suspected infection + SOFA signal",
        "aki": "AKI rule-positive",
        "ventilation": "mechanical ventilation concept-positive",
        "vasopressor": "vasopressor exposure-positive",
        "respiratory": "respiratory support or oxygenation signal",
        "icd": "diagnosis-code rule",
    }
    note = notes.get(preset, "concept rule-positive")
    window = _int_or_none(contract.get("observation_window_hours"))
    return f"{note} · first {window}h window" if window else note


__all__ = ["_eligibility_flow_payload", "_int_or_none"]
