"""A named measurement operation is not a table-summary preference."""

from __future__ import annotations

import pytest

from easyicu.webserver import study_intent


@pytest.mark.parametrize(
    ("question", "concept", "aggregation"),
    [
        ("成人 ICU 患者前24小时最高乳酸与院内死亡", "lact", "max"),
        ("前24小时乳酸的最高值与院内死亡", "lact", "max"),
        ("Peak serum lactate in the first 24 hours and hospital mortality", "lact", "max"),
        ("The minimum creatinine in the first day and mortality", "crea", "min"),
        ("研究首日最低平均动脉压与死亡", "map", "min"),
        ("研究第一天平均心率与死亡", "hr", "mean"),
        ("First bilirubin and hospital mortality", "bili", "first"),
        ("研究乳酸中位数与院内死亡", "lact", "median"),
        ("研究末次乳酸与院内死亡", "lact", "last"),
        ("Cumulative urine output and mortality", "urine", "sum"),
    ],
)
def test_explicit_measurement_operation_has_exact_text_provenance(
    question: str, concept: str, aggregation: str,
) -> None:
    result = study_intent.explicit_exposure_aggregation(question, concept_id=concept)
    assert result is not None
    assert result.concept_id == concept
    assert result.aggregation == aggregation
    assert result.evidence in question


@pytest.mark.parametrize(
    ("question", "concept"),
    [
        ("研究乳酸与院内死亡，表格报告中位数", "lact"),
        ("Study mean arterial pressure and mortality", "map"),
        ("研究平均动脉压与死亡", "map"),
        ("研究峰值乳酸与最低肌酐的关系", "hr"),
        ("研究乳酸最大值与乳酸最小值的差异", "lact"),
        ("Do not use peak lactate; study lactate and death", "lact"),
        ("不研究最高乳酸；研究乳酸与死亡", "lact"),
        ("研究血乳酸趋势与死亡", "lact"),
        ("研究所有指标的最高值与死亡", "lact"),
    ],
)
def test_ambiguous_absent_or_other_variable_operations_remain_unbound(
    question: str, concept: str,
) -> None:
    assert study_intent.explicit_exposure_aggregation(question, concept_id=concept) is None


def test_operations_bind_each_named_concept_without_cross_contamination() -> None:
    text = "研究最高乳酸与最低肌酐的关系"
    lactate = study_intent.explicit_exposure_aggregation(text, concept_id="lact")
    creatinine = study_intent.explicit_exposure_aggregation(text, concept_id="crea")
    assert lactate is not None and lactate.aggregation == "max"
    assert creatinine is not None and creatinine.aggregation == "min"


def test_metadata_planning_coordinates_preserve_named_operation() -> None:
    from easyicu.webserver.research_launch_scientific import (
        _metadata_only_planning_coordinates,
    )

    coordinates = _metadata_only_planning_coordinates(
        question="研究成人 ICU 患者前24小时最高乳酸与院内死亡", database="miiv",
    )
    assert coordinates["primary_exposure"] == "lact"
    assert coordinates["primary_exposure_aggregation"] == "max"
