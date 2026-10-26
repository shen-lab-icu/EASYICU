"""Bounded adapter over the existing population and descriptive kernels."""

from __future__ import annotations

import math

import pandas as pd
import pyarrow.parquet as pq

from easyicu.research_agent.cohort.schema import build_cohort
from easyicu.research_agent.execution.runners.cohort_summary_executor import (
    cohort_summary_table, descriptive_partitions,
)
from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input
from easyicu.research_agent.methods.table_one import build_grouped_table_one
from easyicu.research_agent.planning.cohort_eligibility import predicate_context_column
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.research_context.outbound import SUPPRESSED_COUNT, project_outbound_counts
from easyicu.research_agent.research_context.stay_events import whole_stay_event_columns
from easyicu.research_agent.schema import TableOneSpec, TableOneVariableSpec

from .contracts import QuickStop
from .planning import population_receipt


def _load(source, plan):
    context = source.context.model_copy(update={"research_question": plan.reading.question})
    population = compile_population(plan.effective_reading.population, context)
    if population_receipt(population) != plan.population:
        raise QuickStop("quick_population_changed")
    definition = population.cohort_definition()
    variables = {v.name: v for v in context.variables}
    bindings = {predicate.concept_id: predicate_context_column(
        variables, predicate.concept_id, predicate.aggregation,
    ) for predicate in (*definition.inclusion, *definition.exclusion)}
    columns = sorted({name for name in (plan.measure_column, plan.group_column, source.unit_column,
                                       source.patient_column, *bindings.values()) if name})
    # Inspect Parquet metadata before allocating a frame. This is a quick path
    # on prepared data, not an extraction or an unbounded data scan.
    with source.path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        metadata = parquet.metadata
        selected = {parquet.schema.names.index(name) for name in columns if name in parquet.schema.names}
        if len(selected) != len(columns):
            raise QuickStop("quick_prepared_columns_missing")
        size = sum(metadata.row_group(i).column(j).total_uncompressed_size
                   for i in range(metadata.num_row_groups) for j in selected)
        if metadata.num_rows > 2_000_000 or size > 256 * 1024 * 1024:
            raise QuickStop("quick_prepared_source_too_large")
        frame = parquet.read(columns=columns).to_pandas()
    if len(frame) != context.cohort.n_stays:
        raise QuickStop("quick_source_metadata_mismatch")
    if frame[source.unit_column].isna().any() or frame[source.unit_column].duplicated().any():
        raise QuickStop("quick_unit_identity_invalid")
    frame = build_cohort(definition, frame, column_bindings=bindings,
                         whole_stay_columns=tuple(whole_stay_event_columns(context)))
    if frame.empty:
        raise QuickStop("quick_population_empty")
    if plan.test and (frame[source.patient_column].isna().any() or frame[source.patient_column].duplicated().any()):
        raise QuickStop("quick_repeated_units_require_full_study")
    for name, levels in ((plan.measure_column, plan.measure_levels), (plan.group_column, plan.group_levels)):
        if name is None:
            continue
        if levels:
            if (frame[name].notna() & ~frame[name].isin(levels)).any():
                raise QuickStop("quick_levels_changed")
        else:
            frame[name] = strict_numeric_input(frame[name], name=name).values
    return frame, variables


def _disclose_counts(table):
    # Withhold the whole result if *any* cell is suppressed, including unused
    # levels and missing groups. Reporting totals/percentages alongside a
    # suppressed count would let the conversation reconstruct that count.
    for row in table.to_dict("records"):
        for key in ("numerator", "denominator"):
            value = row.get(key)
            if pd.notna(value) and project_outbound_counts(value, key="count") == SUPPRESSED_COUNT:
                raise QuickStop("quick_small_cells_suppressed")


def _cell(row, statistic, value, *, unit=""):
    if value is None or not math.isfinite(float(value)):
        raise QuickStop("quick_insufficient_observations")
    number = float(value)
    return {
        "statistic": statistic, "variable": str(row.get("variable") or "population"),
        "group": str(row.get("group") or "Overall"),
        "level": "" if pd.isna(row.get("level")) else str(row["level"]),
        "unit": unit, "value": int(number) if number.is_integer() else number,
    }


def descriptive_cells(source, plan):
    """All estimates come from the existing kernels; this code only selects rows."""
    frame, variables = _load(source, plan)
    columns = [plan.measure_column] if plan.measure_column else []
    metadata = {name: variable.model_dump(mode="json") for name, variable in variables.items()}
    summaries = [cohort_summary_table(partition, columns, metadata).assign(group=group)
                 for group, _role, partition in descriptive_partitions(frame, plan.group_column, plan.group_levels)]
    table = pd.concat(summaries, ignore_index=True)
    _disclose_counts(table)
    measure = plan.effective_reading.measure
    cells = []
    for row in table.to_dict("records"):
        statistic = row["statistic"]
        if statistic in {"cohort_n", "nonmissing_n", "missing_n"}:
            cells.append(_cell(row, statistic, row["value"]))
        elif statistic == "level_count" and (measure.level is None or row["level"] == measure.level):
            cells.append(_cell(row, "numerator", row["numerator"]))
            cells.append(_cell(row, "denominator", row["denominator"]))
            if measure.statistic == "proportion":
                cells.append(_cell(row, "percentage", row["percentage"], unit="%"))
        elif statistic in ({"q1", "q3"} if measure.statistic == "iqr" else {measure.statistic}):
            cells.append(_cell(row, statistic, row["value"], unit=row.get("unit") or ""))
    if plan.test:
        variable = TableOneVariableSpec(
            name=plan.measure_column, variable_kind="categorical" if plan.measure_levels else "continuous",
            levels=list(plan.measure_levels), test=plan.test,
            summary="count_percent" if plan.measure_levels else "mean_sd" if measure.statistic == "mean" else "median_iqr",
        )
        comparison = build_grouped_table_one(frame, TableOneSpec(
            group_by=plan.group_column, group_levels=list(plan.group_levels), variables=[variable],
            missing_group_policy="fail_closed",
        ))
        first = comparison.iloc[0]
        cells.append(_cell({"variable": plan.measure_column, "group": "Comparison"},
                           "p_value:" + first["test_name"], first["p_value"]))
    if len(cells) > 160:
        raise QuickStop("quick_answer_too_large")
    return cells
