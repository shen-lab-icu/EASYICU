"""Aggregate-only native-v2 screening; policy is owned by the administrator tool.

Parquet is scanned by one resource-limited DuckDB connection. No patient rows,
IDs, paths or extreme-stay listings leave this owner. Only scalar aggregate
queries are fetched into Python. Exact distinct counts and bounded-memory
approximate quantiles avoid materializing full exports in pandas.
"""

from __future__ import annotations

from easyicu.runtime import duckdb_spill
import json
from pathlib import Path
from typing import Any


from easyicu.concept.catalog import CONCEPT_DICTIONARY, CONCEPT_GROUPS_INTERNAL
from easyicu.concept.export_metadata import metadata_definition_for_concept
from easyicu.resources import load_dictionary
from easyicu.scores.sofa2_aggregate import SOFA2_COMPONENT_NAMES


KEY_CONCEPTS = {
    "hr": ("hr",),
    "map": ("map",),
    "lactate": ("lact",),
    "creatinine": ("crea",),
    "platelets": ("plt",),
    "bilirubin": ("bili",),
    "GCS": ("gcs",),
    "vasopressors": tuple(CONCEPT_GROUPS_INTERNAL["vasopressors"]),
}
SCORE_COMPONENTS = {
    "sofa": tuple(name.replace("sofa2_", "sofa_") for name in SOFA2_COMPONENT_NAMES),
    "sofa2": SOFA2_COMPONENT_NAMES,
    "qsofa": ("gcs_comp", "sbp_comp", "resp_comp"),
    "sirs": ("temp_comp", "hr_comp", "resp_comp", "wbc_comp"),
    "news": (
        "resp_comp",
        "o2sat_comp",
        "temp_comp",
        "sbp_comp",
        "hr_comp",
        "avpu_comp",
        "supp_o2_comp",
    ),
}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _numeric(column: str) -> str:
    return f"TRY_CAST({_q(column)} AS DOUBLE)"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _worst(*verdicts: str) -> str:
    return max(verdicts, key={"pass": 0, "warn": 1, "fail": 2}.__getitem__)


def _check(metrics: dict, policy: dict, value: float | None, **expectation) -> dict:
    verdict = "pass"
    if value is None:
        verdict = "warn"
    elif "fail_above" in policy and value > policy["fail_above"]:
        verdict = "fail"
    elif ("warn_above" in policy and value > policy["warn_above"]) or (
        "warn_below" in policy and value < policy["warn_below"]
    ):
        verdict = "warn"
    return {**metrics, "expectation": {**policy, **expectation}, "verdict": verdict}


def _one(connection, query: str) -> tuple:
    """Fetch one aggregate row. Callers never select an identity or raw value."""
    return connection.execute(query).fetchone()


def _stats(
    connection,
    source: str,
    column: str,
    minimum=None,
    maximum=None,
    *,
    integer: bool = False,
) -> dict:
    value = _numeric(column)
    conditions = []
    if minimum is not None:
        conditions.append(f"v < {float(minimum)}")
    if maximum is not None:
        conditions.append(f"v > {float(maximum)}")
    if integer:
        conditions.append("v <> trunc(v)")
    outside = " OR ".join(conditions) or "false"
    row = _one(
        connection,
        f"""
        SELECT count(*), count(*) FILTER (WHERE raw IS NULL),
               count(*) FILTER (WHERE raw IS NOT NULL AND (v IS NULL OR NOT isfinite(v))),
               count(*) FILTER (WHERE isfinite(v) AND ({outside})),
               approx_quantile(CASE WHEN isfinite(v) THEN v END, [0.01, 0.5, 0.99])
        FROM (SELECT {_q(column)} AS raw, {value} AS v FROM {source})
    """,
    )
    rows, nulls, invalid, out_of_bounds, quantiles = row
    quantiles = quantiles or [None, None, None]
    return {
        "rows": rows,
        "null_count": nulls,
        "null_share": _ratio(nulls, rows),
        "invalid_count": invalid,
        "out_of_bounds": out_of_bounds,
        **dict(zip(("p1", "p50", "p99"), quantiles)),
    }


def _numeric_check(
    connection, source, column, policy, null_policy, *, integer=False
) -> dict:
    stats = _stats(
        connection,
        source,
        column,
        policy.get("minimum"),
        policy.get("maximum"),
        integer=integer,
    )
    available = stats["rows"] > stats["null_count"] + stats["invalid_count"]
    result = _check(stats, policy, stats["out_of_bounds"] + stats["invalid_count"])
    null_check = _check(
        {"null_share": stats["null_share"]}, null_policy, stats["null_share"]
    )
    result["verdict"] = _worst(
        result["verdict"], null_check["verdict"], "pass" if available else "warn"
    )
    return {**result, "null_check": null_check}


def _views(connection, root: Path, dictionary) -> dict[str, dict]:
    root = root.resolve(strict=True)
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "easyicu_native_export_v2"
        or manifest.get("format") != "parquet"
    ):
        raise ValueError("native-v2 manifest required")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files required")
    grouped: dict[str, list[str]] = {}
    seen: set[Path] = set()
    for item in files:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("module"), str)
            or item["module"] not in CONCEPT_GROUPS_INTERNAL
        ):
            raise ValueError("unsupported module")
        relative = item.get("file")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("local parquet member required")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or path.suffix != ".parquet" or path in seen:
            raise ValueError("invalid parquet member")
        seen.add(path)
        grouped.setdefault(item["module"], []).append(str(path))
    modules = {}
    for index, (module, paths) in enumerate(sorted(grouped.items())):
        view = f"module_{index}"
        relation = connection.read_parquet(paths, union_by_name=True)
        columns = set(relation.columns)
        if "stay_id" not in columns:
            raise ValueError("native stay coordinate required")
        relation.create_view(view)
        # Only public concept names may label output. ID/time coordinates and
        # physical availability receipts never count as clinical values.
        values = sorted(
            column
            for column in columns
            if column in CONCEPT_DICTIONARY or dictionary.get(column) is not None
        )
        modules[module] = {"view": view, "columns": columns, "values": values}
    return modules


def _coverage(
    connection, source: str | None, columns: list[str], denominator: int, policy: dict
) -> dict:
    if source and columns:
        observed = " OR ".join(f"{_q(column)} IS NOT NULL" for column in columns)
        numerator = _one(
            connection, f"SELECT count(DISTINCT stay_id) FROM {source} WHERE {observed}"
        )[0]
    else:
        numerator = 0
    share = _ratio(numerator, denominator)
    return _check(
        {
            "stays_with_value": numerator,
            "denominator_stays": denominator,
            "share": share,
        },
        policy,
        share,
    )


def _find(modules: dict, column: str, preferred: str = "") -> dict | None:
    ordered = sorted(modules, key=lambda name: (name != preferred, name))
    return next(
        (modules[name] for name in ordered if column in modules[name]["columns"]), None
    )


def _score_totals(connection, modules: dict, rules: dict) -> dict:
    results = {}
    for total, components in SCORE_COMPONENTS.items():
        preferred = {"sofa": "sofa1_score", "sofa2": "sofa2_score"}.get(
            total, "other_scores"
        )
        module = _find(modules, total, preferred)
        present = [name for name in components if module and name in module["columns"]]
        metrics = {
            "component_count": len(present),
            "expected_component_count": len(components),
            "compared_rows": 0,
            "mismatches": 0,
            "invalid_count": 0,
        }
        if module and present:
            # Compare the same observation, never sums of component maxima.
            expressions = [_numeric(name) for name in present]
            component_sum = " + ".join(f"coalesce({expr}, 0)" for expr in expressions)
            observed = " OR ".join(f"isfinite({expr})" for expr in expressions)
            compared, mismatches = _one(
                connection,
                f"""
                SELECT count(*), count(*) FILTER (WHERE {_numeric(total)} <> ({component_sum}))
                FROM {module["view"]} WHERE isfinite({_numeric(total)}) AND ({observed})
            """,
            )
            metrics.update(compared_rows=compared, mismatches=mismatches)
            invalid = " OR ".join(
                f"({_q(name)} IS NOT NULL AND ({_numeric(name)} IS NULL OR NOT isfinite({_numeric(name)})))"
                for name in (total, *present)
            )
            metrics["invalid_count"] = _one(
                connection, f"SELECT count(*) FROM {module['view']} WHERE {invalid}"
            )[0]
        value = (
            metrics["mismatches"] + metrics["invalid_count"]
            if metrics["compared_rows"] or metrics["invalid_count"]
            else None
        )
        result = _check(metrics, rules["components"], value)
        if len(present) < len(components):
            # A total with omitted component columns cannot be fully verified.
            result["verdict"] = _worst(result["verdict"], "warn")
        results[total] = result
    return results


def _sofa_comparison(connection, modules: dict, rules: dict) -> dict:
    sources = [
        _find(modules, name, preferred)
        for name, preferred in (("sofa", "sofa1_score"), ("sofa2", "sofa2_score"))
    ]
    if all(sources):
        subqueries = [
            f"(SELECT stay_id, max({_numeric(column)}) AS value FROM {module['view']} "
            f"WHERE isfinite({_numeric(column)}) AND stay_id IS NOT NULL GROUP BY stay_id)"
            for column, module in zip(("sofa", "sofa2"), sources)
        ]
        row = _one(
            connection,
            f"""
            SELECT count(*) FILTER (WHERE a.value IS NOT NULL AND b.value IS NOT NULL),
                   count(*) FILTER (WHERE b.value > a.value),
                   count(*) FILTER (WHERE b.value < a.value),
                   count(*) FILTER (WHERE b.value = a.value),
                   approx_quantile(a.value, 0.5) FILTER (WHERE b.value IS NOT NULL),
                   approx_quantile(b.value, 0.5) FILTER (WHERE a.value IS NOT NULL),
                   count(*) FILTER (WHERE a.value IS NULL OR b.value IS NULL)
            FROM {subqueries[0]} a FULL OUTER JOIN {subqueries[1]} b USING (stay_id)
        """,
        )
    else:
        row = (0, 0, 0, 0, None, None, 0)
    paired, higher, lower, equal, median1, median2, unpaired = row
    metrics = {
        "paired_stays": paired,
        "unpaired_stays": unpaired,
        "higher_stays": higher,
        "lower_stays": lower,
        "equal_stays": equal,
        "higher_share": _ratio(higher, paired),
        "lower_share": _ratio(lower, paired),
        "equal_share": _ratio(equal, paired),
        "sofa_p50": median1,
        "sofa2_p50": median2,
    }
    return _check(metrics, rules["sofa2_lower_share"], metrics["lower_share"])


def _sepsis(connection, module, column, stay_count: int, rules: dict) -> dict:
    observed = positive = invalid = 0
    if module:
        v = _numeric(column)
        observed, positive, invalid = _one(
            connection,
            f"""
            SELECT count(DISTINCT stay_id) FILTER (WHERE {v} IN (0, 1)),
                   count(DISTINCT stay_id) FILTER (WHERE {v} = 1),
                   count(*) FILTER (WHERE {_q(column)} IS NOT NULL AND
                       ({v} IS NULL OR NOT isfinite({v}) OR {v} NOT IN (0, 1)))
            FROM {module["view"]}
        """,
        )
    metrics = {
        "observed_stays": observed,
        "positive_stays": positive,
        "denominator_stays": stay_count,
        "invalid_count": invalid,
        "prevalence": _ratio(positive, stay_count) if observed else None,
        "observed_prevalence": _ratio(positive, observed),
    }
    result = _check(metrics, rules["sepsis"], metrics["prevalence"])
    if invalid:
        result["verdict"] = _check({}, rules["bounds"], invalid)["verdict"]
    return result


def _stay_checks(connection, modules: dict, rules: dict) -> dict:
    result = {}
    for name in ("demographics", "outcome"):
        duplicate_stays = extra_rows = 0
        if name in modules:
            duplicate_stays, extra_rows = _one(
                connection,
                f"""
                SELECT count(*), coalesce(sum(n - 1), 0)::BIGINT FROM (
                    SELECT count(*) AS n FROM {modules[name]["view"]}
                    WHERE stay_id IS NOT NULL GROUP BY stay_id HAVING count(*) > 1)
            """,
            )
        result[name] = _check(
            {"duplicate_stays": duplicate_stays, "extra_rows": extra_rows},
            rules["duplicates"],
            duplicate_stays if name in modules else None,
        )
    module = _find(modules, "los_icu", "outcome")
    metrics = {
        "observed_stays": 0,
        "negative_count": 0,
        "invalid_count": 0,
        "p1": None,
        "p50": None,
        "p99": None,
        **{f"above_{days}_days": 0 for days in rules["los"]["count_days"]},
    }
    verdict = "warn"
    if module:
        value = _numeric("los_icu")
        stats = _stats(connection, module["view"], "los_icu", 0)
        # One LOS per stay even when a duplicate outcome row is present.
        source = (
            f"(SELECT stay_id, max({value}) AS los_icu FROM {module['view']} "
            f"WHERE stay_id IS NOT NULL AND isfinite({value}) GROUP BY stay_id)"
        )
        per_stay = _stats(connection, source, "los_icu", 0)
        metrics.update(
            observed_stays=per_stay["rows"],
            negative_count=stats["out_of_bounds"],
            invalid_count=stats["invalid_count"],
            **{key: per_stay[key] for key in ("p1", "p50", "p99")},
        )
        days = sorted(
            set(
                [
                    *rules["los"]["count_days"],
                    rules["los"]["warn_days"],
                    rules["los"]["fail_days"],
                ]
            )
        )
        counts = _one(
            connection,
            "SELECT "
            + ", ".join(
                f"count(*) FILTER (WHERE los_icu > {float(day)})" for day in days
            )
            + f" FROM {source}",
        )
        by_day = dict(zip(days, counts))
        metrics.update(
            {f"above_{day}_days": by_day[day] for day in rules["los"]["count_days"]}
        )
        verdict = "pass" if metrics["observed_stays"] else "warn"
        if by_day[rules["los"]["warn_days"]]:
            verdict = "warn"
        if (
            metrics["negative_count"]
            or metrics["invalid_count"]
            or by_day[rules["los"]["fail_days"]]
        ):
            verdict = "fail"
    result["icu_los"] = {**metrics, "expectation": rules["los"], "verdict": verdict}
    return result


def _scan(connection, root: Path, rules: dict) -> dict:
    dictionary = load_dictionary(include_sofa2=True)
    modules = _views(connection, root, dictionary)
    population = " UNION ".join(
        f"SELECT stay_id FROM {module['view']}" for module in modules.values()
    )
    stay_count = _one(
        connection, f"SELECT count(DISTINCT stay_id) FROM ({population})"
    )[0]
    result: dict[str, Any] = {
        "stay_count": stay_count,
        "module_rows": {},
        "concepts": {},
        "negative_times": {},
        "coverage": {"modules": {}, "key_concepts": {}},
    }
    for name, module in modules.items():
        view = module["view"]
        result["module_rows"][name] = _one(connection, f"SELECT count(*) FROM {view}")[
            0
        ]
        result["coverage"]["modules"][name] = _coverage(
            connection, view, module["values"], stay_count, rules["coverage"]
        )
        if "charttime" in module["columns"]:
            negative = _one(
                connection,
                f"SELECT count(*) FILTER (WHERE {_numeric('charttime')} < 0) FROM {view}",
            )[0]
            result["negative_times"][name] = _check(
                {"negative_count": negative}, rules["negative_times"], negative
            )
        for column in module["values"]:
            definition = metadata_definition_for_concept(column, name, dictionary)
            if definition.minimum is None and definition.maximum is None:
                continue
            policy = {
                **rules["bounds"],
                "minimum": definition.minimum,
                "maximum": definition.maximum,
            }
            result["concepts"][f"{name}.{column}"] = _numeric_check(
                connection, view, column, policy, rules["null_share"]
            )
    for key, columns in KEY_CONCEPTS.items():
        queries = []
        for module in modules.values():
            present = [column for column in columns if column in module["columns"]]
            if present:
                observed = " OR ".join(
                    f"{_q(column)} IS NOT NULL" for column in present
                )
                queries.append(
                    f"SELECT stay_id, 1 AS observed FROM {module['view']} WHERE {observed}"
                )
        source = "(" + " UNION ALL ".join(queries) + ")" if queries else None
        result["coverage"]["key_concepts"][key] = _coverage(
            connection,
            source,
            ["observed"] if queries else [],
            stay_count,
            rules["coverage"],
        )
    ranges = {}
    for column in ("qsofa", "sirs", "news"):
        module = _find(modules, column, "other_scores")
        ranges[column] = (
            _numeric_check(
                connection,
                module["view"],
                column,
                rules[column],
                rules["null_share"],
                integer=True,
            )
            if module
            else _check({}, rules[column], None)
        )
    result["stays"] = _stay_checks(connection, modules, rules)
    result["scores"] = {
        "sofa_comparison": _sofa_comparison(connection, modules, rules),
        "totals": _score_totals(connection, modules, rules),
        "ranges": ranges,
        "sepsis": {
            column: _sepsis(
                connection, _find(modules, column), column, stay_count, rules
            )
            for column in ("sep3_sofa1", "sep3_sofa2")
        },
    }
    return result


def _metrics(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        if key in {"expectation", "null_check"}:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_metrics(item, path))
        elif item is None or isinstance(item, (int, float)):
            result[path] = item
    return result


def _delta(metric: str, old, current, rules: dict) -> dict:
    rate = any(token in metric.rsplit(".", 1)[-1] for token in ("share", "prevalence"))
    policy = rules["drift_rate" if rate else "drift_numeric"]
    difference = current - old if old is not None and current is not None else None
    relative = difference / abs(old) if difference is not None and old else None
    change = (
        abs(difference)
        if rate and difference is not None
        else (
            abs(relative) if relative is not None else (0 if difference == 0 else None)
        )
    )
    result = _check(
        {
            "metric": metric,
            "old": old,
            "current": current,
            "delta": difference,
            "relative_delta": relative,
        },
        policy,
        change,
    )
    if old is None and current is None:
        result["verdict"] = "pass"
    return result


def _summary(value: dict) -> dict:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    if value.get("verdict") in counts:
        counts[value["verdict"]] += 1
    for key, child in value.items():
        if key == "expectation":
            continue
        children = child if isinstance(child, list) else [child]
        for item in children:
            if isinstance(item, dict):
                for verdict, count in _summary(item).items():
                    counts[verdict] += count
    return counts


def build_report(
    export_root: Path,
    *,
    rules: dict,
    compare: Path | None = None,
    threads: int = 2,
    memory_limit: str = "8GB",
) -> dict:
    """Read declared local parquet members and return aggregate-only metrics.

    Policy is supplied by tools/report_export_plausibility.py. One connection
    enforces the same memory/thread limits for current and comparison scans;
    any disk spill is private, temporary, and deleted on success or failure.
    Exceptions are deliberately sanitized at the CLI boundary.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    with duckdb_spill.connection(
        config={
            'memory_limit': memory_limit,
            'threads': threads,
            'autoload_known_extensions': False,
            'autoinstall_known_extensions': False,
        },
    ) as connection:
        effective_memory, effective_threads = _one(
            connection,
            "SELECT current_setting('memory_limit'), current_setting('threads')",
        )
        result = {
            "schema_version": "easyicu_export_plausibility_v1",
            "status": "complete",
            "quantile_method": "DuckDB approx_quantile (approximate, bounded-memory)",
            "runtime": {
                "requested_memory_limit": memory_limit,
                "memory_limit": effective_memory,
                "threads": effective_threads,
            },
            "current": _scan(connection, Path(export_root), rules),
        }
        if compare is not None:
            result["old"] = _scan(connection, Path(compare), rules)
            current_metrics, old_metrics = (
                _metrics(result["current"]),
                _metrics(result["old"]),
            )
            result["comparison"] = {
                "metrics": [
                    _delta(
                        metric,
                        old_metrics.get(metric),
                        current_metrics.get(metric),
                        rules,
                    )
                    for metric in sorted(
                        current_metrics.keys() | old_metrics.keys()
                    )
                ],
                "module_rows": {
                    module: _delta(
                        f"module_rows.{module}",
                        result["old"]["module_rows"].get(module),
                        result["current"]["module_rows"].get(module),
                        rules,
                    )
                    for module in sorted(
                        result["old"]["module_rows"].keys()
                        | result["current"]["module_rows"].keys()
                    )
                },
            }
        result["summary"] = _summary(
            {
                "current": result["current"],
                "comparison": result.get("comparison", {}),
            }
        )
        return result
