from __future__ import annotations


def test_parse_cohort_map_accepts_multiple_pairs(ra):
    from easyicu.research_agent.cli import _parse_cohort_map

    mapping = _parse_cohort_map(["miiv=/tmp/a.parquet", "eicu=/tmp/b.parquet"])
    assert mapping["miiv"].endswith("/tmp/a.parquet")
    assert mapping["eicu"].endswith("/tmp/b.parquet")


def test_parse_cohort_map_rejects_invalid_pair(ra):
    from easyicu.research_agent.cli import _parse_cohort_map

    try:
        _parse_cohort_map(["miiv"])
    except SystemExit as exc:
        assert "--cohort-map must be DB=PATH" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SystemExit for invalid cohort-map input")
