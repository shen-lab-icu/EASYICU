# EasyICU Source Dictionary Coverage Audit

_Generated at 2026-07-02T23:53:37._

## Scope

This audit compares selected EasyICU concepts against local raw source dictionaries and label catalogs. It produces a review queue, not automatic mapping decisions.

## Summary

- Concepts audited: 28
- Databases audited: aumc, mimic, miiv, hirid, eicu
- Candidate rows: 1766
- Unmapped candidates: 1479

## Status Counts

| status | n |
| --- | ---: |
| covered_by_regex | 12 |
| mapped | 275 |
| unmapped_candidate | 1479 |

## Files

- `source_dictionary_coverage_candidates.csv`: all matched candidates.
- `unmapped_candidate_review.csv`: candidates not covered by exact ids or source regexes.
- `summary.json`: machine-readable counts.

## Interpretation

A clean structural dictionary does not prove semantic completeness. Review `unmapped_candidate_review.csv`; only direct equivalents with compatible units and table semantics should be added to the packaged dictionaries.
