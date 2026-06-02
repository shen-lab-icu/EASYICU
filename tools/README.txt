# EasyICU tools directory

This directory contains repository-level command entrypoints. These scripts are
not installed as public Python APIs; they are convenience commands for
benchmarking, validation, and maintenance from the repository root.

## Benchmark tools

| File | Purpose |
|---|---|
| `run_research_agent_bench.py` | Main research-agent benchmark runner. Use `--arms aware` for paper-facing EasyICU workflow runs. |

## Maintenance and validation tools

| File | Purpose |
|---|---|
| `run_analysis_bench_overnight.py` | Batch wrapper around `run_research_agent_bench.py` for long runs. |
| `run_openrouter_fullflow_validation.py` | Real-provider full-flow validation for OpenRouter-compatible models. |
| `evaluate_concept_usage_auditor.py` | Helper for checking concept-usage audit behavior. |
| `fetch_baselines.py` | Fetches external baseline systems into ignored local checkout directories. |

## Archived locally, not uploaded

Historical experiment runners, local-model launchers, manuscript/submission
package builders, and submission-specific result aggregators are kept out of
the tracked repository (see the manuscript/submission entries in `.gitignore`
and `tools/legacy/`). They remain available locally for provenance but are not
part of the clean GitHub surface. Manuscript source-data, figures, and the
draft itself live in a separate private writing workspace, not in this package.
