# EasyICU tools directory

This directory contains repository-level command entrypoints. These scripts are
not installed as public Python APIs; they are convenience commands for
benchmarking, validation, and maintenance from the repository root.

## Current manuscript / benchmark tools

| File | Purpose |
|---|---|
| `run_research_agent_bench.py` | Main research-agent benchmark runner. Use `--arms aware` for paper-facing EasyICU workflow runs. |
| `aggregate_v19_benchmark_results.py` | Builds Figure 4-style source-data tables from benchmark output directories. |

## Maintenance and validation tools

| File | Purpose |
|---|---|
| `run_analysis_bench_overnight.py` | Batch wrapper around `run_research_agent_bench.py` for long runs. |
| `run_openrouter_fullflow_validation.py` | Real-provider full-flow validation for OpenRouter-compatible models. |
| `run_lactate_map_vaso_export_fullflow.py` | Narrow full-flow validation for lactate/MAP/vasopressor export scenarios. |
| `evaluate_concept_usage_auditor.py` | Helper for checking concept-usage audit behavior. |
| `fetch_baselines.py` | Fetches external baseline systems into ignored local checkout directories. |

## Archived locally, not uploaded

Historical v14/v15/v16 experiment runners, local-model launchers, and manuscript
package builders have been moved to `tools/legacy/`, which is ignored by Git.
Those files are kept locally for provenance but are not part of the clean GitHub
surface.

Current Figure 4/5 manuscript source-data and plotting scripts live in the
submission workspace, not in this package tools directory:

From the `EASYICU/` repository root:

- `../easyicu写作/00_当前投稿_20260516/v19_benchmark_runs/`
- `../easyicu写作/00_当前投稿_20260516/01_Nature_Methods/`
