# eICU SOFA-2 constrained-host performance profile (2026-09-20)

## Question and setup

The v6 rebuild runs under an explicit 8,192-MiB resource contract. The first
complete post-integrity-fix eICU benchmark processed 200,859 stays in nine
25,000-stay partitions. It produced 15,720,946 SOFA-2 rows in 2,911.0 seconds,
with a 3,226.6-MiB process-tree peak and a 3,095.0-MiB working-set peak. There
was therefore no evidence that paging or the memory ceiling caused the long
runtime.

One representative source-order-interleaved 25,000-stay partition was profiled
with Python `cProfile`. It used the same concept list and patient partitioning.
The exploratory process imported the host-sized parallel planner before the
isolated writer installed its environment, so its wall time is diagnostic and
must not be read as the formal 8-GiB single-thread runtime. `/usr/bin/time -v`
separately recorded wall time, CPU use, RSS, major faults, and swaps. The
temporary `.prof` and output Parquet files remain outside Git.

## Root cause

Before optimization, the representative partition took 285.90 seconds wall
time and 287.16 profiled seconds. `assess_urine_windows()` consumed 226.14
seconds (78.8% of profiled time). Its patient loop already held positional
integer arrays, but assigned every result through pandas `.loc`. That caused
693,256 `__setitem__` calls and 560,562,180 total function calls. Raw table
loading consumed about 22 seconds and the final SOFA-2 callback about 11
seconds, so neither source I/O nor the score formula was the primary bottleneck.

The 8-GiB execution contract deliberately fixes concept workers, Arrow threads,
and DuckDB threads at one. This protects reproducibility and prevents transient
oversubscription. It also means Python-level per-patient overhead cannot be
hidden by adding cores.

## Change and equivalence evidence

`assess_urine_windows()` now writes patient results into preallocated NumPy
arrays and assigns each output column to the DataFrame once. Window definitions,
coverage rules, patient grouping, weights, thresholds, and output dtypes are
unchanged.

Two deterministic randomized fixtures covered 200 patients, duplicate source
channels, gaps, invalid values, volume bins, and rate intervals. Every output
cell and dtype matched the pre-change implementation exactly. The focused
urine-window, urine-rate, and callback suite also passed (39 tests).

With the same profiled partition, wall time fell to 75.35 seconds and profiled
time to 76.98 seconds. `assess_urine_windows()` fell from 226.14 to 13.78
seconds, and total calls fell from 560.6 million to 40.6 million. Wall time was
3.79 times faster. Maximum RSS changed from 3,612,408 KiB to 3,769,336 KiB;
both runs reported zero swaps and zero major page faults.

## Release consequence

The clean `51d88010` rerun under the formal 8,192-MiB envelope produced exactly
the same 15,720,946 SOFA-2 rows and 17,781 Sepsis-3 rows as the pre-optimization
run. Both Parquet SHA-256 values also matched exactly. SOFA-2 elapsed time fell
from 2,911.0 to 1,731.9 seconds (40.5%); total closure time fell from 2,978.1 to
1,799.4 seconds (39.6%). Process-tree peak RSS rose from 3,226.6 to 3,519.7 MiB,
which remains below half of the 8-GiB contract. The new receipt authorizes a
fixed 25,000-stay batch for both closure modules.
