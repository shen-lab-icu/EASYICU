# Host method

Call chain: descriptive family (`descriptive_measurement_v1`) →
`descriptive.table_one` (`analysis_method_suite.py`, `method_adapter_catalog.py`) →
`execution.runners.table_one_executor.table_one_executor_code` →
`methods.table_one.build_grouped_table_one`. The reference calls that same kernel.

All scientific choices come from `TableOneSpec`: grouping levels and order,
variable levels, summary (`mean_sd`, `median_iqr`, `both`, `count_percent`) and
comparison test. Mean and sample SD use ddof=1; quartiles use NumPy's linear
quantiles. Numeric tests are two-sided Welch t / Welch ANOVA or Mann–Whitney U /
Kruskal–Wallis, exactly as declared. Categorical comparisons use Pearson chi-square
without continuity correction, switching to two-sided Fisher exact only for a
2×2 table with an expected cell <5. P values are unadjusted in /1. /2 follows
its declared descriptive no-test policy. /3 belongs to a different host owner
and is refused.

Missing grouping values either fail or are excluded from Overall and groups
with the excluded count reported. Variable summaries/tests use available values;
missing n (%) uses the full displayed-group denominator, categorical percentages
use nonmissing denominators. No imputation occurs. For two groups the first
declared group is the SMD reference; comparison minus reference is divided by
the equal-weight pooled SD (Bernoulli variances for categorical levels).

The host produces a long-form `table_one.csv` with numeric cells and labels,
without display-string rounding or a figure. The scripts preserve its CSV cells
verbatim. Counts and strings are checked exactly in parity tests.

Dependencies: easyicu, numpy, pandas, pyarrow, pydantic, scipy, matplotlib.
The installed EasyICU package also supplies `skill_packages.reference_io`.
