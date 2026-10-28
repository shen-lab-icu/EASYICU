# Cohort characterization table one

Offline reference scripts using the installed EasyICU host estimators.
Inputs: one prepared cohort (unique `stay_id` in the example) and an explicit
JSON specification naming the variables, levels, missing-data policy and method.
Outputs: result CSV tables, `spec.json`, `step_summary.json`, `manifest.json`,
and effect plots for association/survival. See `references/methods.md` for the
exact estimator configuration and dependencies. Results remain analysis only.

```sh
python -m easyicu.research_agent.skill_packages.cohort_characterization_table_one --example --out ./reference_output
python -m easyicu.research_agent.skill_packages.cohort_characterization_table_one --cohort ./prepared.parquet --spec ./reviewed.json --out ./results
```

`example_data.py` writes synthetic rows only. No database, network or governed
run is needed. For scientific use, prepare and review the cohort and specification
first; the scripts select no study population or scientific defaults.
