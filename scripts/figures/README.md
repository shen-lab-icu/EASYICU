# EasyICU cross-database publication QC

These stable Python entry points audit a completed six-database native-v2 run.
Raw Parquet files and rendered outputs remain outside Git.

```bash
python scripts/figures/QC-A01_cross_database_distributions.py \
  --input-root /path/to/full6_run/exports \
  --output-root /path/to/full6_run/publication_qc \
  --catalog src/easyicu/data/concept-dict.json

python scripts/figures/QC-A02_easyicu_cross_database_reliability_audit.py \
  --export-root /path/to/full6_run/exports \
  --figure-audit /path/to/full6_run/publication_qc/audit/variable_audit.csv \
  --run-metadata /path/to/full6_run/run_metadata.json \
  --output-dir /path/to/full6_run/publication_qc/reliability_audit
```

`QC-A01` produces one paginated atlas per module (maximum 12 panels per
183-mm page), editable SVG/PDF, 600-dpi TIFF, PNG previews, plotting-source CSVs,
and explicit display-tail counts. Continuous variables use record-level
densities; binary variables use stay-level prevalence; ordinal and categorical
variables use probability-mass curves or database-by-category heatmaps.

`QC-A02` checks exact physical schema equality, canonical identifiers and time,
native-v2 manifests, metadata sidecars, provenance, availability and
conservative distribution anomaly signals. A distribution flag is a review
trigger, not proof of a conversion defect; source-table traceback remains
required.
