# EasyICU cross-database publication QC

These stable Python entry points audit a completed six-database native-v2 run.
Raw Parquet files and rendered outputs remain outside Git.
They are the only maintained scripts for this publication-QC figure set; run
directories should contain outputs and provenance, not copied plotting code.

```bash
python scripts/figures/QC-A01_cross_database_distributions.py \
  --input-root /path/to/full6_run/exports \
  --output-root /path/to/full6_run/publication_qc \
  --run-metadata /path/to/full6_run/run_metadata.json \
  --catalog src/easyicu/data/concept-dict.json

python scripts/figures/QC-A02_easyicu_cross_database_reliability_audit.py \
  --export-root /path/to/full6_run/exports \
  --figure-audit /path/to/full6_run/publication_qc/audit/variable_audit.csv \
  --run-metadata /path/to/full6_run/run_metadata.json \
  --output-dir /path/to/full6_run/publication_qc/reliability_audit
```

To refresh labels, units, bounds and figures from the existing lightweight
audit/source CSVs without rescanning Parquet:

```bash
python scripts/figures/QC-A01_cross_database_distributions.py \
  --input-root /path/to/full6_run/exports \
  --output-root /path/to/full6_run/publication_qc \
  --run-metadata /path/to/full6_run/run_metadata.json \
  --catalog src/easyicu/data/concept-dict.json \
  --render-only
```

`QC-A01` produces one paginated atlas per module (maximum 12 panels per
183-mm page), editable SVG/PDF, 600-dpi TIFF, PNG previews, plotting-source CSVs,
and explicit display-tail counts. Continuous variables use record-level
densities; binary variables use stay-level prevalence; ordinal and categorical
variables use probability-mass curves or database-by-category heatmaps.

Both QC manifests bind their outputs to `source_run_id` and the SHA-256 of the
exact source `run_metadata.json`. When QC-A01 is run directly from a Git
checkout, it places that checkout's `src` first so an older editable EasyICU
installation cannot silently supply catalog metadata. Canonical audit units
remain explicit; type markers such as `boolean`, `category` and `datetime` are
suppressed only in reader-facing plot labels.

`QC-A02` checks exact physical schema equality, canonical identifiers and time,
native-v2 manifests, metadata sidecars, provenance, availability and
conservative distribution anomaly signals. A distribution flag is a review
trigger, not proof of a conversion defect; source-table traceback remains
required. Verified source traces are attached to the flag rather than deleting
it, so downstream analyses can still apply database-stratified sensitivity.
Run-specific adjudications require an exact run ID, run-metadata SHA-256 and
anomaly type; reusing a run ID with different metadata leaves the flag
unadjudicated. `QC-A02` also verifies the six root-manifest SHA-256 receipts in
`run_metadata.json` before applying any adjudication.
For every parquet, it additionally verifies the content SHA-256, byte size and the
manifest-bound row-grain receipt: demographics must be unique by `stay_id`, all
other modules by null-equal `(stay_id, charttime)`, and duplicate consolidation
must report zero excess rows after publication. Missing or stale row-grain
receipts are a non-zero audit failure; column presence alone is not treated as
proof of key validity.
The metadata summary reports selected-concept bindings and explicit zero-row
structural placeholders separately; their union must cover every
database-by-module manifest row, while an undocumented metadata gap remains a
non-zero audit failure. Structural closure requires declared and physical zero
rows, the canonical typed schema, an empty selected-concept binding and a
complete per-concept structural-placeholder status. For a curated package
assembled from source-specific corrective
reruns, `run_metadata.json` may declare a `database_commits` mapping; the audit
then verifies each database against its own recorded Git commit.
