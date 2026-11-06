# Running extraction locally / 本地运行指南

## Check that a conversion is complete / 校验转换完整性

Extraction reads only the Parquet files that the converter wrote; a directory
of source CSV files is not a supported runtime layout, so convert it first.
The converter records every output file and the row count of each table in
`conversion_manifest.json` at the root of the database directory. On every
read of an unbucketed table EasyICU checks the files it scans against that
record:

- the scan must cover exactly the files the manifest lists for the table, and
- the rows in their Parquet footers must equal the manifest's row count.

A mismatch stops the extraction with `TableScanError`
(`table_scan_manifest_mismatch`): a chunk is missing, truncated or replaced.
Extraction never continues on part of a table.

To check or repair a converted directory:

```bash
python -m easyicu.io.data_converter /path/to/database --status
```

```bash
python -m easyicu.io.data_converter /path/to/database
```

The first command reports each table's conversion status without converting.
The second re-verifies every table (footer rows against the recorded count)
and converts again only what is missing, damaged or older than its CSV;
verified tables are skipped. Use `--force` only to rebuild everything. Both
need the original CSV files in the directory, since they are the reference.

A directory without `conversion_manifest.json` (an installation converted by
an older EasyICU, or a view assembled by hand) is still checked against its
own files, but a lost chunk cannot be detected there. EasyICU logs
`table_scan_unmanifested` once per table in that case; run the second command
above once to write the manifest.

Bucketed tables (`<table>_bucket/`) carry their own manifest of files and rows
and are verified whenever they are opened. A bucket layout records the source
files it was built from (`derived_from`); if a source file's size or
modification time no longer matches, the layout is refused, or ignored when a
complete flat table sits beside it. Copy an installation with its
modification times (`cp -p`, `rsync -t`), or rebuild the layouts after
copying.

## Recommended storage layout per database / 推荐存储布局

lead-r13 measured each database's layouts on a 3,000-stay sample: 20 module
files, 4 DuckDB threads, 2 workers. Both layouts of a database produced
byte-identical module files; the recommendation is the faster one within the
14 GiB target.

| Database | Recommended layout | Sample time / peak | Other layout |
|---|---|---|---|
| eICU-CRD | variable × stay buckets (two-level, contiguous) | 487 s / 3.9 GiB | hash buckets: same within noise (487 s / 3.9 GiB) |
| SICdb 1.0.8 | variable × stay buckets (`data_float_h`, `laboratory` single-key hash; `medication` two-level) | 236 s / 1.8 GiB | flat Parquet: 261 s / 8.9 GiB |
| MIMIC-III CareVue view | hash buckets | 375 s / 2.6 GiB | variable × stay buckets: 2,165 s / 2.7 GiB (each open re-validates about 5,700 bucket files) |
| MIMIC-IV | TODO(miiv_var): pending the variable-layout measurement | | |
| HiRID | contiguous batches on the current layout | 2,142 s / 3.5 GiB | |
| AmsterdamUMCdb | unchanged | 2,040 s / 6.0 GiB | |

## Memory / 内存

Peak memory of the full-cohort extractions (the 14 GiB local target and the
64 GiB server runs) is recorded in each database's formal 0.2.0
`RUN_RECEIPT`, after the formal run. On the 3,000-stay samples every database
peaked at or below 8.9 GiB (cgroup peak, 2 workers): SICdb flat 8.9 GiB,
MIMIC-IV 6.8 GiB, AmsterdamUMCdb 6.0 GiB, eICU 3.9 GiB, HiRID 3.5 GiB,
MIMIC-III 3.2 GiB, CareVue 2.7 GiB.
