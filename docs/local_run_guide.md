# Running extraction locally / 本地运行指南

## Check that a conversion is complete / 校验转换完整性

Extraction reads the Parquet files that the converter wrote. The converter
records every output file and the row count of each table in
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
and are verified whenever they are opened.

## Memory / 内存

FILL from the lead-r13 14 GiB and 64 GiB full-cohort runs.
