# 2026-07-22 — full0717-v2 grouped native re-export

## Authorization and boundary

The owner explicitly authorized a **new, versioned** six-source native export from the existing converted raw databases on the external drive. Historical `full6_20260717` remains immutable and is not overwritten. This task does not authorize a Provider call, Docker execution, Canonical9 execution, or a paper-authority claim.

## Performance contract

The new path is `easyicu.api.extract_database(..., native_export_v2=True)`, not the Web runner's per-module loop. It keeps the existing 19-module affinity groups, one grouped subprocess per source-table family, `keep_cache`, and the default one-shot batch sentinel. Native metadata finalization reads each newly produced module parquet once; it never re-reads raw ICU tables.

## Implementation and smoke evidence

- Shared producer binding moved to `easyicu.concept.export_metadata`; Web/HTTP export and the grouped extractor use the same typed physical-column contract.
- `native_export_v2=True` publishes a root `_manifest.json` only after every requested module succeeds and every selected concept has one primary typed binding. Any incomplete/error state fails before root-manifest publication.
- A real MIMIC-IV, 10-stay, `demographics` grouped-extractor smoke wrote a fresh private package at `/Volumes/外置硬盘/easyicu_data/full0717-v2_grouped_native_dryrun_miiv_20260722`.
- `open_export_package` verified one physical file, six typed concepts, no missing selected concept, and the content-addressed metadata sidecar. Directory/file permissions were `0700`/`0600`.
- Focused regression: 40 passed; architecture and module-graph gates passed. No Provider, Docker, or Canonical9 run occurred.

## Next

Run all six sources sequentially into a new `full0717-v2_native_20260722` root. For each source: finish all requested modules, open the native package, capture manifest/sidecar digests and row/bounds spot checks, then proceed to the next source. Only after all six packages exist and P4/E2/H2/H3 gates are separately satisfied may the owner consider an aware-arm Canonical9 authorization.
