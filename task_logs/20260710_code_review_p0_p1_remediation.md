# 2026-07-10 EasyICU code-review P0/P1 remediation

Task IDs: `WEBAPP-FASTAPI-NATIVE-QA`, `IDEA-MINING-DISCOVERY-MODULE`, and `DATA-FIX1`.

## Scope and outcome

This pass reviewed and repaired the data foundation, native WebApp and hosted LLM relay, Idea Mining / discovery handoff, MCP entry points, generated-code runners, and manuscript evidence chain. The patch is intentionally limited to correctness, integrity, and high-impact operational safety; it does not attempt a broad redesign of the existing large screen modules.

## Repairs

### Data foundation

- Made an explicitly empty cohort fail closed across ordinary and special concept loaders instead of returning the full database.
- Resolved database paths and patient filters consistently for special loaders.
- Bound concept-cache identity to cohort IDs, source path, row limit, and a 128-bit data fingerprint.
- Applied raw concept bounds before hourly aggregation and added a fail-safe unit-suspect retry for both transformed and untransformed values: when a batch of at least 100 values would be entirely removed, the unbounded result is retained and the manifest records that bounds were skipped.
- Required completed conversion status for both single-file and sharded outputs, validated expected shard inventory, and compared status row counts against Parquet footer metadata before reuse; retained prior manifest entries and bad-row counts during incremental conversion.
- Corrected missing ventilator-free-day handling, readmission ordering, cross-database microbiology denominators, and `ICUTable.to_wide()` column naming.

### Native WebApp and hosted relay

- Changed the hosted relay to loopback-by-default, bearer-authenticated, model-allow-listed behavior with Host, CORS, wildcard, and trusted-proxy checks.
- Enforced loopback clients and valid Host headers for the native WebApp CLI/application.
- Replaced URL fetching with public-address validation plus pinned DNS sockets; redirects are revalidated and HTTPS preserves the original Host/SNI and certificate checks.
- Added strict boolean parsing, pre-decode PDF size limits, bounded 500-stay summaries with predicate/projection pushdown, random session IDs, job backpressure/retention, and atomic source-registry writes.
- Removed server-originated DOM HTML injection in the extraction picker. The new behavior remains in the route owner files (`screens-extraction.js` and `screens-viz.js`); no CSS ownership changes were introduced.

### Research Agent, MCP, and discovery provenance

- Changed the default generated-code backend to `auto`: use a probed immutable Docker image when ready, otherwise use real macOS `sandbox-exec`, and fail before execution on Linux/Windows when no safe backend exists. Unsafe host fallback now requires an explicit development-only opt-in.
- Scrubbed ambient secrets, confined macOS writes to the step directory, and verified that generated code cannot modify run-level evidence.
- Anchored EvidenceStore writes with directory file descriptors, `O_NOFOLLOW`, atomic replacement, and repeated directory validation to reject post-initialization symlink swaps.
- Inspected Docker images once, used the immutable image ID for both `pip freeze` and analysis, derived method capabilities from the container snapshot, and required consistent runtime locks across every step and resume.
- Prevented provider-key forwarding to loopback MCP endpoints; added independent MCP bearer auth for remote binds plus Host, Origin, content type, body-size, SSE, and path-confinement checks.
- Froze the analysis outcome from the confirmed discovery handoff, required exact handoff hashes and explicit human confirmation, and removed the AKI-specific discovery-story shortcut.
- Preserved the discovery trajectory as an explicit resolved JSONL path, retained cohort `Path` semantics in the benchmark launcher, passed only three allow-listed trajectory aliases, and mounted external trajectory files read-only in Docker.
- Required every contract and SVG/PDF/PNG/TIFF export to be registered and hash-valid, linked to code and all source evidence, and format-valid. The SVG validator permits only the fixed W3C SVG 1.1 public declaration used by Matplotlib, strips it before parsing, and still rejects entities, internal subsets, system declarations, oversized SVGs, empty visuals, and escaped symlinks.
- Brought the general `PublicationFigureSkill` onto the same strict registration contract, so its family, association, robustness, step-promotion, and prediction-promotion paths all register renderer code, contract, sources, and every export consistently.

### Operator documentation

- Added README warnings that the native WebApp is local-only and that the hosted relay requires explicit token, bind, model, and proxy configuration.

## Verification

Verification performed during this repair pass (later focused suites cover the final follow-up patches):

```text
Research Agent critical suite:
214 passed in 191.22s

Data + hosted relay + all WebServer tests:
282 passed, 21 skipped, 6 warnings in 83.34s

Final runner/trajectory related suite:
80 passed in 1113.13s; independent final subset 61 passed in 10.15s

Final converter/bounds suite after scoped style restoration:
73 passed in 53.63s; orphan-single + HiRID contract subset 32 passed

PublicationFigureSkill:
43 publication tests + 11 family tests passed; independent real-render inventory check passed

Second read-only P1 closure review:
data/figure 8 passed; runner/trajectory 12 passed plus real macOS sandbox trajectory smoke

ruff check src tests tools:
All checks passed

git diff --check:
no output
```

Additional checks:

- Real Matplotlib renderer -> launcher registration of SVG/PNG/PDF/TIFF -> package inventory completed with `provenance_valid=True` and no unregistered exports.
- `node --check` passed for both changed route-owner JavaScript files.
- Desktop browser QA at 1280x720 covered Home, Data Extraction, and Idea Mining: zero document overflow, zero browser-console warnings/errors, and expected route headings present.
- The WebApp server was stopped and the QA tab was closed after verification.
- Accidental whole-file formatter churn in legacy-style data files was removed while preserving the normalized Python AST; `py_compile`, Ruff, and the data regression suites passed on the restored scoped diff.

## Residual validation limits

- Docker client 29.4.2 is installed, but the Docker daemon is unavailable at `/var/run/docker.sock`; immutable-image execution, real container `pip freeze`, capability snapshot, trajectory mount, and image-build smoke tests are therefore covered by mocks rather than a live container. Linux and Windows `auto` selection were platform-simulated; macOS sandbox execution was exercised for real.
- The 21 skipped tests require local real ICU databases. A six-database export and bounds-manifest spot-check was not run in this pass.
- A real-provider discovery run, real non-loopback MCP/hosted deployment, and the full 3,130-test repository suite were not run. The exercised suites are the changed domains plus one complete local research-agent pipeline test.
- A separately started `bench_e1_baseline_confirm` canonical benchmark was still running during final handoff; it was deliberately not interrupted or counted as verification for this repair pass.
- No commit or push was created.
