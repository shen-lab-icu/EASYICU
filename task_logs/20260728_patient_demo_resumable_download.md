# Patient official Demo resumable-download follow-up

Date: 2026-07-28  
Task: `WEBAPP-FASTAPI-NATIVE-QA` / `PATIENT-CROSSDB-VISUAL-PARITY`

## User-visible problem

The PhysioNet archive could be slow, and the previous downloader used a random
`.part` filename that was deleted on every failure. A retry therefore restarted
from byte zero. At diagnosis time the main MIMIC-IV Demo cache was already fully
prepared and active; eICU Demo remained not downloaded.

## Implemented contract

- `demo_source_storage.py` remains the sole download/cache owner.
- Interrupted bytes use one deterministic private partial plus a release-bound
  receipt containing the expected size and a strong ETag.
- A retry sends `Range` plus `If-Range`, accepts append only for exact `206`,
  matching `Content-Range`, total release size, identity encoding, and the same
  strong ETag.
- A `200` response to a resume request is treated as “Range ignored or validator
  changed”: the old partial is discarded and the full response safely replaces
  it. Misaligned ranges are rejected without appending.
- Redirects are checked before the next connection and restricted to the pinned
  HTTPS PhysioNet release paths.
- Completed bytes still pass exact-size, ZIP, CRC, SHA-256 calculation, and
  same-directory atomic promotion before reuse.
- The path-free catalog reports whether a verified partial can be resumed and
  how many bytes are saved.
- The Patient owner UI shows paused/resumable state, percentage, downloaded and
  total bytes, transfer rate, ETA, a native progress bar, and explicit resume or
  safe-restart wording.

This improves reliability rather than claiming to increase PhysioNet's raw
network throughput. No unofficial mirror or aggressive parallel downloader was
introduced.

## Verification

- Official endpoint probe (bounded Range request): final response `206`,
  `Content-Range: bytes 1048576-2097151/16189661`, 1 MiB transferred.
- Isolated browser E2E:
  - first download reached 6 MiB, then the QA server was intentionally stopped;
  - the partial and strong-ETag receipt remained;
  - after restart the page showed `正在断点续传已校验的官方数据包` at 6 MiB / 38%;
  - the next live sample showed 9 MiB / 58%, 191.3 KB/s, ETA 34 seconds;
  - 1280×720 had no document horizontal overflow or card/progress clipping.
- Focused and owner regression:
  `110 passed, 1 warning` across demo-source backend, Patient demo/ECharts
  executable contracts, static route ownership, and repository packaging.
- `ruff check`, JS syntax, Patient CSS owner presence/absence, brace/comment
  balance, and `git diff --check` passed.
- The isolated QA cache was removed afterward. The user's prepared MIMIC cache
  and active source were not modified; no eICU full-download claim is made.

