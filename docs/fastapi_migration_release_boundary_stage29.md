# FastAPI Migration Release Boundary - Stage29

Date: 2026-06-24

## Decision

The EasyICU Web UI release boundary is now FastAPI native. The legacy Streamlit
package has been removed from the active package, default entrypoint, tests, and
release artifacts.

This document is the PR-readiness summary for the Stage14-28 migration line and
the Stage29 clean-install release verification.

## Release Notes

- `easyicu-webapp` now launches the native FastAPI Web UI.
- The legacy Streamlit WebApp package is removed from the active package
  boundary.
- The `easyicu-webapp-legacy` entrypoint and `webapp-legacy` extra are removed.
- Native FastAPI static assets are packaged in both wheel and sdist:
  - `easyicu/webserver/static/index.html`
  - `easyicu/webserver/static/js/app.js`
  - `easyicu/webserver/static/css/app.css`
- Provider status and agent-provider readiness remain dormant by default. A
  release smoke must keep `ai_enabled=false`, `ready=false`,
  `client_constructed=false`, `network_calls=0`, and `secrets_returned=false`.
- Streamlit can only be recovered from git history or the Stage27 archive patch.
  It is no longer a supported fallback path.

## Migration Summary

FastAPI native route coverage is release-ready for the maintained Web UI:

- Entry, help alias, guided flow, settings, dictionary, and workspace states.
- Extraction advanced filters on registered active exports.
- Patient Review bounded drilldown without row-level identifier leakage.
- Cohort Review aggregate parity from registered source metadata.
- Cross-DB aggregate parity with fail-closed behavior for fewer than two valid
  registered exports.
- Agent run/history/review/signoff/provider-status native paths.
- Numeric evidence audit gate before any future reportable or draft-unlock
  path.

Legacy Streamlit decommission coverage:

- Route split CSS disabled by default, then deleted.
- Shared helpers migrated out of the deleted `easyicu.webapp` package.
- Default entrypoint moved to FastAPI native.
- Legacy Streamlit tests and package removed from the default test and package
  boundary.
- Release archive contract verifies no legacy package is present.

## Commit Boundary

Current branch:

- Branch: `ux/easyicu-web-copilot-agent-projects`
- Base for repository PR context: `origin/main` at `d9d30d9`
- Stage28 HEAD before this Stage29 document: `db898b6`

The branch contains non-WebApp research-agent commits interleaved with the Web UI
work. For a WebApp-only PR, include the WebApp migration commits and exclude the
interleaved research-agent commits listed below.

Recommended WebApp release commit groups:

1. FastAPI native readiness and provider safety:
   - `41eb9f1 Add native FastAPI agent run safety baseline`
   - `b5967ef Support strict provider point-fire schema`
   - `fdd3009 Add native FastAPI fallback readiness path`
2. Legacy CSS baseline, guarded cleanup, and decommission:
   - `c91be32 Add legacy Streamlit CSS fallback baseline`
   - `184aaad Clean legacy Agent CSS stale blocks`
   - `83a8024 Clean additional legacy Agent CSS stale blocks`
   - `9ce6a8a Clean stale Guided CSS blocks`
   - `2ee6b52 Clean source-missing legacy CSS blocks`
   - `7526b52 Clean stale Cohort and States CSS blocks`
   - `45fd02a Guard and clean legacy CSS duplicate cascade`
   - `4c77a67 Extend CSS guard and clean stale residual blocks`
   - `39ba699 Extend legacy CSS stateful guard`
   - `72bfba8 Use stateful guard to clean legacy CSS cascade`
   - `1a92af1 Use stateful guard to clean more legacy CSS`
   - `63bba1c Disable legacy Streamlit route CSS by default`
   - `7fb9f89 Remove inactive legacy Streamlit split CSS`
3. Streamlit package decommission:
   - `88c5e14 Move shared webapp helpers out of Streamlit package`
   - `f02dd36 Deprecate Streamlit webapp entrypoints`
   - `b2cdf14 Mark Streamlit tests and scripts as legacy`
   - `d1da22e Remove legacy Streamlit WebApp package`
   - `f57e685 Clean residual Streamlit decommission references`
4. Release packaging hardening:
   - `db898b6 Harden FastAPI webapp release packaging`
   - Stage29 release-boundary documentation commit

Interleaved non-WebApp commits currently on the branch:

- `e179a4c Surface exposure timing in agent cohort universe`
- `11e2bb9 Surface outcome event time (death_time) in agent cohort universe`
- `ffff381 Teach agent to handle pre-baseline event times and construct onset time-zero`

If the PR is meant to be WebApp-only, split or cherry-pick instead of opening the
entire current branch as-is.

## Stage29 Verification

Clean build:

- Build venv: `/tmp/easyicu_stage29_build_venv`
- Dist directory: `/tmp/easyicu_stage29_dist`
- Artifacts:
  - `/tmp/easyicu_stage29_dist/easyicu-1.0.0-py3-none-any.whl`
  - `/tmp/easyicu_stage29_dist/easyicu-1.0.0.tar.gz`

Archive content checks:

- Wheel static asset count: `28`
- Wheel required native assets present: `true`
- Wheel legacy `easyicu/webapp` count: `0`
- Wheel legacy entrypoint/reference found: `false`
- Sdist static asset count: `30`
- Sdist required native assets present: `true`
- Sdist legacy `src/easyicu/webapp` count: `0`

Clean install:

- Install venv: `/tmp/easyicu_stage29_install_venv`
- Installed wheel with `easyicu[webapp]`.
- `easyicu-webapp --help` displayed the native FastAPI CLI.
- Installed package resource checks:
  - `static_exists=True`
  - `index_exists=True`
  - `app_js_exists=True`
  - `app_css_exists=True`
  - `legacy_webapp_spec=None`

Installed-server smoke:

- Server port: `127.0.0.1:8779`
- Environment included:
  - `EASYICU_DISABLE_PROVIDER_ENV_FILE=1`
  - `EASYICU_RUNTIME_DIR=/tmp/easyicu_stage29_runtime`
- `/api/health`: HTTP 200
- `/api/catalog`: HTTP 200
- Server was stopped after QA; stop returned shell status `130` from Ctrl-C,
  which is expected for foreground uvicorn.

Browser route QA:

- Command:

```bash
python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8779/ --out-dir output/playwright
```

- Report:
  `output/playwright/native_fastapi_route_qa_20260624_202631/route_qa.json`
- Result: passed.
- Console errors: `0`.
- Horizontal overflow: `overflowX=0` for all checked desktop and 393x852 mobile
  routes.
- Unknown hash runtime check rewrote to `#entry` and rendered
  `Welcome to EasyICU`.
- QA still records non-blocking offscreen/clipped counts for complex pages such
  as `agent` and mobile `cohort`; these remain diagnostic counters rather than
  release blockers under the current native route QA gate.

Focused release tests:

```bash
pytest -q tests/test_release_hardening_p0.py tests/test_release_archive_contract.py tests/test_repository_contract.py tests/test_webserver_static_routes.py
```

Result: `30 passed, 1 skipped`.

Additional checks:

- Touched native JS syntax: passed.
- Active legacy import/entrypoint scan:
  - No `easyicu.webapp`
  - No `easyicu-webapp-legacy`
  - No `webapp-legacy`
  - No `--run-legacy-streamlit`
- Provider dormant smoke:
  - `ai_enabled=false`
  - `ready=false`
  - `client_constructed=false`
  - `network_calls=0`
  - `secrets_returned=false`

Archival note:

- `docs/_internal/**` still contains historical path-only references to
  `src/easyicu/webapp`. They are archival notes, not active package, default
  docs, scripts, tests, or release artifacts.

## Clean-Machine Checklist

Run from the `EASYICU/` repo root:

```bash
python -m venv /tmp/easyicu_release_build
/tmp/easyicu_release_build/bin/python -m pip install --upgrade pip build
/tmp/easyicu_release_build/bin/python -m build --outdir /tmp/easyicu_release_dist
```

Inspect archive contents:

```bash
python - <<'PY'
from pathlib import Path
import tarfile, zipfile

dist = Path('/tmp/easyicu_release_dist')
wheel = next(dist.glob('*.whl'))
sdist = next(dist.glob('*.tar.gz'))

with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
    assert 'easyicu/webserver/static/index.html' in names
    assert 'easyicu/webserver/static/js/app.js' in names
    assert 'easyicu/webserver/static/css/app.css' in names
    assert not any(name.startswith('easyicu/webapp/') for name in names)
    entry = next(name for name in names if name.endswith('.dist-info/entry_points.txt'))
    entry_text = zf.read(entry).decode('utf-8')
    assert 'easyicu-webapp-legacy' not in entry_text
    assert 'easyicu.webapp' not in entry_text

with tarfile.open(sdist) as tf:
    names = set(tf.getnames())
    assert 'easyicu-1.0.0/src/easyicu/webserver/static/index.html' in names
    assert 'easyicu-1.0.0/src/easyicu/webserver/static/js/app.js' in names
    assert 'easyicu-1.0.0/src/easyicu/webserver/static/css/app.css' in names
    assert not any(name.startswith('easyicu-1.0.0/src/easyicu/webapp/') for name in names)
PY
```

Clean install and launch:

```bash
python -m venv /tmp/easyicu_release_install
/tmp/easyicu_release_install/bin/python -m pip install --upgrade pip
/tmp/easyicu_release_install/bin/python -m pip install '/tmp/easyicu_release_dist/easyicu-1.0.0-py3-none-any.whl[webapp]'
/tmp/easyicu_release_install/bin/easyicu-webapp --help

EASYICU_DISABLE_PROVIDER_ENV_FILE=1 \
EASYICU_RUNTIME_DIR=/tmp/easyicu_release_runtime \
/tmp/easyicu_release_install/bin/easyicu-webapp run --host 127.0.0.1 --port 8779
```

In another shell:

```bash
python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8779/ --out-dir output/playwright
pytest -q
git diff --check
```

Provider dormant smoke:

```bash
EASYICU_DISABLE_PROVIDER_ENV_FILE=1 python - <<'PY'
from easyicu.webserver import provider_adapter
status = provider_adapter.provider_readiness('openai', ai_enabled=False)
assert status.get('ready') is False
assert status.get('client_constructed') is False
assert status.get('network_calls') == 0
assert status.get('secrets_returned') is False
PY
```

## PR Readiness Verdict

The FastAPI migration line is ready for a release-boundary PR after splitting out
or explicitly accounting for the three interleaved non-WebApp research-agent
commits. No further Streamlit/CSS migration work is required for this release
boundary.
