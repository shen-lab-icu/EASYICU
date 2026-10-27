# EasyICU Desktop

EasyICU Desktop is a thin, local-first Tauri shell around the existing FastAPI
WebApp. It does not duplicate extraction, Idea Mining, Research Agent, evidence,
or publication authority. The desktop process owns only the native window and
the lifecycle of one frozen loopback backend.

## Security and state boundary

- The backend binds only to a dynamically selected `127.0.0.1` port.
- Every launch receives a random 256-bit desktop token. The token is passed to
  the backend through its environment, not its process arguments.
- The native window exchanges the one-time bootstrap query for an HttpOnly,
  SameSite-strict cookie. Requests without the exact token are rejected.
- EasyICU-owned state lives under the operating system application-data root;
  it does not replace the user's real `HOME` or require access to the source
  checkout.
- The loading page receives no shell or filesystem capability. The FastAPI UI
  is loaded as ordinary loopback content and cannot invoke Tauri commands.

Packaging does not change EasyICU's scientific or clinical validation status.
The app remains an evidence-bound research/analysis tool unless a separate
validation and regulatory process establishes a broader claim.

## Build on macOS

Requirements for the build machine only:

- macOS 14+ on Apple Silicon (the minimum is set in `tauri.conf.json` to match
  the locked NumPy/SciPy binaries)
- Rust/Cargo
- A self-contained Node 22.19+ distribution, such as the official Node archive
  (an nvm installation works). Homebrew's dynamically linked Node is rejected
  because copying that executable alone cannot run on a clinician's Mac.
- Python 3.11 (the Python library itself continues to support Python 3.10+)

Run:

```bash
cd desktop
python3.11 scripts/build_macos.py
```

Set `EASYICU_DESKTOP_NODE` to the selected distribution's `bin/node` when it is
not the default, and put that distribution's `bin` directory first on `PATH`
so npm uses the same Node. The build checks macOS library dependencies before
freezing the backend.

The build script creates an isolated build venv, installs hash-checked Python
dependencies from `requirements-macos-arm64-py311.lock`, installs EasyICU as a
wheel, and installs the locked Pi runtime into that installed package. It then
freezes FastAPI and its Python runtime as an installed onedir runtime
with PyInstaller, bundles Node, then builds both `EasyICU.app` and a DMG. The
installed runtime avoids decompressing hundreds of megabytes on every launch.
End users do not need Python, Node, Git, or the EasyICU source tree.

Build from a clean Git checkout. The backend freezes the installed package;
it does not add the source checkout to PyInstaller's import path. Build inputs
(commit, dirty flag, Python/Node versions, and dependency lock hashes) are
recorded in `.build/build-inputs.json`. Keep that receipt with the artifacts.
The macOS build also verifies the installed and frozen execution-kernel source
identities, uses the locked Cargo graph, and records the final DMG SHA-256.

Build caches can be kept outside a synchronized Documents checkout:

```bash
EASYICU_DESKTOP_BUILD_DIR="$TMPDIR/easyicu-build" \
CARGO_TARGET_DIR="$TMPDIR/easyicu-cargo-target" \
python3.11 desktop/scripts/build_macos.py
```

With these overrides, `build-inputs.json` is in `EASYICU_DESKTOP_BUILD_DIR`
and the DMG is in `CARGO_TARGET_DIR/release/bundle/dmg/`. Copy the receipt and
final DMG to a retained release directory before the temporary area is cleared.

## Build on Windows

Target: Windows x64, Python 3.11, Tauri NSIS, **per-user installation**. Build
on Windows with Rust's MSVC toolchain and Visual Studio C++ Build Tools,
Python 3.11 x64, and Node 22.19+ x64 (including npm and its distribution
LICENSE). These are build-machine requirements. End users need no Python,
Node, Git, Docker, source checkout, or administrator rights. The installer
includes the WebView2 offline installer for machines missing that runtime.

`requirements-windows-x64-py311.lock` is the resolved, hash-pinned dependency
set; the build refuses a missing or placeholder lock before invoking any build
tool. To refresh it, run this exact command from the repository root, review
the pins and hashes, and commit the resulting lock:

```powershell
uv pip compile pyproject.toml desktop/build-requirements.in --extra webapp --python-version 3.11 --python-platform x86_64-pc-windows-msvc --generate-hashes --no-emit-package easyicu --output-file desktop/requirements-windows-x64-py311.lock
```

Then, from the repository root:

```powershell
python desktop/scripts/build_windows.py --dry-run
python desktop/scripts/build_windows.py
```

`--dry-run` prints a JSON build plan and lock status without running commands
or creating files; it also works on Linux/macOS. `--backend-only` prepares the
frozen backend and Node without Tauri packaging. `EASYICU_DESKTOP_PYTHON` and
`EASYICU_DESKTOP_NODE` can select specific executable paths. The default build
venv and receipt are in `desktop/.build/windows/`; override with
`EASYICU_DESKTOP_BUILD_DIR`. The NSIS installer is under
`desktop/src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/`, or under
`CARGO_TARGET_DIR/x86_64-pc-windows-msvc/release/bundle/nsis/` when set.
The shared build owner stages `pi-coding-agent` as the root package at the short
`p/` resource path, so its private dependencies live directly in `p/node_modules/`.
The gateway and skill remain under `p/src/`, resolved through `packaged_app_dir`.
Node's package self-reference loads the agent from its root exports. Gateway
metadata that collides with the agent's metadata lives in `p/gateway/`; the
runtime owner reads the original gateway metadata and unchanged lock bytes for
integrity checks and explicit private-runtime installs.
The owner prunes only
staged `node_modules`: TypeScript declarations, `dist-types/`, source maps, and
docs/tests/examples directories at package roots without runtime
`main`/`exports`/`bin` entries. Nested directories with these names remain intact:
compiled code can reach them through relative imports or `require`, as in
`yaml/dist/compose/composer.js` loading `../doc/directives.js`.
It also drops `*.ts`/`*.mts`/`*.cts` sources from packages whose runtime entries
do not use TypeScript. Runtime-referenced directories and source siblings remain.
Root staging preserves the agent's private graph, retains required outer
fallback packages and checks every selected package's declared dependency/peer
versions before moving files. It discards unused outer copies. A version change
fails with `desktop_pi_root_stage_failed`. The staged tree is then hoisted:
duplicate leaves must match in version and
payload, and moves into empty slots preserve dependency and peer versions.
Conflicts, private dependencies and packages with bin shims stay in place.
The wheel and source checkout's dependency trees stay intact. Bundled Node
checks the gateway's syntax and imports the Pi agent, Pi AI, typebox and the
compiled provider adapters, including the lazy OpenAI completions adapter.
It resolves Pi AI from both the gateway and agent using ESM import conditions;
`desktop_pi_import_failed` identifies a failing entry or adapter specifier.
This smoke does not make a provider call.

Before Tauri packaging, the build measures paths relative to the installation
root using Tauri's resource destinations. Windows fails with
`desktop_bundle_path_budget_exceeded` above **160 UTF-16 characters**, reserving
100 characters of MAX_PATH for the installation root. macOS reports the same
overflow as a warning. `longest-paths.json` records the ten longest paths before
the check can fail, and `build-inputs.json` records pruned file counts/bytes,
root-staging and hoisted/deduplicated package counts and the path report. Both drivers expose
the staging and check plan with `--dry-run`.
Keep staging/cache paths short as build tools can also impose path limits.

The pinned gateway lock explains why the prior hoister retained the deepest
SDKs: nested/top-level versions differ for `@smithy/core` (3.24.3/3.31.1),
`@aws-sdk/core` (3.974.11/3.977.6), and `@aws-sdk/nested-clients`
(3.997.9/3.997.41); those entries declare no peers. For example, nested Smithy
core depends on `@smithy/types ^4.14.2` and `@aws-crypto/crc32 5.2.0`, while
the outer copy depends on `@smithy/types ^4.16.1`. Promoting the shrinkwrapped
agent removes the stable `node_modules/@earendil-works/pi-coding-agent/` prefix
(45 characters) without substituting newer outer SDK versions or raising the budget.

Both platform drivers share the wheel installation, locked Pi installation,
Node bundling, freezing, source preservation, and build receipt helpers.
The Windows venv is isolated, Python dependencies use `--require-hashes`,
EasyICU is installed as a wheel with no editable/source-checkout import path,
and `pip check` verifies dependencies. Cargo builds with `--locked`.
The backend is a PyInstaller **onedir** runtime, with its Python runtime,
installed EasyICU `.py` sources and package data under `_internal`. Preserving
the exact installed sources is necessary for `build_execution_kernel_identity`;
the Windows build compares the installed and frozen source identities and
records the result in `build-inputs.json`, alongside the commit, dirty flag,
versions, dependency lock hashes, and installer SHA-256. Keep the receipt with
the installer. There is no runtime unpacking or modification of executables.

Windows application state, runtime files (including the WebView2 profile), and logs live under
`%LOCALAPPDATA%\EasyICU\{state,runtime,logs}`. The NSIS product is named
**EasyICU Desktop**, separating its default installation directory from this
data root. Keep a custom installation directory separate from user data too.
Arguments are passed as arrays, preserving spaces and non-ASCII characters;
Windows state paths use extended drive/UNC paths (`\\?\`) and the application
and backend manifests declare long-path awareness. External tools may still
have their own path-length limits.

Uninstall removes the application and retains the user's state and exports.
To explicitly remove retained data, close EasyICU, uninstall, then delete
`%LOCALAPPDATA%\EasyICU`; separately delete any exports saved elsewhere.
The installer does not silently erase those files. Production Windows releases
should use a maintainer's Authenticode signing setup; this lane produces an
unsigned internal-test installer and does not claim SmartScreen/Defender approval.

`.github/workflows/desktop_windows_build.yml` runs on every dev9 push, on `v*`
tags and on manual dispatch. Its Windows job runs on `windows-latest`: it builds
the installer, silently installs into a temporary profile path containing
spaces and Chinese characters, and runs the installed app with `--smoke-test`. This mode creates no WebView;
it launches the bundled loopback backend, waits for the authenticated
`/api/catalog` readiness endpoint, checks rejection of an invalid token, and
stops/reaps the backend. The smoke process has no Python, Node, Git, or Docker
on PATH. The workflow then silently uninstalls and checks that tiny synthetic
state/export sentinels remain. It uploads the installer and receipt only after
success; synthetic smoke diagnostics, including `longest-paths.json` even if
the build fails before installation, are retained on failure. This uses the
runner account with temporary profile/environment paths, not a new Windows
account. When startup fails, the job log shows the end of the backend log and
names the exception the backend raised.

The same workflow's macOS job runs on `macos-15` (Apple Silicon). It builds
the DMG, copies `EasyICU.app` out of it into a temporary profile, verifies the
signature, and starts the bundled backend with the arguments and token header
the shell uses (`desktop/scripts/smoke_macos.py`). It checks the same
readiness, invalid-token rejection and frozen durable startup with only system
paths on PATH. The macOS shell has no headless mode, so this does not open the
app window.

## Updating Python dependencies

The resolved committed lock is for macOS arm64 / Python 3.11 only. Its initial runtime
pins were seeded from the existing product development environment. Installing
the lock checks the distribution hashes; `pip check` then verifies the installed
dependency relationships. A lock update still requires packaging and user-flow
validation before release.

From the repository root, using uv:

```bash
uv pip compile pyproject.toml desktop/build-requirements.in \
  --extra webapp --python-version 3.11 --python-platform aarch64-apple-darwin \
  --generate-hashes --no-emit-package easyicu \
  --output-file desktop/requirements-macos-arm64-py311.lock
```

Existing pins are retained unless an upgrade is requested. Review intentional
updates with `--upgrade-package <name>` and commit the resulting lock. Node uses
the two committed `package-lock.json` files, and Rust uses `Cargo.lock`. Build
tools are maintainer dependencies; they are not required on an end user's Mac.

## Distribution

The app inside a locally built DMG receives an ad-hoc signature and is suitable
for internal testing. The Tauri target `.app` is an intermediate; distribute the
DMG. Its payload is copied into the system temporary directory before metadata
cleanup and signing, because File Provider can restore Finder metadata inside
a Documents checkout. Source files are untouched.
Public distribution requires an Apple Developer ID, hardened-runtime signing,
and notarization; set `APPLE_SIGNING_IDENTITY` and use the standard Tauri/Apple
release credentials when producing a public release.

Releases come from tags. Pushing a `v*` tag builds and smoke-tests both
installers at that commit, binds each to the tag's commit through its build
receipt (`desktop/scripts/verify_desktop_artifact.py`), and creates a **draft**
GitHub release with the Windows installer, the macOS DMG and one
`SHA256SUMS.txt`. A `v0.*` tag is labelled Pre-release. A maintainer reviews
the draft and publishes it. The version comes from `pyproject.toml`,
`desktop/package.json`, `src-tauri/tauri.conf.json` and `src-tauri/Cargo.toml`;
change all four together before tagging.

Both signing hooks stay disabled until certificates exist. An unsigned Windows
installer triggers a SmartScreen warning, and an ad-hoc-signed, un-notarized
DMG must be opened from Finder's context menu (or allowed under Privacy &
Security) the first time.

## Current platform boundary

Windows x64 and macOS Apple Silicon each have a CI build, install and
readiness lane, as described above. Neither has **yet been validated on a
clinician's PC**. After a lane's first successful run, describe its artifact
as "built and smoke-tested in CI, not yet validated on a clinician's PC"; do
not claim that result in advance.

Desktop startup requires no Docker. Research execution without Docker depends
on the host runner delivered separately in task `t7a`; packaging and the
readiness smoke test do not establish that research analyses run successfully.
Before claiming Windows support, validate normal GUI startup, WebView2 setup
under a standard user, clinical workflows, the host runner, upgrades/uninstall,
and long/Unicode user paths on an actual Windows PC.
