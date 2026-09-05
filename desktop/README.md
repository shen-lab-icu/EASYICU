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
- Node 22.19+
- Python 3.11 (the Python library itself continues to support Python 3.10+)

Run:

```bash
cd desktop
python3.11 scripts/build_macos.py
```

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

## Updating Python dependencies

The committed lock is for macOS arm64 / Python 3.11 only. Its initial runtime
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

Local builds receive an ad-hoc signature and are suitable for internal testing.
The signing step clears extended attributes from the generated app bundle,
including Finder metadata that macOS refuses to sign; source files are untouched.
Public distribution requires an Apple Developer ID, hardened-runtime signing,
and notarization; set `APPLE_SIGNING_IDENTITY` and use the standard Tauri/Apple
release credentials when producing a public release.

## Current platform boundary

The build target is macOS Apple Silicon. The
Tauri shell and Python entry point are platform-neutral, but Windows artifacts
must be built and tested on Windows before they are claimed as supported.
