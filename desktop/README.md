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

- macOS 12+
- Rust/Cargo
- Node 22.19+
- Python 3.10+

Run:

```bash
cd desktop
python3 scripts/build_macos.py
```

The build script creates an isolated build venv, installs the exact locked Pi
runtime, freezes FastAPI and its Python runtime as an installed onedir runtime
with PyInstaller, bundles Node, then builds both `EasyICU.app` and a DMG. The
installed runtime avoids decompressing hundreds of megabytes on every launch.
End users do not need Python, Node, Git, or the EasyICU source tree.

Local builds receive an ad-hoc signature and are suitable for internal testing.
Public distribution requires an Apple Developer ID, hardened-runtime signing,
and notarization; set `APPLE_SIGNING_IDENTITY` and use the standard Tauri/Apple
release credentials when producing a public release.

## Current platform boundary

This branch produces and verifies the macOS Apple Silicon distribution. The
Tauri shell and Python entry point are platform-neutral, but Windows artifacts
must be built and tested on Windows before they are claimed as supported.
