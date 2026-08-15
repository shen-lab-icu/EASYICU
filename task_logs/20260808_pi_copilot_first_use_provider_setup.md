# Pi Copilot first-use model-service setup

Date: 2026-08-08

Branch: `feat/pi-copilot-shell`

Baseline: `e34a4a8`

Implementation commit: `2a9d107` — `feat(web): gate Pi chat on verified provider setup`

## Outcome

Guided Copilot now opens with a model-service setup gate when Pi is not
verified. The chat composer and session creation path are unavailable until
the user explicitly authorizes external AI use, supplies the service address,
model and API credential, and the local FastAPI owner verifies both
authentication and exact model availability through the bounded `/models`
endpoint.

On success, the UI immediately creates the governed Pi session. Existing users
can reopen the same setup surface from **Model service** in the chat header.
The deterministic local Guided workflow remains an explicit fallback.

## Security and ownership

- `pi_copilot/provider_config.py` owns typed Pi shell configuration,
  verification, the private credential file and verification receipt.
- The credential file and receipt are written atomically with mode `0600`.
  The receipt contains only a configuration fingerprint and verification
  metadata; it never contains the API credential.
- Failed authentication, unavailable models, rejected destinations and invalid
  responses write neither file.
- The browser does not put the credential in state or local storage. It clears
  the password input immediately after constructing the local setup request.
- Responses, runtime status, sessions, logs and Pi tool results never return
  credential values.
- A changed credential, endpoint, model, provider or transport invalidates the
  receipt and closes the chat gate again.
- Process-level `EASYICU_PI_*` values remain an operator override but cannot
  bypass receipt verification.
- Credential-bearing URL validation now has the dependency-neutral owner
  `webserver/provider_url_security.py`; both the Pi setup and scientific
  provider adapter consume that contract without importing one another's
  private implementation.
- Reconfiguration restarts only the Pi sidecar configuration and preserves
  independent context/output/session-token budget settings.

## Verification

- Pi/Web/provider-boundary focused gate: **163 passed**.
- Ruff on changed Python and test owners: passed.
- Node syntax checks for `api.js` and `screens-guided-pi.js`: passed.
- Import Linter: **7 kept / 0 broken**.
- deptry: no dependency issues.
- Wheel and sdist build: passed; wheel contains the provider configuration,
  shared URL-security and Guided Pi frontend owners.
- Browser QA at 1440×900 and 1024×768: first-use setup is the only Pi center
  surface, password input is empty with `type=password` and
  `autocomplete=new-password`, document width equals viewport width, and the
  console has zero errors/warnings.

The repository-wide suite was not rerun for this scoped Web/Pi change. The
previous canonical full baseline remains `9855 passed / 13 skipped / 0
failed`; this task used owner, adjacent boundary, privacy, static-route and
browser gates.

## Real connection boundary

No real credential was read or copied and no model/provider request was sent
during implementation. The user can now perform the authorized check directly
in Guided Copilot: the local endpoint and model defaults are prefilled, while
the credential must be entered by the user. The previously planned governed
tool canary remains the next release evidence after that setup succeeds.
