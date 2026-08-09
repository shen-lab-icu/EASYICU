# Pi exact model ID and real conversation canary

- Date: 2026-08-08
- Branch: `feat/pi-copilot-shell`
- Trigger: browser review showed the local service reported `gpt-5.6-luna` while the Pi setup default was incorrectly written as `gpt5.6 luna`.

## Correction

The previous handoff overstated completion: it had verified the UI, typed API contracts, ownership boundaries, and browser layout, but had not sent a real Pi model message. This was not sufficient evidence for a working conversation.

All Pi CLIProxyAPI defaults are now the exact model identifier reported by the configured service: `gpt-5.6-luna`. The strict `/models` membership check remains unchanged; the fix does not add fuzzy model substitution or silently select another model.

Updated owners:

- provider configuration default
- typed provider setup route default
- Python gateway fallback
- pinned Node sidecar fallback/runtime status
- Guided Pi first-use form and CLIProxyAPI preset
- sidecar setup documentation and cache-buster regression

## Real browser canary

Using the real FastAPI WebApp at `http://127.0.0.1:8765/#guided` and the user-authorized local CLIProxyAPI service:

1. Selected an existing EasyICU research project.
2. Submitted the corrected exact model ID through the first-use verification form.
3. Observed successful provider verification and creation of a Pi AgentSession showing `gpt-5.6-luna`.
4. Sent a no-tool, PHI-free first message and received a real model response.
5. Sent a second message requesting the exact reply `真实对话已通过。`; the model returned `真实对话已通过。`.
6. Reloaded the page, selected the same project, and observed both turns restored from the Pi session.

No Configure/Run/Cancel grant was enabled and no EasyICU tool was invoked. The credential value is not recorded in this evidence file, project files, browser storage, Pi transcript, logs, or API responses; the verified provider owner saved it only in the private local credential file.

## Automated verification

- Focused Pi provider/static/routes/contracts/gateway suite passed.
- Ruff passed for the changed Python owners and tests.
- Node syntax checks passed for the sidecar and Guided Pi frontend.
- `git diff --check` passed.
