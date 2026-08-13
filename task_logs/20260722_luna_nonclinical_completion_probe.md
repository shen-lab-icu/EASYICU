# Luna non-clinical completion probe

Date: 2026-07-22 EDT  
Task: `FIG2-CANONICAL9-GATE` transport readiness only  
Branch: `refactor/agent-control-plane@2516b36`

## Scope

At the owner's request, the existing private loopback configuration was used to
verify only the local `gpt-5.6-luna` transport. The probe first verified that
the authenticated loopback `/v1/models` endpoint was available, then sent the
fixed non-clinical request `Reply exactly OK` with `max_tokens=4` and
`temperature=0`.

Result: HTTP 200 and exact sentinel `OK`.

## Explicit exclusions

The probe did not load or enumerate full0717, read a patient row, send clinical
data, create an Agent pipeline, invoke Docker, create a Canonical9 run, or
produce a paper-facing result. It is a transport readiness check only and does
not weaken P4: production input authority remains `0/9` pending source-bound
typed materialization plus the required data/identity/clinical/methods review.
