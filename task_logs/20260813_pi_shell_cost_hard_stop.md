# Pi shell conservative dollar-cost hard stop

Date: 2026-08-13  
Branch: `fix/pi-workspace-review-20260809`  
Scope: external-review finding 8 only; development checkpoint, not formal experiment evidence.

## Adjudication

Confirmed (major before stable multi-user use). The Pi conversational shell
already bounded tokens and provider-call counts, but it exposed
`pricing_available=false` and had no provider-price-bound per-message or
per-session dollar ceiling.

## Fix

- Added an all-or-none, operator-supplied pricing contract: input/output USD per
  million tokens plus per-message/per-session cost ceilings.
- Every provider call now reserves a conservative upper-bound cost from the
  verified input-token upper bound and maximum authorized output before any
  network request is sent.
- The cumulative reservation and a SHA-256 pricing binding are persisted in the
  Pi session JSONL. Restart preserves the session ceiling; changing or removing
  pricing from a cost-bound session fails closed.
- Historical sessions without cost receipts cannot silently adopt pricing after
  they have already made provider calls.
- When a local proxy does not publish trustworthy prices, pricing remains
  unavailable and the existing token/call ceilings remain the explicit hard-stop
  fallback. EasyICU does not invent a price for `gpt-5.6-luna`.

## Verification

- `tests/test_pi_copilot_gateway.py`: 23 passed.
- Node syntax checks passed for `main.mjs` and `shell-budget.mjs`.
- Ruff and `git diff --check` passed.
- Architecture ratchet: no lower-is-better metric regressed.
- Full exact-head CI was intentionally not run during Web E1 iteration.

