# Pi Copilot reviewer-grade complete research demo

**Task ID:** `PI-COPILOT-REVIEWER-GRADE-PRODUCT-DEMO`  
**Date:** 2026-08-11  
**Implementation commit:** `a1e5d8b`  
**Branch:** `fix/pi-workspace-review-20260809`

## Outcome

The read-only Pi Copilot product demo now presents one inspectable research journey from a natural-language question through Idea Mining, literature review, study setup, extraction, Plan review, analysis, result interpretation, and a manuscript draft. It remains an engineering product demonstration rather than formal paper evidence.

The historical E1 source run used `sep3_sofa2_max`. The UI no longer presents this as standard Sepsis-3:

- every selected exposure/result label now says **experimental SOFA-2 sensitivity indicator**;
- the persistent demo note says it is **not standard Sepsis-3**;
- the current live-product rule is stated explicitly: generic Sepsis-3 maps to SOFA-1, and SOFA-2 is used only when requested;
- the unchanged historical figure is retained byte-for-byte, with a visible projection note explaining its legacy label.

The workflow count is now `7/8`, not `7/7`: the automated stages are complete, while the manuscript remains locked at `human_review_required`. Completed stage reason codes now have user-facing descriptions instead of the fallback “waiting for the preceding stage.”

## Browser UAT

Desktop viewport: `1327 × 969`.

- 11/11 distinct product/artifact entry points opened their intended right-side preview.
- 9/9 literature records opened individually: Singer, STROBE, RECORD, Suissa, Anderson, Durrleman, Sterne, ricu, and MIMIC-IV.
- “Start my own study” exited the read-only demo, and the complete demo could be reopened.
- `document.scrollWidth == viewport width == 1327`; horizontal overflow was false.
- the final workflow projection showed `7/8` and `Draft locked pending clinical and methods review`.

Accepted screenshots:

- `output/ui_qa/20260811_web_copilot_reviewer_demo/01-literature-evidence.png`
- `output/ui_qa/20260811_web_copilot_reviewer_demo/02-evidence-bound-plan.png`
- `output/ui_qa/20260811_web_copilot_reviewer_demo/03-agent-result-figure.png`
- `output/ui_qa/20260811_web_copilot_reviewer_demo/04-locked-manuscript.png`

## Focused verification

```text
.venv/bin/pytest -q tests/test_pi_copilot_static.py tests/test_webserver_static_routes.py
91 passed

node --check src/easyicu/webserver/static/js/screens-guided-pi-demo.js
node --check src/easyicu/webserver/static/js/screens-guided-pi.js
git diff --check
all passed
```

Earlier in the same development iteration, the focused Copilot/literature/Plan contract suite completed with 245 cases after three fixture/signature corrections. No full exact-head CI matrix and no Canonical9 provider batch were started for this small Web iteration.

## Remaining boundary

This demonstrates the complete product interaction and inspectable outputs. It does **not** convert the historical engineering canary into scientific authority, and it does not replace the remaining real-project Plan approval, human scientific review, or formal Canonical9 execution gates.
