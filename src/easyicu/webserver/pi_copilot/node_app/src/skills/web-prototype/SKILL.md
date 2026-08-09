---
name: web-prototype
description: Build or revise a small, local, self-contained web artifact in the current EasyICU project workspace.
---

# Web prototype workflow

Use this workflow when the user asks for a webpage, calculator, dashboard, form,
interactive explanation, or another browser-viewable artifact.

1. List the current project files before choosing names.
2. Prefer one self-contained HTML file with inline CSS and JavaScript so the
   governed preview can render it without network access.
3. Label simulated values and unvalidated formulas explicitly. Never invent a
   clinically validated model, treatment recommendation, effect estimate, or
   patient-specific claim.
4. Write the artifact through the project workspace tool. Do not paste a full
   substitute code block into chat when the requested file can be created.
   Read the current file first before replacing or editing it, then pass the
   returned `sha256` as `expected_sha256` so another session cannot be silently
   overwritten.
5. Read the saved file back, run the bounded static check, then request the web
   preview tool.
6. Summarize what was actually written and checked. If a tool was blocked, say
   that no file was changed.

Keep all files relative to the isolated EasyICU project workspace. Do not ask
for or embed patient rows, identifiers, credentials, private source paths, or
external tracking scripts. Workspace file contents read by Pi may be sent to the
configured Pi model service, so never place PHI, patient rows, credentials, or
private clinical data in this workspace.
