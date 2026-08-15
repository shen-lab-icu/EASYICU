# Security Policy

## Supported versions

EasyICU is research software under active development. Security fixes are
prepared against the current default branch; older commits, tags, and local
research snapshots receive best-effort support only.

| Version | Security fixes |
|---|---|
| Current default branch | Yes |
| Older commits, tags, and snapshots | Best effort |

A software version is not a clinical-validation claim. Clinical mapping and
publication authority remain governed by the shipped clinical contracts and
release evidence.

## Report a vulnerability privately

Use GitHub's
[private vulnerability reporting](https://github.com/shen-lab-icu/EASYICU/security/advisories/new).
If that form is unavailable, open a public issue that asks a maintainer for a
private channel without including vulnerability details.

Please include, when available:

- the affected EasyICU version or exact commit;
- the affected entry point and deployment context;
- minimal reproduction steps using synthetic data;
- the expected and observed security boundary;
- likely impact and any known mitigations.

Do not include patient-level data, protected health information, credentials,
tokens, database exports, private prompts, or institution-confidential paths in
any report. Replace them with synthetic fixtures and redact logs before upload.

## Security boundaries

- The native Web app is intended for loopback-only use unless an operator adds
  an independently authenticated access layer.
- LLM prompt minimization does not provide execution isolation. Generated code
  may still process the prepared cohort made available to its runner.
- The host subprocess runner is not a strong security sandbox. Use the
  digest-bound Docker runner and its fail-closed capability checks when process,
  filesystem, and network isolation are required.
- Evidence hashes establish artifact identity; they do not prove that arbitrary
  code, clinical mappings, or statistical interpretations are correct.
- Never use production patient data in a public issue, pull request, test
  fixture, demo bundle, or vulnerability proof of concept.

Clinical-definition or data-mapping errors that could change research results
should normally be reported through the issue tracker with synthetic evidence.
Use private reporting when disclosure would expose a security weakness,
sensitive configuration, or confidential data location.

## Coordinated handling

Maintainers will acknowledge a complete private report, reproduce it with safe
fixtures, classify the affected owner boundary, and coordinate disclosure after
a fix and focused regression test exist. Timelines depend on severity and
maintainer availability; this document does not promise a response-time SLA.
