# Agent experiment package registry

Every paper-facing or architecture-development run receives a stable ID such
as `FIG2-E2-DEV-001`, `FIG2-E3-CANONICAL-001`, or
`FIG2-H2-NEGCTRL-001`. The original `research_output/.../run_*` directory is
immutable evidence. A local package under `research_output/_packages/<ID>/`
provides a clean numbered view:

- `code/` — generated step scripts;
- `results/` — registered table/statistic evidence;
- `figures/` — registered figure evidence;
- `reports/` — readiness and human review reports;
- `provenance/` — manifest, plan, context, costs, and run identity;
- `package.json` — SHA-indexed inventory joining everything to the code commit.

The package uses relative links, so it does not duplicate large data. Final
submission artifacts can later be copied from an accepted package, while old
development runs remain separate and clearly labelled.

Create a package with:

```bash
python tools/agent_experiment_package.py \
  --experiment-id FIG2-E2-DEV-001 \
  --run-dir research_output/.../run_...
```

IDs are never reused. `DEV` means architecture development, `CANONICAL` means
the frozen full-data paper run, and `NEGCTRL` means an intentional scientific
fail-close such as an unanswerable exposure contrast.
