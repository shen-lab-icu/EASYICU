# Benchmark assets

This is the single repository owner for benchmark fixtures, case protocols,
frozen experiment suites, and benchmark-only catalogs.  Production code under
`src/easyicu/` must not import these packages.

- `cases/`: reusable development case plug-ins.
- `catalogs/`: source-derived catalogs used by benchmark and discovery tools.
- `idea_mining/`: frozen Idea Mining evaluation and validation inputs.
- `figure2_canonical9/`: the formal Canonical9 paper protocol and evaluator.
- `meta_generalization/`: cross-case generalization fixtures.
- `agent_experiments/`: experiment-package registry metadata.

There is deliberately no parallel top-level `benchmark/` directory.  New
benchmark assets must be placed under one of the owners above.
