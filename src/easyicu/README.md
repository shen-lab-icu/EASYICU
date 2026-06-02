# `easyicu` — package module map

This is the import-facing map of the `easyicu` package for contributors.
For installation, usage, and the project pitch, read the repository
[`README.md`](../../README.md) at the repo root first. This file only
explains *how the code is layered* so cross-file changes are feasible.

EasyICU is a layered toolkit. The top level is ~75 modules, but they sort
into five layers; the boundaries below are the ones that matter when a
change touches more than one file.

## 1. Source / concept abstraction (the foundation)

**Concepts — not database-specific variable names — are the unit of
cross-database analysis.** Adding a clinical variable means editing the
dictionary plus its callbacks, not adding a code path per database.

- `config.py`, `datasource.py`, `resources.py` — `ICUDataSource`,
  `DataSourceConfig` / `Registry`, and loaders for `data/data-sources.json`.
  Map each of the 6 supported databases (MIMIC-III, MIMIC-IV, eICU-CRD,
  AmsterdamUMCdb, HiRID, SICdb) to local tables.
- `concept.py` + `concept_schema.py` + `concept_expr_parser.py` +
  `concept_loader.py` + `concept_callback_apply.py` +
  [`data/concept-dict.json`](data/concept-dict.json) — the
  dictionary-driven concept layer. `concept_schema.py` holds the pure
  data classes; `concept_expr_parser.py` the R-style expression helpers;
  `concept_loader.py` the dictionary-loading shims;
  `concept_callback_apply.py` the source-level callback dispatcher;
  `concept.py` still hosts `ConceptResolver` and re-exports the split
  modules, so `from easyicu.concept import X` keeps working.
- `concept_callbacks.py`, `callbacks.py`, `callbacks_missing.py` —
  callback chains that normalise raw rows into concept values.
- `table.py` — `ICUTable`, `IdTbl`, `TsTbl`, `WinTbl`, `PvalTbl` and the
  rbind/cbind/merge helpers. The typed pandas wrapper used everywhere
  downstream.

> **Three callback dispatchers, by design.** New callbacks go in exactly
> one of: `concept_callback_apply._apply_callback` (source-level),
> `concept_callbacks.CALLBACK_REGISTRY` (concept-level derived scores), or
> `ConceptResolver._load_single_concept` (special-cases). Do **not**
> replicate dispatch across them.

## 2. Convert step (raw → prepared)

- `data_converter.py`, `setup_data.py` — `DataConverter` is the **single
  converter engine**; both the extraction API and the webapp call
  `DataConverter.convert_all()`. It turns raw CSV / CSV.GZ / tar.gz dumps
  into the prepared, ricu-style sharded Parquet layout (end-to-end
  pyarrow by default, zstd compression). **Every extraction API assumes
  data has already been converted — never bypass this** when reasoning
  about a user-facing flow.

## 3. Public Python API

- `api.py` (`load_concepts`, `load_sofa`, `load_sofa2`, `load_sepsis3`,
  `load_vitals`, `load_labs`, domain loaders) — the surface external users
  hit. `load_concepts.py` and the `easy.py` one-liners sit alongside it.

## 4. Clinical score implementations

- `sofa2.py`, `sepsis_sofa2.py`, `sepsis.py`, `kdigo_aki.py`, `scores.py`,
  `circ_failure.py` — clinical scores layered on top of the concept layer.
  [`data/sofa2-dict.json`](data/sofa2-dict.json) extends the main
  dictionary with the SOFA-2 dependency closure (see
  [`data/README.md`](data/README.md)).

## 5. Sub-packages (each has its own README)

- [`webapp/`](webapp/README.md) — the Streamlit web layer; clinician /
  reviewer no-code entry point. Owns the AI opt-in gate invariant.
- [`research_agent/`](research_agent/README.md) — the optional,
  evidence-bound analysis-agent layer (question + cohort → auditable
  manuscript scaffold).
- [`data/`](data/README.md) — the JSON concept dictionaries that drive
  the concept layer.
- `visualization/` — plotting helpers used by the webapp and notebooks.

## Invariants the package relies on

- **Prepared data (post-conversion) is the shared contract.** Raw
  CSV/CSV.GZ/tar.gz is not a valid `data_path` for the extraction APIs.
- **AI-assistant changes stay explicitly advisory and human-confirmed.**
- The `data/*.bak_before_*.json` snapshots are intentional history — do
  not sweep them while doing unrelated refactors, and do not commit new
  ones casually.
