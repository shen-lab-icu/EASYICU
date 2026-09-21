# EasyICU concept dictionaries

This folder ships the JSON files that drive EasyICU's
dictionary-driven concept layer. They are loaded via
`easyicu.concept_loader.load_dictionary()` and
`easyicu.resources.load_concept_dictionaries(...)`.

The primary concept and data-source configuration includes modified material
derived from the GPL-3-licensed
[ricu](https://github.com/eth-mds/ricu) project. The combined EasyICU
distribution is therefore licensed under `GPL-3.0-only`; see the repository's
`LICENSE`, `COPYING`, and `THIRD_PARTY_NOTICES.md` files.

## Files

### `concept-dict.json` — primary dictionary

The canonical concept catalog: vital signs, labs, vasopressors,
medications, scores (SOFA-1, qSOFA, etc.), outcomes, and demographics.
Every cross-database analysis path expects this dictionary to be
loaded.

| Key shape | Example | Meaning |
|---|---|---|
| `concept_name` → callbacks per data source | `hr` → `{miiv: {…}, eicu: {…}, …}` | Per-database extraction rules |

Exact base, overlay, merged, database, clinical-contract, and capability counts
are generated from the shipped registries in
[`docs/catalog_summary.md`](../../../docs/catalog_summary.md). Do not maintain a
second hand-written count here.
The web-side catalog may report a larger loadable total because it also exposes
derived concepts and overlays that are computed by callbacks rather than stored
as standalone entries in this file.

### `sofa2-dict.json` — SOFA-2 overlay (NOT a peer file)

**This is not a second dictionary.** It is an **overlay** — a small set
of additions and overrides applied on top of `concept-dict.json` only
when an analysis explicitly opts into SOFA-2 semantics.

Two roles:

1. **13 new concepts** unique to SOFA-2: `sofa2`, `sofa2_resp`,
   `sofa2_coag`, `sofa2_liver`, `sofa2_cardio`, `sofa2_cns`,
   `sofa2_renal`, `sep3_sofa2`, plus the supporting concepts
   `adv_resp`, `delirium_positive`, `delirium_tx_proxy`,
   `delirium_tx_evidence`, deprecated alias `delirium_tx`, `motor_response`,
   `other_vaso`.
2. **8 overrides** of existing keys (`ecmo`, `ecmo_indication`,
   `mech_circ_support`, `rrt`, `rrt_criteria`, `uo_6h`, `uo_12h`,
   `uo_24h`). SOFA-2 needs subtly different callbacks for these — the
   overlay replaces the main-dict entry only when loaded together.

### Load patterns

```python
# Standard analysis — SOFA-1 / sepsis-3 / cohort assembly:
load_dictionary()                          # concept-dict.json only

# SOFA-2 sensitivity analysis:
load_concept_dictionaries(include_sofa2=True)
# or equivalently:
load_concept_dictionaries(extra_names=["sofa2-dict"])

# api.py / load_concepts.py auto-detect requested SOFA-2 concepts
# and pull in the overlay transparently — manual flag only needed
# when calling load_dictionary directly.
```

### Why not merge `sofa2-dict.json` into `concept-dict.json`?

The 8 override entries can't live in one flat dictionary without an
in-dict variant mechanism (e.g. `sofa1_callback` vs `sofa2_callback`
branches per concept). The current overlay design:

- Keeps `concept-dict.json` minimal and stable.
- Lets a SOFA-1-only analysis skip 21 unused keys.
- Lets the SOFA-2 override behavior live next to the SOFA-2-only
  additions, instead of bloating each affected concept entry.

Merging would force an extra abstraction layer (variants, dispatch by
analysis type) inside every modified concept entry. The pilot-2
reproducibility tooling depends on the loaded dictionary being
analysis-specific; the overlay keeps that contract explicit.

## History snapshots (`*.bak_before_*.json`)

The numerous `concept-dict.json.bak_before_*.json` files are
**intentional checkpoints** capturing the dictionary state before each
medication-batch ingest (batch2 → batch8, fentanyl/midazolam rate,
etc.). They are referenced by tests
(`tests/core/test_batch{2..8}_medications.py`) and by the data-engineering
review log. **Do not delete them and do not casually add new ones** —
project CLAUDE.md spells this rule out.
