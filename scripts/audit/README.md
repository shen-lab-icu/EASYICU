# EasyICU concept-dict audit pipeline

Scripts for continuous auditing of `src/easyicu/data/concept-dict.json` and
`src/easyicu/data/sofa2-dict.json` mappings against the raw ICU databases.

## Files

| Script | Purpose |
|---|---|
| `regenerate_user_concept_tracker.py` | Diffs EasyICU concept-dict vs ricu's installed concept-dict, regenerates `docs/user_added_concepts_tracker.md`. Holds `AUDIT` and `AUDIT_HISTORY` dicts as source-of-truth for per-concept × per-DB status. |
| `verify_concept_dict_changes.py` | End-to-end verification that every itemid in concept-dict resolves in the raw DB dictionary tables (d_items / variable_reference / d_references / numericitems / drugitems). Catches fabricated/typo'd itemids. Outputs `docs/concept_dict_change_verification.json`. |
| `audit_cvp_comprehensive.py` | Synonym-based search for CVP across all source tables of all 6 DBs (English + Dutch + German). Pattern is reusable: copy and adapt SYN regex for any concept. |
| `audit_concept_strict.py` | Generic strict (word-boundary + exclude lists) per-concept × per-DB itemid completeness check. Use `--concepts X Y` or `--category ventilator` or `--all-user-added`. |

## Workflow when adding/auditing a concept

1. **Identify candidates** — run `audit_concept_strict.py --concepts new_concept` (after adding to concept-dict.json) to see what raw items the dictionary search finds vs what's currently mapped.
2. **Verify candidates** — manually inspect the dict labels; filter out wrong-concept matches (e.g. Apache scores, alarms, other drugs in same class).
3. **Edit `concept-dict.json`** — add the missing itemids; preserve `_comment` audit trail.
4. **Run `verify_concept_dict_changes.py`** — confirms all itemids resolve. Catches fabricated IDs.
5. **Smoke-extract** — `from easyicu.api import extract_database; extract_database(database='X', modules=['vitals'], max_patients=100)` should return non-empty data for the new concept.
6. **Update tracker** — edit `AUDIT` dict in `regenerate_user_concept_tracker.py`, add an `AUDIT_HISTORY` entry, run the regenerator.

## Initial audit results (2026-05-27)

See `docs/concept_dict_audit_log.md` for the full audit log:
- 77 user-added concepts × 6 DBs all audited
- Discovered + fixed: AUMC CVP missing 4.96M-row variant 20926; eICU CVP missing nurseCharting; MIMIC-III peep/pip/ps mismapped to PIP/Plateau/Vt itemids; HiRID Pharma 20 drugs systematically added; SIC 15 drugs added; AUMC drugitems 8 drugs added
- Bug caught by verification: 4 fabricated HiRID propofol IDs replaced with real ones
- Final medication coverage: 36-44/45 per DB (up from 16-28/45 starting state)
