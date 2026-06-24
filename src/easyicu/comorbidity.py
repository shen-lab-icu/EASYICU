"""Charlson & Elixhauser comorbidity coding (Quan 2005 algorithm).

This module turns a table of ICD diagnosis codes into per-admission
comorbidity flags and the two standard weighted summary indices:

* **Charlson Comorbidity Index (CCI)** — 17 conditions, original
  Charlson (1987) weights.
* **Elixhauser** — 31 conditions, with the van Walraven (2009)
  weighted point score.

Code sets are the Quan 2005 *enhanced* ICD-9-CM and ICD-10 coding
algorithms (Quan H, et al. *Med Care* 2005;43(11):1130-9). This is a
published, fixed standard — not a heuristic. Codes are matched as
**dot-free prefixes**, version-aware (ICD-9 vs ICD-10), against all
diagnoses of an admission ("all hospital diagnoses" window — the
convention used by the source paper and most ICU studies).

Pure / data-source agnostic: :func:`flag_comorbidities` takes a tidy
``DataFrame[id, code, version]`` and returns one row per id. The
EasyICU loader wiring lives in ``api.py`` / the concept layer; this
module has no EasyICU imports so it stays unit-testable in isolation.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


# --------------------------------------------------------------------------
# Range-expansion helpers (keep the code tables readable)
# --------------------------------------------------------------------------
def _icd9_range(start: int, end: int) -> List[str]:
    """Inclusive 3-digit ICD-9 prefix range, e.g. 430..438 -> ['430',...]."""
    return [f"{n:03d}" for n in range(start, end + 1)]


def _icd10_range(letter: str, start: int, end: int) -> List[str]:
    """Inclusive ICD-10 prefix range within a letter, e.g. C00..C26."""
    return [f"{letter}{n:02d}" for n in range(start, end + 1)]


def _norm(code: object) -> str:
    """Normalise an ICD code to a dot-free, upper-case, stripped token."""
    if code is None:
        return ""
    s = str(code).strip().upper().replace(".", "").replace(" ", "")
    return s


# --------------------------------------------------------------------------
# Charlson — Quan 2005 enhanced code sets (17 conditions)
# Keyed condition -> {"icd9": [...prefixes...], "icd10": [...prefixes...]}
# --------------------------------------------------------------------------
CHARLSON: Dict[str, Dict[str, List[str]]] = {
    "mi": {
        "icd9": ["410", "412"],
        "icd10": ["I21", "I22", "I252"],
    },
    "chf": {
        "icd9": ["39891", "40201", "40211", "40291", "40401", "40403",
                 "40411", "40413", "40491", "40493", "4254", "4255",
                 "4256", "4257", "4258", "4259", "428"],
        "icd10": ["I099", "I110", "I130", "I132", "I255", "I420", "I425",
                  "I426", "I427", "I428", "I429", "I43", "I50", "P290"],
    },
    "pvd": {
        "icd9": ["0930", "4373", "440", "441", "4431", "4432", "4433",
                 "4434", "4435", "4436", "4437", "4438", "4439", "4471",
                 "5571", "5579", "V434"],
        "icd10": ["I70", "I71", "I731", "I738", "I739", "I771", "I790",
                  "I792", "K551", "K558", "K559", "Z958", "Z959"],
    },
    "cevd": {
        "icd9": ["36234"] + _icd9_range(430, 438),
        "icd10": ["G45", "G46", "H340"] + _icd10_range("I", 60, 69),
    },
    "dementia": {
        "icd9": ["290", "2941", "3312"],
        "icd10": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    },
    "cpd": {
        "icd9": ["4168", "4169"] + _icd9_range(490, 505) + ["5064", "5081", "5088"],
        "icd10": ["I278", "I279"] + _icd10_range("J", 40, 47)
                 + _icd10_range("J", 60, 67) + ["J684", "J701", "J703"],
    },
    "rheum": {
        "icd9": ["4465", "7100", "7101", "7102", "7103", "7104", "7140",
                 "7141", "7142", "7148", "725"],
        "icd10": ["M05", "M06", "M315", "M32", "M33", "M34", "M351",
                  "M353", "M360"],
    },
    "pud": {
        "icd9": _icd9_range(531, 534),
        "icd10": ["K25", "K26", "K27", "K28"],
    },
    "mild_liver": {
        "icd9": ["07022", "07023", "07032", "07033", "07044", "07054",
                 "0706", "0709", "570", "571", "5733", "5734", "5738",
                 "5739", "V427"],
        "icd10": ["B18", "K700", "K701", "K702", "K703", "K709", "K713",
                  "K714", "K715", "K717", "K73", "K74", "K760", "K762",
                  "K763", "K764", "K768", "K769", "Z944"],
    },
    "diab": {  # diabetes without chronic complication
        "icd9": ["2500", "2501", "2502", "2503", "2508", "2509"],
        "icd10": ["E100", "E101", "E106", "E108", "E109", "E110", "E111",
                  "E116", "E118", "E119", "E120", "E121", "E126", "E128",
                  "E129", "E130", "E131", "E136", "E138", "E139", "E140",
                  "E141", "E146", "E148", "E149"],
    },
    "diabwc": {  # diabetes with chronic complication
        "icd9": ["2504", "2505", "2506", "2507"],
        "icd10": ["E102", "E103", "E104", "E105", "E107", "E112", "E113",
                  "E114", "E115", "E117", "E122", "E123", "E124", "E125",
                  "E127", "E132", "E133", "E134", "E135", "E137", "E142",
                  "E143", "E144", "E145", "E147"],
    },
    "para": {  # hemiplegia or paraplegia
        "icd9": ["3341", "342", "343", "3440", "3441", "3442", "3443",
                 "3444", "3445", "3446", "3449"],
        "icd10": ["G041", "G114", "G801", "G802", "G81", "G82", "G830",
                  "G831", "G832", "G833", "G834", "G839"],
    },
    "renal": {
        "icd9": ["40301", "40311", "40391", "40402", "40403", "40412",
                 "40413", "40492", "40493", "582", "5830", "5831", "5832",
                 "5834", "5836", "5837", "585", "586", "5880", "V420",
                 "V451", "V56"],
        "icd10": ["I120", "I131", "N032", "N033", "N034", "N035", "N036",
                  "N037", "N052", "N053", "N054", "N055", "N056", "N057",
                  "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940",
                  "Z992"],
    },
    "malignancy": {
        "icd9": (_icd9_range(140, 172) + _icd9_range(174, 195)
                 + _icd9_range(200, 208) + ["2386"]),
        "icd10": (_icd10_range("C", 0, 26) + _icd10_range("C", 30, 34)
                  + _icd10_range("C", 37, 41) + ["C43"]
                  + _icd10_range("C", 45, 58) + _icd10_range("C", 60, 76)
                  + _icd10_range("C", 81, 85) + ["C88"]
                  + _icd10_range("C", 90, 97)),
    },
    "msld": {  # moderate or severe liver disease
        "icd9": ["4560", "4561", "4562", "5722", "5723", "5724", "5725",
                 "5726", "5727", "5728"],
        "icd10": ["I850", "I859", "I864", "I982", "K704", "K711", "K721",
                  "K729", "K765", "K766", "K767"],
    },
    "metacanc": {  # metastatic solid tumour
        "icd9": _icd9_range(196, 199),
        "icd10": _icd10_range("C", 77, 80),
    },
    "aids": {
        "icd9": ["042", "043", "044"],
        "icd10": ["B20", "B21", "B22", "B24"],
    },
}

# Original Charlson (1987) weights
CHARLSON_WEIGHTS: Dict[str, int] = {
    "mi": 1, "chf": 1, "pvd": 1, "cevd": 1, "dementia": 1, "cpd": 1,
    "rheum": 1, "pud": 1, "mild_liver": 1, "diab": 1, "diabwc": 2,
    "para": 2, "renal": 2, "malignancy": 2, "msld": 3, "metacanc": 6,
    "aids": 6,
}

# Hierarchy: the more severe condition supersedes the milder one when
# both are present (standard Charlson handling).
CHARLSON_HIERARCHY = {
    "diab": "diabwc",        # if diabwc present, drop diab
    "mild_liver": "msld",    # if msld present, drop mild_liver
    "malignancy": "metacanc",  # if metacanc present, drop malignancy
}


# --------------------------------------------------------------------------
# Elixhauser — Quan 2005 enhanced code sets (31 conditions)
# --------------------------------------------------------------------------
ELIXHAUSER: Dict[str, Dict[str, List[str]]] = {
    "chf": {
        "icd9": ["39891", "40201", "40211", "40291", "40401", "40403",
                 "40411", "40413", "40491", "40493", "4254", "4255",
                 "4256", "4257", "4258", "4259", "428"],
        "icd10": ["I099", "I110", "I130", "I132", "I255", "I420", "I425",
                  "I426", "I427", "I428", "I429", "I43", "I50", "P290"],
    },
    "arrhythmia": {
        "icd9": ["4260", "42613", "4267", "4269", "42610", "42612", "4270",
                 "4271", "4272", "4273", "4274", "4276", "4278", "4279",
                 "7850", "99601", "99604", "V450", "V533"],
        "icd10": ["I441", "I442", "I443", "I456", "I459", "I47", "I48",
                  "I49", "R000", "R001", "R008", "T821", "Z450", "Z950"],
    },
    "valvular": {
        "icd9": ["0932", "394", "395", "396", "397", "424", "7463", "7464",
                 "7465", "7466", "V422", "V433"],
        "icd10": ["A520", "I05", "I06", "I07", "I08", "I091", "I098", "I34",
                  "I35", "I36", "I37", "I38", "I39", "Q230", "Q231", "Q232",
                  "Q233", "Z952", "Z953", "Z954"],
    },
    "pulmcirc": {
        "icd9": ["4150", "4151", "416", "4170", "4178", "4179"],
        "icd10": ["I26", "I27", "I280", "I288", "I289"],
    },
    "pvd": {
        "icd9": ["0930", "4373", "440", "441", "4431", "4432", "4433",
                 "4434", "4435", "4436", "4437", "4438", "4439", "4471",
                 "5571", "5579", "V434"],
        "icd10": ["I70", "I71", "I731", "I738", "I739", "I771", "I790",
                  "I792", "K551", "K558", "K559", "Z958", "Z959"],
    },
    "htn_unc": {  # hypertension, uncomplicated
        "icd9": ["401"],
        "icd10": ["I10"],
    },
    "htn_comp": {  # hypertension, complicated
        "icd9": ["402", "403", "404", "405"],
        "icd10": ["I11", "I12", "I13", "I15"],
    },
    "paralysis": {
        "icd9": ["3341", "342", "343", "3440", "3441", "3442", "3443",
                 "3444", "3445", "3446", "3449"],
        "icd10": ["G041", "G114", "G801", "G802", "G81", "G82", "G830",
                  "G831", "G832", "G833", "G834", "G839"],
    },
    "neuro": {  # other neurological disorders
        "icd9": ["3319", "3320", "3321", "3334", "3335", "33392", "334",
                 "335", "3362", "340", "341", "345", "3481", "3483", "7803",
                 "7843"],
        "icd10": ["G10", "G11", "G12", "G13", "G20", "G21", "G22", "G254",
                  "G255", "G312", "G318", "G319", "G32", "G35", "G36", "G37",
                  "G40", "G41", "G931", "G934", "R470", "R56"],
    },
    "chronic_pulm": {
        "icd9": ["4168", "4169"] + _icd9_range(490, 505) + ["5064", "5081", "5088"],
        "icd10": ["I278", "I279"] + _icd10_range("J", 40, 47)
                 + _icd10_range("J", 60, 67) + ["J684", "J701", "J703"],
    },
    "diab_unc": {
        "icd9": ["2500", "2501", "2502", "2503"],
        "icd10": ["E100", "E101", "E109", "E110", "E111", "E119", "E120",
                  "E121", "E129", "E130", "E131", "E139", "E140", "E141",
                  "E149"],
    },
    "diab_comp": {
        "icd9": ["2504", "2505", "2506", "2507", "2508", "2509"],
        "icd10": ["E102", "E103", "E104", "E105", "E106", "E107", "E108",
                  "E112", "E113", "E114", "E115", "E116", "E117", "E118",
                  "E132", "E133", "E134", "E135", "E136", "E137", "E138",
                  "E142", "E143", "E144", "E145", "E146", "E147", "E148"],
    },
    "hypothyroid": {
        "icd9": ["2409", "243", "244", "2461", "2468"],
        "icd10": ["E00", "E01", "E02", "E03", "E890"],
    },
    "renal_fail": {
        "icd9": ["40301", "40311", "40391", "40402", "40403", "40412",
                 "40413", "40492", "40493", "585", "586", "5880", "V420",
                 "V451", "V56"],
        "icd10": ["I120", "I131", "N18", "N19", "N250", "Z490", "Z491",
                  "Z492", "Z940", "Z992"],
    },
    "liver": {
        "icd9": ["07022", "07023", "07032", "07033", "07044", "07054",
                 "0706", "0709", "456", "5710", "5712", "5713", "5714",
                 "5715", "5716", "5718", "5719", "5723", "5728", "5733",
                 "5734", "5738", "5739", "V427"],
        "icd10": ["B18", "I85", "I864", "I982", "K70", "K711", "K713",
                  "K714", "K715", "K717", "K72", "K73", "K74", "K760",
                  "K762", "K763", "K764", "K765", "K766", "K767", "K768",
                  "K769", "Z944"],
    },
    "pud_nobleed": {
        "icd9": ["5317", "5319", "5327", "5329", "5337", "5339", "5347",
                 "5349"],
        "icd10": ["K257", "K259", "K267", "K269", "K277", "K279", "K287",
                  "K289"],
    },
    "aids": {
        "icd9": ["042", "043", "044"],
        "icd10": ["B20", "B21", "B22", "B24"],
    },
    "lymphoma": {
        "icd9": _icd9_range(200, 202) + ["2030", "2386"],
        "icd10": _icd10_range("C", 81, 85) + ["C88", "C96", "C900", "C902"],
    },
    "metacanc": {
        "icd9": _icd9_range(196, 199),
        "icd10": _icd10_range("C", 77, 80),
    },
    "solidtumor": {
        "icd9": _icd9_range(140, 172) + _icd9_range(174, 195),
        "icd10": (_icd10_range("C", 0, 26) + _icd10_range("C", 30, 34)
                  + _icd10_range("C", 37, 41) + ["C43", "C45"]
                  + _icd10_range("C", 45, 58) + _icd10_range("C", 60, 76)),
    },
    "rheum": {
        "icd9": ["7010", "7100", "7101", "7102", "7103", "7104", "7108",
                 "7109", "7112", "714", "7193", "720", "725", "7285",
                 "72889", "72930"],
        "icd10": ["L940", "L941", "L943", "M05", "M06", "M08", "M120",
                  "M123", "M30", "M31", "M32", "M33", "M34", "M35", "M45",
                  "M461", "M468", "M469"],
    },
    "coag": {
        "icd9": ["286", "2871", "2873", "2874", "2875"],
        "icd10": ["D65", "D66", "D67", "D68", "D691", "D693", "D694",
                  "D695", "D696"],
    },
    "obesity": {
        "icd9": ["2780"],
        "icd10": ["E66"],
    },
    "weightloss": {
        "icd9": _icd9_range(260, 263) + ["7832", "7994"],
        "icd10": ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634",
                  "R64"],
    },
    "fluid": {
        "icd9": ["2536", "276"],
        "icd10": ["E222", "E86", "E87"],
    },
    "blanemia": {  # blood loss anemia
        "icd9": ["2800"],
        "icd10": ["D500"],
    },
    "defanemia": {  # deficiency anemia
        "icd9": ["2801", "2808", "2809", "281"],
        "icd10": ["D508", "D509", "D51", "D52", "D53"],
    },
    "alcohol": {
        "icd9": ["2911", "2912", "2913", "2915", "2918", "2919", "30390",
                 "30393", "30500", "30503", "V113"],
        "icd10": ["F10", "E52", "G621", "I426", "K292", "K700", "K703",
                  "K709", "T51", "Z502", "Z714", "Z721"],
    },
    "drug": {
        "icd9": ["2920", "29282", "29289", "2929", "304", "30520", "30523",
                 "30530", "30533", "30590", "30593"],
        "icd10": ["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19",
                  "Z715", "Z722"],
    },
    "psychoses": {
        "icd9": ["29304", "29381", "29382", "295", "29811", "29814",
                 "2989", "29910", "29911"],
        "icd10": ["F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302",
                  "F312", "F315"],
    },
    "depression": {
        "icd9": ["2962", "2963", "2965", "3004", "309", "311"],
        "icd10": ["F204", "F313", "F314", "F315", "F32", "F33", "F341",
                  "F412", "F432"],
    },
}

# van Walraven (2009) weights for Elixhauser conditions
ELIXHAUSER_VW_WEIGHTS: Dict[str, int] = {
    "chf": 7, "arrhythmia": 5, "valvular": -1, "pulmcirc": 4, "pvd": 2,
    "htn_unc": 0, "htn_comp": 0, "paralysis": 7, "neuro": 6, "chronic_pulm": 3,
    "diab_unc": 0, "diab_comp": 0, "hypothyroid": 0, "renal_fail": 5,
    "liver": 11, "pud_nobleed": 0, "aids": 0, "lymphoma": 9,
    "metacanc": 12, "solidtumor": 4, "rheum": 0, "coag": 3, "obesity": -4,
    "weightloss": 6, "fluid": 5, "blanemia": -2, "defanemia": -2,
    "alcohol": 0, "drug": -7, "psychoses": 0, "depression": -3,
}


def _build_prefix_lookup(table: Dict[str, Dict[str, List[str]]]):
    """Pre-normalise + index code sets for fast version-aware matching.

    Returns (cond_order, {version: [(prefix, condition)...] sorted by len desc}).
    """
    cond_order = list(table.keys())
    by_version: Dict[int, List[tuple]] = {9: [], 10: []}
    for cond, sets in table.items():
        for pref in sets.get("icd9", []):
            by_version[9].append((_norm(pref), cond))
        for pref in sets.get("icd10", []):
            by_version[10].append((_norm(pref), cond))
    # longest prefix first so the most specific code set wins
    for v in by_version:
        by_version[v].sort(key=lambda t: len(t[0]), reverse=True)
    return cond_order, by_version


_CHARLSON_LOOKUP = _build_prefix_lookup(CHARLSON)
_ELIX_LOOKUP = _build_prefix_lookup(ELIXHAUSER)


def _match_one(code: str, version: int, lookup_by_version) -> List[str]:
    """Return all conditions whose prefix matches a single normalised code."""
    prefixes = lookup_by_version.get(int(version) if version in (9, 10, "9", "10") else version, [])
    hits = []
    for pref, cond in prefixes:
        if pref and code.startswith(pref):
            hits.append(cond)
    return hits


def flag_comorbidities(
    diagnoses: pd.DataFrame,
    *,
    system: str = "charlson",
    id_col: str = "id",
    code_col: str = "code",
    version_col: str = "version",
    default_version: int = 9,
) -> pd.DataFrame:
    """Map ICD diagnosis rows to per-id comorbidity flags + weighted index.

    Args:
        diagnoses: tidy frame, one ICD code per row.
        system: ``"charlson"`` or ``"elixhauser"``.
        id_col / code_col / version_col: column names in ``diagnoses``.
        default_version: ICD version assumed when ``version_col`` is
            absent or null (MIMIC-III is all ICD-9 -> 9).

    Returns:
        One row per id: boolean flag per condition plus, for Charlson,
        ``charlson_index`` (CCI), and for Elixhauser
        ``elixhauser_vw`` (van Walraven score) + ``elixhauser_count``.
    """
    system = system.lower()
    if system == "charlson":
        cond_order, lookup, weights = (
            _CHARLSON_LOOKUP[0], _CHARLSON_LOOKUP[1], CHARLSON_WEIGHTS)
    elif system == "elixhauser":
        cond_order, lookup, weights = (
            _ELIX_LOOKUP[0], _ELIX_LOOKUP[1], ELIXHAUSER_VW_WEIGHTS)
    else:
        raise ValueError("system must be 'charlson' or 'elixhauser'")

    if diagnoses.empty:
        cols = [id_col] + cond_order
        return pd.DataFrame(columns=cols)

    df = diagnoses[[id_col, code_col]].copy()
    if version_col in diagnoses.columns:
        ver = pd.to_numeric(diagnoses[version_col], errors="coerce").fillna(default_version).astype(int)
    else:
        ver = pd.Series(default_version, index=diagnoses.index)
    df["_code"] = diagnoses[code_col].map(_norm)
    df["_ver"] = ver.values

    ids = df[id_col].unique()

    # Match on DISTINCT (code, version) pairs — tens of thousands instead
    # of millions of rows — then join back to ids. A code can belong to
    # several conditions (e.g. distinct Charlson and Elixhauser-shared
    # categories), so we keep every (code, condition) hit.
    distinct = df[["_code", "_ver"]].drop_duplicates()
    code_cond_rows: List[tuple] = []
    for version, sub in distinct.groupby("_ver"):
        prefixes = lookup.get(int(version), lookup.get(default_version, []))
        if not prefixes:
            continue
        for code in sub["_code"].values:
            if not code:
                continue
            for pref, cond in prefixes:
                if pref and code.startswith(pref):
                    code_cond_rows.append((code, int(version), cond))

    out = pd.DataFrame({id_col: ids})
    if code_cond_rows:
        code_cond = pd.DataFrame(code_cond_rows, columns=["_code", "_ver", "_cond"])
        present = (
            df.merge(code_cond, on=["_code", "_ver"])[[id_col, "_cond"]]
            .drop_duplicates()
        )
        pres_wide = (
            present.assign(_v=True)
            .pivot_table(index=id_col, columns="_cond", values="_v", aggfunc="any")
            .reindex(index=ids, columns=cond_order)
        )
        for cond in cond_order:
            out[cond] = pres_wide[cond].fillna(False).astype(bool).values
    else:
        for cond in cond_order:
            out[cond] = False

    if system == "charlson":
        # apply severity hierarchy before weighting
        weighted = out.copy()
        for milder, severe in CHARLSON_HIERARCHY.items():
            drop = weighted[severe]
            weighted.loc[drop, milder] = False
        idx = pd.Series(0, index=out.index, dtype=int)
        for cond, w in weights.items():
            idx += weighted[cond].astype(int) * w
        out["charlson_index"] = idx.values
        # expose the hierarchy-adjusted flags (so index == sum(flags*w))
        for milder in CHARLSON_HIERARCHY:
            out[milder] = weighted[milder].values
    else:
        score = pd.Series(0, index=out.index, dtype=int)
        count = pd.Series(0, index=out.index, dtype=int)
        for cond, w in weights.items():
            present = out[cond].astype(int)
            score += present * w
            count += present
        out["elixhauser_vw"] = score.values
        out["elixhauser_count"] = count.values

    return out


# --------------------------------------------------------------------------
# EasyICU loader — wires flag_comorbidities to each database's diagnosis
# table and the ICU stay-id schema. Per-database because the id structure
# and diagnosis storage differ (MIMIC: hadm_id ICD table -> map to stay_id;
# eICU: per-stay comma-joined ICD-9/10 string; SICdb: single ICD10Main).
# HiRID/AmsterdamUMCdb ship no ICD diagnoses -> documented N/A.
# --------------------------------------------------------------------------
# ICU stay-id column produced per database (matches the concept layer).
_STAY_ID_COL = {
    "miiv": "stay_id", "miiv_demo": "stay_id",
    "mimic": "icustay_id", "mimic_demo": "icustay_id",
    "eicu": "patientunitstayid", "eicu_demo": "patientunitstayid",
    "sic": "CaseID", "sic_demo": "CaseID",
}
# Databases with no usable ICD diagnosis source.
_NO_ICD_DATABASES = {"hirid", "aumc"}


def _build_datasource(database: str, data_path: object):
    from .datasource import ICUDataSource
    from .resources import load_data_sources

    config = load_data_sources().get(database)
    if config is None:
        raise ValueError(f"Unknown database: {database}")
    if data_path is None:
        import os

        root = os.environ.get("EASYICU_DATA_PATH", "")
        if root:
            try:
                from .data_paths import find_database_path

                data_path = find_database_path(root, database)
            except Exception:
                data_path = os.path.join(root, database)
    return ICUDataSource(config=config, base_path=str(data_path) if data_path else None)


def _table_df(data_source, name: str, columns=None) -> pd.DataFrame:
    tbl = data_source.load_table(name, columns=columns)
    df = getattr(tbl, "data", None)
    if df is None and hasattr(tbl, "to_df"):
        df = tbl.to_df()
    if df is None:
        df = tbl  # already a DataFrame
    return df


def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _explode_eicu_codes(series: pd.Series) -> pd.DataFrame:
    """eICU diagnosis.icd9code is a comma-joined mix, e.g. '414.00, I25.10'.

    Split, infer ICD version per token (leading letter -> ICD-10, else
    ICD-9), and return a long frame of (idx, code, version) preserving the
    original row index for a join back to patientunitstayid.
    """
    exploded = series.fillna("").str.split(",")
    rows = []
    for idx, tokens in exploded.items():
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            version = 10 if tok[:1].isalpha() else 9
            rows.append((idx, tok, version))
    return pd.DataFrame(rows, columns=["_row", "code", "version"])


def load_comorbidity(
    database: str,
    data_path: object = None,
    *,
    system: str = "charlson",
    patient_ids=None,
    max_patients: object = None,
    verbose: bool = False,
):
    """Load per-ICU-stay comorbidity flags + index for a database.

    Returns a DataFrame keyed by the database's ICU stay id with one
    boolean column per condition plus the weighted index
    (``charlson_index`` or ``elixhauser_vw``/``elixhauser_count``).
    Databases without ICD diagnoses (HiRID, AmsterdamUMCdb) return an
    empty frame — comorbidity is genuinely unavailable there.
    """
    db = database.lower()
    if db in _NO_ICD_DATABASES:
        if verbose:
            print(f"[comorbidity] {database} ships no ICD diagnoses — N/A")
        return pd.DataFrame()

    ds = _build_datasource(database, data_path)

    if db in ("miiv", "miiv_demo", "mimic", "mimic_demo"):
        dx = _lower_cols(_table_df(ds, "diagnoses_icd"))
        code_col = "icd_code" if "icd_code" in dx.columns else "icd9_code"
        ver_col = "icd_version" if "icd_version" in dx.columns else None
        dx = dx.rename(columns={code_col: "code"})
        if ver_col is None:
            dx["version"] = 9  # MIMIC-III is ICD-9 only
        else:
            dx = dx.rename(columns={ver_col: "version"})
        flags = flag_comorbidities(
            dx[["hadm_id", "code", "version"]], system=system,
            id_col="hadm_id", code_col="code", version_col="version",
        )
        stays = _lower_cols(_table_df(ds, "icustays"))
        # MIMIC-III icustays keys the ICU stay as icustay_id; MIMIC-IV as stay_id.
        stay_col = "stay_id" if "stay_id" in stays.columns else "icustay_id"
        stays = stays[["hadm_id", stay_col]]
        out = stays.merge(flags, on="hadm_id", how="inner").drop(columns="hadm_id")

    elif db in ("eicu", "eicu_demo"):
        dx = _lower_cols(_table_df(ds, "diagnosis"))
        long = _explode_eicu_codes(dx["icd9code"])
        long = long.merge(
            dx[["patientunitstayid"]].reset_index().rename(columns={"index": "_row"}),
            on="_row", how="left",
        )
        flags = flag_comorbidities(
            long[["patientunitstayid", "code", "version"]], system=system,
            id_col="patientunitstayid", code_col="code", version_col="version",
        )
        out = flags

    elif db in ("sic", "sic_demo"):
        cases = _table_df(ds, "cases")
        # CamelCase columns in SICdb cases
        colmap = {c.lower(): c for c in cases.columns}
        cid, icd = colmap.get("caseid"), colmap.get("icd10main")
        dx = cases[[cid, icd]].rename(columns={cid: "CaseID", icd: "code"})
        dx["version"] = 10
        flags = flag_comorbidities(
            dx[["CaseID", "code", "version"]], system=system,
            id_col="CaseID", code_col="code", version_col="version",
        )
        out = flags

    else:
        if verbose:
            print(f"[comorbidity] no diagnosis adapter for {database}")
        return pd.DataFrame()

    if patient_ids:
        stay_col = _STAY_ID_COL.get(db)
        if stay_col and stay_col in out.columns:
            out = out[out[stay_col].isin(list(patient_ids))]
    return out.reset_index(drop=True)


__all__ = [
    "CHARLSON", "ELIXHAUSER", "CHARLSON_WEIGHTS", "ELIXHAUSER_VW_WEIGHTS",
    "flag_comorbidities", "load_comorbidity",
]
