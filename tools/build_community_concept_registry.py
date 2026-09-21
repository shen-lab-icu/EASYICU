#!/usr/bin/env python3
"""Build the audited concept registry for the four community ICU sources.

The compact declarations below are the reviewable source of truth.  The
generated JSON is consumed by :func:`easyicu.resources.load_dictionary`, while
the coverage artefacts make every standard concept auditable even when a
source cannot support it.  Generation never reads patient rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from easyicu.concept import ConceptDictionary


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "easyicu" / "data"
DOCS = ROOT / "docs"
DATABASES = ("nwicu", "zhejiang_eicu", "jinhua", "zigong")


def source(
    table: str | None,
    *,
    ids: object | None = None,
    sub_var: str | None = None,
    value_var: str | None = None,
    index_var: str | None = None,
    dur_var: str | None = None,
    callback: str | None = None,
    regex: str | None = None,
    class_name: str | None = None,
    win_type: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if table is not None:
        out["table"] = table
    for key, value in (
        ("ids", ids),
        ("sub_var", sub_var),
        ("value_var", value_var),
        ("index_var", index_var),
        ("dur_var", dur_var),
        ("callback", callback),
        ("regex", regex),
        ("class_name", class_name),
        ("win_type", win_type),
        ("_comment", comment),
    ):
        if value is not None:
            out[key] = value
    return out


def add(
    registry: dict[str, dict[str, Any]],
    concept: str,
    database: str,
    *sources: dict[str, Any],
) -> None:
    registry.setdefault(concept, {"sources": {}})["sources"][database] = list(
        sources
    )


def numeric(
    table: str,
    ids: object,
    sub_var: str,
    *,
    callback: str | None = None,
    value_var: str | None = None,
    index_var: str | None = None,
) -> dict[str, Any]:
    return source(
        table,
        ids=ids,
        sub_var=sub_var,
        value_var=value_var,
        index_var=index_var,
        callback=callback,
    )


def build_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}

    # Demographics, stay-level outcomes, and native vital signs.
    add(registry, "age", "nwicu", source("patients", value_var="anchor_age", class_name="col_itm"))
    add(registry, "age", "zhejiang_eicu", source("ptadmitable", value_var="Age_cut", callback="apply_map(c(`(0,18]` = 9, `(18,30]` = 24, `(30,40]` = 35, `(40,50]` = 45, `(50,60]` = 55, `(60,70]` = 65, `(70,80]` = 75, `(80,90]` = 85, `(90,150]` = 95))", class_name="col_itm"))
    add(registry, "age", "jinhua", source("medical_record_front_page", value_var="age", class_name="col_itm"))
    add(registry, "age", "zigong", source("dtbaseline", value_var="Age", class_name="col_itm"))
    add(registry, "sex", "nwicu", source("patients", value_var="gender", callback="apply_map(c(M = 'Male', F = 'Female'))", class_name="col_itm"))
    add(registry, "sex", "zhejiang_eicu", source("ptadmitable", value_var="Sex", class_name="col_itm"))
    add(registry, "sex", "jinhua", source("medical_record_front_page", value_var="patient_gender_en", class_name="col_itm"))
    add(registry, "sex", "zigong", source("dtbaseline", value_var="SEX", class_name="col_itm"))
    add(registry, "icu_unit_type", "nwicu", source("icustays", value_var="first_careunit", class_name="col_itm"))
    add(registry, "death", "nwicu", source("nwicu_outcomes", index_var="dischtime", value_var="hospital_expire_flag", callback="transform_fun(comp_na(`==`, 1))", class_name="col_itm"))
    add(registry, "death", "zhejiang_eicu", source("ptadmitable", index_var="Discharge_DateTime", value_var="StatusOnDischarge", callback="transform_fun(comp_na(`==`, 'Dead'))", class_name="col_itm"))
    add(registry, "death", "jinhua", source("patient_expire_hospital", value_var="expire_flag", callback="transform_fun(comp_na(`==`, 1))", class_name="col_itm"))
    add(registry, "los_hosp", "nwicu", source(None, callback="los_callback", class_name="fun_itm", win_type="hadm", comment="Native hospital admission/discharge interval."))
    add(registry, "los_hosp", "zhejiang_eicu", source("ptadmitable", value_var="DaysHospitalStay", class_name="col_itm"))
    add(registry, "los_hosp", "jinhua", source("medical_record_front_page", value_var="length_stay", class_name="col_itm"))
    for database in ("nwicu", "jinhua", "zigong"):
        add(registry, "los_icu", database, source(None, callback="los_callback", class_name="fun_itm", win_type="icustay", comment="Native/proxy semantics are recorded in community_preparation_manifest.json."))

    nw_vitals = {
        "hr": 320045,
        "temp": 323761,
        "resp": 320210,
        "spo2": 320277,
        "o2sat": 320277,
        "sbp": [320050, 320179],
        "dbp": [320051, 320180],
        "weight": 326531,
        "height": 326707,
    }
    for concept, ids in nw_vitals.items():
        callback = None
        if concept == "temp":
            callback = "convert_unit(fahr_to_cels, 'C', 'f')"
        elif concept == "weight":
            callback = "convert_unit(binary_op(`/`, 35.274), 'kg')"
        elif concept == "height":
            callback = "convert_unit(binary_op(`*`, 2.54), 'cm')"
        add(registry, concept, "nwicu", numeric("chartevents", ids, "itemid", callback=callback))

    zh_vitals = {
        "hr": ["Heart Rate", "Pulse rate"],
        "temp": "Temperature",
        "resp": "Respiratory rate",
        "spo2": "Oxygen saturation (Pulse Oxymetry)",
        "o2sat": "Oxygen saturation (Pulse Oxymetry)",
        "sbp": "Systolic Blood pressure",
        "dbp": "Diastolic Blood pressure",
        "weight": "Body weight",
        "height": "Height",
    }
    for concept, ids in zh_vitals.items():
        add(registry, concept, "zhejiang_eicu", numeric("vitalsign", ids, "VitalSign_DESC"))
    # The nursing SpO2 stream adds temporal coverage to the formal vital stream.
    registry["spo2"]["sources"]["zhejiang_eicu"].append(numeric("nursingchart_vitalsign", "SpO2", "NursingEvent_item"))
    registry["o2sat"]["sources"]["zhejiang_eicu"].append(numeric("nursingchart_vitalsign", "SpO2", "NursingEvent_item"))

    jh_vitals = {
        "hr": "Pulse",
        "temp": "Temperature",
        "resp": "Respiratory",
        "spo2": "Spo2",
        "o2sat": "Spo2",
        "sbp": "Sbp",
        "dbp": "Dbp",
        "weight": "Weight",
        "height": "Height",
        "bmi": "BMI",
    }
    for concept, ids in jh_vitals.items():
        add(registry, concept, "jinhua", numeric("vital_signs", ids, "subcategory_name_en"))

    zg_vitals = {
        "hr": "heart_rate",
        "temp": "temperature",
        "resp": "breathing",
        "spo2": "Blood_oxygen_saturation",
        "o2sat": "Blood_oxygen_saturation",
        "sbp": "Blood_pressure_high",
        "dbp": "Blood_pressure_low",
        "map": "MAP",
        "cvp": "Central_venous_pressure",
        "urine": "Urine_volume",
    }
    for concept, column in zg_vitals.items():
        add(registry, concept, "zigong", source("dtnursingchart", value_var=column, class_name="col_itm"))

    # NWICU laboratory dictionary: identifiers are stable for release 0.1.0.
    nw_labs: dict[str, tuple[object, str | None]] = {
        "glu": ([100001, 100045], None), "crea": (100002, None),
        "ca": (100003, None), "bun": (100004, None), "mch": (100005, None),
        "hct": (100006, None), "hgb": (100007, None), "rdw": (100008, None),
        "mg": (100009, None), "na": ([100010, 100050], None),
        "k": ([100011, 100047], None), "potassium": ([100011, 100047], None),
        "cl": (100012, None), "bicar": (100013, None), "bicarb": (100013, None),
        "tco2": (100013, None), "plt": (100014, None), "phos": (100015, None),
        "wbc": (100016, None), "mcv": (100017, None), "mchc": (100018, None),
        "rbc": (100019, None), "bili": (100020, None), "alb": (100021, None),
        "neut": (100022, None), "lymph": (100023, None), "eos": (100024, None),
        "basos": (100025, None), "monos": (100026, None),
        "total_protein": (100027, None), "fio2": (100029, None),
        "inr_pt": (100030, None), "lact": (100031, None), "ast": (100032, None),
        "alp": (100033, None), "pt": (100034, None),
        "anion_gap": ([100040, 100041], None), "alt": (100042, None),
        "cai": (100044, None), "ptt": (100046, None), "fgn": (100048, None),
        "bili_dir": (100049, None), "ferritin": (100052, None),
        "crp": (100053, "convert_unit(binary_op(`*`, 10), 'mg/L', '^mg/dL$')"),
        "trig": (100056, None), "tnt": (100057, None),
        "tri": (100059, None),
        "peep": (100060, "convert_unit(binary_op(`*`, 1), 'cmH2O', 'cm of H2O')"),
        "uric_acid": (100069, None), "ntprobnp": (100071, None),
        "d_dimer": (100075, None), "tidal_vol": (100077, None),
        "hba1c": (100079, None), "ck": (100085, None), "hdl": (100086, None),
        "cholesterol": (100088, None), "osmolality": (100089, None),
        "ldl": ([100090, 100248], None), "lipase": (100091, None),
        "tsh": (100092, None), "transferrin": (100094, None),
        "ammonia": (100095, None), "ckmb": (100099, None), "iron": (100102, None),
        "ft4": (100103, None), "sao2": (100109, None), "tibc": (100116, None),
        "cortisol": (100128, None), "amylase": (100137, None),
        "methb": (100145, None), "hbco": (100150, None), "ggt": (100161, None),
        "t4": (100276, None), "ph": (100339, None),
    }
    for concept, (ids, callback) in nw_labs.items():
        add(registry, concept, "nwicu", numeric("labevents", ids, "itemid", value_var="valuenum", index_var="charttime", callback=callback))

    # Zhejiang laboratory names and units were audited against actual rows.
    zh_labs: dict[str, tuple[object, float | None, str | None]] = {
        "be": ("Actual alkali residue", None, None),
        "cai": ("Standard ion calcium", None, None),
        "fio2": ("Inhalation oxygen concentration", None, None),
        "lact": ("lactic acid", None, None), "pco2": ("Partial pressure of carbon dioxide", None, None),
        "ph": ("PH", None, None), "po2": ("Blood oxygen partial pressure", None, None),
        "sao2": ("Blood oxygen saturation", None, None),
        "tco2": ("Total carbon dioxide", None, None),
        "alb": ("albumin", 0.1, "g/dL"), "alp": ("alkaline phosphatase", None, None),
        "alt": ("Alanine aminotransferase", None, None), "ast": ("Aspartate aminotransferase", None, None),
        "bicar": (["Actual bicarbonate", "Standard bicarbonate"], None, None),
        "bicarb": (["Actual bicarbonate", "Standard bicarbonate"], None, None),
        "bili": ("total bilirubin", 0.058467, "mg/dL"),
        "bili_dir": ("Direct bilirubin", 0.058467, "mg/dL"),
        "bun": ("urea", 2.801, "mg/dL"), "ca": ("calcium", 4.008, "mg/dL"),
        "ck": ("creatine kinase", None, None), "crea": ("creatinine", 0.011309, "mg/dL"),
        "crp": ("Hypersensitivity C-reactive protein", None, None),
        "glu": ("glucose", 18.016, "mg/dL"), "phos": ("phosphorus", 3.097, "mg/dL"),
        "tri": ("Cardiac troponin I", None, None), "amylase": ("Total amylase", None, None),
        "d_dimer": ("D-dimer quantification", None, None), "ldh": ("lactate dehydrogenase", None, None),
        "ggt": ("Glutamyl transpeptidase", None, None),
        "trig": ("triglyceride", 88.57, "mg/dL"), "tsh": (["TSH", "Thyroid stimulating hormone (TSH)"], None, None),
        "total_protein": ("Total protein", 0.1, "g/dL"), "pct": ("Procalcitonin", None, None),
        "uric_acid": (None, None, None), "cholesterol": ("total cholesterol", 38.67, "mg/dL"),
        "hdl": ("High density lipoprotein cholesterol", 38.67, "mg/dL"),
        "ldl": (["Low density lipoprotein cholesterol", "*Low density lipoprotein cholesterol"], 38.67, "mg/dL"),
        "ft4": ("Free thyroxine (FT4)", 0.1, "ng/dL"),
        "prealbumin": ("Prealbumin", 0.1, "mg/dL"), "t4": ("Total thyroxine (TT4)", 0.1, "ug/dL"),
        "esr": ("Erythrocyte sedimentation rate", None, None),
        "hba1c": ("Glycated hemoglobin A1c", None, None),
        "hct": ("Hematocrit", 100.0, "%"), "hgb": ("hemoglobin", 0.1, "g/dL"),
        "inr_pt": ("International standardization ratio", None, None),
        "lymph": ("Lymphocyte classification", None, None), "mch": ("Average hemoglobin level", None, None),
        "mchc": ("Average hemoglobin concentration", 0.1, "g/dL"),
        "mcv": ("Average red blood cell volume", None, None), "neut": ("Classification of neutrophils", None, None),
        "plt": ("platelet count", None, None), "pt": ("prothrombin time", None, None),
        "ptt": ("Partial thromboplastin time", None, None), "rbc": ("Red blood cell count", None, None),
        "rdw": ("Red blood cell distribution width", None, None), "wbc": ("White blood cell count", None, None),
        "basos": ("Classification of basophils", None, None), "eos": ("Classification of eosinophils", None, None),
        "monos": ("Monocyte classification", None, None), "mpv": ("Average platelet volume", None, None),
        "retic": ("Reticulocyte ratio", None, None),
    }
    for concept, (ids, factor, unit) in zh_labs.items():
        if ids is None:
            continue
        callback = f"convert_unit(binary_op(`*`, {factor}), '{unit}')" if factor is not None else None
        add(registry, concept, "zhejiang_eicu", numeric("lab", ids, "Lab_itemName_Eng", callback=callback))

    # Jinhua has no explicit unit column; these transformations are therefore
    # limited to labels whose observed distributions and source definitions
    # establish one unambiguous conventional unit.
    jh_labs: dict[str, tuple[object, float | None, str | None]] = {
        "be": (["Actual base excess (ABE) - arterial blood", "Base excess (BE)-arterial blood"], None, None),
        "cai": (["Calcium ions (Ca2+)-arterial blood", "Calcium ion (Ca2+)-unknown"], None, None),
        "fio2": ("Oxygen concentration (FiO2)-arterial blood", None, None),
        "hbco": ("Carboxyhemoglobin percentage (FCOHb%) - arterial blood", None, None),
        "lact": ("Lactate-arterial blood", None, None),
        "methb": ("Methemoglobin percentage (FMetHb%) - arterial blood", None, None),
        "pco2": ("Partial pressure of carbon dioxide (PaCO2)-arterial blood", None, None),
        "ph": ("PH-arterial blood", None, None), "po2": ("Partial pressure of oxygen (PaO2)-arterial blood", None, None),
        "sao2": ("Oxygen saturation (SaO2)-arterial blood", None, None),
        "tco2": ("Blood carbon dioxide content ctCO2(B)-arterial blood", None, None),
        "alb": ("Albumin (ALB)-Unknown", 0.1, "g/dL"), "alp": ("Alkaline phosphatase (ALP) - venous blood", None, None),
        "alt": ("Alanine aminotransferase (ALT) - venous blood", None, None), "ast": ("Aspartate aminotransferase (AST) - venous blood", None, None),
        "bicar": ("Actual bicarbonate (AB) - arterial blood", None, None), "bicarb": ("Actual bicarbonate (AB) - arterial blood", None, None),
        "bili": ("Total bilirubin (TBIL) - venous blood", 0.058467, "mg/dL"),
        "bili_dir": ("Direct bilirubin (DBIL) - venous blood", 0.058467, "mg/dL"),
        "bun": ("Urea Nitrogen (BUN)-Unknown", 2.801, "mg/dL"),
        "ca": ("Total calcium (Ca) - venous blood", 4.008, "mg/dL"), "ck": ("Creatine kinase (CK) - venous blood", None, None),
        "cl": (["Chloride ion (Cl-)-Unknown", "Chloride ion (Cl-)-arterial blood"], None, None),
        "crea": ("Creatinine (Crea) - unknown", 0.011309, "mg/dL"),
        "crp": ("High-sensitivity C-reactive protein (hs-CRP)-venous blood", None, None),
        "glu": (["Glucose (Glu)-Unknown", "Glucose (Glu)-arterial blood"], 18.016, "mg/dL"),
        "k": (["Potassium ion (K+)-unknown", "Potassium ion (K+)-arterial blood"], None, None),
        "potassium": (["Potassium ion (K+)-unknown", "Potassium ion (K+)-arterial blood"], None, None),
        "mg": ("Magnesium ion (Mg2+)-Unknown", 2.4305, "mg/dL"),
        "na": (["Sodium ion (Na+)-unknown", "Sodium ions (Na+)-arterial blood"], None, None),
        "phos": ("Phosphorus (P)-Unknown", 3.097, "mg/dL"),
        "tri": ("Cardiac troponin I (cTnI) - venous blood", None, None), "ammonia": ("Ammonia (Amm) - venous blood", None, None),
        "amylase": ("Amylase (Amy)-Unknown", None, None), "d_dimer": ("D-Dimer-venous blood", None, None),
        "ferritin": ("Ferritin (Ferr)-Unknown", None, None), "ldh": ("Lactate dehydrogenase (LDH) - unknown", None, None),
        "lipase": ("Lipase-unknown", None, None), "ggt": ("Gamma-glutamyl transferase (GGT)-venous blood", None, None),
        "trig": ("Triglycerides (TG) - venous blood", 88.57, "mg/dL"), "tsh": ("Thyroid Stimulating Hormone (TSH)-Venous Blood", None, None),
        "total_protein": ("Total protein (TP)-unknown", 0.1, "g/dL"), "ntprobnp": ("N-terminal pro-brain natriuretic peptide (NT-ProBNP) - venous blood", None, None),
        "pct": ("Procalcitonin (PCT) - unknown", None, None), "bnp": ("Brain natriuretic peptide (BNP) - venous blood", None, None),
        "uric_acid": ("Uric acid (UA) - unknown", 1 / 59.48, "mg/dL"),
        "cholesterol": ("Total cholesterol (TC) - venous blood", 38.67, "mg/dL"),
        "hdl": ("High-density lipoprotein cholesterol (HDL-C)-venous blood", 38.67, "mg/dL"),
        "ldl": ("Low-density lipoprotein cholesterol (LDL-C) - venous blood", 38.67, "mg/dL"),
        "iron": ("Iron (Fe)-Unknown", 5.584, "ug/dL"), "tibc": ("Total Iron Binding Capacity (TIBC)-Venous Blood", 5.584, "ug/dL"),
        "transferrin": ("Transferrin (TRF)-Unknown", 100.0, "mg/dL"),
        "ft4": ("Free thyroxine (FT4) - venous blood", 0.07769, "ng/dL"),
        "prealbumin": ("Prealbumin (PA) - venous blood", 0.1, "mg/dL"), "myoglobin": ("Myoglobin (Mb) - venous blood", None, None),
        "t4": ("Total thyroxine (TT4) - venous blood", 0.0777, "ug/dL"),
        "fgn": ("Fibrinogen (Fbg)-venous blood", 100.0, "mg/dL"), "hba1c": ("Glycated hemoglobin A1c (HbA1c) - venous blood", None, None),
        "hct": ("Hematocrit (Hct) - venous blood", None, None), "hgb": ("Hemoglobin (Hb) - venous blood", 0.1, "g/dL"),
        "inr_pt": ("Prothrombin international normalized ratio (PT-INR) - venous blood", None, None),
        "lymph": ("Lymphocyte percentage (Lymph%) - venous blood", None, None),
        "mch": ("Mean corpuscular hemoglobin content (MCH) - venous blood", None, None),
        "mchc": ("Mean corpuscular hemoglobin concentration (MCHC) - venous blood", 0.1, "g/dL"),
        "mcv": ("Mean corpuscular volume (MCV) - venous blood", None, None),
        "neut": ("Neutrophil percentage (Neut%) - venous blood", None, None),
        "plt": ("Platelet Count (PLT#)-Venous Blood", None, None), "pt": ("Prothrombin time (PT) - venous blood", None, None),
        "ptt": ("Activated partial thromboplastin time (APTT) - venous blood", None, None),
        "rbc": ("Red blood cell count (RBC#) - venous blood", None, None),
        "rdw": ("Red blood cell volume distribution width CV (RDW-CV) - venous blood", None, None),
        "wbc": ("White blood cell count (WBC#) - venous blood", None, None),
        "basos": ("Basophil percentage (Baso%) - venous blood", None, None),
        "eos": ("Eosinophil percentage (Eos%) - venous blood", None, None),
        "monos": ("Mononuclear cell percentage (Mono%) - venous blood", None, None),
        "mpv": ("Mean platelet volume (MPV) - venous blood", None, None),
        "retic": ("Reticulocyte percentage (RET%) - venous blood", None, None),
    }
    for concept, (ids, factor, unit) in jh_labs.items():
        callback = f"transform_fun(binary_op(`*`, {factor}))" if factor is not None else None
        add(registry, concept, "jinhua", numeric("laboratory_test", ids, "inspection_subproject_name_en", callback=callback))

    zg_labs: dict[str, tuple[object, float | None, str | None]] = {
        "be": ("Actual alkali surplus (ABE)", None, None), "cai": (["Calcium ion concentration (caca2 +)", "Free calcium (Ca + +)"], None, None),
        "hbco": ("Carboxyhemoglobin (fcohb)", None, None), "lact": (["Lactic acid (LAC)", "Lactic acid concentration (CLAC)"], None, None),
        "methb": ("Methemoglobin (fmethb)", None, None), "pco2": (["Arterial partial pressure of carbon dioxide (PCO2)", "Carbon dioxide partial pressure (PCO2)"], None, None),
        "ph": ("PH (PH)", None, None), "po2": (["Arterial oxygen partial pressure (PO2)", "Oxygen partial pressure (PO2)"], None, None),
        "sao2": ("Oxygen saturation (SO2)", None, None), "tco2": (["TCO2(P)(TCO2(P))", "Total carbon dioxide (ctco2)"], None, None),
        "alb": ("Albumin (ALB)", 0.1, "g/dL"), "alp": ("Alkaline phosphatase (ALP)", None, None),
        "alt": ("Alanine aminotransferase (ALT)", None, None), "anion_gap": (["Anion gap", "Anion gap (angap)"], None, None),
        "ast": ("Aspartate aminotransferase (AST)", None, None),
        "bicar": (["HCO3 - measured bicarbonate (HCO3 -)", "Measured bicarbonate (hco3act)"], None, None),
        "bicarb": (["HCO3 - measured bicarbonate (HCO3 -)", "Measured bicarbonate (hco3act)"], None, None),
        "bili": ("Total bilirubin (TBIL)", 0.058467, "mg/dL"), "bili_dir": ("Direct bilirubin (DBIL)", 0.058467, "mg/dL"),
        "bun": ("Urea (urea)", 2.801, "mg/dL"), "ca": ("Calcium (CA)", 4.008, "mg/dL"),
        "ck": ("Creatine kinase (CK)", None, None), "cl": (["Chloride ion (Cl -)", "Chloride ion concentration (CCL -)"], None, None),
        "crea": ("Creatinine (enzymatic method) (CRE)", 0.011309, "mg/dL"), "crp": (["C-reactive protein (CRP)", "High sensitivity C-reactive protein (hsCRP)"], None, None),
        "glu": (["Glucose (Glu)", "Glucose concentration (cglu)"], 18.016, "mg/dL"),
        "k": (["Potassium (k)", "Potassium ion (K +)", "Potassium ion concentration (CK +)"], None, None),
        "potassium": (["Potassium (k)", "Potassium ion (K +)", "Potassium ion concentration (CK +)"], None, None),
        "mg": ("Serum magnesium (mg)", 2.4305, "mg/dL"),
        "na": (["Sodium (NA)", "Sodium ion (Na +)", "Sodium ion concentration (CNA +)"], None, None),
        "phos": ("Inorganic phosphorus (P)", 3.097, "mg/dL"), "tri": ("High sensitivity troponin I (tn-i)", None, None),
        "d_dimer": ("D-Dimer (D2)", 1000.0, "ng/mL"), "ldh": ("Lactate dehydrogenase (LDH)", None, None),
        "ggt": ("γ- Glutamyltransferase (GGT)", None, None), "trig": ("Triglyceride (TG)", 88.57, "mg/dL"),
        "total_protein": ("Total protein (TP)", 0.1, "g/dL"), "bnp": ("Brain natriuretic peptide (BNP)", None, None),
        "uric_acid": ("Uric acid (UA)", 1 / 59.48, "mg/dL"), "cholesterol": ("Total cholesterol (TCHO)", 38.67, "mg/dL"),
        "hdl": ("High density lipoprotein cholesterol (HDL)", 38.67, "mg/dL"), "ldl": ("Low density lipoprotein cholesterol (LDL)", 38.67, "mg/dL"),
        "prealbumin": ("Prealbumin (PA)", 0.1, "mg/dL"), "myoglobin": ("Myoglobin (myo)", None, None),
        "esr": ("ESR (SR)", None, None), "fgn": ("Fibrinogen (FIB)", 100.0, "mg/dL"),
        "hct": (["Hematocrit (HCT)", "Hct(Hct)"], None, None), "hgb": ("Hemoglobin (Hgb)", 0.1, "g/dL"),
        "inr_pt": ("International normalized ratio (INR)", None, None), "lymph": ("Lymphocyte ratio (lym%)", None, None),
        "mch": ("Mean hemoglobin (MCH)", None, None), "mchc": ("Mean hemoglobin concentration (MCHC)", 0.1, "g/dL"),
        "mcv": ("Mean corpuscular volume (MCV)", None, None), "neut": ("Neutrophil ratio (neu%)", None, None),
        "plt": ("Platelet (PLT)", None, None), "pt": ("Prothrombin time (PT)", None, None),
        "ptt": ("Activated partial thromboplastin time (APTT)", None, None), "rbc": ("Red blood cell (RBC)", None, None),
        "rdw": ("Coefficient of variation of erythrocyte distribution width (rdwcv)", None, None),
        "wbc": ("White blood cell (WBC)", None, None), "basos": ("Basophil ratio (BAS%)", None, None),
        "eos": ("Eosinophil ratio (EOS%)", None, None), "monos": ("Monocyte ratio (Mon%)", None, None),
        "mpv": ("Mean platelet volume (MPV)", None, None),
    }
    for concept, (ids, factor, unit) in zg_labs.items():
        callback = f"convert_unit(binary_op(`*`, {factor}), '{unit}')" if factor is not None else None
        add(registry, concept, "zigong", numeric("dtlab", ids, "Item", callback=callback))

    # Output, neurological and ventilator measurements.
    add(registry, "urine", "zhejiang_eicu", numeric("nursingchart_io", ["小便", "尿量"], "NursingIO_item"))
    zh_vent = {
        "cvp": ("中心静脉压", "convert_unit(binary_op(`*`, 0.735559), 'mmHg', '^cmH2O$')"),
        "minute_vol": ("分钟通气量", None), "ps": ("压力支持", None),
        "tidal_vol": (["吸入潮气量", "呼出潮气量"], None), "peep": ("呼吸末正压", None),
        "vent_rate": ("呼吸频率(设)", None), "pip": ("气道峰压", None),
        "fio2": ("氧浓度", None), "tidal_vol_set": ("潮气量(设)", None),
    }
    for concept, (ids, callback) in zh_vent.items():
        add(registry, concept, "zhejiang_eicu", numeric("nursingchart_vitalsign", ids, "NursingEvent_item", callback=callback))
    for concept, callback in (("vent_mode", "community_vent_mode_control"), ("vent_breath_seq", "community_vent_mode_seq")):
        add(registry, concept, "zhejiang_eicu", numeric("nursingchart_vitalsign", "呼吸机模式", "NursingEvent_item", value_var="NursingEvent_val", callback=callback))
    add(registry, "adv_resp", "zhejiang_eicu", numeric("nursingchart_vitalsign", "呼吸机模式", "NursingEvent_item", callback="transform_fun(set_val(TRUE))"))

    add(registry, "vent_rate", "zigong", source("dtnursingchart", value_var="F", class_name="col_itm"))
    add(registry, "tidal_vol_set", "zigong", source("dtnursingchart", value_var="Vt_setting", class_name="col_itm"))
    add(registry, "tidal_vol", "zigong", source("dtnursingchart", value_var="Vt_Supervisor", class_name="col_itm"))
    for concept, callback in (("vent_mode", "community_vent_mode_control"), ("vent_breath_seq", "community_vent_mode_seq")):
        add(registry, concept, "zigong", source("dtnursingchart", value_var="Breathing_pattern", callback=callback, class_name="col_itm"))
    add(registry, "adv_resp", "zigong", source("dtnursingchart", value_var="Breathing_pattern", callback="transform_fun(set_val(TRUE))", class_name="col_itm"))
    add(registry, "egcs", "zigong", source("dtnursingchart", value_var="open_one's_eyes", callback="transform_fun(extract_leading_number)", class_name="col_itm"))
    add(registry, "mgcs", "zigong", source("dtnursingchart", value_var="motion", callback="transform_fun(extract_leading_number)", class_name="col_itm"))
    add(registry, "vgcs", "zigong", source("dtnursingchart", value_var="language", callback="transform_fun(extract_leading_number)", class_name="col_itm"))
    add(registry, "rass", "zigong", source("dtnursingchart", value_var="RASS_sedation_score", class_name="col_itm"))

    # Procedure evidence.  Access-only and billing-only rows are deliberately excluded.
    add(registry, "rrt", "nwicu", numeric("procedureevents", [704890, 740477, 772042], "itemid", value_var="itemid", callback="transform_fun(set_val(TRUE))"))
    add(registry, "ecmo", "nwicu", numeric("procedureevents", 736876, "itemid", value_var="itemid", callback="transform_fun(set_val(TRUE))"))
    add(registry, "mech_vent", "nwicu", source("procedureevents", ids=[787541, 792843], sub_var="itemid", value_var="itemid", callback="apply_map(c(`787541` = 'invasive', `792843` = 'noninvasive'), var = 'sub_var')", comment="Point evidence only: the published NWICU rows have no endtime, so no continuous ventilation duration is inferred."))
    add(registry, "vent_start", "nwicu", numeric("procedureevents", 753461, "itemid", value_var="itemid", callback="transform_fun(set_val(TRUE))"))
    add(registry, "vent_end", "nwicu", numeric("procedureevents", 763189, "itemid", value_var="itemid", callback="transform_fun(set_val(TRUE))"))
    add(registry, "platelets", "nwicu", numeric("procedureevents", 767372, "itemid", value_var="itemid", callback="transform_fun(set_val(TRUE))"))

    add(registry, "rrt", "zhejiang_eicu", numeric("nursingchart_io", "CRRT滤出液", "NursingIO_item", callback="transform_fun(set_val(TRUE))"))
    add(registry, "rrt", "jinhua", source("orders", sub_var="order_content_en", value_var="order_content_en", regex=r"^(Hemodialysis|Peritoneal dialysis \(manual\)|Peritoneal dialysis)$", callback="transform_fun(set_val(TRUE))", class_name="rgx_itm"))
    add(registry, "ecmo", "jinhua", source("orders", sub_var="order_content_en", value_var="order_content_en", regex=r"ECMO.*(operation monitoring|installation)|Extracorporeal membrane oxygenation.*monitoring", callback="transform_fun(set_val(TRUE))", class_name="rgx_itm"))
    add(registry, "mech_vent", "jinhua", source("orders", ids=["Ventilator mechanical ventilation (invasive)", "Ventilator mechanical ventilation", "Ventilator assisted breathing", "Ventilator mechanical ventilation (non-invasive)", "Use of non-invasive ventilator", "Non-invasive assisted ventilation"], sub_var="order_content_en", value_var="order_content_en", dur_var="endtime_base", callback="community_jinhua_mech_vent"))
    for concept, regex in {
        "packed_rbc": r"^Actual blood transfusion.*(suspended red blood cells|red blood cells)",
        "ffp": r"^Actual blood transfusion.*(fresh frozen plasma|frozen plasma)",
        "platelets": r"^Actual blood transfusion.*platelet apheresis",
    }.items():
        add(registry, concept, "jinhua", source("orders", sub_var="order_content_en", value_var="order_content_en", regex=regex, callback="transform_fun(set_val(TRUE))", class_name="rgx_itm"))

    # A culture record establishes sample collection; it does not imply positivity.
    add(registry, "samp", "zhejiang_eicu", source("microbiologyculture", value_var="MicrobiologyCulture_Finding", callback="transform_fun(set_val(TRUE))", class_name="col_itm"))
    add(registry, "samp", "jinhua", source("microbiology_culture", value_var="culture_results", callback="transform_fun(set_val(TRUE))", class_name="col_itm"))

    # Medication exposure regexes.  IV-only concepts use route-restricted orders
    # in NWICU/Zhejiang/Jinhua and formulation-restricted names in Zigong.
    medication_patterns = {
        "abx": r"aztreonam|bactrim|cephalexin|chloramphenicol|cipro|flagyl|metronidazole|nitrofurantoin|tazobactam|rifamp|sulfadiazine|trimethoprim|amikacin|gentamicin|vancomycin|amoxicillin|ampicillin|dicloxacillin|nafcillin|oxacillin|penicillin|piperacillin|azithromycin|clarithromycin|erythromycin|clindamycin|streptomycin|tobramycin|cefazolin|ceftazidime|cefadroxil|cefepime|cefotetan|cefotaxime|cefpodoxime|cefuroxime|doxycycline|minocycline|tetracycline|levofloxacin|moxifloxacin|ofloxacin|meropenem|imipenem|ertapenem|ceftriaxone",
        "albumin_iv": r"human (serum )?albumin|albumin, human", "amiodarone": r"amiodarone|cordarone",
        "apixaban": r"apixaban|eliquis", "aspirin": r"aspirin|acetylsalicylic", "bicarbonate": r"sodium bicarbonate",
        "calcium_iv": r"calcium (gluconate|chloride)", "cisatracurium": r"cisatracurium|nimbex",
        "cort": r"hydrocortisone|prednisone|prednisolone|methylprednisolone|dexamethasone",
        "dex": r"dextrose.*10|10%.*dextrose|glucose.*10", "dexamethasone": r"dexamethasone",
        "dexmedetomidine": r"dexmedetomidine|precedex|dexdor", "dextrose50": r"dextrose.*50|50%.*dextrose|glucose.*50",
        "diltiazem": r"diltiazem|cardizem", "enoxaparin": r"enoxaparin|nadroparin|dalteparin|tinzaparin",
        "esmolol": r"esmolol|brevibloc", "fentanyl": r"(^|[^a-z])fentanyl", "furosemide": r"furosemide|lasix",
        "heparin": r"(^|[^a-z])heparin", "insulin": r"insulin", "ketamine": r"ketamine",
        "labetalol": r"labetalol", "levetiracetam": r"levetiracetam|keppra", "lorazepam": r"lorazepam|ativan",
        "magnesium_iv": r"magnesium sulfate", "mannitol": r"mannitol", "meropenem": r"meropenem",
        "midazolam": r"midazolam|versed|dormicum", "milrinone": r"milrinone|enoximone", "morphine": r"(^|[^a-z])morphine",
        "neostigmine": r"neostigmine", "nicardipine": r"nicardipine|cardene", "nitroglycerin": r"nitroglycerin",
        "octreotide": r"octreotide|sandostatin", "pantoprazole": r"pantoprazole|protonix",
        "phenytoin": r"phenytoin|fosphenytoin|dilantin|cerebyx", "potassium_iv": r"potassium chloride|potassium phosphate|(^|[^a-z])kcl([^a-z]|$)|kphos",
        "propofol": r"propofol|diprivan", "rocuronium": r"rocuronium|esmeron", "vancomycin": r"vancomycin|vancocin",
        "vecuronium": r"vecuronium|norcuron", "warfarin": r"warfarin|coumadin|phenprocoumon|acenocoumarol",
    }
    iv_only = {
        "albumin_iv", "amiodarone", "bicarbonate", "calcium_iv", "cisatracurium", "dex",
        "dexmedetomidine", "dextrose50", "esmolol", "fentanyl", "furosemide", "heparin",
        "ketamine", "levetiracetam", "magnesium_iv", "mannitol", "midazolam", "milrinone",
        "morphine", "nicardipine", "nitroglycerin", "pantoprazole", "potassium_iv", "propofol",
        "rocuronium", "vecuronium",
    }
    iv_routes = {
        "nwicu": ["Intravenous", "Injection"],
        "zhejiang_eicu": ["Micropump injection", "Intravenous drip", "Static push", "Pump injection", "vein"],
        "jinhua": ["Intravenous drip", "Micropump injection", "Slow intravenous push", "Micropump intravenous drip", "Intravenous injection", "Intravenous infusion"],
    }
    table_contract = {
        "nwicu": ("prescriptions", "drug", "route"),
        "zhejiang_eicu": ("medication", "Med_DESC_Eng", "Med_route_Eng"),
        "jinhua": ("medication", "drug_name_en", "dose_type_en"),
        "zigong": ("dtdrugs", "DrugName", None),
    }
    absent_or_route_incompatible = {
        ("apixaban", "jinhua"), ("apixaban", "zigong"),
        ("ketamine", "jinhua"), ("labetalol", "jinhua"),
        ("labetalol", "zigong"), ("levetiracetam", "zhejiang_eicu"),
        ("lorazepam", "jinhua"), ("nicardipine", "jinhua"),
        ("phenytoin", "zigong"), ("rocuronium", "jinhua"),
    }
    zigong_iv_suffix = r".*(injection|for injection|human (serum )?albumin)"
    for concept, pattern in medication_patterns.items():
        for database, (table, drug_col, route_col) in table_contract.items():
            if (concept, database) in absent_or_route_incompatible:
                continue
            if database == "zigong" and concept in iv_only:
                pattern_for_db = rf"(?:{pattern}){zigong_iv_suffix}|human (serum )?albumin"
            else:
                pattern_for_db = pattern
            if concept == "mannitol" and database == "zigong":
                continue  # no matching source label in release 1.1
            kwargs: dict[str, Any] = {
                "sub_var": drug_col,
                "value_var": drug_col,
                "regex": pattern_for_db,
                "callback": "transform_fun(set_val(TRUE))",
                "class_name": "rgx_itm",
            }
            if concept in iv_only and route_col is not None:
                kwargs["sub_var"] = route_col
                kwargs["ids"] = iv_routes[database]
            add(registry, concept, database, source(table, **kwargs))

    return registry


def load_complete_dictionary(registry: dict[str, dict[str, Any]]) -> ConceptDictionary:
    base = ConceptDictionary.from_payload(json.loads((DATA / "concept-dict.json").read_text()))
    base.update(ConceptDictionary.from_payload(registry))
    base.update(ConceptDictionary.from_payload(json.loads((DATA / "sofa2-dict.json").read_text())))
    return base


def build_coverage(registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dictionary = load_complete_dictionary(registry)
    memo: dict[tuple[str, str], str] = {}

    def status(concept: str, database: str, stack: tuple[str, ...] = ()) -> str:
        key = (concept, database)
        if key in memo:
            return memo[key]
        definition = dictionary[concept]
        if database in definition.sources:
            memo[key] = "direct"
            return "direct"
        dependencies = list(definition.sub_concepts or definition.depends_on or [])
        if not dependencies or concept in stack:
            memo[key] = "unavailable"
            return "unavailable"
        child = [status(dep, database, (*stack, concept)) for dep in dependencies if dep in dictionary]
        if child and all(item in {"direct", "derived"} for item in child):
            result = "derived"
        elif any(item in {"direct", "derived", "partial"} for item in child):
            result = "partial"
        else:
            result = "unavailable"
        memo[key] = result
        return result

    concepts: dict[str, Any] = {}
    for name in sorted(dictionary.keys()):
        definition = dictionary[name]
        cells: dict[str, Any] = {}
        for database in DATABASES:
            cell_status = status(name, database)
            dependencies = list(definition.sub_concepts or definition.depends_on or [])
            cell: dict[str, Any] = {"status": cell_status}
            if cell_status == "direct":
                cell["source_count"] = len(definition.sources[database])
                cell["reason"] = "validated native field/event mapping"
            elif dependencies:
                missing = [dep for dep in dependencies if status(dep, database) == "unavailable"]
                partial = [dep for dep in dependencies if status(dep, database) == "partial"]
                cell["dependencies"] = dependencies
                if missing:
                    cell["missing_dependencies"] = missing
                if partial:
                    cell["partial_dependencies"] = partial
                cell["reason"] = {
                    "derived": "all declared dependencies are available",
                    "partial": "some declared dependencies are available; full definition is not",
                    "unavailable": "declared dependencies are unavailable",
                }[cell_status]
            else:
                cell["reason"] = "no validated native field/event in this release"
            cells[database] = cell
        concepts[name] = {
            "category": definition.category,
            "description": definition.description,
            "databases": cells,
        }

    counts = {
        database: dict(Counter(concepts[name]["databases"][database]["status"] for name in concepts))
        for database in DATABASES
    }
    return {
        "schema_version": 1,
        "concept_count": len(concepts),
        "databases": list(DATABASES),
        "status_definitions": {
            "direct": "validated native source mapping",
            "derived": "all declared dependency concepts are available",
            "partial": "some but not all declared dependency concepts are available",
            "unavailable": "no validated source or usable dependency path",
        },
        "counts": counts,
        "concepts": concepts,
    }


def coverage_markdown(coverage: dict[str, Any]) -> str:
    lines = [
        "# Community database concept coverage",
        "",
        "Generated from `tools/build_community_concept_registry.py`. The matrix covers every standard",
        f"concept (base dictionary plus SOFA-2 overlay): **{coverage['concept_count']} concepts × {len(DATABASES)} databases**.",
        "",
        "`partial` is deliberately not advertised as extractable: it records useful ingredients while preventing an incomplete score or phenotype from being mistaken for a complete definition.",
        "",
        "## Summary",
        "",
        "| Database | Direct | Derived | Partial | Unavailable |",
        "|---|---:|---:|---:|---:|",
    ]
    for database in DATABASES:
        counts = coverage["counts"][database]
        lines.append(f"| {database} | {counts.get('direct', 0)} | {counts.get('derived', 0)} | {counts.get('partial', 0)} | {counts.get('unavailable', 0)} |")
    lines.extend(["", "## Complete matrix", "", "| Concept | Category | NWICU | Zhejiang eICU | Jinhua | Zigong |", "|---|---|---|---|---|---|"])
    for concept, record in coverage["concepts"].items():
        cells = [record["databases"][database]["status"] for database in DATABASES]
        lines.append(f"| `{concept}` | {record['category'] or ''} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    registry = build_registry()
    coverage = build_coverage(registry)
    outputs = {
        DATA / "community-concept-sources.json": json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        DATA / "community-concept-coverage.json": json.dumps(coverage, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        DOCS / "community_concept_coverage.md": coverage_markdown(coverage),
    }
    stale = [str(path.relative_to(ROOT)) for path, content in outputs.items() if not path.is_file() or path.read_text(encoding="utf-8") != content]
    if args.check:
        if stale:
            raise SystemExit("stale generated community registry: " + ", ".join(stale))
        return
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"concepts": coverage["concept_count"], "counts": coverage["counts"]}, indent=2))


if __name__ == "__main__":
    main()
