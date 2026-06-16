#!/usr/bin/env python
"""Execute the two doable mined ideas end-to-end (data-layer completion).

Idea A (obesity -> ICU mortality, "obesity paradox"): cohort by the EasyICU
`bmi` concept (more accurate than under-coded ICD obesity); in-hospital mortality
by BMI category on MIMIC-IV full cohort.
Idea B (severe Legionnaires' -> mortality): ICD-defined cohort (A48.1/A48.2/482.84),
descriptive in-hospital mortality (exposure 'antimicrobial regimen' did not
resolve; small-n). Human-confirmed ICD code-set per the Case-1 QC gate.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("EASYICU_FORCE_INPROCESS_BATCH","1")
import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd
from easyicu.api import load_concepts
DB="/Volumes/外置硬盘/databases/mimiciv"
out={}

# ---- Idea A: obesity paradox via bmi concept ----
print("[A] obesity -> mortality (BMI concept, miiv full)...", flush=True)
df=load_concepts(["bmi","death","age"],database="miiv",data_path=DB,batch_size=2_000_000)
idc=df.columns[0]
g=df.groupby(idc).agg(bmi=("bmi","max"),death=("death","max"),age=("age","max"))
g=g.dropna(subset=["bmi"])
g=g[(g.bmi>=10)&(g.bmi<=80)]
g["death"]=g["death"].fillna(0).clip(0,1)
cats=[("underweight",0,18.5),("normal",18.5,25),("overweight",25,30),("obese I",30,35),("obese II-III",35,80)]
rowsA=[]
for name,lo,hi in cats:
    m=(g.bmi>=lo)&(g.bmi<hi)
    n=int(m.sum()); d=int(g.death[m].sum())
    rowsA.append({"bmi_category":name,"n":n,"deaths":d,"mortality_pct":round(100*d/n,1) if n else None})
out["idea_A_obesity_paradox"]={"cohort_def":"EasyICU bmi concept, worst per stay, 10-80 range","n_total":int(g.shape[0]),"by_bmi_category":rowsA}
print(pd.DataFrame(rowsA).to_string(index=False))

# ---- Idea B: Legionnaires ICD cohort mortality ----
print("\n[B] severe Legionnaires' -> mortality (ICD cohort)...", flush=True)
dx=pd.read_parquet(f"{DB}/hosp/diagnoses_icd.parquet",columns=["hadm_id","icd_code","icd_version"])
leg_codes={("48284",9),("A481",10),("A482",10)}
dx["key"]=list(zip(dx.icd_code.astype(str),dx.icd_version.astype(int)))
leg_hadm=set(dx.loc[dx.key.isin(leg_codes),"hadm_id"].unique())
adm=pd.read_parquet(f"{DB}/hosp/admissions.parquet",columns=["hadm_id","hospital_expire_flag"])
cohort=adm[adm.hadm_id.isin(leg_hadm)]
n=int(cohort.shape[0]); d=int(cohort.hospital_expire_flag.sum())
out["idea_B_legionnaires"]={"cohort_def":"ICD A48.1/A48.2/482.84 (human-confirmed), hadm-level","n_hadm":n,"deaths":d,"in_hosp_mortality_pct":round(100*d/n,1) if n else None,"caveat":"exposure 'antimicrobial regimen' unresolved -> descriptive only; small n, underpowered; ICD proxy"}
print(f"  Legionnaires hadm n={n}, deaths={d}, in-hosp mortality={round(100*d/n,1) if n else None}%")

from pathlib import Path
Path("research_output/mined_ideas_completed_20260616.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nwrote research_output/mined_ideas_completed_20260616.json")
