# Community database concept coverage

Generated from `tools/build_community_concept_registry.py`. The matrix covers every standard
concept (base dictionary plus SOFA-2 overlay): **274 concepts × 4 databases**.

`partial` is deliberately not advertised as extractable: it records useful ingredients while preventing an incomplete score or phenotype from being mistaken for a complete definition.

## Summary

| Database | Direct | Derived | Partial | Unavailable |
|---|---:|---:|---:|---:|
| nwicu | 137 | 20 | 25 | 92 |
| zhejiang_eicu | 128 | 25 | 22 | 99 |
| jinhua | 135 | 19 | 24 | 96 |
| zigong | 125 | 19 | 33 | 97 |

## Complete matrix

| Concept | Category | NWICU | Zhejiang eICU | Jinhua | Zigong |
|---|---|---|---|---|---|
| `abx` | medications | direct | direct | direct | direct |
| `acute_rrt_input` | renal | derived | derived | derived | unavailable |
| `adh_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `adm` | demographics | unavailable | unavailable | unavailable | unavailable |
| `adv_resp` | respiratory | unavailable | direct | unavailable | direct |
| `age` | demographics | direct | direct | direct | direct |
| `alb` | chemistry | direct | direct | direct | direct |
| `albumin_iv` | medications | direct | direct | direct | direct |
| `alp` | chemistry | direct | direct | direct | direct |
| `alt` | chemistry | direct | direct | direct | direct |
| `amiodarone` | medications | direct | direct | direct | direct |
| `ammonia` | chemistry | direct | unavailable | direct | unavailable |
| `amylase` | chemistry | direct | direct | direct | unavailable |
| `anion_gap` | chemistry | direct | partial | derived | direct |
| `apache_iv` | severity | unavailable | unavailable | unavailable | unavailable |
| `apache_iv_pred_hosp_mort` | severity | unavailable | unavailable | unavailable | unavailable |
| `apixaban` | medications | direct | direct | unavailable | unavailable |
| `aspirin` | medications | direct | direct | direct | direct |
| `ast` | chemistry | direct | direct | direct | direct |
| `avpu` | neurological | unavailable | unavailable | unavailable | partial |
| `basos` | hematology | direct | direct | direct | direct |
| `be` | blood gas | unavailable | direct | direct | direct |
| `bicar` | chemistry | direct | direct | direct | direct |
| `bicarb` | chemistry | direct | direct | direct | direct |
| `bicarbonate` | medications | direct | direct | direct | direct |
| `bili` | chemistry | direct | direct | direct | direct |
| `bili_dir` | chemistry | direct | direct | direct | direct |
| `bmi` | demographics | derived | derived | direct | unavailable |
| `bnd` | hematology | unavailable | unavailable | unavailable | unavailable |
| `bnp` | chemistry | unavailable | unavailable | direct | direct |
| `bun` | chemistry | direct | direct | direct | direct |
| `bun_creatinine_ratio` | renal | derived | derived | derived | derived |
| `ca` | chemistry | direct | direct | direct | direct |
| `cai` | blood gas | direct | direct | direct | direct |
| `calcium_iv` | medications | direct | direct | direct | direct |
| `cholesterol` | chemistry | direct | direct | direct | direct |
| `cisatracurium` | medications | direct | direct | direct | direct |
| `ck` | chemistry | direct | direct | direct | direct |
| `ckmb` | chemistry | direct | unavailable | unavailable | unavailable |
| `cl` | chemistry | direct | unavailable | direct | direct |
| `co` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `compliance` | ventilator | unavailable | unavailable | unavailable | unavailable |
| `corrected_calcium` | chemistry | derived | derived | derived | derived |
| `cort` | medications | direct | direct | direct | direct |
| `cortisol` | chemistry | direct | unavailable | unavailable | unavailable |
| `crea` | chemistry | direct | direct | direct | direct |
| `crp` | chemistry | direct | direct | direct | direct |
| `crrt_mode_input` | renal | unavailable | unavailable | unavailable | unavailable |
| `cvp` | vitals | unavailable | direct | unavailable | direct |
| `d_dimer` | chemistry | direct | direct | direct | direct |
| `dbp` | vitals | direct | direct | direct | direct |
| `death` | outcome | direct | direct | direct | unavailable |
| `delirium_positive` | neurology | unavailable | unavailable | unavailable | unavailable |
| `delirium_tx` | neurology | unavailable | unavailable | unavailable | unavailable |
| `delirium_tx_evidence` | neurology | unavailable | unavailable | unavailable | unavailable |
| `delirium_tx_proxy` | neurology | unavailable | unavailable | unavailable | unavailable |
| `dex` | medications | direct | direct | direct | direct |
| `dexamethasone` | medications | direct | direct | direct | direct |
| `dexmedetomidine` | medications | direct | direct | direct | direct |
| `dextrose50` | medications | direct | direct | direct | direct |
| `diastolic_shock_index` | cardiovascular | derived | derived | derived | derived |
| `diltiazem` | medications | direct | direct | direct | direct |
| `dobu60` | medications | unavailable | unavailable | unavailable | unavailable |
| `dobu_dur` | medications | unavailable | unavailable | unavailable | unavailable |
| `dobu_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `dopa60` | medications | unavailable | unavailable | unavailable | unavailable |
| `dopa_dur` | medications | unavailable | unavailable | unavailable | unavailable |
| `dopa_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `driving_pres` | ventilator | partial | partial | unavailable | unavailable |
| `driving_pres_controlled` | ventilator | partial | partial | unavailable | partial |
| `ecmo` | respiratory | direct | unavailable | direct | unavailable |
| `ecmo_indication` | respiratory | unavailable | unavailable | unavailable | unavailable |
| `egcs` | neurological | unavailable | unavailable | unavailable | direct |
| `egfr` | renal | derived | derived | derived | derived |
| `enoxaparin` | medications | direct | direct | direct | direct |
| `eos` | hematology | direct | direct | direct | direct |
| `epi60` | medications | unavailable | unavailable | unavailable | unavailable |
| `epi_dur` | medications | unavailable | unavailable | unavailable | unavailable |
| `epi_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `esmolol` | medications | direct | direct | direct | direct |
| `esr` | hematology | unavailable | direct | unavailable | direct |
| `etco2` | ventilator | unavailable | unavailable | unavailable | unavailable |
| `ett_gcs` | respiratory | unavailable | unavailable | unavailable | unavailable |
| `fentanyl` | medications | direct | direct | direct | direct |
| `fentanyl_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `ferritin` | chemistry | direct | unavailable | direct | unavailable |
| `ffp` | medications | unavailable | unavailable | direct | unavailable |
| `fgn` | hematology | direct | unavailable | direct | direct |
| `fio2` | blood gas | direct | direct | direct | unavailable |
| `fluid_balance` | output | unavailable | partial | unavailable | partial |
| `fluid_balance_cumulative` | output | unavailable | partial | unavailable | partial |
| `ft4` | chemistry | direct | direct | direct | unavailable |
| `furosemide` | medications | direct | direct | direct | direct |
| `gcs` | neurological | unavailable | unavailable | unavailable | partial |
| `ggt` | chemistry | direct | direct | direct | direct |
| `glu` | chemistry | direct | direct | direct | direct |
| `hba1c` | hematology | direct | direct | direct | unavailable |
| `hbco` | blood gas | direct | unavailable | direct | direct |
| `hct` | hematology | direct | direct | direct | direct |
| `hdl` | chemistry | direct | direct | direct | direct |
| `height` | demographics | direct | direct | direct | unavailable |
| `heparin` | medications | direct | direct | direct | direct |
| `hgb` | hematology | direct | direct | direct | direct |
| `hr` | vitals | direct | direct | direct | direct |
| `icp` | neurological | unavailable | unavailable | unavailable | unavailable |
| `icu_unit_type` | demographics | direct | unavailable | unavailable | unavailable |
| `infection_icd` | outcome | unavailable | unavailable | unavailable | unavailable |
| `inr_pt` | hematology | direct | direct | direct | direct |
| `ins` | medications | unavailable | unavailable | unavailable | unavailable |
| `insulin` | medications | direct | direct | direct | direct |
| `iron` | chemistry | direct | unavailable | direct | unavailable |
| `k` | chemistry | direct | unavailable | direct | direct |
| `kdigo_aki` | renal | partial | partial | partial | partial |
| `kdigo_creat` | renal | derived | derived | derived | derived |
| `kdigo_creatinine_input` | renal | derived | derived | derived | derived |
| `kdigo_uo` | renal | partial | derived | partial | partial |
| `kdigo_urine_input` | renal | unavailable | derived | unavailable | derived |
| `ketamine` | medications | direct | direct | unavailable | direct |
| `labetalol` | medications | direct | direct | unavailable | unavailable |
| `lact` | blood gas | direct | direct | direct | direct |
| `ldh` | chemistry | unavailable | direct | direct | direct |
| `ldl` | chemistry | direct | direct | direct | direct |
| `levetiracetam` | medications | direct | unavailable | direct | direct |
| `lipase` | chemistry | direct | unavailable | direct | unavailable |
| `lorazepam` | medications | direct | direct | unavailable | direct |
| `los_hosp` | outcome | direct | direct | direct | unavailable |
| `los_icu` | outcome | direct | unavailable | direct | direct |
| `lymph` | hematology | direct | direct | direct | direct |
| `magnesium_iv` | medications | direct | direct | direct | direct |
| `mannitol` | medications | direct | direct | direct | unavailable |
| `map` | vitals | unavailable | unavailable | unavailable | direct |
| `mch` | hematology | direct | direct | direct | direct |
| `mchc` | hematology | direct | direct | direct | direct |
| `mcv` | hematology | direct | direct | direct | direct |
| `mean_airway_pres` | ventilator | unavailable | unavailable | unavailable | unavailable |
| `mech_circ_support` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `mech_vent` | respiratory | direct | unavailable | direct | unavailable |
| `meropenem` | medications | direct | direct | direct | direct |
| `methb` | blood gas | direct | unavailable | direct | direct |
| `mews` | outcome | partial | partial | partial | partial |
| `mg` | chemistry | direct | unavailable | direct | direct |
| `mgcs` | neurological | unavailable | unavailable | unavailable | direct |
| `midazolam` | medications | direct | direct | direct | direct |
| `midazolam_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `milrinone` | medications | direct | direct | direct | direct |
| `minute_vol` | ventilator | unavailable | direct | unavailable | unavailable |
| `modified_shock_index` | cardiovascular | partial | partial | partial | derived |
| `monos` | hematology | direct | direct | direct | direct |
| `morphine` | medications | direct | direct | direct | direct |
| `motor_response` | neurology | unavailable | unavailable | unavailable | unavailable |
| `mpv` | hematology | unavailable | direct | direct | direct |
| `myoglobin` | chemistry | unavailable | unavailable | direct | direct |
| `na` | chemistry | direct | unavailable | direct | direct |
| `neostigmine` | medications | direct | direct | direct | direct |
| `neut` | hematology | direct | direct | direct | direct |
| `news` | outcome | partial | partial | partial | partial |
| `nicardipine` | medications | direct | direct | unavailable | direct |
| `nitroglycerin` | medications | direct | direct | direct | direct |
| `nlr` | hematology | derived | derived | derived | derived |
| `norepi60` | medications | unavailable | unavailable | unavailable | unavailable |
| `norepi_dur` | medications | unavailable | unavailable | unavailable | unavailable |
| `norepi_equiv` | medications | unavailable | unavailable | unavailable | unavailable |
| `norepi_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `ntprobnp` | chemistry | direct | unavailable | direct | unavailable |
| `o2sat` | respiratory | direct | direct | direct | direct |
| `octreotide` | medications | direct | direct | direct | direct |
| `osmolality` | chemistry | direct | unavailable | unavailable | unavailable |
| `other_vaso` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `oxygenation_index` | respiratory | partial | partial | partial | partial |
| `packed_rbc` | medications | unavailable | unavailable | direct | unavailable |
| `pafi` | respiratory | partial | derived | derived | partial |
| `pantoprazole` | medications | direct | direct | direct | direct |
| `pap_dia` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `pap_mean` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `pap_sys` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `pawp` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `pco2` | blood gas | unavailable | direct | direct | direct |
| `pct` | chemistry | unavailable | direct | direct | unavailable |
| `peep` | ventilator | direct | direct | unavailable | unavailable |
| `persistent_critical_illness` | outcome | derived | unavailable | derived | derived |
| `ph` | blood gas | direct | direct | direct | direct |
| `phenytoin` | medications | direct | direct | direct | unavailable |
| `phn_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `phos` | chemistry | direct | direct | direct | direct |
| `pip` | ventilator | unavailable | direct | unavailable | unavailable |
| `plateau_pres` | ventilator | unavailable | unavailable | unavailable | unavailable |
| `platelets` | medications | direct | unavailable | direct | unavailable |
| `plr` | hematology | derived | derived | derived | derived |
| `plt` | hematology | direct | direct | direct | direct |
| `po2` | blood gas | unavailable | direct | direct | direct |
| `potassium` | chemistry | direct | unavailable | direct | direct |
| `potassium_iv` | medications | direct | direct | direct | direct |
| `prealbumin` | chemistry | unavailable | direct | direct | direct |
| `propofol` | medications | direct | direct | direct | direct |
| `propofol_rate` | medications | unavailable | unavailable | unavailable | unavailable |
| `ps` | ventilator | unavailable | direct | unavailable | unavailable |
| `pt` | hematology | direct | direct | direct | direct |
| `ptt` | hematology | direct | direct | direct | direct |
| `pulse_pressure` | vitals | derived | derived | derived | derived |
| `qsofa` | outcome | partial | partial | partial | partial |
| `rass` | neurological | unavailable | unavailable | unavailable | direct |
| `rbc` | hematology | direct | direct | direct | direct |
| `rdw` | hematology | direct | direct | direct | direct |
| `resp` | respiratory | direct | direct | direct | direct |
| `retic` | hematology | unavailable | direct | direct | unavailable |
| `rocuronium` | medications | direct | direct | unavailable | direct |
| `rrt` | renal | direct | direct | direct | unavailable |
| `rrt_criteria` | renal | partial | partial | partial | partial |
| `safi` | respiratory | derived | derived | derived | partial |
| `samp` | microbiology | unavailable | direct | direct | unavailable |
| `sao2` | respiratory | direct | direct | direct | direct |
| `saps3` | severity | unavailable | unavailable | unavailable | unavailable |
| `sbp` | vitals | direct | direct | direct | direct |
| `scvo2` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `sedated_gcs` | neurology | unavailable | unavailable | unavailable | unavailable |
| `sep3` | outcome | partial | partial | partial | partial |
| `sep3_sofa2` | outcome | partial | partial | partial | partial |
| `sex` | demographics | direct | direct | direct | direct |
| `shock_index` | cardiovascular | derived | derived | derived | derived |
| `sirs` | outcome | partial | partial | partial | partial |
| `sofa` | outcome | partial | partial | partial | partial |
| `sofa2` | outcome | partial | partial | partial | partial |
| `sofa2_cardio` | outcome | partial | unavailable | partial | partial |
| `sofa2_cns` | outcome | unavailable | unavailable | unavailable | partial |
| `sofa2_cns_ascertainment` | outcome | unavailable | unavailable | unavailable | partial |
| `sofa2_cns_delirium_tx_ascertainment` | outcome | unavailable | unavailable | unavailable | partial |
| `sofa2_cns_proxy_sensitivity` | outcome | unavailable | unavailable | unavailable | partial |
| `sofa2_coag` | outcome | derived | derived | derived | derived |
| `sofa2_liver` | outcome | derived | derived | derived | derived |
| `sofa2_renal` | outcome | partial | partial | partial | partial |
| `sofa2_resp` | outcome | partial | partial | partial | partial |
| `sofa_cardio` | outcome | unavailable | unavailable | unavailable | partial |
| `sofa_cns` | outcome | unavailable | unavailable | unavailable | partial |
| `sofa_coag` | outcome | derived | derived | derived | derived |
| `sofa_liver` | outcome | derived | derived | derived | derived |
| `sofa_renal` | outcome | partial | derived | partial | derived |
| `sofa_resp` | outcome | partial | partial | partial | partial |
| `spo2` | respiratory | direct | direct | direct | direct |
| `supp_o2` | respiratory | derived | partial | partial | unavailable |
| `susp_inf` | outcome | partial | partial | partial | partial |
| `svo2` | cardiovascular | unavailable | unavailable | unavailable | unavailable |
| `t4` | chemistry | direct | direct | direct | unavailable |
| `tco2` | blood gas | direct | direct | direct | direct |
| `temp` | vitals | direct | direct | direct | direct |
| `tgcs` | neurological | unavailable | unavailable | unavailable | unavailable |
| `tibc` | chemistry | direct | unavailable | direct | unavailable |
| `tidal_vol` | ventilator | direct | direct | unavailable | direct |
| `tidal_vol_set` | ventilator | unavailable | direct | unavailable | direct |
| `tnt` | chemistry | direct | unavailable | unavailable | unavailable |
| `total_input_ml` | output | unavailable | unavailable | unavailable | unavailable |
| `total_protein` | chemistry | direct | direct | direct | direct |
| `transferrin` | chemistry | direct | unavailable | direct | unavailable |
| `tri` | chemistry | direct | direct | direct | direct |
| `trig` | chemistry | direct | direct | direct | direct |
| `tsh` | chemistry | direct | direct | direct | unavailable |
| `uo_12h` | output | partial | derived | partial | partial |
| `uo_24h` | output | partial | derived | partial | partial |
| `uo_6h` | output | partial | derived | partial | partial |
| `uric_acid` | chemistry | direct | unavailable | direct | direct |
| `urine` | output | unavailable | direct | unavailable | direct |
| `urine24` | output | unavailable | derived | unavailable | derived |
| `vancomycin` | medications | direct | direct | direct | direct |
| `vaso_ind` | medications | unavailable | unavailable | unavailable | unavailable |
| `vecuronium` | medications | direct | direct | direct | direct |
| `vent_breath_seq` | ventilator | unavailable | direct | unavailable | direct |
| `vent_end` | respiratory | direct | unavailable | unavailable | unavailable |
| `vent_ind` | respiratory | derived | unavailable | partial | unavailable |
| `vent_mode` | ventilator | unavailable | direct | unavailable | direct |
| `vent_rate` | ventilator | unavailable | direct | unavailable | direct |
| `vent_start` | respiratory | direct | unavailable | unavailable | unavailable |
| `vgcs` | neurological | unavailable | unavailable | unavailable | direct |
| `warfarin` | medications | direct | direct | direct | direct |
| `wbc` | hematology | direct | direct | direct | direct |
| `weight` | demographics | direct | direct | direct | unavailable |
