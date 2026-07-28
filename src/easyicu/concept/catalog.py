"""Shared concept catalog metadata for EasyICU runtimes."""

from __future__ import annotations

from ..concept_output_sources import (
    COMPOSITE_CONCEPT_OUTPUT_SOURCES as COMPOSITE_CONCEPT_OUTPUT_SOURCES,
)

# 数据字典定义 - 特征缩写及其含义
CONCEPT_DICTIONARY = {
    # 生命体征
    'hr': ('Heart Rate', '心率', 'bpm'),
    'map': ('Mean Arterial Pressure', '平均动脉压', 'mmHg'),
    'sbp': ('Systolic Blood Pressure', '收缩压', 'mmHg'),
    'dbp': ('Diastolic Blood Pressure', '舒张压', 'mmHg'),
    'pulse_pressure': ('Pulse Pressure (SBP - DBP)', '脉压差 (收缩压 - 舒张压)', 'mmHg'),
    'cvp': ('Central Venous Pressure', '中心静脉压', 'mmHg'),
    'shock_index': ('Shock Index', '休克指数', 'ratio'),
    'modified_shock_index': ('Modified Shock Index', '改良休克指数', 'ratio'),
    'diastolic_shock_index': ('Diastolic Shock Index', '舒张压休克指数', 'ratio'),
    'temp': ('Temperature', '体温', '°C'),
    'etco2': ('End-Tidal CO2', '呼气末二氧化碳', 'mmHg'),
    'resp': ('Respiratory Rate', '呼吸频率', 'breaths/min'),

    # 呼吸系统
    'pafi': ('PaO2/FiO2 Ratio', '氧合指数', 'mmHg'),
    'safi': ('SpO2/FiO2 Ratio', '脉氧/吸氧比', ''),
    'supp_o2': ('Supplemental Oxygen', '辅助吸氧', 'boolean'),
    'vent_ind': ('Ventilation Duration Windows', '机械通气时间窗', 'boolean'),
    'o2sat': ('Oxygen Saturation (SpO2)', '血氧饱和度', '%'),
    'sao2': ('Arterial Oxygen Saturation', '动脉血氧饱和度', '%'),
    'mech_vent': ('Mechanical Ventilation Mode', '机械通气模式', 'category'),
    'vent_mode': ('Ventilator Breath-Control Mode', '呼吸机控制类型', 'category'),
    'vent_breath_seq': ('Ventilator Breath Sequencing', '呼吸机呼吸序列', 'category'),
    'driving_pres_controlled': ('Driving Pressure (controlled vent)', '驱动压(控制通气)', 'cmH2O'),
    'ett_gcs': ('Intubation/Tracheostomy Status', '气管插管/切开状态', 'boolean'),
    'fio2': ('Fraction of Inspired Oxygen', '吸入氧浓度', '%'),

    # 血气分析
    'be': ('Base Excess', '碱剩余', 'mEq/L'),
    'cai': ('Ionized Calcium', '离子钙', 'mmol/L'),
    'hbco': ('Carboxyhemoglobin', '碳氧血红蛋白', '%'),
    'lact': ('Lactate', '乳酸', 'mmol/L'),
    'methb': ('Methemoglobin', '高铁血红蛋白', '%'),
    'pco2': ('Partial Pressure of CO2', '二氧化碳分压', 'mmHg'),
    'ph': ('Blood pH', '血液pH值', ''),
    'po2': ('Partial Pressure of O2', '氧分压', 'mmHg'),
    'tco2': ('Total CO2', '总二氧化碳', 'mEq/L'),

    # 实验室检查
    'alb': ('Albumin', '白蛋白', 'g/dL'),
    'alp': ('Alkaline Phosphatase', '碱性磷酸酶', 'IU/L'),
    'alt': ('Alanine Aminotransferase', '谷丙转氨酶', 'IU/L'),
    'ast': ('Aspartate Aminotransferase', '谷草转氨酶', 'IU/L'),
    'bicar': ('Bicarbonate', '碳酸氢根', 'mEq/L'),
    'bili': ('Total Bilirubin', '总胆红素', 'mg/dL'),
    'bili_dir': ('Direct Bilirubin', '直接胆红素', 'mg/dL'),
    'bun': ('Blood Urea Nitrogen', '血尿素氮', 'mg/dL'),
    'ca': ('Calcium', '钙', 'mg/dL'),
    'ck': ('Creatine Kinase', '肌酸激酶', 'IU/L'),
    'ckmb': ('CK-MB', '肌酸激酶同工酶', 'ng/mL'),
    'cl': ('Chloride', '氯', 'mEq/L'),
    'crea': ('Creatinine', '肌酐', 'mg/dL'),
    'crp': ('C-Reactive Protein', 'C反应蛋白', 'mg/L'),
    'glu': ('Glucose', '血糖', 'mg/dL'),
    'k': ('Potassium', '钾', 'mEq/L'),
    'mg': ('Magnesium', '镁', 'mg/dL'),
    'na': ('Sodium', '钠', 'mEq/L'),
    'anion_gap': ('Anion Gap (Na - Cl - HCO3)', '阴离子间隙', 'mEq/L'),
    'phos': ('Phosphorus', '磷', 'mg/dL'),
    'tnt': ('Troponin T', '肌钙蛋白T', 'ng/mL'),

    # 血液学
    'bnd': ('Band Neutrophils', '杆状核中性粒细胞', '%'),
    'esr': ('Erythrocyte Sedimentation Rate', '红细胞沉降率', 'mm/hr'),
    'fgn': ('Fibrinogen', '纤维蛋白原', 'mg/dL'),
    'hgb': ('Hemoglobin', '血红蛋白', 'g/dL'),
    'inr_pt': ('INR (Prothrombin Time)', '国际标准化比值', ''),
    'lymph': ('Lymphocytes', '淋巴细胞', '%'),
    'mch': ('Mean Corpuscular Hemoglobin', '平均红细胞血红蛋白含量', 'pg'),
    'mchc': ('Mean Corpuscular Hemoglobin Concentration', '平均红细胞血红蛋白浓度', 'g/dL'),
    'mcv': ('Mean Corpuscular Volume', '平均红细胞体积', 'fL'),
    'neut': ('Neutrophils', '中性粒细胞', '%'),
    'plt': ('Platelets', '血小板', '×10³/μL'),
    'ptt': ('Partial Thromboplastin Time', '部分凝血活酶时间', 'sec'),
    'wbc': ('White Blood Cells', '白细胞', '×10³/μL'),

    # 药物治疗
    'abx': ('Antibiotics', '抗生素使用', 'boolean'),
    'adh_rate': ('Vasopressin Rate', '血管加压素速率', 'units/min'),
    'cort': ('Corticosteroids', '糖皮质激素', 'boolean'),
    'dex': ('Dextrose (D10)', '葡萄糖（10%）', 'mL/hr'),
    'dobu_dur': ('Dobutamine Duration', '多巴酚丁胺持续时间', 'hours'),
    'dobu_rate': ('Dobutamine Rate', '多巴酚丁胺速率', 'mcg/kg/min'),
    'dobu60': ('Dobutamine Rate (>60min)', '多巴酚丁胺速率（持续>60分钟）', 'mcg/kg/min'),
    'epi_dur': ('Epinephrine Duration', '肾上腺素持续时间', 'hours'),
    'epi_rate': ('Epinephrine Rate', '肾上腺素速率', 'mcg/kg/min'),
    'furosemide': ('Furosemide (Lasix)', '呋塞米 (速尿)', 'boolean'),
    'propofol': ('Propofol (Diprivan)', '丙泊酚', 'boolean'),
    'propofol_rate': ('Propofol Rate', '丙泊酚速率', 'mcg/kg/min'),
    'midazolam': ('Midazolam (Versed)', '咪达唑仑', 'boolean'),
    'midazolam_rate': ('Midazolam Rate', '咪达唑仑速率', 'mg/hour'),
    'dexmedetomidine': ('Dexmedetomidine (Precedex)', '右美托咪定', 'boolean'),
    'fentanyl': ('Fentanyl', '芬太尼', 'boolean'),
    'fentanyl_rate': ('Fentanyl Rate', '芬太尼速率', 'mcg/hour'),
    'lorazepam': ('Lorazepam (Ativan)', '劳拉西泮', 'boolean'),
    'ketamine': ('Ketamine', '氯胺酮', 'boolean'),
    'vecuronium': ('Vecuronium', '维库溴铵', 'boolean'),
    'cisatracurium': ('Cisatracurium (Nimbex)', '顺阿曲库铵', 'boolean'),
    'nitroglycerin': ('Nitroglycerin', '硝酸甘油', 'boolean'),
    'pantoprazole': ('Pantoprazole (Protonix)', '泮托拉唑', 'boolean'),
    'vancomycin': ('Vancomycin', '万古霉素', 'boolean'),
    'meropenem': ('Meropenem', '美罗培南', 'boolean'),
    'calcium_iv': ('Calcium IV', '静脉钙剂', 'boolean'),
    'potassium_iv': ('Potassium IV', '静脉钾剂', 'boolean'),
    'magnesium_iv': ('Magnesium IV', '静脉镁剂', 'boolean'),
    'albumin_iv': ('Albumin IV', '静脉白蛋白', 'boolean'),
    'packed_rbc': ('Packed RBC Transfusion', '红细胞输注', 'boolean'),
    'bicarbonate': ('Sodium Bicarbonate', '碳酸氢钠', 'boolean'),
    'dextrose50': ('Dextrose 50% (D50)', '50%葡萄糖', 'boolean'),
    'ffp': ('Fresh Frozen Plasma', '新鲜冰冻血浆', 'boolean'),
    'platelets': ('Platelet Transfusion', '血小板输注', 'boolean'),
    'levetiracetam': ('Levetiracetam (Keppra)', '左乙拉西坦', 'boolean'),
    'dexamethasone': ('Dexamethasone', '地塞米松', 'boolean'),
    'octreotide': ('Octreotide (Sandostatin)', '奥曲肽', 'boolean'),
    'neostigmine': ('Neostigmine', '新斯的明', 'boolean'),
    'phenytoin': ('Phenytoin (Dilantin)', '苯妥英', 'boolean'),
    'labetalol': ('Labetalol', '拉贝洛尔', 'boolean'),
    'esmolol': ('Esmolol (Brevibloc)', '艾司洛尔', 'boolean'),
    'diltiazem': ('Diltiazem (Cardizem)', '地尔硫卓', 'boolean'),
    'nicardipine': ('Nicardipine (Cardene)', '尼卡地平', 'boolean'),
    'warfarin': ('Warfarin (Coumadin)', '华法林', 'boolean'),
    'apixaban': ('Apixaban (Eliquis)', '阿哌沙班', 'boolean'),
    'enoxaparin': ('Enoxaparin (Lovenox)', '依诺肝素', 'boolean'),
    'aspirin': ('Aspirin', '阿司匹林', 'boolean'),
    'insulin': ('Insulin (boolean)', '胰岛素 (布尔)', 'boolean'),
    'morphine': ('Morphine', '吗啡', 'boolean'),
    'heparin': ('Heparin', '肝素', 'boolean'),
    'mannitol': ('Mannitol', '甘露醇', 'boolean'),
    'amiodarone': ('Amiodarone (Cordarone)', '胺碘酮', 'boolean'),
    'milrinone': ('Milrinone (Primacor)', '米力农', 'boolean'),
    'rocuronium': ('Rocuronium (Esmeron)', '罗库溴铵', 'boolean'),
    'ins': ('Insulin', '胰岛素', 'units/hr'),
    'norepi_dur': ('Norepinephrine Duration', '去甲肾上腺素持续时间', 'hours'),
    'norepi_equiv': ('Norepinephrine Equivalent', '去甲肾上腺素当量', 'mcg/kg/min'),
    'norepi_rate': ('Norepinephrine Rate', '去甲肾上腺素速率', 'mcg/kg/min'),
    'vaso_ind': ('Vasopressor Indicator', '血管活性药物指示', 'boolean'),

    # 尿量
    'urine': ('Urine Output', '尿量', 'mL'),
    'urine24': ('24h Urine Output', '24小时尿量', 'mL/24h'),
    'total_input_ml': ('Total IV Fluid Input', '总输液量', 'mL/hr'),
    'fluid_balance': ('Hourly Fluid Balance', '每小时液体平衡', 'mL/hr'),
    'fluid_balance_cumulative': ('Cumulative Fluid Balance', '累计液体平衡', 'mL'),
    # 神经系统
    'avpu': ('AVPU Scale', 'AVPU意识评分', ''),
    'egcs': ('Eye Component of GCS', 'GCS眼睛评分', ''),
    'gcs': ('Glasgow Coma Scale', '格拉斯哥昏迷评分', ''),
    'mgcs': ('Motor Component of GCS', 'GCS运动评分', ''),
    'rass': ('Richmond Agitation-Sedation Scale', 'RASS镇静评分', ''),
    'tgcs': ('Total GCS', 'GCS总分', ''),
    'vgcs': ('Verbal Component of GCS', 'GCS语言评分', ''),

    # 人口统计
    'age': ('Age', '年龄', 'years'),
    'bmi': ('Body Mass Index', '体重指数', 'kg/m²'),
    'height': ('Height', '身高', 'cm'),
    'sex': ('Sex', '性别', ''),
    'weight': ('Weight', '体重', 'kg'),

    # SOFA-1 评分
    'sofa': ('SOFA Score (Total)', 'SOFA总分', '0-24'),
    'sofa_resp': ('SOFA Respiratory', 'SOFA呼吸评分', '0-4'),
    'sofa_coag': ('SOFA Coagulation', 'SOFA凝血评分', '0-4'),
    'sofa_liver': ('SOFA Liver', 'SOFA肝脏评分', '0-4'),
    'sofa_cardio': ('SOFA Cardiovascular', 'SOFA心血管评分', '0-4'),
    'sofa_cns': ('SOFA Central Nervous System', 'SOFA神经评分', '0-4'),
    'sofa_renal': ('SOFA Renal', 'SOFA肾脏评分', '0-4'),
    'qsofa': ('Quick SOFA', '快速SOFA评分', '0-3'),
    'sirs': ('SIRS Criteria', 'SIRS标准', '0-4'),
    'mews': ('Modified Early Warning Score', '改良早期预警评分', '0-14'),
    'news': ('National Early Warning Score', '国家早期预警评分', '0-20'),
    'death': ('In-hospital Mortality', '院内死亡', 'boolean'),
    'los_icu': ('ICU Length of Stay', 'ICU住院时长', 'days'),
    'los_hosp': ('Hospital Length of Stay', '住院时长', 'days'),

    # SOFA-2 评分 (2025年新标准)
    'sofa2': ('SOFA-2 Score (Total)', 'SOFA-2总分 (2025新标准)', '0-24'),
    'sofa2_resp': ('SOFA-2 Respiratory', 'SOFA-2呼吸评分', '0-4'),
    'sofa2_coag': ('SOFA-2 Coagulation', 'SOFA-2凝血评分', '0-4'),
    'sofa2_liver': ('SOFA-2 Liver', 'SOFA-2肝脏评分', '0-4'),
    'sofa2_cardio': ('SOFA-2 Cardiovascular', 'SOFA-2心血管评分', '0-4'),
    'sofa2_cns': ('SOFA-2 Central Nervous System', 'SOFA-2神经评分', '0-4'),
    'sofa2_renal': ('SOFA-2 Renal', 'SOFA-2肾脏评分', '0-4'),

    # Sepsis 诊断
    'sep3_sofa1': ('Sepsis-3 (SOFA-1 based)', 'Sepsis-3诊断 (基于传统SOFA)', 'boolean'),
    'sep3_sofa2': ('Sepsis-3 (SOFA-2 based)', 'Sepsis-3诊断 (基于SOFA-2, 2025新标准)', 'boolean'),
    'susp_inf': ('Suspected Infection (ICD or Abx+Culture timing)', '疑似感染 (ICD诊断码或抗生素+培养时间窗)', 'boolean'),
    'infection_icd': ('ICD Infection Diagnosis (eICU only, Angus 2001)', 'ICD感染诊断 (仅eICU, Angus标准)', 'boolean'),

    # 呼吸系统 (扩展)
    'spo2': ('Peripheral Oxygen Saturation', '脉搏血氧饱和度', '%'),
    'vent_start': ('Ventilation Start Time', '通气开始时间', 'datetime'),
    'vent_end': ('Ventilation End Time', '通气结束时间', 'datetime'),
    'ecmo': ('ECMO in Use', 'ECMO使用中', 'boolean'),
    'ecmo_indication': ('ECMO Indication', 'ECMO适应症 (呼吸/心血管)', ''),
    'adv_resp': ('Advanced Respiratory Support', '高级呼吸支持 (IMV/NIV/HFNC)', 'boolean'),
    'oxygenation_index': ('Oxygenation Index', '氧合指数 (OI)', ''),

    # 呼吸机参数 (Ventilator Parameters)
    'peep': ('Positive End-Expiratory Pressure', '呼气末正压', 'cmH2O'),
    'tidal_vol': ('Tidal Volume (Observed)', '潮气量（实测）', 'mL'),
    'tidal_vol_set': ('Tidal Volume (Set)', '潮气量（设定）', 'mL'),
    'pip': ('Peak Inspiratory Pressure', '吸气峰压', 'cmH2O'),
    'plateau_pres': ('Plateau Pressure', '平台压', 'cmH2O'),
    'mean_airway_pres': ('Mean Airway Pressure', '平均气道压', 'cmH2O'),
    'minute_vol': ('Minute Ventilation', '分钟通气量', 'L/min'),
    'vent_rate': ('Ventilator Respiratory Rate', '呼吸机频率', '/min'),
    'compliance': ('Static Compliance', '静态肺顺应性', 'mL/cmH2O'),
    'driving_pres': ('Driving Pressure', '驱动压', 'cmH2O'),
    'ps': ('Pressure Support', '压力支持', 'cmH2O'),

    # 血液学 (扩展)
    'basos': ('Basophils', '嗜碱性粒细胞', '%'),
    'eos': ('Eosinophils', '嗜酸性粒细胞', '%'),
    'hba1c': ('Hemoglobin A1C', '糖化血红蛋白', '%'),
    'hct': ('Hematocrit', '红细胞压积', '%'),
    'pt': ('Prothrombin Time', '凝血酶原时间', 'sec'),
    'rbc': ('Red Blood Cell Count', '红细胞计数', '×10⁶/μL'),
    'rdw': ('Red Cell Distribution Width', '红细胞分布宽度', '%'),
    'nlr': ('Neutrophil-to-Lymphocyte Ratio', '中性粒/淋巴细胞比值', 'ratio'),
    'plr': ('Platelet-to-Lymphocyte Ratio', '血小板/淋巴细胞比值', 'ratio'),

    # 生化 (扩展)
    'tri': ('Troponin I', '肌钙蛋白I', 'ng/mL'),

    # 药物 (扩展)
    'dopa_rate': ('Dopamine Rate', '多巴胺速率', 'mcg/kg/min'),
    'dopa_dur': ('Dopamine Duration', '多巴胺持续时间', 'hours'),
    'dopa60': ('Dopamine Rate (>60min)', '多巴胺速率（持续>60分钟）', 'mcg/kg/min'),
    'norepi60': ('Norepinephrine Rate (>60min)', '去甲肾上腺素速率（持续>60分钟）', 'mcg/kg/min'),
    'epi60': ('Epinephrine Rate (>60min)', '肾上腺素速率（持续>60分钟）', 'mcg/kg/min'),
    'phn_rate': ('Phenylephrine Rate', '去氧肾上腺素速率', 'mcg/kg/min'),

    # 肾脏与尿量率
    'rrt': ('Renal Replacement Therapy', '肾脏替代治疗', 'boolean'),
    'rrt_criteria': ('RRT Criteria Met', '满足RRT标准', 'boolean'),
    'uo_6h': ('Average Urine Output Rate (past 6h)', '过去6小时平均尿量率', 'mL/kg/h'),
    'uo_12h': ('Average Urine Output Rate (past 12h)', '过去12小时平均尿量率', 'mL/kg/h'),
    'uo_24h': ('Average Urine Output Rate (past 24h)', '过去24小时平均尿量率', 'mL/kg/h'),

    # KDIGO AKI (急性肾损伤) - 🔧 2026-02-04: 移除重复的 kdigo_aki/kdigo_creat/kdigo_uo
    'aki': ('Acute Kidney Injury', '急性肾损伤', 'boolean'),
    'aki_stage': ('AKI Stage (KDIGO)', 'AKI分期（KDIGO标准）', '0-3'),
    'aki_stage_creat': ('AKI Stage (Creatinine)', 'AKI分期（肌酐）', '0-3'),
    'aki_stage_uo': ('AKI Stage (Urine Output)', 'AKI分期（尿量）', '0-3'),
    'aki_stage_rrt': ('AKI Stage (RRT)', 'AKI分期（RRT）', '0-3'),
    # 🔧 2026-02-12: 添加规范化后的 KDIGO 扩展列
    'creat_low_past_48hr': ('Lowest Creatinine in Past 48h', '过去48小时内最低肌酐', 'mg/dL'),
    'creat_low_past_7day': ('Baseline Creatinine (7-day lowest)', '基线肌酐（7天内最低值）', 'mg/dL'),
    'uo_rt_6hr': ('Urine Output Rate (6h rolling window)', '尿量率（6小时滚动窗口）', 'mL/kg/h'),
    'uo_rt_12hr': ('Urine Output Rate (12h rolling window)', '尿量率（12小时滚动窗口）', 'mL/kg/h'),
    'uo_rt_24hr': ('Urine Output Rate (24h rolling window)', '尿量率（24小时滚动窗口）', 'mL/kg/h'),
    'bun_creatinine_ratio': ('BUN-to-Creatinine Ratio', '尿素氮/肌酐比值', 'ratio'),
    'egfr': ('Estimated Glomerular Filtration Rate', '估算肾小球滤过率', 'mL/min/1.73m²'),

    # 神经 (扩展)
    'sedated_gcs': ('GCS Before Sedation', '镇静前GCS', ''),

    # 心血管 (扩展)
    'mech_circ_support': ('Mechanical Circulatory Support', '机械循环支持 (IABP/LVAD/Impella)', 'boolean'),
    'other_vaso': ('Other Vasopressors', '其他血管活性药物', 'boolean'),
    'circ_failure': ('Circulatory Failure', '循环衰竭', 'boolean'),
    'circ_event': ('Circulatory Failure Event Level', '循环衰竭事件等级', '0-3'),

    # 神经系统 SOFA-2 扩展
    'motor_response': ('GCS Motor Response', 'GCS运动反应', '1-6'),
    'delirium_positive': ('Delirium Positive (CAM-ICU)', '谵妄阳性（CAM-ICU）', 'boolean'),
    'delirium_tx': ('Delirium Treatment', '谵妄治疗', 'boolean'),

    # 人口统计 (扩展)
    'adm': ('Admission Type', '入院类型', ''),

    # 微生物
    'samp': ('Body Fluid Sampling (for infection workup)', '体液采样 (用于感染检查)', 'boolean'),
    'culture_positive': ('Any Positive Culture', '任意培养阳性', 'boolean'),
    'bld_culture_positive': ('Positive Blood Culture', '血培养阳性', 'boolean'),

    # 严重度评分 (扩展)
    'apache_iv': ('APACHE IVa Score', 'APACHE IVa 评分', 'points'),
    'apache_iv_pred_hosp_mort': ('APACHE IVa Predicted Hospital Mortality', 'APACHE IVa 预测院内死亡率', 'fraction'),
    'saps3': ('SAPS-3 Score', 'SAPS-3 评分', 'points'),

    # 合并症指数
    'charlson': ('Charlson Comorbidity Index', 'Charlson 合并症指数', 'points'),
    'elixhauser': ('Elixhauser (van Walraven) Score', 'Elixhauser (van Walraven) 评分', 'points'),

    # 复合结局终点
    'mort_28d': ('28-day Mortality', '28天死亡率', 'boolean'),
    'mort_90d': ('90-day Mortality', '90天死亡率', 'boolean'),
    'mort_365d': ('1-year Mortality', '1年死亡率', 'boolean'),
    'icu_free_days_28': ('ICU-free Days (to day 28)', 'ICU-free 天数 (至第28天)', 'days'),
    'vent_free_days_28': ('Ventilator-free Days (to day 28)', '无机械通气天数 (至第28天)', 'days'),
    'icu_readmission': ('ICU Readmission (same hospitalisation)', 'ICU 再入 (同次住院)', 'boolean'),
    'persistent_critical_illness': ('Persistent Critical Illness', '持续危重症', 'boolean'),

    # 实验室检查 (扩展)
    'ammonia': ('Ammonia', '氨', 'umol/L'),
    'amylase': ('Amylase', '淀粉酶', 'U/L'),
    'd_dimer': ('D-dimer', 'D-二聚体', 'ng/mL'),
    'ferritin': ('Ferritin', '铁蛋白', 'ng/mL'),
    'ldh': ('Lactate Dehydrogenase', '乳酸脱氢酶', 'U/L'),
    'lipase': ('Lipase', '脂肪酶', 'U/L'),
    'osmolality': ('Serum Osmolality', '血清渗透压', 'mOsm/kg'),
    'corrected_calcium': ('Albumin-Corrected Calcium', '白蛋白校正钙', 'mg/dL'),

    # 2026-07-04 recall/completeness audit — newly added concepts
    'pap_sys': ('Pulmonary Artery Pressure, Systolic', '肺动脉收缩压', 'mmHg'),
    'pap_dia': ('Pulmonary Artery Pressure, Diastolic', '肺动脉舒张压', 'mmHg'),
    'pap_mean': ('Pulmonary Artery Pressure, Mean', '肺动脉平均压', 'mmHg'),
    'co': ('Cardiac Output', '心输出量', 'L/min'),
    'ggt': ('Gamma-Glutamyl Transferase', 'γ-谷氨酰转移酶', 'U/L'),
    'trig': ('Triglycerides', '甘油三酯', 'mg/dL'),
    'tsh': ('Thyroid Stimulating Hormone', '促甲状腺激素', 'mIU/L'),
    'total_protein': ('Total Protein', '总蛋白', 'g/dL'),
    'ntprobnp': ('NT-proBNP', 'N末端脑钠肽前体', 'pg/mL'),
    'monos': ('Monocytes', '单核细胞', '%'),
    'mpv': ('Mean Platelet Volume', '平均血小板体积', 'fL'),
    'icp': ('Intracranial Pressure', '颅内压', 'mmHg'),
    'svo2': ('Mixed Venous O₂ Saturation', '混合静脉血氧饱和度', '%'),
    'scvo2': ('Central Venous O₂ Saturation', '中心静脉血氧饱和度', '%'),
    'pawp': ('Pulmonary Artery Wedge Pressure', '肺动脉楔压', 'mmHg'),
    'cortisol': ('Cortisol', '皮质醇', 'µg/dL'),
    'pct': ('Procalcitonin', '降钙素原', 'ng/mL'),
    'bnp': ('B-type Natriuretic Peptide', 'B型利钠肽', 'pg/mL'),
    'uric_acid': ('Uric Acid', '尿酸', 'mg/dL'),
    'cholesterol': ('Total Cholesterol', '总胆固醇', 'mg/dL'),
    'hdl': ('HDL Cholesterol', '高密度脂蛋白胆固醇', 'mg/dL'),
    'ldl': ('LDL Cholesterol', '低密度脂蛋白胆固醇', 'mg/dL'),
    'iron': ('Iron (Fe)', '血清铁', 'µg/dL'),
    'tibc': ('Total Iron Binding Capacity', '总铁结合力', 'µg/dL'),
    'transferrin': ('Transferrin', '转铁蛋白', 'mg/dL'),
    'ft4': ('Free Thyroxine (FT4)', '游离甲状腺素', 'ng/dL'),
    'prealbumin': ('Prealbumin', '前白蛋白', 'mg/dL'),
    'myoglobin': ('Myoglobin', '肌红蛋白', 'ng/mL'),
    't4': ('Total Thyroxine (T4)', '总甲状腺素', 'µg/dL'),
    'retic': ('Reticulocyte %', '网织红细胞百分比', '%'),
}

# 特征详细描述（英文和中文）
CONCEPT_DESCRIPTIONS = {
    # SOFA-2
    'sofa2': ('Total SOFA-2 score (2025 new standard), sum of 6 organ systems (0-24)', 'SOFA-2总分（2025年新标准），6个器官系统评分之和（0-24分）'),
    'sofa2_resp': ('Respiratory: PaO2/FiO2 (or SpO2/FiO2 if unavailable), scores 3-4 require advanced respiratory support (IMV/NIV/HFNC) or ECMO', '呼吸系统：基于氧合指数，3-4分需要高级呼吸支持（IMV/NIV/HFNC）或ECMO'),
    'sofa2_coag': ('Coagulation: platelet count with updated thresholds (≤50→4, ≤80→3, ≤100→2, ≤150→1)', '凝血系统：基于血小板计数，使用更新的阈值（≤50→4分，≤80→3分，≤100→2分，≤150→1分）'),
    'sofa2_liver': ('Liver: bilirubin with relaxed 1-point threshold (>1.2 mg/dL instead of >1.9)', '肝脏：基于胆红素，1分阈值放宽（>1.2 mg/dL，原为>1.9）'),
    'sofa2_cardio': ('Cardiovascular: combined NE+Epi dose, other vasopressors/inotropes, or mechanical circulatory support (IABP/LVAD/Impella)', '心血管：基于去甲肾+肾上腺素联合剂量、其他血管活性药物或机械循环支持'),
    'sofa2_cns': ('Neurological: GCS score, with delirium (CAM-ICU+ or treatment) adding 1 point if GCS=15', '神经系统：基于GCS评分，若GCS=15但有谵妄（CAM-ICU阳性或接受治疗）则加1分'),
    'sofa2_renal': ('Renal: creatinine and urine output (6h/12h/24h windows), score 4 for RRT or meeting RRT criteria', '肾脏：基于肌酐和尿量（6h/12h/24h窗口），接受RRT或满足RRT标准则为4分'),

    # Sepsis
    'sep3_sofa2': ('Sepsis-3 diagnosis: suspected infection + SOFA-2 ≥2 point increase from baseline', '基于SOFA-2的Sepsis-3诊断：疑似感染 + SOFA-2较基线升高≥2分'),
    'sep3_sofa1': ('Sepsis-3 diagnosis: suspected infection + traditional SOFA ≥2 point increase', '基于传统SOFA的Sepsis-3诊断：疑似感染 + SOFA较基线升高≥2分'),
    'susp_inf': ('Suspected infection: (1) ICD infection diagnosis codes (eICU only) OR (2) antibiotics started within 72h of culture OR culture within 24h of antibiotics. Combines infection_icd, abx, and samp concepts.', '疑似感染：(1) ICD感染诊断码（仅eICU可用）或 (2) 培养后72小时内开始抗生素 或 抗生素后24小时内进行培养。由infection_icd、abx和samp概念组合而成'),
    'infection_icd': ('Infection diagnosis based on Angus 2001 ICD criteria (explicit infection codes). ONLY available in eICU database.', '基于Angus 2001 ICD标准的感染诊断（显性感染编码）。仅eICU数据库可用'),
    'samp': ('Body fluid sampling (blood, urine, sputum, etc.) for culture-based infection workup. Used as a marker for suspected infection when combined with antibiotic timing.', '体液采样（血液、尿液、痰液等）用于培养检查。与抗生素时间窗结合作为疑似感染的标志'),

    # Vitals
    'hr': ('Heart rate in beats per minute', '每分钟心跳次数'),
    'map': ('Mean arterial pressure = (SBP + 2×DBP) / 3', '平均动脉压 = (收缩压 + 2×舒张压) / 3'),
    'sbp': ('Systolic blood pressure (peak pressure during heartbeat)', '收缩压（心脏收缩时的最高压力）'),
    'dbp': ('Diastolic blood pressure (pressure between heartbeats)', '舒张压（心脏舒张时的最低压力）'),
    'pulse_pressure': ('Pulse pressure = SBP - DBP; narrow (<25) suggests shock/tamponade, wide (>60) suggests sepsis/aortic regurgitation', '脉压差 = 收缩压 - 舒张压；窄脉压(<25)提示休克/心包填塞，宽脉压(>60)提示脓毒症/主动脉瓣关闭不全'),
    'temp': ('Body temperature in Celsius', '体温（摄氏度）'),
    'resp': ('Respiratory rate (breaths per minute)', '呼吸频率（每分钟呼吸次数）'),

    # Respiratory
    'pafi': ('PaO2/FiO2 ratio - key oxygenation index for ARDS/SOFA scoring', '氧合指数 - ARDS/SOFA评分的关键指标'),
    'safi': ('SpO2/FiO2 ratio - non-invasive alternative to PaFi (used when SpO2<98%)', '脉氧/吸氧比 - PaFi的非侵入性替代（当SpO2<98%时使用）'),
    'fio2': ('Fraction of inspired oxygen (21-100%)', '吸入氧浓度（21-100%）'),
    'vent_ind': ('Mechanical ventilation indicator (boolean)', '机械通气指示（布尔值）'),
    'ecmo_indication': ("ECMO indication type: 'respiratory' (lung failure) or 'cardiovascular' (heart failure). Any ECMO auto-scores 4 in SOFA-2 resp; cardiovascular indication also scores in SOFA-2 cardio as mech_circ_support", "ECMO适应症类型：'respiratory'（肺衰竭）或'cardiovascular'（心衰）。任何ECMO均使SOFA-2呼吸评分为4分；心血管适应症还计入SOFA-2心血管的机械循环支持"),
    'adv_resp': ('Advanced respiratory support indicator: IMV (invasive mechanical ventilation), NIV (non-invasive ventilation), HFNC (high-flow nasal cannula), CPAP, or BiPAP - required for SOFA-2 respiratory scores 3-4', '高级呼吸支持指示：IMV（有创机械通气）、NIV（无创通气）、HFNC（经鼻高流量）、CPAP或BiPAP - SOFA-2呼吸评分3-4分的必要条件'),

    # Blood gas
    'lact': ('Lactate - marker of tissue hypoperfusion and shock', '乳酸 - 组织低灌注和休克的标志物'),
    'ph': ('Blood acidity/alkalinity (normal 7.35-7.45)', '血液酸碱度（正常7.35-7.45）'),
    'pco2': ('Partial pressure of CO2 in arterial blood', '动脉血中二氧化碳分压'),
    'po2': ('Partial pressure of O2 in arterial blood', '动脉血中氧分压'),

    # Labs
    'crea': ('Serum creatinine - kidney function marker, key for SOFA renal scoring', '血清肌酐 - 肾功能标志物，SOFA肾脏评分关键指标'),
    'bili': ('Total bilirubin - liver function marker, key for SOFA liver scoring', '总胆红素 - 肝功能标志物，SOFA肝脏评分关键指标'),
    'plt': ('Platelet count - coagulation marker, key for SOFA coagulation scoring', '血小板计数 - 凝血功能标志物，SOFA凝血评分关键指标'),
    'wbc': ('White blood cell count - infection/inflammation marker', '白细胞计数 - 感染/炎症标志物'),

    # Vasopressors
    'norepi_rate': ('Norepinephrine infusion rate in μg/kg/min (weight-adjusted)', '去甲肾上腺素输注速率（μg/kg/min，体重校正）'),
    'norepi_equiv': ('Norepinephrine equivalent dose - standardized vasopressor potency', '去甲肾上腺素当量 - 标准化血管活性药物效价'),
    'vaso_ind': ('Any vasopressor use indicator (boolean)', '任何血管活性药物使用指示（布尔值）'),
    'other_vaso': ('Other vasopressors/inotropes: vasopressin, phenylephrine, milrinone (combined with dobutamine in SOFA-2 cardio scoring as "has_other_vaso")', '其他血管活性药物：血管加压素、去氧肾上腺素、米力农（在SOFA-2心血管评分中与多巴酚丁胺合并为"has_other_vaso"）'),

    # Neurological
    'gcs': ('Glasgow Coma Scale total score (3-15), key for SOFA CNS scoring', '格拉斯哥昏迷评分总分（3-15分），SOFA神经评分关键指标'),

    # Outcomes
    'death': ('In-hospital mortality (0=survived, 1=died)', '院内死亡（0=存活，1=死亡）'),
    'los_icu': ('ICU length of stay in days', 'ICU住院时长（天）'),
    'los_hosp': ('Hospital length of stay in days', '总住院时长（天）'),

    # AKI
    'aki': ('Acute Kidney Injury (KDIGO Stage ≥1)', '急性肾损伤（KDIGO分期≥1）'),
    'aki_stage': ('KDIGO AKI stage (0-3): max of creatinine and urine output criteria', 'KDIGO AKI分期（0-3）：肌酐和尿量标准的最大值'),
    'aki_stage_creat': ('AKI stage based on creatinine: ≥1.5x baseline or ≥0.3 mg/dL increase in 48h', '基于肌酐的AKI分期：较基线升高≥1.5倍 或 48h内升高≥0.3 mg/dL'),
    'aki_stage_uo': ('AKI stage based on urine output: <0.5 mL/kg/h for 6h (Stage 1), 12h (Stage 2), or <0.3 for 24h (Stage 3)', '基于尿量的AKI分期：<0.5 mL/kg/h持续6h(1期)、12h(2期) 或 <0.3持续24h(3期)'),

    # Circulatory failure
    'circ_failure': ('Circulatory failure (circEWS definition): lactate ≥2 mmol/L with hypotension/vasopressors', '循环衰竭（circEWS定义）：乳酸≥2 mmol/L伴低血压或血管活性药物'),
    'circ_event': ('Circulatory failure event level (0-3): based on lactate, MAP, and vasopressor tier', '循环衰竭事件等级（0-3）：基于乳酸、MAP和血管活性药物等级'),

    # Other scores
    'qsofa': ('Quick SOFA (0-3): RR≥22 + altered mental status + SBP≤100', '快速SOFA（0-3分）：呼吸频率≥22 + 意识改变 + 收缩压≤100'),
    'sirs': ('SIRS criteria (0-4): temp + HR + RR/PaCO2 + WBC/bands', 'SIRS标准（0-4分）：体温 + 心率 + 呼吸/PaCO2 + 白细胞/杆状核'),
}

# 🔧 FIX (2026-02-09): 随机患者采样，避免 eICU 等多中心数据库的采样偏差
# 使用固定种子保证可复现
def _sample_patient_ids_random(all_ids: list, n: int, seed: int = 42) -> list:
    """从患者ID列表中随机采样n个，使用固定种子保证可复现。

    修复 eICU 等多中心数据库的采样偏差问题：
    - 旧方法：all_ids[:n] 按ID排序取前N个 → 可能全部来自同一家医院
    - 新方法：随机采样 → 覆盖多家医院，确保各种特征（GCS、血管活性药等）有数据
    """
    import random
    if len(all_ids) <= n:
        return all_ids
    rng = random.Random(seed)
    return sorted(rng.sample(all_ids, n))


def _get_patient_id_table_files(database: str) -> list:
    """返回数据库特定的患者ID表文件查找列表。

    不同数据库的患者ID存储在不同的表中：
    - MIIV/MIMIC-III: icu/icustays.parquet or icustays.parquet
    - eICU: patient.parquet
    - AUMC: admissions.parquet
    - HiRID: general.parquet
    - SICdb: cases.parquet

    返回按优先级排序的文件列表，确保所有数据库都能正确找到患者ID。
    """
    # 数据库特定的主表
    db_specific = {
        'miiv': ['icu/icustays.parquet'],
        'mimic': ['icu/icustays.parquet'],
        'hirid': ['general.parquet'],
        'sic': ['cases.parquet'],
        'aumc': ['admissions.parquet'],
    }
    specific = db_specific.get(database, [])
    # 通用查找列表
    generic = ['icustays.parquet', 'patient.parquet', 'admissions.parquet', 'general.parquet', 'cases.parquet']
    # 合并：先查数据库特定的，再查通用的（去重）
    result = list(specific)
    for f in generic:
        if f not in result:
            result.append(f)
    return result


# 全局特征分组定义 - 供侧边栏和数据字典共用
# 使用英文key，并提供双语显示名称
CONCEPT_GROUPS_INTERNAL = {
    'sofa2_score': ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
    'sofa1_score': ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
    'sepsis3_sofa2': ['sep3_sofa2'],  # 🔧 共享概念移到单独的 sepsis_shared 模块
    'sepsis3_sofa1': ['sep3_sofa1'],  # 🔧 共享概念移到单独的 sepsis_shared 模块
    'sepsis_shared': ['susp_inf', 'infection_icd', 'samp', 'culture_positive', 'bld_culture_positive'],  # Sepsis共享概念（已移除sep3）
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'pulse_pressure', 'cvp', 'temp', 'spo2', 'resp', 'shock_index', 'modified_shock_index', 'diastolic_shock_index'],  # 🔧 etco2 移到 ventilator；cvp(中心静脉压,measured 血流动力学 vital,dict category=vitals)接入 vitals 模块,不再走单独 cvp_extraction
    'respiratory': ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication', 'adv_resp', 'oxygenation_index'],
    'ventilator': ['peep', 'tidal_vol', 'tidal_vol_set', 'pip', 'plateau_pres', 'mean_airway_pres', 'minute_vol', 'vent_rate', 'etco2', 'compliance', 'driving_pres', 'ps', 'vent_mode', 'vent_breath_seq', 'driving_pres_controlled'],
    'blood_gas': ['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2'],
    'chemistry': ['alb', 'alp', 'alt', 'ast', 'anion_gap', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri', 'ammonia', 'amylase', 'd_dimer', 'ferritin', 'ldh', 'lipase', 'osmolality', 'corrected_calcium', 'ggt', 'trig', 'tsh', 'total_protein', 'ntprobnp', 'cortisol', 'pct', 'bnp', 'uric_acid', 'cholesterol', 'hdl', 'ldl', 'iron', 'tibc', 'transferrin', 'ft4', 'prealbumin', 'myoglobin', 't4'],
    'hematology': ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt', 'ptt', 'rbc', 'rdw', 'wbc', 'nlr', 'plr', 'monos', 'mpv', 'retic'],
    'vasopressors': ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind', 'other_vaso'],
    'medications': ['abx', 'albumin_iv', 'bicarbonate', 'calcium_iv', 'cort', 'dex', 'dexamethasone', 'dextrose50', 'ffp', 'ins', 'amiodarone', 'cisatracurium', 'dexmedetomidine', 'fentanyl', 'fentanyl_rate', 'furosemide', 'heparin', 'ketamine', 'levetiracetam', 'lorazepam', 'magnesium_iv', 'mannitol', 'meropenem', 'midazolam', 'midazolam_rate', 'milrinone', 'morphine', 'neostigmine', 'nitroglycerin', 'octreotide', 'packed_rbc', 'pantoprazole', 'platelets', 'potassium_iv', 'propofol', 'propofol_rate', 'rocuronium', 'vancomycin', 'vecuronium', 'apixaban', 'aspirin', 'diltiazem', 'enoxaparin', 'esmolol', 'insulin', 'labetalol', 'nicardipine', 'phenytoin', 'warfarin'],
    # 🔧 2026-02-04: 移除重复的 kdigo_aki/kdigo_creat/kdigo_uo，只保留 aki_* 规范名
    'renal': ['urine', 'urine24', 'uo_6h', 'uo_12h', 'uo_24h', 'rrt', 'rrt_criteria', 'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
              # 规范化后的列名（从 kdigo_* 展开列规范化而来）
              'creat_low_past_48hr', 'creat_low_past_7day', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
              # 液体平衡（入量/出量衍生）
              'fluid_balance', 'fluid_balance_cumulative', 'total_input_ml',
              # 衍生肾功能指数 (Tier 1, 2026-06-22)
              'bun_creatinine_ratio', 'egfr'],
    'neurological': ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'tgcs', 'vgcs', 'sedated_gcs', 'motor_response', 'delirium_positive', 'delirium_tx', 'icp'],
    'circulatory': ['mech_circ_support', 'circ_failure', 'circ_event', 'pap_sys', 'pap_dia', 'pap_mean', 'co', 'svo2', 'scvo2', 'pawp'],  # 🔧 添加循环衰竭特征 + 肺动脉压/心输出量 + 静脉血氧/楔压 (2026-07-04)
    'demographics': ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
    'other_scores': ['qsofa', 'sirs', 'mews', 'news', 'apache_iv', 'apache_iv_pred_hosp_mort', 'saps3', 'charlson', 'elixhauser'],
    'outcome': ['death', 'los_icu', 'los_hosp', 'mort_28d', 'mort_90d', 'mort_365d', 'icu_free_days_28', 'vent_free_days_28', 'icu_readmission', 'persistent_critical_illness'],
}

# Concepts present in the extraction dictionary but intentionally hidden from
# the user-facing catalog.  They are loader entry points or legacy/source aliases
# whose outputs are surfaced through canonical web concepts above.
HIDDEN_DICTIONARY_CONCEPTS = {
    'bicarb',
    'kdigo_aki',
    'kdigo_creat',
    'kdigo_uo',
    'potassium',
    'sep3',
}

# 双语显示名称映射（优化：更清晰的命名区分评分vs诊断，包含准确特征数量）
CONCEPT_GROUP_NAMES = {
    'sofa2_score': ('⭐ SOFA-2 Scores', '⭐ SOFA-2 评分'),
    'sofa1_score': ('📊 SOFA-1 Scores', '📊 SOFA-1 评分'),
    'sepsis3_sofa2': ('🦠 Sepsis-3 (SOFA-2 based)', '🦠 Sepsis-3 (基于SOFA-2)'),
    'sepsis3_sofa1': ('🦠 Sepsis-3 (SOFA-1 based)', '🦠 Sepsis-3 (基于SOFA-1)'),
    'sepsis_shared': ('🦠 Sepsis Shared Concepts', '🦠 Sepsis 共享概念'),
    'vitals': ('❤️ Vital Signs', '❤️ 生命体征'),
    'respiratory': ('💨 Respiratory System', '💨 呼吸系统'),
    'ventilator': ('🌬️ Ventilator Parameters', '🌬️ 呼吸机参数'),
    'blood_gas': ('🩸 Blood Gas Analysis', '🩸 血气分析'),
    'chemistry': ('🧪 Lab - Chemistry', '🧪 实验室-生化'),
    'hematology': ('🔬 Lab - Hematology', '🔬 实验室-血液学'),
    'vasopressors': ('💉 Vasopressors', '💉 血管活性药物'),
    'medications': ('💊 Other Medications', '💊 其他药物'),
    'renal': ('🚰 Renal & Urine Output', '🚰 肾脏与尿量'),
    'neurological': ('🧠 Neurological', '🧠 神经系统'),
    'circulatory': ('❤️‍🩹 Circulatory System', '❤️‍🩹 循环系统'),
    'demographics': ('👤 Demographics', '👤 人口统计'),
    'other_scores': ('📈 Other Scores', '📈 其他评分'),
    'outcome': ('🎯 Outcome', '🎯 结局'),
}

# 用于时序分析页面的显示名称映射（英文版本）
CONCEPT_GROUPS_DISPLAY = {
    'sofa2_score': '⭐ SOFA-2 Scores',
    'sofa1_score': '📊 SOFA-1 Scores',
    'sepsis3_sofa2': '🦠 Sepsis-3 (SOFA-2)',
    'sepsis3_sofa1': '🦠 Sepsis-3 (SOFA-1)',
    'sepsis_shared': '🦠 Sepsis Shared',
    'vitals': '❤️ Vital Signs',
    'respiratory': '💨 Respiratory',
    'ventilator': '🌬️ Ventilator',
    'blood_gas': '🩸 Blood Gas',
    'chemistry': '🧪 Chemistry',
    'hematology': '🔬 Hematology',
    'vasopressors': '💉 Vasopressors',
    'medications': '💊 Medications',
    'renal': '🚰 Renal',
    'neurological': '🧠 Neurological',
    'circulatory': '❤️‍🩹 Circulatory',
    'demographics': '👤 Demographics',
    'other_scores': '📈 Other Scores',
    'outcome': '🎯 Outcome',
}

MODULE_PREVIEW_SUMMARIES = {
    'renal': {
        'en': "AKI staging, urine output, RRT, and creatinine-derived context in one module preview.",
        'zh': "将 AKI 分期、尿量、RRT 和肌酐基线线索放在同一模块预览中。",
    },
    'respiratory': {
        'en': "Respiratory support, oxygenation, and ventilation status in a single preview.",
        'zh': "在同一预览中查看呼吸支持、氧合和通气状态。",
    },
    'vitals': {
        'en': "Core bedside vital signs aligned into a compact longitudinal preview.",
        'zh': "将核心床旁生命体征汇总到紧凑的纵向预览中。",
    },
    'chemistry': {
        'en': "Key chemistry measurements grouped for quick sanity checks before deeper analysis.",
        'zh': "将关键生化指标汇总，便于深入分析前快速核查。",
    },
}

MODULE_PREVIEW_TAG_PRIORITY = {
    'renal': ['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'rrt', 'uo_6h', 'uo_12h', 'creat_low_past_48hr'],
    'respiratory': ['mech_vent', 'vent_ind', 'fio2', 'pafi', 'safi', 'spo2', 'resp'],
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    'chemistry': ['crea', 'bun', 'na', 'k', 'glu', 'bicar', 'lact'],
}

MODULE_PREVIEW_COLUMN_PRIORITY = {
    'renal': ['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'rrt', 'uo_6h', 'uo_12h', 'creat_low_past_48hr'],
    'respiratory': ['mech_vent', 'vent_ind', 'fio2', 'pafi', 'safi', 'spo2', 'resp'],
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    'chemistry': ['crea', 'bun', 'na', 'k', 'glu', 'bicar', 'lact'],
}

PREVIEW_TIME_COLUMNS = [
    'charttime', 'time', 'starttime', 'start', 'endtime', 'itemtime',
    'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat',
    'givenat', 'enteredentryat', 'intakeoutputoffset', 'observationoffset',
    'nursingchartoffset', 'labresultoffset', 'respchartoffset'
]

# ============ 临床阈值线（用于时序图表默认标注） ============
# Optional ``source`` field documents the clinical guideline behind each
# threshold so readers can see provenance (2026-05 Phase D polish).
CLINICAL_THRESHOLDS = {
    'hr':   {'lines': [60, 100], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Bradycardia', 'Tachycardia'], 'unit': 'bpm', 'source': 'AHA adult HR norms; SCCM 2021'},
    'map':  {'lines': [65], 'colors': ['#ef4444'], 'labels': ['Hypotension'], 'unit': 'mmHg', 'source': 'Surviving Sepsis Campaign 2021'},
    'sbp':  {'lines': [90, 140], 'colors': ['#ef4444', '#f59e0b'], 'labels': ['Hypotension', 'Hypertension'], 'unit': 'mmHg', 'source': 'SOFA 1996 (CV); ACC/AHA 2017 (HTN)'},
    'spo2': {'lines': [94], 'colors': ['#ef4444'], 'labels': ['Hypoxemia'], 'unit': '%', 'source': 'BTS emergency oxygen 2017'},
    'temp': {'lines': [36, 38], 'colors': ['#3b82f6', '#ef4444'], 'labels': ['Hypothermia', 'Fever'], 'unit': '°C', 'source': 'Sepsis-3 / SIRS 1992'},
    'resp': {'lines': [12, 20], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Bradypnea', 'Tachypnea'], 'unit': '/min', 'source': 'qSOFA / SIRS criteria'},
    'lact': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Elevated'], 'unit': 'mmol/L', 'source': 'Sepsis-3 / Surviving Sepsis Campaign'},
    'crea': {'lines': [1.2], 'colors': ['#f59e0b'], 'labels': ['Elevated'], 'unit': 'mg/dL', 'source': 'KDIGO 2012 AKI'},
    'ph':   {'lines': [7.35, 7.45], 'colors': ['#ef4444', '#ef4444'], 'labels': ['Acidosis', 'Alkalosis'], 'unit': '', 'source': 'standard arterial pH range'},
    'glu':  {'lines': [70, 180], 'colors': ['#ef4444', '#f59e0b'], 'labels': ['Hypoglycemia', 'Hyperglycemia'], 'unit': 'mg/dL', 'source': 'ADA in-hospital glycemic targets 2024'},
    'k':    {'lines': [3.5, 5.0], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Hypokalemia', 'Hyperkalemia'], 'unit': 'mEq/L', 'source': 'standard reference range'},
    'na':   {'lines': [135, 145], 'colors': ['#f59e0b', '#f59e0b'], 'labels': ['Hyponatremia', 'Hypernatremia'], 'unit': 'mEq/L', 'source': 'standard reference range'},
    'anion_gap': {'lines': [8, 16], 'colors': ['#3b82f6', '#ef4444'], 'labels': ['Low AG', 'High AG (metabolic acidosis)'], 'unit': 'mEq/L', 'source': 'standard chemistry'},
    'pulse_pressure': {'lines': [25, 60], 'colors': ['#ef4444', '#f59e0b'], 'labels': ['Narrow PP (shock)', 'Wide PP'], 'unit': 'mmHg', 'source': 'hemodynamic textbook'},
    'cvp':  {'lines': [12], 'colors': ['#ef4444'], 'labels': ['Venous congestion (↑AKI risk)'], 'unit': 'mmHg', 'source': 'venous congestion / CVP-AKI literature'},
    'plt':  {'lines': [150], 'colors': ['#ef4444'], 'labels': ['Thrombocytopenia'], 'unit': '×10³/µL', 'source': 'SOFA coag component'},
    'hgb':  {'lines': [7], 'colors': ['#ef4444'], 'labels': ['Severe Anemia'], 'unit': 'g/dL', 'source': 'WHO anemia; TRICC transfusion'},
    'inr_pt': {'lines': [1.5], 'colors': ['#f59e0b'], 'labels': ['Coagulopathy'], 'unit': '', 'source': 'clinical coagulopathy threshold'},
    'pafi': {'lines': [300, 200, 100], 'colors': ['#f59e0b', '#ef4444', '#7f1d1d'], 'labels': ['Mild ARDS', 'Moderate ARDS', 'Severe ARDS'], 'unit': 'mmHg', 'source': 'Berlin Definition 2012'},
    'bili': {'lines': [1.2], 'colors': ['#f59e0b'], 'labels': ['Elevated'], 'unit': 'mg/dL', 'source': 'SOFA liver component'},
    'sofa': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Organ Dysfunction'], 'unit': 'points', 'source': 'Sepsis-3 (Singer 2016)'},
    'sofa2': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Organ Dysfunction'], 'unit': 'points', 'source': 'Sepsis-3 / SOFA-2 update'},
    'gcs':  {'lines': [8], 'colors': ['#ef4444'], 'labels': ['Severe Impairment'], 'unit': 'points', 'source': 'Teasdale 1974; severe TBI ≤8'},
    'qsofa': {'lines': [2], 'colors': ['#ef4444'], 'labels': ['Positive qSOFA'], 'unit': 'points', 'source': 'Sepsis-3 (Seymour 2016)'},
}

# 临床概念分道映射
CLINICAL_LANES = {
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'cvp', 'temp', 'spo2', 'resp'],
    'labs': ['lact', 'crea', 'bili', 'plt', 'hgb', 'wbc', 'inr_pt', 'glu', 'k', 'na', 'alb', 'crp', 'tnt', 'ph', 'po2', 'pco2'],
    'interventions': ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'fio2', 'peep', 'ins', 'abx', 'cort', 'rrt'],
    'scores': ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'mews', 'news', 'pafi', 'safi'],
}

# 跨库概念可用性（用于 harmonization badge）
CONCEPT_DB_COVERAGE = {
    'hr': 6, 'map': 6, 'sbp': 6, 'dbp': 6, 'resp': 6, 'spo2': 6, 'temp': 6,
    'glu': 6, 'crea': 6, 'bili': 6, 'plt': 6, 'hgb': 6, 'wbc': 6, 'na': 6, 'k': 6,
    'anion_gap': 6, 'pulse_pressure': 6,
    'age': 6, 'sex': 6, 'weight': 6, 'height': 6, 'death': 6, 'los_icu': 6,
    'sofa': 6, 'sofa2': 6, 'gcs': 6,
    'lact': 5, 'alb': 5, 'crp': 5, 'fio2': 5, 'po2': 5, 'pco2': 5, 'ph': 5,
    'pafi': 5, 'safi': 5, 'urine': 5,
    'peep': 4, 'tidal_vol': 4, 'ins': 4,
    'mech_vent': 3, 'vent_ind': 3, 'ecmo': 2, 'rrt': 4,
    'vent_mode': 4, 'vent_breath_seq': 4,  # miiv/mimic/aumc/hirid (eICU 332 stays, SIC none)
    'driving_pres_controlled': 4,  # plateau+mode overlap: miiv/mimic/aumc/hirid
    'furosemide': 6,
    'propofol': 6, 'midazolam': 6, 'dexmedetomidine': 5,
    'fentanyl': 6, 'morphine': 6, 'heparin': 6,
    'mannitol': 5, 'amiodarone': 6, 'milrinone': 6, 'rocuronium': 5,
    # Rate concepts (2026-05-13): HiRID pharma has no propofol reference;
    # SIC removed pending AmountPerMinute unit audit (see
    # audit_reports/sic_amount_per_minute_unit_audit_20260513.md)
    'propofol_rate': 4,
    # MIIV+MIMIC only for now; other DBs need non-kg mass-rate callback (TODO)
    'fentanyl_rate': 5,
    'midazolam_rate': 5,
    # Batch 2 (2026-05-13; HiRID additions audited 2026-05-27)
    'lorazepam': 6,
    'ketamine': 6,
    'vecuronium': 5,
    'cisatracurium': 4,
    'nitroglycerin': 6,
    # Batch 3 (2026-05-13; round-3 additions audited 2026-05-27)
    'pantoprazole': 5,
    'vancomycin': 6,
    'meropenem': 6,
    'calcium_iv': 6,
    # Batch 4 (2026-05-13; HiRID additions audited 2026-05-27)
    'potassium_iv': 6,
    'magnesium_iv': 6,
    'albumin_iv': 5,
    'packed_rbc': 6,
    # Batch 5 (2026-05-13; round-3 additions audited 2026-05-27)
    'bicarbonate': 6,
    'dextrose50': 5,
    'ffp': 5,
    'platelets': 6,
    # Batch 6 (2026-05-13; prescriptions/HiRID/AUMC additions audited 2026-05-27)
    'levetiracetam': 6,
    'dexamethasone': 6,
    'octreotide': 5,
    'neostigmine': 4,
    # Batch 7 (2026-05-14; HiRID/eICU additions audited 2026-05-27)
    'phenytoin': 5,
    'labetalol': 6,
    'esmolol': 6,
    'diltiazem': 5,
    'nicardipine': 4,
    # Batch 8 (2026-05-14; prescriptions/HiRID additions audited 2026-05-27)
    'warfarin': 6,
    'apixaban': 1,
    'enoxaparin': 6,
    'aspirin': 6,
    'insulin': 5,  # MIIV+MIMIC+eICU+AUMC+HiRID
    # Fluid balance (2026-05-14)
    'total_input_ml': 4,  # MIIV + MIMIC-III MV + AUMC
    'fluid_balance': 4,
    'fluid_balance_cumulative': 4,
    # Renal rolling-window UO rates (2026-05-13): derived from urine + weight
    'uo_6h': 6,
    'uo_12h': 6,
    'uo_24h': 6,
}

SUPPORTED_DB_KEYS = ('miiv', 'mimic', 'eicu', 'aumc', 'hirid', 'sic')


# 🔧 ADD (2026-02-05): 支持时序分析的模块（排除静态数据模块）
# 静态数据模块（demographics, outcome）的值不是连续变化的，不适合时序分析
TIME_SERIES_COMPATIBLE_MODULES = {
    'sofa2_score',      # SOFA评分随时间变化
    'sofa1_score',
    'sepsis3_sofa2',    # Sepsis状态随时间变化
    'sepsis3_sofa1',
    'sepsis_shared',
    'vitals',           # 生命体征（心率、血压等）
    'respiratory',      # 呼吸系统
    'ventilator',       # 呼吸机参数
    'blood_gas',        # 血气分析
    'chemistry',        # 生化检验
    'hematology',       # 血液学
    'vasopressors',     # 血管活性药物
    'medications',      # 药物
    'renal',            # 肾脏与尿量
    'neurological',     # 神经系统（GCS等）
    'circulatory',      # 循环系统
    'other_scores',     # 其他评分
    # 排除: 'demographics' - 静态数据（年龄、性别、身高、体重等）
    # 排除: 'outcome' - 静态数据（死亡、住院时长等）
}

SCREENSHOT_TIMESERIES_PRIORITY = [
    'hr', 'map', 'sbp', 'spo2', 'temp', 'resp',
    'crea', 'plt', 'wbc', 'lact', 'sofa2', 'sofa',
]

SCREENSHOT_QUALITY_PRIORITY = [
    'crea', 'lact', 'hr', 'map', 'sbp', 'temp', 'wbc', 'hgb', 'plt', 'bili', 'sofa2', 'sofa',
]

QUALITY_DEMOGRAPHIC_STATIC = {
    'death', 'los_icu', 'los_hosp', 'age', 'weight', 'height', 'sex', 'bmi'
}

QUALITY_EVENT_TIME_SERIES = {
    'circ_failure', 'circ_event',
    'sep3_sofa2', 'sep3_sofa1', 'sepsis_sofa2',
    'susp_inf', 'infection_icd', 'samp',
    'rrt', 'rrt_criteria',
    'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt',
    'mech_vent', 'vent_ind', 'vent_start', 'vent_end',
    'ecmo', 'ecmo_indication', 'mech_circ_support',
    'abx', 'cort',
    'vaso_ind',
}

QUALITY_STATIC_BOOLEAN_EVENTS = {
    'ecmo', 'ecmo_indication', 'mech_circ_support',
    'cort', 'abx', 'vaso_ind',
}

QUALITY_TIME_CANDIDATES = [
    'time', 'charttime', 'datetime', 'measuredat', 'measuredat_minutes',
    'observationoffset', 'Offset', 'starttime', 'endtime', 'givenat', 'timestamp',
]

QUALITY_EXCLUDE_COLUMNS = {
    'stay_id', 'hadm_id', 'icustay_id', 'time', 'index',
    'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
    'patientunitstayid', 'admissionid', 'patientid', 'CaseID',
}

PRIMARY_VALUE_COLUMN_HINTS = {
    'abp': ['map', 'sbp', 'dbp'],
    'bp': ['map', 'sbp', 'dbp'],
    'fio2': ['fio2'],
    'sofa': ['sofa'],
    'sofa2': ['sofa2'],
}

PHYSIOLOGIC_RANGES = {
    'hr': (20.0, 250.0),
    'resp': (4.0, 80.0),
    'sbp': (40.0, 300.0),
    'dbp': (20.0, 200.0),
    'pulse_pressure': (0.0, 200.0),
    'map': (30.0, 220.0),
    'temp': (25.0, 45.0),
    'spo2': (0.0, 100.0),
    'o2sat': (0.0, 100.0),
    'fio2': (0.0, 100.0),
    'ph': (6.8, 7.8),
    'po2': (20.0, 600.0),
    'pco2': (10.0, 150.0),
    'pafi': (20.0, 800.0),
    'safi': (20.0, 800.0),
    'glu': (20.0, 1500.0),
    'crea': (0.1, 20.0),
    'creat': (0.1, 20.0),
    'lact': (0.0, 30.0),
    'plt': (0.0, 2000.0),
    'wbc': (0.0, 500.0),
    'hgb': (0.0, 25.0),
    'inr_pt': (0.5, 20.0),
    'na': (100.0, 200.0),
    'anion_gap': (-10.0, 50.0),
    'k': (1.0, 10.0),
    'bili': (0.0, 80.0),
    'alb': (0.5, 8.0),
    'bun': (1.0, 250.0),
}
