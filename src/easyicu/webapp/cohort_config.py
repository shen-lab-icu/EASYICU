"""Cohort filter configuration constants for the EasyICU webapp."""

from __future__ import annotations


DISEASE_COHORT_CONFIG = {
    'sepsis': {
        'label_en': 'Sepsis-3 cohort',
        'label_zh': '脓毒症队列（Sepsis-3）',
        'description_en': 'Use Sepsis-3 labels (`sep3_sofa2` preferred, fallback `sep3_sofa1`) to keep only septic patients.',
        'description_zh': '使用 Sepsis-3 标签（优先 `sep3_sofa2`，回退 `sep3_sofa1`）仅保留脓毒症患者。',
        'required_modules': {'sepsis3_sofa2', 'sepsis3_sofa1', 'sepsis_shared', 'sofa2_score', 'sofa1_score'},
        'concept_priority': ['sep3_sofa2', 'sep3_sofa1'],
    },
    'aki': {
        'label_en': 'AKI cohort (KDIGO)',
        'label_zh': 'AKI 队列（KDIGO）',
        'description_en': 'Use KDIGO-AKI outputs (`aki_stage` preferred, fallback `aki`) to keep AKI-positive patients.',
        'description_zh': '使用 KDIGO-AKI 输出（优先 `aki_stage`，回退 `aki`）仅保留 AKI 患者。',
        'required_modules': {'renal'},
        'concept_priority': ['aki_stage', 'aki'],
    },
    'circ_failure': {
        'label_en': 'Circulatory failure cohort',
        'label_zh': '循环衰竭队列',
        'description_en': 'Use `circ_failure` or `circ_event` to keep patients with circulatory failure evidence.',
        'description_zh': '使用 `circ_failure` 或 `circ_event` 仅保留存在循环衰竭证据的患者。',
        'required_modules': {'circulatory'},
        'concept_priority': ['circ_failure', 'circ_event'],
    },
    'mech_vent': {
        'label_en': 'Mechanical ventilation cohort',
        'label_zh': '机械通气队列',
        'description_en': 'Use `mech_vent` or `vent_ind` to keep ventilated ICU stays.',
        'description_zh': '使用 `mech_vent` 或 `vent_ind` 仅保留机械通气 ICU 住院记录。',
        'required_modules': {'respiratory'},
        'concept_priority': ['mech_vent', 'vent_ind'],
    },
    'rrt': {
        'label_en': 'Renal replacement therapy cohort',
        'label_zh': '肾脏替代治疗队列',
        'description_en': 'Use `rrt` or `rrt_criteria` to keep ICU stays receiving renal replacement therapy.',
        'description_zh': '使用 `rrt` 或 `rrt_criteria` 仅保留接受肾脏替代治疗的 ICU 住院记录。',
        'required_modules': {'renal'},
        'concept_priority': ['rrt', 'rrt_criteria'],
    },
    'ards': {
        'label_en': 'ARDS cohort',
        'label_zh': 'ARDS 队列',
        'description_en': 'ICD-backed ARDS template for databases with diagnosis codes. Use for acute respiratory distress syndrome cohorts.',
        'description_zh': '适用于带诊断编码数据库的 ARDS 模板队列，可用于急性呼吸窘迫综合征研究。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['J80', '51882'],
    },
    'pneumonia': {
        'label_en': 'Pneumonia cohort',
        'label_zh': '肺炎队列',
        'description_en': 'ICD-backed pneumonia template for infectious respiratory cohorts.',
        'description_zh': '适用于呼吸系统感染研究的 ICD 肺炎模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', '481', '482', '483', '485', '486'],
    },
    'heart_failure': {
        'label_en': 'Heart failure cohort',
        'label_zh': '心力衰竭队列',
        'description_en': 'ICD-backed heart-failure template for decompensated heart-failure or cardiogenic cohorts.',
        'description_zh': '适用于失代偿心衰或心源性相关研究的 ICD 心衰模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['I50', '428'],
    },
    'ami': {
        'label_en': 'Acute myocardial infarction cohort',
        'label_zh': '急性心肌梗死队列',
        'description_en': 'ICD-backed AMI template for STEMI / NSTEMI style cohorts.',
        'description_zh': '适用于 STEMI / NSTEMI 等急性心肌梗死研究的 ICD 模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['I21', 'I22', '410'],
    },
    'stroke': {
        'label_en': 'Stroke cohort',
        'label_zh': '卒中队列',
        'description_en': 'ICD-backed stroke template covering ischemic and hemorrhagic stroke codes.',
        'description_zh': '覆盖缺血性与出血性卒中的 ICD 模板队列。',
        'required_modules': set(),
        'concept_priority': [],
        'icd_tokens': ['I60', 'I61', 'I63', 'I64', '430', '431', '434', '436'],
    },
}


SEPSIS_MODE_CONFIG = {
    "auto": {
        "label_en": "Auto by database",
        "label_zh": "按数据库自动选择",
        "desc_en": "Recommended default. eICU uses `ICD + antibiotics`; other databases default to `ABX + sampling`.",
        "desc_zh": "推荐默认值。eICU 使用 `ICD + 抗生素`；其他数据库默认使用 `抗生素 + 采样`。",
    },
    "and": {
        "label_en": "ABX + sampling (strict window)",
        "label_zh": "抗生素 + 采样（严格时间窗）",
        "desc_en": "Classic Sepsis-3 style suspected infection: antibiotics and body-fluid sampling must co-occur within windows.",
        "desc_zh": "经典 Sepsis-3 风格的疑似感染定义：抗生素与体液采样需在时间窗内共同出现。",
    },
    "or": {
        "label_en": "ABX or sampling",
        "label_zh": "抗生素或采样",
        "desc_en": "More permissive suspected infection proxy. Keeps either antibiotics or sampling events.",
        "desc_zh": "更宽松的疑似感染代理定义。只要出现抗生素或采样事件即可。",
    },
    "abx": {
        "label_en": "Antibiotics only",
        "label_zh": "仅抗生素",
        "desc_en": "Antibiotic-only proxy, useful when sampling coverage is sparse.",
        "desc_zh": "仅抗生素代理定义，适用于采样覆盖较差的数据集。",
    },
    "samp": {
        "label_en": "Sampling only",
        "label_zh": "仅采样",
        "desc_en": "Body-fluid sampling only. Useful for exploratory sensitivity analyses.",
        "desc_zh": "仅使用体液采样事件，适合做敏感性分析。",
    },
    "icd_abx": {
        "label_en": "ICD infection + antibiotics",
        "label_zh": "感染 ICD + 抗生素",
        "desc_en": "eICU-oriented fallback: infection ICD identifies patients, antibiotics provide event time.",
        "desc_zh": "偏 eICU 的替代方案：感染 ICD 先定人，再用抗生素时间定时点。",
    },
}


ICD_FILTER_DATABASES = {'miiv', 'mimic', 'eicu'}

