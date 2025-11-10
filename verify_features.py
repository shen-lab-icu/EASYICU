#!/usr/bin/env python3
"""
特征提取验证脚本

对比原始表数据和pyricu提取的特征，确保数据完全对应
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, 'src')

from pyricu import load_concepts
from pyricu.fst_reader import read_fst


def verify_miiv_features(data_path: str, patient_id: int):
    """验证MIMIC-IV数据库的特征提取"""
    print("=" * 80)
    print(f"🔬 MIIV 特征验证: 患者 {patient_id}")
    print("=" * 80)
    
    data_path_obj = Path(data_path)
    
    # MIIV数据可能是fst或parquet格式
    def read_table(table_name):
        """读取表，支持fst和parquet格式"""
        # 先尝试parquet
        parquet_file = data_path_obj / f'{table_name}.parquet'
        if parquet_file.exists():
            return pd.read_parquet(parquet_file)
        
        # 再尝试fst
        fst_file = data_path_obj / f'{table_name}.fst'
        if fst_file.exists():
            return read_fst(fst_file)
        
        # 检查分区目录
        partition_dir = data_path_obj / table_name
        if partition_dir.exists() and partition_dir.is_dir():
            frames = []
            for file in sorted(partition_dir.glob('*.fst')):
                frames.append(read_fst(file))
            if frames:
                return pd.concat(frames, ignore_index=True)
            for file in sorted(partition_dir.glob('*.parquet')):
                frames.append(pd.read_parquet(file))
            if frames:
                return pd.concat(frames, ignore_index=True)
        
        return None
    
    # 1. 获取患者的subject_id
    icustays = read_table('icustays')
    if icustays is None:
        print("⚠️ 无法读取icustays数据")
        return
    patient_info = icustays[icustays['stay_id'] == patient_id].iloc[0]
    subject_id = patient_info['subject_id']
    
    print(f"\n患者信息:")
    print(f"  stay_id: {patient_id}")
    print(f"  subject_id: {subject_id}")
    print(f"  intime: {patient_info['intime']}")
    print(f"  outtime: {patient_info['outtime']}")
    
    # 2. 读取原始chartevents数据
    print(f"\n{'='*80}")
    print("📊 步骤1: 读取原始chartevents数据")
    print("="*80)
    
    chartevents = read_table('chartevents')
    if chartevents is None:
        print("⚠️ 无法读取chartevents数据")
        chart_df = pd.DataFrame()
    else:
        chart_df = chartevents[chartevents['stay_id'] == patient_id]
    print(f"总记录数: {len(chart_df)}")
    print(f"时间范围: {chart_df['charttime'].min()} ~ {chart_df['charttime'].max()}")
    
    # 关键itemid
    hr_itemids = [220045]  # 心率
    sbp_itemids = [220050, 220179]  # 收缩压 (动脉和无创)
    temp_itemids = [223761, 223762]  # 体温
    
    print(f"\n关键生命体征itemid:")
    for itemid in hr_itemids:
        data = chart_df[chart_df['itemid'] == itemid]
        if len(data) > 0:
            print(f"  HR ({itemid}): {len(data)} 条, 值范围 {data['valuenum'].min():.1f}-{data['valuenum'].max():.1f}")
            print(f"    样本: {data[['charttime', 'valuenum']].head(3).to_dict('records')}")
    
    for itemid in sbp_itemids:
        data = chart_df[chart_df['itemid'] == itemid]
        if len(data) > 0:
            print(f"  SBP ({itemid}): {len(data)} 条, 值范围 {data['valuenum'].min():.1f}-{data['valuenum'].max():.1f}")
            print(f"    样本: {data[['charttime', 'valuenum']].head(3).to_dict('records')}")
    
    # 3. 读取原始labevents数据
    print(f"\n{'='*80}")
    print("📊 步骤2: 读取原始labevents数据")
    print("="*80)
    
    labevents = read_table('labevents')
    if labevents is None:
        print("⚠️ 无法读取labevents数据")
        lab_df = pd.DataFrame()
    else:
        lab_df = labevents[labevents['subject_id'] == subject_id]
    print(f"总记录数: {len(lab_df)}")
    
    # SOFA相关实验室指标
    bili_itemid = 50885  # Bilirubin
    crea_itemid = 50912  # Creatinine
    plt_itemid = 51265   # Platelet
    
    print(f"\nSOFA实验室指标:")
    for itemid, name in [(bili_itemid, 'Bilirubin'), (crea_itemid, 'Creatinine'), (plt_itemid, 'Platelet')]:
        data = lab_df[lab_df['itemid'] == itemid]
        if len(data) > 0:
            print(f"  {name} ({itemid}): {len(data)} 条, 值范围 {data['valuenum'].min():.2f}-{data['valuenum'].max():.2f}")
            print(f"    样本: {data[['charttime', 'valuenum']].head(3).to_dict('records')}")
    
    # 4. 使用pyricu提取特征
    print(f"\n{'='*80}")
    print("📊 步骤3: 使用pyricu提取特征")
    print("="*80)
    
    # 提取生命体征
    vitals = load_concepts(['hr', 'sbp', 'temp'], database='miiv', data_path=data_path, patient_ids=[patient_id], verbose=False)
    print(f"\n提取的生命体征: {len(vitals)} 行")
    print(f"列名: {vitals.columns.tolist()}")
    print(f"前5行:")
    print(vitals.head())
    
    # 提取实验室指标
    labs = load_concepts(['bili', 'crea', 'plt'], database='miiv', data_path=data_path, patient_ids=[patient_id], verbose=False)
    print(f"\n提取的实验室指标: {len(labs)} 行")
    print(f"列名: {labs.columns.tolist()}")
    print(f"前5行:")
    print(labs.head())
    
    # 提取SOFA评分和组件
    print(f"\n提取SOFA评分:")
    sofa_df = load_concepts(['sofa'], database='miiv', data_path=data_path, patient_ids=[patient_id], verbose=False)
    print(f"  SOFA总分: {len(sofa_df)} 行")
    print(f"  列名: {sofa_df.columns.tolist()}")
    print(f"  前5行:")
    print(sofa_df.head())
    
    # 5. 数据对比验证
    print(f"\n{'='*80}")
    print("✅ 步骤4: 数据对比验证")
    print("="*80)
    
    # 验证心率
    hr_raw_count = len(chart_df[chart_df['itemid'].isin(hr_itemids)])
    hr_raw_notnull = chart_df[chart_df['itemid'].isin(hr_itemids)]['valuenum'].notna().sum()
    hr_extracted = vitals['hr'].dropna() if 'hr' in vitals.columns else pd.Series()
    print(f"\n心率 (HR):")
    print(f"  原始记录数: {hr_raw_count}, 非空值: {hr_raw_notnull}")
    print(f"  提取非空值: {len(hr_extracted)}")
    if len(hr_extracted) > 0:
        print(f"  提取值范围: {hr_extracted.min():.1f}-{hr_extracted.max():.1f}")
        print(f"  样本: {hr_extracted.head(3).tolist()}")
    
    # 验证收缩压
    sbp_raw_count = len(chart_df[chart_df['itemid'].isin(sbp_itemids)])
    sbp_raw_notnull = chart_df[chart_df['itemid'].isin(sbp_itemids)]['valuenum'].notna().sum()
    sbp_extracted = vitals['sbp'].dropna() if 'sbp' in vitals.columns else pd.Series()
    print(f"\n收缩压 (SBP):")
    print(f"  原始记录数: {sbp_raw_count}, 非空值: {sbp_raw_notnull}")
    print(f"  提取非空值: {len(sbp_extracted)}")
    if len(sbp_extracted) > 0:
        print(f"  提取值范围: {sbp_extracted.min():.1f}-{sbp_extracted.max():.1f}")
    
    # 验证体温 - 检查空值原因
    temp_itemids = [223761, 223762]
    temp_raw = chart_df[chart_df['itemid'].isin(temp_itemids)]
    temp_raw_notnull = temp_raw['valuenum'].notna().sum()
    temp_extracted = vitals['temp'].dropna() if 'temp' in vitals.columns else pd.Series()
    print(f"\n体温 (Temperature):")
    print(f"  原始记录数: {len(temp_raw)}, 非空值: {temp_raw_notnull}")
    print(f"  提取非空值: {len(temp_extracted)}")
    if len(temp_raw) > 0:
        print(f"  原始数据样本 (前5条):")
        print(temp_raw[['charttime', 'itemid', 'value', 'valuenum', 'valueuom']].head())
    if len(temp_extracted) > 0:
        print(f"  提取值范围: {temp_extracted.min():.1f}-{temp_extracted.max():.1f}")
    
    # 验证Creatinine
    crea_raw = lab_df[lab_df['itemid'] == crea_itemid]
    crea_raw_notnull = crea_raw['valuenum'].notna().sum()
    crea_extracted = labs['crea'].dropna() if 'crea' in labs.columns else pd.Series()
    print(f"\nCreatinine:")
    print(f"  原始记录数: {len(crea_raw)}, 非空值: {crea_raw_notnull}")
    print(f"  提取非空值: {len(crea_extracted)}")
    if len(crea_raw) > 0:
        print(f"  原始值范围: {crea_raw['valuenum'].min():.2f}-{crea_raw['valuenum'].max():.2f}")
    if len(crea_extracted) > 0:
        print(f"  提取值范围: {crea_extracted.min():.2f}-{crea_extracted.max():.2f}")
    
    # 验证Platelet
    plt_raw = lab_df[lab_df['itemid'] == plt_itemid]
    plt_raw_notnull = plt_raw['valuenum'].notna().sum()
    plt_extracted = labs['plt'].dropna() if 'plt' in labs.columns else pd.Series()
    print(f"\nPlatelet:")
    print(f"  原始记录数: {len(plt_raw)}, 非空值: {plt_raw_notnull}")
    print(f"  提取非空值: {len(plt_extracted)}")
    if len(plt_raw) > 0:
        print(f"  原始值范围: {plt_raw['valuenum'].min():.0f}-{plt_raw['valuenum'].max():.0f}")
    if len(plt_extracted) > 0:
        print(f"  提取值范围: {plt_extracted.min():.0f}-{plt_extracted.max():.0f}")
    
    # 6. SOFA/SOFA2 组件详细验证
    print(f"\n{'='*80}")
    print("🔍 步骤5: SOFA/SOFA2 组件详细验证")
    print("="*80)
    
    # 提取SOFA-2评分
    try:
        from pyricu import load_sofa2
        sofa2_df = load_sofa2(database='miiv', data_path=data_path, patient_ids=[patient_id], 
                              interval='1h', win_length='24h', keep_components=False, verbose=False)
        print(f"\nSOFA-2评分: {len(sofa2_df)} 行")
        if len(sofa2_df) > 0:
            print(f"  列名: {sofa2_df.columns.tolist()}")
            print(f"  前5行:")
            print(sofa2_df.head())
            print(f"  SOFA-2平均分: {sofa2_df['sofa2'].mean():.2f}")
            
            # 对比SOFA和SOFA2的差异
            if len(sofa_df) > 0 and 'sofa' in sofa_df.columns:
                print(f"\n  SOFA vs SOFA-2 对比:")
                print(f"    SOFA平均分: {sofa_df['sofa'].mean():.2f}")
                print(f"    SOFA-2平均分: {sofa2_df['sofa2'].mean():.2f}")
    except Exception as e:
        print(f"⚠️  SOFA-2加载失败: {e}")
    
    # 7. SOFA2新增特征验证
    print(f"\n{'='*80}")
    print("🔍 步骤6: SOFA2 新增特征验证")
    print("="*80)
    
    # 7.1 RRT (肾脏替代治疗) - 用于肾脏评分
    print(f"\n【肾脏替代治疗 RRT】")
    try:
        rrt_df = load_concepts(['rrt'], database='miiv', data_path=data_path, patient_ids=[patient_id], verbose=False)
        print(f"  提取RRT数据: {len(rrt_df)} 行")
        if len(rrt_df) > 0:
            rrt_positive = rrt_df['rrt'].notna().sum()
            print(f"    RRT阳性记录: {rrt_positive}")
            print(f"    样本数据:")
            print(rrt_df[rrt_df['rrt'].notna()].head())
        
        # 检查原始procedureevents中的RRT记录
        procedureevents = read_table('procedureevents')
        if procedureevents is not None:
            patient_proc = procedureevents[procedureevents['stay_id'] == patient_id]
            # MIIV中RRT相关的itemid: 225802, 225803, 225805等
            rrt_itemids = [225802, 225803, 225805, 224270]
            rrt_raw = patient_proc[patient_proc['itemid'].isin(rrt_itemids)]
            print(f"  原始procedureevents中RRT记录: {len(rrt_raw)}")
            if len(rrt_raw) > 0:
                print(f"    RRT itemid分布:")
                print(rrt_raw['itemid'].value_counts())
        else:
            print("  ⚠️ procedureevents表不存在")
    except Exception as e:
        print(f"  ⚠️  RRT验证失败: {e}")
    
    # 7.2 ECMO (体外膜肺氧合) - 用于呼吸评分
    print(f"\n【ECMO 体外膜肺氧合】")
    try:
        ecmo_df = load_concepts(['ecmo'], database='miiv', data_path=data_path, patient_ids=[patient_id], verbose=False)
        print(f"  提取ECMO数据: {len(ecmo_df)} 行")
        if len(ecmo_df) > 0:
            ecmo_positive = ecmo_df['ecmo'].notna().sum()
            print(f"    ECMO阳性记录: {ecmo_positive}")
            if ecmo_positive > 0:
                print(f"    样本数据:")
                print(ecmo_df[ecmo_df['ecmo'].notna()].head())
        
        # 检查原始procedureevents中的ECMO记录
        ecmo_itemids = [228169, 229270]  # ECMO相关itemid
        ecmo_raw = patient_proc[patient_proc['itemid'].isin(ecmo_itemids)]
        print(f"  原始procedureevents中ECMO记录: {len(ecmo_raw)}")
        if len(ecmo_raw) > 0:
            print(f"    样本数据:")
            print(ecmo_raw[['starttime', 'itemid', 'value']].head())
    except Exception as e:
        print(f"  ⚠️  ECMO验证失败: {e}")
    
    # 7.3 高级循环支持 (Advanced Circulatory Support)
    print(f"\n【高级循环支持】")
    try:
        # 检查IABP (主动脉内球囊反搏)
        iabp_itemids = [225908]  # IABP
        iabp_raw = patient_proc[patient_proc['itemid'].isin(iabp_itemids)]
        print(f"  IABP记录: {len(iabp_raw)}")
        if len(iabp_raw) > 0:
            print(f"    样本数据:")
            print(iabp_raw[['starttime', 'itemid', 'value']].head())
        
        # 检查Impella等其他高级循环支持设备
        impella_itemids = [229267]  # Impella
        impella_raw = patient_proc[patient_proc['itemid'].isin(impella_itemids)]
        print(f"  Impella记录: {len(impella_raw)}")
    except Exception as e:
        print(f"  ⚠️  高级循环支持验证失败: {e}")
    
    # 7.4 谵妄 (Delirium) - 用于CNS评分
    print(f"\n【谵妄 Delirium】")
    try:
        # 检查CAM-ICU评估
        # MIIV中谵妄相关itemid: 228334 (CAM-ICU), 227750等
        delirium_itemids = [228334, 227750]
        delirium_raw = chart_df[chart_df['itemid'].isin(delirium_itemids)]
        print(f"  原始chartevents中谵妄评估记录: {len(delirium_raw)}")
        if len(delirium_raw) > 0:
            print(f"    谵妄评估值分布:")
            print(delirium_raw['value'].value_counts())
            print(f"    样本数据:")
            print(delirium_raw[['charttime', 'itemid', 'value']].head())
    except Exception as e:
        print(f"  ⚠️  谵妄验证失败: {e}")
    
    # 7.5 镇静评分 (RASS) - 用于CNS评分
    print(f"\n【镇静评分 RASS】")
    try:
        rass_itemids = [228096]  # RASS
        rass_raw = chart_df[chart_df['itemid'].isin(rass_itemids)]
        print(f"  原始chartevents中RASS记录: {len(rass_raw)}")
        if len(rass_raw) > 0:
            print(f"    RASS值分布:")
            print(rass_raw['valuenum'].value_counts().sort_index())
            print(f"    样本数据:")
            print(rass_raw[['charttime', 'valuenum']].head())
    except Exception as e:
        print(f"  ⚠️  RASS验证失败: {e}")
    
    # 7.6 PaO2/FiO2比值数据 - SOFA2改进的呼吸评分
    print(f"\n【PaO2/FiO2 比值】")
    try:
        pao2_itemids = [50821]  # PaO2 from labevents
        fio2_itemids = [223835, 220277]  # FiO2
        
        pao2_raw = lab_df[lab_df['itemid'].isin(pao2_itemids)]
        fio2_raw = chart_df[chart_df['itemid'].isin(fio2_itemids)]
        
        print(f"  原始PaO2记录: {len(pao2_raw)}")
        if len(pao2_raw) > 0:
            print(f"    PaO2值范围: {pao2_raw['valuenum'].min():.1f} - {pao2_raw['valuenum'].max():.1f}")
            print(f"    样本: {pao2_raw['valuenum'].head(3).tolist()}")
        
        print(f"  原始FiO2记录: {len(fio2_raw)}")
        if len(fio2_raw) > 0:
            print(f"    FiO2值范围: {fio2_raw['valuenum'].min():.1f} - {fio2_raw['valuenum'].max():.1f}")
            print(f"    样本: {fio2_raw['valuenum'].head(3).tolist()}")
    except Exception as e:
        print(f"  ⚠️  PaO2/FiO2验证失败: {e}")
    
    # 7.7 尿量 (Urine Output) - 用于肾脏评分
    print(f"\n【尿量 Urine Output】")
    try:
        outputevents = read_table('outputevents')
        if outputevents is not None:
            patient_output = outputevents[outputevents['stay_id'] == patient_id]
            # 尿量相关itemid
            urine_itemids = [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558]
            urine_raw = patient_output[patient_output['itemid'].isin(urine_itemids)]
            
            print(f"  原始outputevents中尿量记录: {len(urine_raw)}")
            if len(urine_raw) > 0:
                total_urine = urine_raw['value'].sum()
                print(f"    总尿量: {total_urine:.1f} mL")
                print(f"    尿量itemid分布:")
                print(urine_raw['itemid'].value_counts())
                print(f"    样本数据:")
                print(urine_raw[['charttime', 'itemid', 'value']].head())
        else:
            print("  ⚠️ outputevents表不存在")
    except Exception as e:
        print(f"  ⚠️  尿量验证失败: {e}")
    
    # 检查GCS数据（用于CNS评分）
    gcs_itemids = [223900, 223901, 220739]  # GCS-Verbal, GCS-Motor, GCS-Eyes
    gcs_data = chart_df[chart_df['itemid'].isin(gcs_itemids)]
    print(f"\nGCS数据 (用于CNS评分):")
    print(f"  原始记录数: {len(gcs_data)}, 非空值: {gcs_data['valuenum'].notna().sum()}")
    if len(gcs_data) > 0:
        for itemid in gcs_itemids:
            data = gcs_data[gcs_data['itemid'] == itemid]
            if len(data) > 0:
                print(f"    itemid {itemid}: {len(data)} 条, 样本值: {data['valuenum'].dropna().head(3).tolist()}")
    
    # 检查机械通气数据（用于呼吸评分）
    vent_itemids = [225792, 225794]  # 有创、无创通气
    vent_data = chart_df[chart_df['itemid'].isin(vent_itemids)]
    print(f"\n机械通气数据 (用于呼吸评分):")
    print(f"  原始记录数: {len(vent_data)}")
    if len(vent_data) > 0:
        print(f"  样本数据:")
        print(vent_data[['charttime', 'itemid', 'value']].head())
    
    # 检查血管升压药数据
    inputevents = read_table('inputevents')
    if inputevents is not None:
        patient_input = inputevents[inputevents['stay_id'] == patient_id]
        vaso_itemids = [221906, 221289, 221662, 221653]  # 常见血管升压药
        vaso_data = patient_input[patient_input['itemid'].isin(vaso_itemids)]
        print(f"\n血管升压药数据 (用于循环评分):")
        print(f"  原始记录数: {len(vaso_data)}")
        if len(vaso_data) > 0:
            print(f"  样本数据:")
            print(vaso_data[['starttime', 'itemid', 'rate', 'rateuom']].head())
    else:
        print(f"\n血管升压药数据:")
        print("  ⚠️ inputevents表不存在")
    
    print(f"\n{'='*80}")
    print("✅ 验证完成")
    print("="*80)


def verify_eicu_features(data_path: str, patient_id: int):
    """验证eICU数据库的特征提取"""
    print("=" * 80)
    print(f"🔬 eICU 特征验证: 患者 {patient_id}")
    print("=" * 80)
    
    data_path_obj = Path(data_path)
    
    # eICU可能是parquet格式
    def read_table(table_name):
        """读取表，支持fst和parquet格式"""
        fst_file = data_path_obj / f'{table_name}.fst'
        parquet_file = data_path_obj / f'{table_name}.parquet'
        
        if fst_file.exists():
            return read_fst(fst_file)
        elif parquet_file.exists():
            return pd.read_parquet(parquet_file)
        else:
            return None
    
    # 1. 读取原始vitalPeriodic数据
    print(f"\n{'='*80}")
    print("📊 步骤1: 读取原始vitalPeriodic数据")
    print("="*80)
    
    vital_df = read_table('vitalperiodic')
    if vital_df is not None:
        patient_data = vital_df[vital_df['patientunitstayid'] == patient_id]
        
        print(f"总记录数: {len(patient_data)}")
        if len(patient_data) > 0:
            print(f"列名: {patient_data.columns.tolist()}")
            print(f"前5行:")
            print(patient_data.head())
            
            if 'temperature' in patient_data.columns:
                temp_data = patient_data['temperature'].dropna()
                if len(temp_data) > 0:
                    print(f"\n体温数据: {len(temp_data)} 条, 值范围 {temp_data.min():.1f}-{temp_data.max():.1f}")
                    print(f"  样本: {temp_data.head(3).tolist()}")
            
            if 'heartrate' in patient_data.columns:
                hr_data = patient_data['heartrate'].dropna()
                if len(hr_data) > 0:
                    print(f"心率数据: {len(hr_data)} 条, 值范围 {hr_data.min():.0f}-{hr_data.max():.0f}")
                    print(f"  样本: {hr_data.head(3).tolist()}")
    else:
        print("⚠️  vitalPeriodic表不存在")
    
    # 2. 读取原始lab数据
    print(f"\n{'='*80}")
    print("📊 步骤2: 读取原始lab数据")
    print("="*80)
    
    lab_df = read_table('lab')
    if lab_df is not None:
        patient_data = lab_df[lab_df['patientunitstayid'] == patient_id]
        
        print(f"总记录数: {len(patient_data)}")
        if len(patient_data) > 0:
            print(f"唯一实验室项目: {patient_data['labname'].nunique()}")
            print(f"实验室项目列表:")
            for labname in patient_data['labname'].unique()[:10]:
                count = len(patient_data[patient_data['labname'] == labname])
                sample_val = patient_data[patient_data['labname'] == labname]['labresult'].iloc[0]
                print(f"  {labname}: {count} 条, 样本值={sample_val}")
    else:
        print("⚠️  lab表不存在")
    
    # 3. 使用pyricu提取特征
    print(f"\n{'='*80}")
    print("📊 步骤3: 使用pyricu提取特征")
    print("="*80)
    
    vitals = load_concepts(['hr', 'temp'], database='eicu', data_path=data_path, patient_ids=[patient_id], verbose=False)
    print(f"\n提取的生命体征: {len(vitals)} 行")
    if len(vitals) > 0:
        print(f"列名: {vitals.columns.tolist()}")
        print(f"前5行:")
        print(vitals.head())
        
        # 提取SOFA评分
        print(f"\n提取SOFA评分:")
        sofa_df = load_concepts(['sofa'], database='eicu', data_path=data_path, patient_ids=[patient_id], verbose=False)
        print(f"  SOFA总分: {len(sofa_df)} 行")
        if len(sofa_df) > 0:
            print(f"  列名: {sofa_df.columns.tolist()}")
            print(f"  SOFA平均分: {sofa_df['sofa'].mean():.2f}")
        
        # 4. 数据对比验证
        print(f"\n{'='*80}")
        print("✅ 步骤4: 数据对比验证")
        print("="*80)
        
        if vital_df is not None and 'heartrate' in vital_df.columns:
            hr_raw = vital_df[vital_df['patientunitstayid'] == patient_id]['heartrate'].dropna()
            hr_raw_count = len(vital_df[vital_df['patientunitstayid'] == patient_id])
            hr_raw_notnull = len(hr_raw)
            hr_extracted = vitals['hr'].dropna() if 'hr' in vitals.columns else pd.Series()
            print(f"\n心率 (HR):")
            print(f"  原始记录数: {hr_raw_count}, 非空值: {hr_raw_notnull}")
            print(f"  提取非空值: {len(hr_extracted)}")
            if len(hr_raw) > 0:
                print(f"  原始值范围: {hr_raw.min():.0f}-{hr_raw.max():.0f}")
            if len(hr_extracted) > 0:
                print(f"  提取值范围: {hr_extracted.min():.0f}-{hr_extracted.max():.0f}")
        
        if vital_df is not None and 'temperature' in vital_df.columns:
            temp_raw_all = vital_df[vital_df['patientunitstayid'] == patient_id]
            temp_raw = temp_raw_all['temperature'].dropna()
            temp_raw_count = len(temp_raw_all)
            temp_raw_notnull = len(temp_raw)
            temp_extracted = vitals['temp'].dropna() if 'temp' in vitals.columns else pd.Series()
            print(f"\n体温 (Temperature):")
            print(f"  原始记录数: {temp_raw_count}, 非空值: {temp_raw_notnull}")
            print(f"  提取非空值: {len(temp_extracted)}")
            if temp_raw_notnull == 0:
                print(f"  ⚠️  原始数据中temperature列全为空值")
                # 检查是否在其他列
                print(f"  检查原始数据样本:")
                print(temp_raw_all[['observationoffset', 'temperature', 'heartrate']].head())
            if len(temp_raw) > 0:
                print(f"  原始值范围: {temp_raw.min():.1f}-{temp_raw.max():.1f}")
            if len(temp_extracted) > 0:
                print(f"  提取值范围: {temp_extracted.min():.1f}-{temp_extracted.max():.1f}")
        
        # 检查实验室数据用于SOFA
        if lab_df is not None:
            patient_labs = lab_df[lab_df['patientunitstayid'] == patient_id]
            
            # 查找creatinine
            crea_data = patient_labs[patient_labs['labname'].str.contains('creatinine', case=False, na=False)]
            print(f"\nCreatinine (用于肾脏评分):")
            print(f"  原始记录数: {len(crea_data)}")
            if len(crea_data) > 0:
                print(f"  样本数据:")
                print(crea_data[['labresultoffset', 'labname', 'labresult']].head())
            
            # 查找bilirubin
            bili_data = patient_labs[patient_labs['labname'].str.contains('bilirubin', case=False, na=False)]
            print(f"\nBilirubin (用于肝脏评分):")
            print(f"  原始记录数: {len(bili_data)}")
            if len(bili_data) > 0:
                print(f"  样本数据:")
                print(bili_data[['labresultoffset', 'labname', 'labresult']].head())
            
            # 查找platelet
            plt_data = patient_labs[patient_labs['labname'].str.contains('platelet', case=False, na=False)]
            print(f"\nPlatelet (用于凝血评分):")
            print(f"  原始记录数: {len(plt_data)}")
            if len(plt_data) > 0:
                print(f"  样本数据:")
                print(plt_data[['labresultoffset', 'labname', 'labresult']].head())
    
        # 5. SOFA2 新增特征验证
        print(f"\n{'='*80}")
        print("🔍 步骤5: eICU SOFA2 新增特征验证")
        print("="*80)
        
        # 提取SOFA-2评分
        try:
            from pyricu import load_sofa2
            sofa2_df = load_sofa2(database='eicu', data_path=data_path, patient_ids=[patient_id], 
                                  interval='1h', win_length='24h', keep_components=False, verbose=False)
            print(f"\nSOFA-2评分: {len(sofa2_df)} 行")
            if len(sofa2_df) > 0:
                print(f"  SOFA-2平均分: {sofa2_df['sofa2'].mean():.2f}")
        except Exception as e:
            print(f"⚠️  SOFA-2加载失败: {e}")
        
        # 检查RRT数据
        print(f"\n【RRT 肾脏替代治疗】")
        treatment_df = read_table('treatment')
        if treatment_df is not None:
            patient_treatment = treatment_df[treatment_df['patientunitstayid'] == patient_id]
            # eICU中RRT相关的treatment
            rrt_treatments = patient_treatment[patient_treatment['treatmentstring'].str.contains('dialysis|CRRT|hemofiltration', case=False, na=False)]
            print(f"  RRT治疗记录: {len(rrt_treatments)}")
            if len(rrt_treatments) > 0:
                print(f"    治疗类型:")
                print(rrt_treatments['treatmentstring'].value_counts())
        
        # 检查呼吸机数据
        print(f"\n【机械通气】")
        resp_care_df = read_table('respiratorycare')
        if resp_care_df is not None:
            patient_resp = resp_care_df[resp_care_df['patientunitstayid'] == patient_id]
            print(f"  呼吸机记录: {len(patient_resp)}")
            if len(patient_resp) > 0:
                print(f"    样本数据:")
                print(patient_resp[['respCareStatusoffset', 'airwaytype', 'airwaysize']].head())
        
        # 检查药物数据（血管升压药）
        print(f"\n【血管升压药】")
        infusion_df = read_table('infusiondrug')
        if infusion_df is not None:
            patient_infusion = infusion_df[infusion_df['patientunitstayid'] == patient_id]
            vaso_drugs = patient_infusion[patient_infusion['drugname'].str.contains('Norepinephrine|Epinephrine|Dopamine|Vasopressin', case=False, na=False)]
            print(f"  血管升压药记录: {len(vaso_drugs)}")
            if len(vaso_drugs) > 0:
                print(f"    药物分布:")
                print(vaso_drugs['drugname'].value_counts())
                print(f"    样本数据:")
                print(vaso_drugs[['drugstartoffset', 'drugname', 'drugrate']].head())
    else:
        print("无数据")
    
    print(f"\n{'='*80}")
    print("✅ 验证完成")
    print("="*80)


def verify_aumc_features(data_path: str, patient_id: int):
    """验证AUMC数据库的特征提取"""
    print("=" * 80)
    print(f"🔬 AUMC 特征验证: 患者 {patient_id}")
    print("=" * 80)
    
    data_path_obj = Path(data_path)
    
    # AUMC可能是parquet格式
    def read_table(table_name):
        """读取表，支持fst和parquet格式"""
        fst_file = data_path_obj / f'{table_name}.fst'
        parquet_file = data_path_obj / f'{table_name}.parquet'
        
        if fst_file.exists():
            return read_fst(fst_file)
        elif parquet_file.exists():
            return pd.read_parquet(parquet_file)
        else:
            return None
    
    # 1. 读取原始numericitems数据
    print(f"\n{'='*80}")
    print("📊 步骤1: 读取原始numericitems数据")
    print("="*80)
    
    numeric_df = read_table('numericitems')
    if numeric_df is not None:
        patient_data = numeric_df[numeric_df['admissionid'] == patient_id]
        
        print(f"总记录数: {len(patient_data)}")
        if len(patient_data) > 0:
            print(f"列名: {patient_data.columns.tolist()}")
            print(f"唯一itemid数: {patient_data['itemid'].nunique()}")
            
            # AUMC关键itemid (需要查阅AUMC字典)
            # 这里展示前10个最常见的itemid
            itemid_counts = patient_data['itemid'].value_counts().head(10)
            print(f"\n前10个常见itemid:")
            for itemid, count in itemid_counts.items():
                sample_val = patient_data[patient_data['itemid'] == itemid]['value'].iloc[0]
                print(f"  itemid {itemid}: {count} 条, 样本值={sample_val}")
    else:
        print("⚠️  numericitems表不存在")
    
    # 2. 读取原始listitems数据
    print(f"\n{'='*80}")
    print("📊 步骤2: 读取原始listitems数据")
    print("="*80)
    
    list_df = read_table('listitems')
    if list_df is not None:
        patient_data = list_df[list_df['admissionid'] == patient_id]
        
        print(f"总记录数: {len(patient_data)}")
        if len(patient_data) > 0:
            print(f"唯一itemid数: {patient_data['itemid'].nunique()}")
            
            itemid_counts = patient_data['itemid'].value_counts().head(5)
            print(f"\n前5个常见itemid:")
            for itemid, count in itemid_counts.items():
                sample_val = patient_data[patient_data['itemid'] == itemid]['value'].iloc[0]
                print(f"  itemid {itemid}: {count} 条, 样本值={sample_val}")
    else:
        print("⚠️  listitems表不存在")
    
    # 3. 使用pyricu提取特征
    print(f"\n{'='*80}")
    print("📊 步骤3: 使用pyricu提取特征")
    print("="*80)
    
    try:
        vitals = load_concepts(['hr', 'temp'], database='aumc', data_path=data_path, patient_ids=[patient_id], verbose=False)
        print(f"\n提取的生命体征: {len(vitals)} 行")
        if len(vitals) > 0:
            print(f"列名: {vitals.columns.tolist()}")
            print(f"前5行:")
            print(vitals.head())
            
            # 4. 数据对比验证
            print(f"\n{'='*80}")
            print("✅ 步骤4: 数据对比验证")
            print("="*80)
            
            hr_extracted = vitals['hr'].dropna() if 'hr' in vitals.columns else pd.Series()
            temp_extracted = vitals['temp'].dropna() if 'temp' in vitals.columns else pd.Series()
            
            print(f"\n心率 (HR):")
            print(f"  提取非空值: {len(hr_extracted)}")
            if len(hr_extracted) > 0:
                print(f"  提取值范围: {hr_extracted.min():.0f}-{hr_extracted.max():.0f}")
                print(f"  样本: {hr_extracted.head(3).tolist()}")
            
            print(f"\n体温 (Temperature):")
            print(f"  提取非空值: {len(temp_extracted)}")
            if len(temp_extracted) > 0:
                print(f"  提取值范围: {temp_extracted.min():.1f}-{temp_extracted.max():.1f}")
                print(f"  样本: {temp_extracted.head(3).tolist()}")
        else:
            print("无数据")
    except Exception as e:
        print(f"⚠️  特征提取失败: {e}")
    
    # 5. SOFA2 新增特征验证
    print(f"\n{'='*80}")
    print("🔍 步骤5: AUMC SOFA2 新增特征验证")
    print("="*80)
    
    # 提取SOFA-2评分
    try:
        from pyricu import load_sofa2
        sofa2_df = load_sofa2(database='aumc', data_path=data_path, patient_ids=[patient_id], 
                              interval='1h', win_length='24h', keep_components=False, verbose=False)
        print(f"\nSOFA-2评分: {len(sofa2_df)} 行")
        if len(sofa2_df) > 0:
            print(f"  SOFA-2平均分: {sofa2_df['sofa2'].mean():.2f}")
    except Exception as e:
        print(f"⚠️  SOFA-2加载失败: {e}")
    
    # 检查RRT数据 - AUMC在procedureorderitems表中
    print(f"\n【RRT 肾脏替代治疗】")
    proc_df = read_table('procedureorderitems')
    if proc_df is not None:
        patient_proc = proc_df[proc_df['admissionid'] == patient_id]
        # AUMC中RRT相关的procedure
        rrt_procs = patient_proc[patient_proc['item'].str.contains('dialyse|hemofiltratie|CVVH', case=False, na=False)]
        print(f"  RRT操作记录: {len(rrt_procs)}")
        if len(rrt_procs) > 0:
            print(f"    操作类型:")
            print(rrt_procs['item'].value_counts())
    
    # 检查listitems中的特殊治疗
    print(f"\n【高级循环支持 & ECMO】")
    list_df = read_table('listitems')
    if list_df is not None:
        patient_list = list_df[list_df['admissionid'] == patient_id]
        # 查找ECMO和其他循环支持
        support_items = patient_list[patient_list['item'].str.contains('ECMO|IABP|Impella', case=False, na=False)]
        print(f"  循环支持记录: {len(support_items)}")
        if len(support_items) > 0:
            print(f"    设备类型:")
            print(support_items['item'].value_counts())
    
    # 检查药物数据（血管升压药）
    print(f"\n【血管升压药】")
    drug_df = read_table('drugitems')
    if drug_df is not None:
        patient_drugs = drug_df[drug_df['admissionid'] == patient_id]
        vaso_drugs = patient_drugs[patient_drugs['item'].str.contains('Noradrenaline|Adrenaline|Dopamine', case=False, na=False)]
        print(f"  血管升压药记录: {len(vaso_drugs)}")
        if len(vaso_drugs) > 0:
            print(f"    药物分布:")
            print(vaso_drugs['item'].value_counts())
            print(f"    样本数据:")
            print(vaso_drugs[['start', 'item', 'duration']].head())
    
    print(f"\n{'='*80}")
    print("✅ 验证完成")
    print("="*80)


if __name__ == '__main__':
    # 测试MIIV - 使用包含SOFA2特征的患者
    print("\n" + "🔬 " * 40)
    print("📋 MIIV患者: 30005000 (有RRT+血管加压药+谵妄评估)")
    try:
        verify_miiv_features('test_data_miiv', 30005000)
    except Exception as e:
        print(f"❌ MIIV验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试eICU - 使用包含SOFA2特征的患者
    print("\n" + "🔬 " * 40)
    print("📋 eICU患者: 243334 (有RRT+血管加压药)")
    try:
        verify_eicu_features('test_data_eicu', 243334)
    except Exception as e:
        print(f"❌ eICU验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试AUMC - 使用包含SOFA2特征的患者（包括ECMO）
    print("\n" + "🔬 " * 40)
    print("📋 AUMC患者: 3441 (有RRT+ECMO+血管加压药)")
    try:
        verify_aumc_features('test_data_aumc', 3441)
    except Exception as e:
        print(f"❌ AUMC验证失败: {e}")
        import traceback
        traceback.print_exc()

