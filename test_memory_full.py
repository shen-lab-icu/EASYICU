"""全量 MIIV 94K 患者测试 — 验证 16GB 兼容性。

模拟 webapp execute_sidebar_export() 的完整流程:
- 19 个模块全部导出
- 子进程隔离（主进程零 DataFrame 操作）
- DuckDB 4 线程

基线：旧方案 661GB RSS; 新方案预期 <1GB RSS
"""
import os, sys, time, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def get_rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0

def main():
    database = 'miiv'
    data_path = '/home/zhuhb/icudb/mimiciv/3.1/'
    export_format = 'parquet'
    id_col = 'stay_id'
    
    ALL_MODULES = {
        'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
        'sofa1_score': ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
        'sofa2_score': ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
        'chemistry': ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri'],
        'hematology': ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt', 'ptt', 'rbc', 'rdw', 'wbc'],
        'blood_gas': ['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2'],
        'respiratory': ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication', 'adv_resp'],
        'vasopressors': ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind', 'other_vaso'],
        'medications': ['abx', 'cort', 'dex', 'ins'],
        'renal': ['urine', 'urine24', 'uo_6h', 'uo_12h', 'uo_24h', 'rrt', 'rrt_criteria', 'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'aki_stage_rrt', 'creat_low_past_48hr', 'creat_low_past_7day', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
        'neurological': ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'tgcs', 'vgcs', 'sedated_gcs', 'motor_response', 'delirium_positive', 'delirium_tx'],
        'demographics': ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
        'other_scores': ['qsofa', 'sirs', 'mews', 'news'],
        'outcome': ['death', 'los_icu', 'los_hosp'],
        'ventilator': ['peep', 'tidal_vol', 'tidal_vol_set', 'pip', 'plateau_pres', 'mean_airway_pres', 'minute_vol', 'vent_rate', 'etco2', 'compliance', 'driving_pres', 'ps'],
        'circulatory': ['mech_circ_support', 'circ_failure', 'circ_event'],
        'sepsis_shared': ['susp_inf', 'infection_icd', 'samp'],
        'sepsis3_sofa1': ['sep3_sofa1'],
        'sepsis3_sofa2': ['sep3_sofa2'],
    }
    
    import pandas as pd
    from pathlib import Path
    icustays = pd.read_parquet(Path(data_path) / 'icustays.parquet', columns=[id_col])
    all_pids = sorted(icustays[id_col].unique().tolist())
    n_patients = len(all_pids)
    patient_ids_filter = {id_col: all_pids}
    
    print(f"=== 全量测试 === database={database}, N={n_patients}")
    rss_baseline = get_rss_mb()
    print(f"Baseline RSS: {rss_baseline:.1f} MB")
    
    export_dir = tempfile.mkdtemp(prefix='easyicu_full_export_')
    print(f"Export dir: {export_dir}")
    
    from easyicu.webapp.app import _subprocess_load_and_export_module
    import multiprocessing as mp
    
    total_exported = 0
    total_rows = 0
    rss_peak = rss_baseline
    t0_all = time.time()
    
    module_order = [
        'vitals', 'demographics', 'outcome',
        'chemistry', 'hematology', 'blood_gas',
        'medications', 'ventilator', 'respiratory',
        'vasopressors', 'renal', 'neurological',
        'other_scores', 'circulatory',
        'sofa1_score', 'sofa2_score',
        'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared',
    ]
    
    for mod_idx, mod_key in enumerate(module_order):
        mod_concepts = list(ALL_MODULES.get(mod_key, []))
        if not mod_concepts:
            continue
        
        rss_before = get_rss_mb()
        t0 = time.time()
        
        proc = mp.Process(
            target=_subprocess_load_and_export_module,
            args=(mod_concepts, database, data_path,
                  patient_ids_filter, None,
                  export_dir, export_format, mod_key,
                  None, False, '', None, None),
            daemon=True
        )
        proc.start()
        
        while proc.is_alive():
            proc.join(timeout=10)
            elapsed = time.time() - t0
            rss_now = get_rss_mb()
            print(f"\r  [{mod_key}] running... {elapsed:.0f}s, RSS={rss_now:.0f}MB", end='', flush=True)
        
        elapsed = time.time() - t0
        rss_after = get_rss_mb()
        rss_delta = rss_after - rss_before
        rss_peak = max(rss_peak, rss_after)
        
        manifest_path = os.path.join(export_dir, f'_manifest_{mod_key}.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                meta = json.load(f)
            rows = meta.get('rows', 0)
            total_rows += rows
            n_concepts = len(meta.get('concepts', []))
            n_empty = len(meta.get('empty_concepts', []))
            exported_file = meta.get('exported_file', '')
            file_mb = os.path.getsize(exported_file) / 1024 / 1024 if exported_file and os.path.exists(exported_file) else 0
            total_exported += 1
            print(f"\r  {mod_key}: {n_concepts} concepts, {rows:,} rows, "
                  f"file={file_mb:.1f}MB, time={elapsed:.1f}s, "
                  f"RSS Δ={rss_delta:+.1f}MB (RSS={rss_after:.0f}MB), "
                  f"empty={n_empty}, exit={proc.exitcode}")
            os.unlink(manifest_path)
        else:
            print(f"\r  {mod_key}: NO MANIFEST (exit={proc.exitcode}), "
                  f"RSS Δ={rss_delta:+.1f}MB, time={elapsed:.1f}s")
    
    total_time = time.time() - t0_all
    rss_final = get_rss_mb()
    
    # 统计磁盘使用
    disk_mb = sum(
        os.path.getsize(os.path.join(export_dir, f))
        for f in os.listdir(export_dir) if not f.startswith('_')
    ) / 1024 / 1024
    
    print(f"\n=== 结果 ===")
    print(f"Exported: {total_exported}/{len(module_order)} modules")
    print(f"Total rows: {total_rows:,}")
    print(f"Disk: {disk_mb:.1f} MB")
    print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"RSS: {rss_baseline:.0f} → {rss_final:.0f} MB (peak={rss_peak:.0f}MB, delta={rss_final-rss_baseline:+.0f}MB)")
    print(f"Files: {len([f for f in os.listdir(export_dir) if not f.startswith('_')])}")
    
    # 16GB 兼容性检查
    if rss_peak < 2000:
        print("✅ 16GB 兼容: 主进程 RSS < 2GB")
    elif rss_peak < 4000:
        print("⚠️ 16GB 可能兼容: 主进程 RSS < 4GB（子进程额外需要 2-4GB）")
    else:
        print(f"❌ 16GB 不兼容: 主进程 RSS = {rss_peak:.0f}MB")
    
    shutil.rmtree(export_dir, ignore_errors=True)

if __name__ == '__main__':
    main()
