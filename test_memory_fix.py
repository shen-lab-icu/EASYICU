"""测试 _subprocess_load_and_export_module 内存使用。

验证: 主进程 RSS 在模块加载后不增长（所有 DataFrame 在子进程中处理）。
"""
import os, sys, time, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def get_rss_mb():
    """获取当前进程 RSS (MB)"""
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
    n_patients = 10000
    export_format = 'parquet'
    id_col = 'stay_id'
    
    # 内联定义测试模块（避免从 webapp.app 导入触发 Streamlit 初始化）
    TEST_MODULES = {
        'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
        'chemistry': ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri'],
        'hematology': ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt', 'ptt', 'rbc', 'rdw', 'wbc'],
        'demographics': ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
        'outcome': ['death', 'los_icu', 'los_hosp'],
        'sofa1_score': ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
        'vasopressors': ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind', 'other_vaso'],
        'respiratory': ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication', 'adv_resp'],
    }
    
    # 获取患者ID
    import pandas as pd
    from pathlib import Path
    icustays = pd.read_parquet(Path(data_path) / 'icustays.parquet', columns=[id_col])
    all_pids = sorted(icustays[id_col].unique().tolist())
    test_pids = all_pids[:n_patients]
    patient_ids_filter = {id_col: test_pids}
    
    print(f"=== 内存修复测试 === database={database}, N={n_patients}")
    rss_baseline = get_rss_mb()
    print(f"Baseline RSS: {rss_baseline:.1f} MB")
    
    # 测试模块列表（包含重量级 SOFA/vasopressors）
    test_modules = ['vitals', 'chemistry', 'hematology', 'demographics', 'outcome',
                    'sofa1_score', 'vasopressors', 'respiratory']
    
    export_dir = tempfile.mkdtemp(prefix='easyicu_test_export_')
    print(f"Export dir: {export_dir}")
    
    # 导入子进程函数（会触发 streamlit import 但不会启动 server）
    from easyicu.webapp.app import _subprocess_load_and_export_module
    import multiprocessing as mp
    
    total_exported = 0
    for mod_key in test_modules:
        mod_concepts = list(TEST_MODULES.get(mod_key, []))
        if not mod_concepts:
            continue
        
        rss_before = get_rss_mb()
        t0 = time.time()
        
        proc = mp.Process(
            target=_subprocess_load_and_export_module,
            args=(mod_concepts, database, data_path,
                  patient_ids_filter, None,  # no batch_size
                  export_dir, export_format, mod_key,
                  None,  # no cohort_exclude_ids
                  False,  # no overwrite
                  '',  # no cohort_suffix
                  None,  # no dep_concepts_to_cache
                  None),  # no deps_cache_dir
            daemon=True
        )
        proc.start()
        proc.join()
        
        elapsed = time.time() - t0
        rss_after = get_rss_mb()
        rss_delta = rss_after - rss_before
        
        # 读取 manifest
        manifest_path = os.path.join(export_dir, f'_manifest_{mod_key}.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                meta = json.load(f)
            rows = meta.get('rows', 0)
            n_concepts = len(meta.get('concepts', []))
            n_empty = len(meta.get('empty_concepts', []))
            exported_file = meta.get('exported_file', '')
            file_size_mb = os.path.getsize(exported_file) / 1024 / 1024 if exported_file and os.path.exists(exported_file) else 0
            total_exported += 1
            print(f"  {mod_key}: {n_concepts} concepts, {rows} rows, "
                  f"file={file_size_mb:.1f}MB, time={elapsed:.1f}s, "
                  f"RSS delta={rss_delta:+.1f}MB (RSS={rss_after:.1f}MB), "
                  f"empty={n_empty}, exit={proc.exitcode}")
            os.unlink(manifest_path)
        else:
            print(f"  {mod_key}: NO MANIFEST (exit={proc.exitcode}), "
                  f"RSS delta={rss_delta:+.1f}MB, time={elapsed:.1f}s")
    
    rss_final = get_rss_mb()
    print(f"\n=== 结果 ===")
    print(f"Exported: {total_exported}/{len(test_modules)} modules")
    print(f"RSS: {rss_baseline:.1f} → {rss_final:.1f} MB (delta={rss_final-rss_baseline:+.1f}MB)")
    print(f"Files: {os.listdir(export_dir)}")
    
    # 验证导出文件内容
    import pandas as pd_verify
    print(f"\n=== 文件验证 ===")
    for f in sorted(os.listdir(export_dir)):
        if f.endswith('.parquet'):
            fpath = os.path.join(export_dir, f)
            df = pd_verify.read_parquet(fpath)
            mod_name = f.split('_')[0]
            concepts = [c for c in df.columns if c not in ('stay_id', 'charttime')]
            id_col_name = 'stay_id' if 'stay_id' in df.columns else df.columns[0]
            n_patients = df[id_col_name].nunique()
            print(f"  {mod_name}: {df.shape[0]:,} rows × {len(df.columns)} cols, "
                  f"{n_patients} patients, concepts={concepts[:5]}")
    
    # 清理
    shutil.rmtree(export_dir, ignore_errors=True)

if __name__ == '__main__':
    main()
