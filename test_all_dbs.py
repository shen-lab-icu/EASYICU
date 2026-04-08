"""全 6 库测试 — 验证 _subprocess_load_and_export_module 在所有数据库上的正确性和性能。

每库 5000 名患者 × 核心模块，重点关注:
1. 每个库是否能正确导出
2. 各模块耗时分布（识别瓶颈）
3. 主进程 RSS 是否零增长
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

DB_CONFIGS = {
    'miiv':  {'path': '/home/zhuhb/icudb/mimiciv/3.1/',  'id_col': 'stay_id',             'id_table': 'icustays.parquet'},
    'eicu':  {'path': '/home/zhuhb/icudb/eicu/2.0.1/',   'id_col': 'patientunitstayid',   'id_table': 'patient.parquet'},
    'aumc':  {'path': '/home/zhuhb/icudb/aumc/1.0.2/',   'id_col': 'admissionid',         'id_table': 'admissions.parquet'},
    'hirid': {'path': '/home/zhuhb/icudb/hirid/1.1.1/',  'id_col': 'patientid',           'id_table': 'general.parquet'},
    'mimic': {'path': '/home/zhuhb/icudb/mimiciii/1.4/',  'id_col': 'icustay_id',          'id_table': 'icustays.parquet'},
    'sic':   {'path': '/home/zhuhb/icudb/sic/',           'id_col': 'CaseID',              'id_table': 'cases.parquet'},
}

# 核心模块（覆盖快中慢）+ 一个重量级 SOFA
CORE_MODULES = {
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    'chemistry': ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri'],
    'demographics': ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
    'outcome': ['death', 'los_icu', 'los_hosp'],
    'sofa1_score': ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
    'vasopressors': ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind', 'other_vaso'],
    'respiratory': ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication', 'adv_resp'],
    'other_scores': ['qsofa', 'sirs', 'mews', 'news'],
}

def main():
    n_patients = 5000
    export_format = 'parquet'
    
    import pandas as pd
    from pathlib import Path
    
    # 延迟导入（触发 streamlit warning 但不启动）
    from easyicu.webapp.app import _subprocess_load_and_export_module
    import multiprocessing as mp
    
    print(f"{'='*80}")
    print(f"全 6 库测试 — N={n_patients} × {len(CORE_MODULES)} 模块")
    print(f"{'='*80}")
    
    all_results = {}
    
    for db_name, db_cfg in DB_CONFIGS.items():
        data_path = db_cfg['path']
        id_col = db_cfg['id_col']
        id_table = db_cfg['id_table']
        
        if not Path(data_path).exists():
            print(f"\n⚠️ {db_name}: 路径不存在 {data_path}")
            continue
        
        # 获取患者 ID
        id_path = Path(data_path) / id_table
        if not id_path.exists():
            # 尝试子目录
            for sub in ['icu', 'hosp', '']:
                test_path = Path(data_path) / sub / id_table if sub else id_path
                if test_path.exists():
                    id_path = test_path
                    break
        
        try:
            icustays = pd.read_parquet(id_path, columns=[id_col])
            all_pids = sorted(icustays[id_col].unique().tolist())
        except Exception as e:
            print(f"\n⚠️ {db_name}: 无法读取 {id_path}: {e}")
            continue
        
        total_available = len(all_pids)
        test_pids = all_pids[:n_patients]
        patient_ids_filter = {id_col: test_pids}
        
        export_dir = tempfile.mkdtemp(prefix=f'easyicu_bench_{db_name}_')
        
        print(f"\n{'─'*60}")
        print(f"📊 {db_name.upper()} — {total_available} available, testing {len(test_pids)}")
        print(f"{'─'*60}")
        
        rss_before_db = get_rss_mb()
        db_start = time.time()
        db_results = {}
        
        for mod_key, mod_concepts in CORE_MODULES.items():
            t0 = time.time()
            rss_before = get_rss_mb()
            
            proc = mp.Process(
                target=_subprocess_load_and_export_module,
                args=(mod_concepts, db_name, data_path,
                      patient_ids_filter, None,
                      export_dir, export_format, mod_key,
                      None, False, '', None, None),
                daemon=True
            )
            proc.start()
            proc.join(timeout=600)  # 10min timeout per module
            
            if proc.is_alive():
                proc.terminate()
                proc.join()
                elapsed = time.time() - t0
                print(f"  {mod_key:20s}: ⏰ TIMEOUT ({elapsed:.0f}s)")
                db_results[mod_key] = {'status': 'timeout', 'time': elapsed}
                continue
            
            elapsed = time.time() - t0
            rss_after = get_rss_mb()
            rss_delta = rss_after - rss_before
            
            manifest_path = os.path.join(export_dir, f'_manifest_{mod_key}.json')
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    meta = json.load(f)
                rows = meta.get('rows', 0)
                n_concepts = len(meta.get('concepts', []))
                n_empty = len(meta.get('empty_concepts', []))
                exported_file = meta.get('exported_file', '')
                file_mb = os.path.getsize(exported_file) / 1024 / 1024 if exported_file and os.path.exists(exported_file) else 0
                
                print(f"  {mod_key:20s}: ✅ {n_concepts:3d} concepts, {rows:>10,} rows, "
                      f"{file_mb:5.1f}MB, {elapsed:6.1f}s, RSS Δ={rss_delta:+.0f}MB")
                db_results[mod_key] = {
                    'status': 'ok', 'time': elapsed, 'rows': rows,
                    'concepts': n_concepts, 'empty': n_empty, 'file_mb': file_mb
                }
                os.unlink(manifest_path)
            elif proc.exitcode != 0:
                print(f"  {mod_key:20s}: ❌ EXIT={proc.exitcode}, {elapsed:6.1f}s")
                db_results[mod_key] = {'status': 'error', 'time': elapsed, 'exitcode': proc.exitcode}
            else:
                print(f"  {mod_key:20s}: ⚠️ no manifest, {elapsed:6.1f}s")
                db_results[mod_key] = {'status': 'empty', 'time': elapsed}
        
        db_elapsed = time.time() - db_start
        rss_after_db = get_rss_mb()
        
        ok_count = sum(1 for v in db_results.values() if v['status'] == 'ok')
        total_rows = sum(v.get('rows', 0) for v in db_results.values())
        total_disk = sum(v.get('file_mb', 0) for v in db_results.values())
        
        print(f"  {'─'*55}")
        print(f"  {db_name.upper()} 合计: {ok_count}/{len(CORE_MODULES)} OK, "
              f"{total_rows:,} rows, {total_disk:.1f}MB disk, "
              f"{db_elapsed:.0f}s ({db_elapsed/60:.1f}min), "
              f"RSS Δ={rss_after_db-rss_before_db:+.0f}MB")
        
        all_results[db_name] = db_results
        shutil.rmtree(export_dir, ignore_errors=True)
    
    # 汇总
    print(f"\n{'='*80}")
    print(f"汇总 — 各模块平均耗时排名（across DBs）")
    print(f"{'='*80}")
    
    mod_times = {}
    for db_name, db_results in all_results.items():
        for mod_key, result in db_results.items():
            if result['status'] == 'ok':
                if mod_key not in mod_times:
                    mod_times[mod_key] = []
                mod_times[mod_key].append((db_name, result['time']))
    
    # 按平均时间排序
    mod_avg = {}
    for mod_key, entries in mod_times.items():
        times = [t for _, t in entries]
        mod_avg[mod_key] = sum(times) / len(times)
    
    for rank, (mod_key, avg_time) in enumerate(sorted(mod_avg.items(), key=lambda x: -x[1]), 1):
        entries = mod_times[mod_key]
        details = ', '.join(f"{db}={t:.0f}s" for db, t in sorted(entries, key=lambda x: -x[1]))
        print(f"  {rank}. {mod_key:20s}: avg={avg_time:6.1f}s | {details}")
    
    print(f"\n主进程最终 RSS: {get_rss_mb():.0f} MB")

if __name__ == '__main__':
    main()
