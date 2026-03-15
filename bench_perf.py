#!/usr/bin/env python3
"""性能基准测试 — 对比不同患者规模的提取速度和内存"""
import sys, time, tracemalloc, logging, os, gc, argparse
sys.path.insert(0, 'src')

logging.basicConfig(level=logging.WARNING)
for name in ['pyricu', 'pyricu.datasource', 'pyricu.parallel_config']:
    logging.getLogger(name).setLevel(logging.WARNING)

from pyricu.api import load_concepts, clear_global_loader

DBS = [
    ('miiv',  '/home/zhuhb/icudb/mimiciv/3.1'),
    ('eicu',  '/home/zhuhb/icudb/eicu/2.0.1'),
    ('aumc',  '/home/zhuhb/icudb/aumc/1.0.2'),
    ('hirid', '/home/zhuhb/icudb/hirid/1.1.1'),
    ('mimic', '/home/zhuhb/icudb/mimiciii/1.4'),
    ('sic',   '/home/zhuhb/icudb/sicdb/1.0.6'),
]

CONCEPTS = ['hr', 'map', 'sbp', 'dbp', 'temp', 'resp', 'spo2',
            'po2', 'pco2', 'fio2', 'o2sat', 'pafi', 'safi',
            'bili', 'plt']

def run_benchmark(n_patients=None):
    label = f"{n_patients} patients" if n_patients else "ALL patients"
    print(f"\nPyRICU 性能基准 — {label} x {len(CONCEPTS)} concepts")
    print(f"{'DB':8s} {'Patients':>9s} {'Time':>8s} {'Peak MB':>9s} {'Rows':>12s}")
    print("=" * 52)
    sys.stdout.flush()

    total_time = 0
    for db_name, db_path in DBS:
        if not os.path.isdir(db_path):
            print(f"{db_name:8s} {'N/A':>9s} (path missing)")
            sys.stdout.flush()
            continue

        clear_global_loader()
        gc.collect()
        tracemalloc.start()
        t0 = time.time()

        try:
            # 使用顶层 api.load_concepts，正确处理 n_patients 采样
            kwargs = dict(
                database=db_name,
                data_path=db_path,
                ricu_compatible=True,
                verbose=False,
            )
            if n_patients is not None:
                kwargs['max_patients'] = n_patients

            r = load_concepts(CONCEPTS, **kwargs)
            elapsed = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            rows = len(r) if r is not None else 0
            n_actual = r.iloc[:, 0].nunique() if rows > 0 else 0

            print(f"{db_name:8s} {n_actual:>9d} {elapsed:>7.1f}s {peak/1048576:>8.0f}MB {rows:>12,d}")
            total_time += elapsed
        except Exception as e:
            elapsed = time.time() - t0
            try: tracemalloc.stop()
            except: pass
            print(f"{db_name:8s} {'ERR':>9s} {elapsed:>7.1f}s  {str(e)[:60]}")
            total_time += elapsed
        sys.stdout.flush()

    print("=" * 52)
    print(f"{'TOTAL':8s} {'':>9s} {total_time:>7.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-patients', type=int, default=None,
                        help='Number of patients (default: all)')
    args = parser.parse_args()
    run_benchmark(args.n_patients)
