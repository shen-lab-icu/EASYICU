#!/usr/bin/env python3
"""Profile pyricu concept loading to find bottlenecks."""
import sys, time, cProfile, pstats, io, os, argparse, logging, gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
logging.basicConfig(level=logging.WARNING)
for name in ['pyricu', 'pyricu.datasource', 'pyricu.parallel_config']:
    logging.getLogger(name).setLevel(logging.WARNING)

from pyricu.api import load_concepts, clear_global_loader

DB_PATHS = {
    'miiv': '/home/zhuhb/icudb/mimiciv/3.1',
    'eicu': '/home/zhuhb/icudb/eicu/2.0.1',
    'aumc': '/home/zhuhb/icudb/aumc/1.0.2',
    'hirid': '/home/zhuhb/icudb/hirid/1.1.1',
    'mimic': '/home/zhuhb/icudb/mimiciii/1.4',
    'sic': '/home/zhuhb/icudb/sicdb/1.0.6',
}

def run_one_db(db_name, n_patients, concepts):
    clear_global_loader()
    gc.collect()
    t0 = time.time()
    result = load_concepts(
        concepts,
        database=db_name,
        data_path=DB_PATHS[db_name],
        max_patients=n_patients,
        ricu_compatible=True,
        verbose=False,
    )
    elapsed = time.time() - t0
    rows = len(result) if result is not None else 0
    return elapsed, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='miiv', help='Database name or "all"')
    parser.add_argument('-n', type=int, default=1000, help='Number of patients')
    parser.add_argument('--profile', action='store_true', help='Enable cProfile')
    parser.add_argument('--concepts', default='hr,map,sbp,dbp,temp,resp,spo2,po2,pco2,fio2,o2sat,pafi,safi,bili,plt')
    args = parser.parse_args()

    concepts = [c.strip() for c in args.concepts.split(',')]
    
    dbs = list(DB_PATHS.keys()) if args.db == 'all' else [args.db]
    
    total_time = 0
    for db_name in dbs:
        if db_name not in DB_PATHS:
            print(f"Unknown db: {db_name}")
            continue
        print(f"\n=== {db_name} | {args.n} patients | {len(concepts)} concepts ===")

        if args.profile:
            pr = cProfile.Profile()
            pr.enable()
            elapsed, rows = run_one_db(db_name, args.n, concepts)
            pr.disable()

            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(80)
            print(s.getvalue())

            s2 = io.StringIO()
            ps2 = pstats.Stats(pr, stream=s2)
            ps2.sort_stats('tottime')
            ps2.print_stats(80)
            print("\n=== Sorted by tottime ===")
            print(s2.getvalue())
        else:
            elapsed, rows = run_one_db(db_name, args.n, concepts)

        total_time += elapsed
        print(f"Result: {rows} rows in {elapsed:.1f}s")
    
    if len(dbs) > 1:
        print(f"\nTOTAL: {total_time:.1f}s")


if __name__ == '__main__':
    main()
