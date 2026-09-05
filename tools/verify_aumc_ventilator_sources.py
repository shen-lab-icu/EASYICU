"""Verify AUMC ventilator against raw sources; persist aggregate counts only.

Run after extraction, not concurrently with a module benchmark. DuckDB is
limited to one thread / 512 MB and uses an isolated temporary spill directory.
"""
import argparse
import json
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('aggregate receipt must be fresh')
    with tempfile.TemporaryDirectory(prefix='easyicu_ventilator_oracle_') as spill:
        with duckdb.connect() as con:
            con.execute("SET memory_limit='512MB'")
            con.execute('SET threads=1')
            con.execute('SET temp_directory=?', [spill])
            verify(args, output, con)


def verify(args, output, con):
    con.read_parquet(args.before).create_view('old')
    con.read_parquet(args.after).create_view('new')
    raw = Path(args.raw)
    con.read_parquet(str(raw / 'admissions.parquet')).create_view('admissions')
    con.read_parquet(str(raw / 'listitems/*.parquet')).create_view('modes_raw')
    con.read_parquet(str(raw / 'numericitems/*.parquet')).create_view('numeric_raw')
    mapping_path = Path(__file__).resolve().parents[1] / 'src/easyicu/data/vent_mode_map.json'
    mapping = json.loads(mapping_path.read_text())['aumc']['map']
    con.register('mapping', pd.DataFrame([
        {'native': key, 'control': value['control'], 'seq': value['seq']}
        for key, value in mapping.items()
    ]))
    # Choose one original record BEFORE deriving either axis. The admission
    # offset is necessary for repeated ICU admissions with a nonzero origin.
    con.execute('''CREATE TEMP TABLE mode_oracle AS
        SELECT r.admissionid stay_id,
          floor((r.measuredat-a.admittedat)/3600000.) charttime,
          m.control, m.seq
        FROM modes_raw r JOIN admissions a USING(admissionid)
        JOIN mapping m ON trim(r.value)=m.native
        WHERE r.itemid IN (12290,12347,9534,6685,12376)
          AND r.measuredat BETWEEN a.admittedat-86400000. AND a.dischargedat+86400000.
        QUALIFY row_number() OVER (PARTITION BY stay_id,charttime
          ORDER BY r.measuredat,m.native,cast(r.itemid AS VARCHAR))=1''')
    mode_stats = con.execute('''SELECT count(*),
        count(*) FILTER (WHERE o.stay_id IS NULL),
        count(*) FILTER (WHERE n.vent_mode IS DISTINCT FROM o.control
          OR n.vent_breath_seq IS DISTINCT FROM o.seq)
        FROM new n LEFT JOIN mode_oracle o USING(stay_id,charttime)
        WHERE n.vent_mode IS NOT NULL OR n.vent_breath_seq IS NOT NULL''').fetchone()
    # Independently pool raw mL and converted L sources and apply the existing
    # 0..2000 bounds BEFORE aggregation. Do not call a production loader.
    con.execute('''CREATE TEMP TABLE tidal_oracle AS
        WITH normalized AS (
          SELECT r.admissionid stay_id,
            floor((r.measuredat-a.admittedat)/3600000.) charttime,
            CASE WHEN r.itemid IN (8871,9669)
              AND regexp_matches(coalesce(cast(r.unit AS VARCHAR),''),'(?i)l')
              THEN r.value*1000. ELSE r.value END AS normalized_value
          FROM numeric_raw r JOIN admissions a USING(admissionid)
          WHERE r.itemid IN (12275,12277,16243,8872,12373,12358,12360,8871,9669)
            AND r.measuredat BETWEEN a.admittedat-86400000. AND a.dischargedat+86400000.
        ) SELECT stay_id,charttime,median(normalized_value) tidal_vol
          FROM normalized WHERE normalized_value BETWEEN 0 AND 2000 GROUP BY stay_id,charttime''')
    tidal_stats = con.execute('''SELECT count(*),
        count(*) FILTER (WHERE o.stay_id IS NULL),
        count(*) FILTER (WHERE n.tidal_vol IS DISTINCT FROM o.tidal_vol)
        FROM new n LEFT JOIN tidal_oracle o USING(stay_id,charttime)
        WHERE n.tidal_vol IS NOT NULL''').fetchone()
    missing_mode_outputs = con.execute('''SELECT count(*) FROM mode_oracle o
        LEFT JOIN new n USING(stay_id,charttime)
        WHERE n.vent_mode IS NULL OR n.vent_breath_seq IS NULL''').fetchone()[0]
    missing_tidal_outputs = con.execute('''SELECT count(*) FROM tidal_oracle o
        LEFT JOIN new n USING(stay_id,charttime) WHERE n.tidal_vol IS NULL''').fetchone()[0]
    tidal_delta = con.execute('''SELECT max(abs(n.tidal_vol-o.tidal_vol)),
        count(*) FILTER (WHERE abs(n.tidal_vol-o.tidal_vol)>0.001),
        count(*) FILTER (WHERE abs(n.tidal_vol-o.tidal_vol)>1),
        count(*) FILTER (WHERE n.tidal_vol IS DISTINCT FROM cast(cast(o.tidal_vol AS FLOAT) AS DOUBLE))
        FROM new n JOIN tidal_oracle o USING(stay_id,charttime)''').fetchone()
    columns = [name for name in pq.read_schema(args.after).names
               if name not in {'stay_id','charttime'}]
    differences = {name: con.execute(f'''SELECT count(*) FROM old o FULL JOIN new n
        USING(stay_id,charttime) WHERE o."{name}" IS DISTINCT FROM n."{name}"''').fetchone()[0]
        for name in columns}
    key_counts = con.execute('''SELECT
        (SELECT count(*) FROM old), (SELECT count(*) FROM new),
        (SELECT count(*) FROM (SELECT stay_id,charttime FROM old EXCEPT SELECT stay_id,charttime FROM new)),
        (SELECT count(*) FROM (SELECT stay_id,charttime FROM new EXCEPT SELECT stay_id,charttime FROM old)),
        (SELECT count(*)-count(DISTINCT(stay_id,charttime)) FROM new)''').fetchone()
    unchanged = {k:v for k,v in differences.items() if k not in {
        'vent_mode','vent_breath_seq','driving_pres_controlled','tidal_vol'}}
    report = {
        'before':args.before, 'after':args.after,
        'rows_before':key_counts[0], 'rows_after':key_counts[1],
        'removed_keys':key_counts[2], 'added_keys':key_counts[3], 'duplicate_keys':key_counts[4],
        'schema_equal':pq.read_schema(args.before).equals(pq.read_schema(args.after),check_metadata=False),
        'changed_values':differences,
        'mode_nonnull_keys':mode_stats[0], 'mode_missing_source_keys':mode_stats[1],
        'mode_source_mismatches':mode_stats[2],
        'tidal_nonnull_keys':tidal_stats[0], 'tidal_missing_source_keys':tidal_stats[1],
        'tidal_source_mismatches':tidal_stats[2],
        'tidal_max_absolute_difference_ml':tidal_delta[0],
        'tidal_differences_above_0_001_ml':tidal_delta[1],
        'tidal_differences_above_1_ml':tidal_delta[2],
        'tidal_float32_source_mismatches':tidal_delta[3],
        'missing_mode_outputs':missing_mode_outputs,
        'missing_tidal_outputs':missing_tidal_outputs,
        'unaffected_columns_equal':not any(unchanged.values()),
    }
    report['passed'] = (
        report['schema_equal'] and key_counts[4]==0 and not any(unchanged.values())
        # The hourly loader stores this signal in float32 before native-v2
        # widens it to double. Compare to that exact representable value, not
        # an ad hoc clinical tolerance; retain double-oracle discrepancies.
        and mode_stats[1:]==(0,0) and tidal_stats[1]==0 and tidal_delta[3]==0
        and missing_mode_outputs==0 and missing_tidal_outputs==0
    )
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
