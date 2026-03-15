#!/usr/bin/env python3
"""Profile callers of key pandas functions."""
import sys, os, cProfile, pstats, io, gc, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
logging.basicConfig(level=logging.WARNING)
for name in ['pyricu', 'pyricu.datasource', 'pyricu.parallel_config']:
    logging.getLogger(name).setLevel(logging.WARNING)

from pyricu.api import load_concepts, clear_global_loader

def run():
    clear_global_loader()
    gc.collect()
    return load_concepts(
        ['hr','map','sbp','dbp','temp','resp','spo2','po2','pco2','fio2','o2sat','pafi','safi','bili','plt'],
        database='eicu',
        data_path='/home/zhuhb/icudb/eicu/2.0.1',
        max_patients=10000,
        ricu_compatible=True,
        verbose=False,
    )

pr = cProfile.Profile()
pr.enable()
result = run()
pr.disable()

import logging
logging.basicConfig(level=logging.WARNING)

import pyricu.ts_utils as _ts
_orig_change_interval = _ts.change_interval

def _tracked_change_interval(table, *args, **kwargs):
    _pa = getattr(table, '_pre_aggregated', False)
    _name = getattr(table, 'value_column', '?')
    _rows = len(table.data) if hasattr(table, 'data') else '?'
    print(f"  change_interval: {_name} rows={_rows} _pre_aggregated={_pa}")
    return _orig_change_interval(table, *args, **kwargs)

_ts.change_interval = _tracked_change_interval

# Also hook _resolve to check wide_table_batch_results
from pyricu.api import load_concepts, clear_global_loader

def run():
    clear_global_loader()
    gc.collect()
    return load_concepts(
        ['hr','map','sbp','dbp','temp','resp','spo2','po2','pco2','fio2','o2sat','pafi','safi','bili','plt'],
        database='eicu',
        data_path='/home/zhuhb/icudb/eicu/2.0.1',
        max_patients=2000,
        ricu_compatible=True,
        verbose=True,  # Enable verbose to see batch loading messages
    )

result = run()
print(f"\nResult: {len(result)} rows")
