#!/usr/bin/env python3
"""Detailed profiling to find exact call sites for dropna/merge_asof."""
import sys, os, time, logging, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
logging.basicConfig(level=logging.WARNING)
for name in ['pyricu', 'pyricu.datasource', 'pyricu.parallel_config']:
    logging.getLogger(name).setLevel(logging.WARNING)

import cProfile
import pstats
from pyricu.api import load_concepts, clear_global_loader

# Patch dropna and merge_asof to count calls with traceback
import pandas as pd
import traceback

_dropna_count = 0
_dropna_callers = {}
_orig_dropna = pd.DataFrame.dropna

def _patched_dropna(self, *args, **kwargs):
    global _dropna_count
    _dropna_count += 1
    # Get caller info (skip this wrapper, get real caller)
    frame = sys._getframe(1)
    caller = f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"
    # Trim to just filename
    fname = os.path.basename(frame.f_code.co_filename)
    key = f"{fname}:{frame.f_lineno}:{frame.f_code.co_name}"
    _dropna_callers[key] = _dropna_callers.get(key, 0) + 1
    return _orig_dropna(self, *args, **kwargs)

_merge_asof_count = 0
_merge_asof_callers = {}
_orig_merge_asof = pd.merge_asof

def _patched_merge_asof(*args, **kwargs):
    global _merge_asof_count
    _merge_asof_count += 1
    frame = sys._getframe(1)
    fname = os.path.basename(frame.f_code.co_filename)
    key = f"{fname}:{frame.f_lineno}:{frame.f_code.co_name}"
    _merge_asof_callers[key] = _merge_asof_callers.get(key, 0) + 1
    return _orig_merge_asof(*args, **kwargs)

_copy_count = 0
_copy_callers = {}
_orig_copy = pd.DataFrame.copy

def _patched_copy(self, *args, **kwargs):
    global _copy_count
    _copy_count += 1
    frame = sys._getframe(1)
    fname = os.path.basename(frame.f_code.co_filename)
    key = f"{fname}:{frame.f_lineno}:{frame.f_code.co_name}"
    _copy_callers[key] = _copy_callers.get(key, 0) + 1
    return _orig_copy(self, *args, **kwargs)

_sort_count = 0
_sort_callers = {}
_orig_sort = pd.DataFrame.sort_values

def _patched_sort(self, *args, **kwargs):
    global _sort_count
    _sort_count += 1
    frame = sys._getframe(1)
    fname = os.path.basename(frame.f_code.co_filename)
    key = f"{fname}:{frame.f_lineno}:{frame.f_code.co_name}"
    _sort_callers[key] = _sort_callers.get(key, 0) + 1
    return _orig_sort(self, *args, **kwargs)

_concat_count = 0
_concat_callers = {}
_orig_concat = pd.concat

def _patched_concat(*args, **kwargs):
    global _concat_count
    _concat_count += 1
    frame = sys._getframe(1)
    fname = os.path.basename(frame.f_code.co_filename)
    key = f"{fname}:{frame.f_lineno}:{frame.f_code.co_name}"
    _concat_callers[key] = _concat_callers.get(key, 0) + 1
    return _orig_concat(*args, **kwargs)

# Apply patches
pd.DataFrame.dropna = _patched_dropna
pd.merge_asof = _patched_merge_asof
pd.DataFrame.copy = _patched_copy
pd.DataFrame.sort_values = _patched_sort
pd.concat = _patched_concat

# Run
clear_global_loader()
gc.collect()
t0 = time.time()
result = load_concepts(
    ['hr', 'map', 'sbp', 'dbp', 'temp', 'resp', 'spo2', 'po2', 'pco2', 'fio2', 'o2sat', 'pafi', 'safi', 'bili', 'plt'],
    database='miiv',
    data_path='/home/zhuhb/icudb/mimiciv/3.1',
    max_patients=2000,
    ricu_compatible=True,
    verbose=False,
)
elapsed = time.time() - t0

print(f"\nResult: {len(result)} rows in {elapsed:.1f}s")

print(f"\n=== dropna: {_dropna_count} total calls ===")
for k, v in sorted(_dropna_callers.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v:6d}  {k}")

print(f"\n=== merge_asof: {_merge_asof_count} total calls ===")
for k, v in sorted(_merge_asof_callers.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v:6d}  {k}")

print(f"\n=== DataFrame.copy: {_copy_count} total calls ===")
for k, v in sorted(_copy_callers.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v:6d}  {k}")

print(f"\n=== sort_values: {_sort_count} total calls ===")
for k, v in sorted(_sort_callers.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v:6d}  {k}")

print(f"\n=== pd.concat: {_concat_count} total calls ===")
for k, v in sorted(_concat_callers.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v:6d}  {k}")
