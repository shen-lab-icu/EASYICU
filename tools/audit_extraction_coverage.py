#!/usr/bin/env python3
"""跨库 special-concept 覆盖完整性审计 —— 关闭"数值分布检查"的盲区。

背景 / 为什么需要它
-------------------
数值分布一致性检查(比较各库已有概念的均值/分位数/量纲)对"某库整组概念
根本没被提取(整列缺失/整列空)"是天然隐形的——不存在的列产生不了分布。
2026-07 的一个真实事故: EasyICU load_concepts 的分批分支在 special-concept
重挂之前 return, 导致触发分批的模块(eICU 自动分批 / 或显式 batch_size)
静默丢掉整个 special 组(KDIGO/CIRC/COMORB/OUTCOME/MICRO)。分布检查查不出,
一个"存在性/覆盖率"矩阵一眼就能看出(eICU renal 14 列 vs 其余库 24 列)。

设计要点(自维护, 不硬编"合法名单")
-----------------------------------
每个 special 组的"合法缺失库"直接 **import 各 loader 的排除集**:
  comorbidity._NO_ICD_DATABASES / microbiology._NO_MICRO_DATABASES /
  outcomes._FOLLOWUP_DATABASES
=> loader 改了排除策略, 本审计自动跟随, 不会漂移。

判定
----
对 (db, special_group): 该组概念应整组在 该库的对应模块 parquet 里。
  - db ∈ 该组 loader 排除集      -> LEGITIMATE(合法空, 期望缺)
  - 该库根本没提取该模块          -> SKIP(N/A)
  - 组在                          -> OK
  - 组缺 且 db ∉ 排除集           -> BUG(fail loud, 退出码 1)

用法
----
  python tools/audit_extraction_coverage.py [EXTRACTION_ROOT]
默认 EXTRACTION_ROOT = full6_20260717。退出码 1 = 发现 BUG。
"""
from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pyarrow.parquet as pq
from easyicu.scores.comorbidity import _NO_ICD_DATABASES
from easyicu.scores.microbiology import _NO_MICRO_DATABASES
from easyicu.scores.outcomes import _FOLLOWUP_DATABASES

DEFAULT_ROOT = "/Volumes/外置硬盘/easyicu_data/phd_thesis_module_reextract/full6_20260717"

# special 组 -> (所在模块, 组内概念, 合法缺失库集合)
# 合法缺失库来自各 loader 的排除常量(import 而非硬编, 自动跟随 loader 变更)。
FOLLOWUP_MORT = {"mort_28d", "mort_90d", "mort_365d"}
ALL_DBS = {"aumc", "eicu", "hirid", "miiv", "mimic", "sic"}
SPECIAL_GROUPS = {
    "KDIGO": {
        "module": "renal",
        "concepts": ["aki", "aki_stage", "aki_stage_creat", "aki_stage_uo",
                     "aki_stage_rrt", "uo_rt_6hr", "uo_rt_12hr", "uo_rt_24hr",
                     "creat_low_past_48hr", "creat_low_past_7day"],
        "excluded_dbs": set(),  # KDIGO 无 loader 排除 => 6 库都应产出
    },
    "CIRC": {
        "module": "circulatory",
        "concepts": ["circ_failure", "circ_event"],
        "excluded_dbs": set(),  # 无排除 => 6 库都应产出
    },
    "COMORB": {
        "module": "other_scores",
        "concepts": ["charlson", "elixhauser"],
        "excluded_dbs": {d for d in _NO_ICD_DATABASES if d in ALL_DBS},
    },
    "MICRO": {
        "module": "sepsis_shared",
        "concepts": ["culture_positive", "bld_culture_positive"],
        "excluded_dbs": {d for d in _NO_MICRO_DATABASES if d in ALL_DBS},
    },
    "OUTCOME_MORT": {
        "module": "outcome",
        "concepts": ["mort_28d", "mort_90d", "mort_365d"],
        # 只有 _FOLLOWUP_DATABASES 能出随访死亡 => 其余库合法缺
        "excluded_dbs": ALL_DBS - {d for d in _FOLLOWUP_DATABASES if d in ALL_DBS},
    },
}


def module_cols(root, db, module):
    p = os.path.join(root, db, module + ".parquet")
    if not os.path.isfile(p):
        return None  # 该库未提取该模块
    return set(pq.ParquetFile(p).schema.names)


def audit(root):
    dbs = sorted(d for d in ALL_DBS if os.path.isdir(os.path.join(root, d)))
    print(f"审计提取根: {root}")
    print(f"库: {dbs}\n" + "=" * 78)
    bugs = []
    for gname, g in SPECIAL_GROUPS.items():
        mod, concepts, excl = g["module"], g["concepts"], g["excluded_dbs"]
        print(f"\n[{gname}] @ {mod}  概念={concepts}")
        print(f"    合法缺失库(loader排除): {sorted(excl) or '无'}")
        for db in dbs:
            cols = module_cols(root, db, mod)
            if cols is None:
                print(f"    {db:6s} SKIP  (未提取 {mod} 模块)")
                continue
            have = [c for c in concepts if c in cols]
            miss = [c for c in concepts if c not in cols]
            if not miss:
                print(f"    {db:6s} OK    (全组在)")
            elif db in excl:
                print(f"    {db:6s} LEGIT (loader排除, 合法缺 {len(miss)})")
            else:
                verdict = "BUG★整组缺" if not have else "BUG★部分缺"
                print(f"    {db:6s} {verdict}  缺={miss}")
                bugs.append((gname, db, mod, miss))
    print("\n" + "=" * 78)
    if bugs:
        print(f"❌ 发现 {len(bugs)} 处 special 组丢失(非合法缺失):")
        for gname, db, mod, miss in bugs:
            print(f"    {db}/{mod} [{gname}] 缺 {miss}")
        return 1
    print("✅ 所有 special 组覆盖完整(缺失均为 loader 声明的合法缺失)")
    return 0


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    sys.exit(audit(root))
