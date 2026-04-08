#!/usr/bin/env python3
"""真实数据库提取性能基准脚本。

用途：
1. 对 `load_concepts()` 在 6 个数据库上的真实提取速度做对比
2. 通过子进程隔离每个 case，避免上一个 case 的内存残留污染结果
3. 记录 elapsed / peak RSS / rows / throughput，并输出 JSON 报告

示例：
	python demo/benchmark_real_extract.py \
		--db-path miiv=/data/mimiciv_parquet \
		--db-path eicu=/data/eicu_parquet \
		--patients 2000 \
		--cases vitals sofa aki \
		--output output/benchmark_report.json

也支持环境变量：
	EASYICU_MIIV_PATH=/data/mimiciv_parquet
	EASYICU_EICU_PATH=/data/eicu_parquet
	EASYICU_AUMC_PATH=/data/aumc_parquet
	EASYICU_HIRID_PATH=/data/hirid_parquet
	EASYICU_MIMIC_PATH=/data/mimiciii_parquet
	EASYICU_SIC_PATH=/data/sicdb_parquet
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import psutil


SUPPORTED_DATABASES = ("miiv", "eicu", "aumc", "hirid", "mimic", "sic")

CASE_PRESETS: Dict[str, Dict[str, object]] = {
	"vitals": {
		"concepts": ["hr", "map", "sbp", "resp", "spo2"],
		"kwargs": {"interval": "1h", "merge": True},
		"description": "基础生命体征，多表读取 + 常规对齐",
	},
	"sofa": {
		"concepts": ["sofa"],
		"kwargs": {"interval": "6h", "win_length": "24h", "merge": True},
		"description": "SOFA 总分，典型复杂评分链路",
	},
	"aki": {
		"concepts": ["kdigo_aki"],
		"kwargs": {"interval": "6h", "merge": True},
		"description": "KDIGO AKI，典型高成本窗口计算",
	},
	"sepsis": {
		"concepts": ["sep3"],
		"kwargs": {"interval": "6h", "win_length": "24h", "merge": True},
		"description": "Sepsis-3，复合特征链路",
	},
	"sepsis_sofa2": {
		"concepts": ["sep3_sofa2"],
		"kwargs": {"interval": "6h", "win_length": "24h", "merge": True, "use_sofa2": True},
		"description": "Sepsis-3 with SOFA-2，复合特征链路",
	},
}


def _parse_db_path_items(items: Iterable[str]) -> Dict[str, str]:
	mapping: Dict[str, str] = {}
	for item in items:
		if "=" not in item:
			raise ValueError(f"db-path 参数格式错误: {item}，应为 db=/path")
		db, path = item.split("=", 1)
		db = db.strip().lower()
		if db not in SUPPORTED_DATABASES:
			raise ValueError(f"不支持的数据库: {db}")
		mapping[db] = path.strip()
	return mapping


def _discover_env_db_paths() -> Dict[str, str]:
	env_map = {}
	for db in SUPPORTED_DATABASES:
		value = os.environ.get(f"EASYICU_{db.upper()}_PATH")
		if value:
			env_map[db] = value
	return env_map


def _result_rows(result) -> int:
	import pandas as pd

	if isinstance(result, pd.DataFrame):
		return len(result)
	if isinstance(result, dict):
		total = 0
		for value in result.values():
			if isinstance(value, pd.DataFrame):
				total += len(value)
		return total
	return 0


def _run_worker(payload: Dict[str, object]) -> Dict[str, object]:
	from easyicu.api import load_concepts

	concepts = payload["concepts"]
	kwargs = dict(payload["kwargs"])
	for key in ("parallel_workers", "concept_workers", "chunk_size"):
		value = payload.get(key)
		if value is not None:
			kwargs[key] = value
	t0 = time.perf_counter()
	result = load_concepts(
		concepts=concepts,
		database=payload["database"],
		data_path=payload["data_path"],
		max_patients=payload["patients"],
		memory_efficient=payload["memory_efficient"],
		verbose=False,
		**kwargs,
	)
	elapsed = time.perf_counter() - t0
	rows = _result_rows(result)

	return {
		"database": payload["database"],
		"case": payload["case"],
		"patients": payload["patients"],
		"elapsed_s": elapsed,
		"rows": rows,
		"concepts": concepts,
		"memory_efficient": payload["memory_efficient"],
	}


def _spawn_case(payload: Dict[str, object]) -> Dict[str, object]:
	cmd = [sys.executable, __file__, "--worker", json.dumps(payload, ensure_ascii=False)]
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	ps_proc = psutil.Process(proc.pid)
	peak_rss = 0.0

	while proc.poll() is None:
		try:
			peak_rss = max(peak_rss, ps_proc.memory_info().rss / 1024 / 1024)
		except psutil.Error:
			pass
		time.sleep(0.05)

	stdout, stderr = proc.communicate()
	if proc.returncode != 0:
		raise RuntimeError(
			f"case 失败: db={payload['database']} case={payload['case']} rc={proc.returncode}\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
		)

	lines = [line for line in stdout.splitlines() if line.strip()]
	if not lines:
		raise RuntimeError(f"case 未输出结果: db={payload['database']} case={payload['case']}")
	result = json.loads(lines[-1])
	result["peak_rss_mb"] = round(peak_rss, 1)
	result["throughput_rows_per_s"] = round(result["rows"] / result["elapsed_s"], 2) if result["elapsed_s"] > 0 else None
	if stderr.strip():
		result["stderr"] = stderr.strip()
	return result


def _summarize(results: List[Dict[str, object]]) -> Dict[str, object]:
	if not results:
		return {"results": [], "slowest": [], "highest_peak_rss": [], "fastest": []}

	slowest = sorted(results, key=lambda item: item["elapsed_s"], reverse=True)[:5]
	memory_top = sorted(results, key=lambda item: item["peak_rss_mb"], reverse=True)[:5]
	fastest = sorted(results, key=lambda item: item["throughput_rows_per_s"], reverse=True)[:5]

	by_case: Dict[str, List[float]] = {}
	for item in results:
		by_case.setdefault(item["case"], []).append(item["elapsed_s"])

	case_stats = {
		case: {
			"runs": len(values),
			"mean_elapsed_s": round(statistics.mean(values), 3),
			"median_elapsed_s": round(statistics.median(values), 3),
		}
		for case, values in by_case.items()
	}

	return {
		"results": results,
		"slowest": slowest,
		"highest_peak_rss": memory_top,
		"fastest": fastest,
		"case_stats": case_stats,
	}


def _print_console_summary(summary: Dict[str, object]) -> None:
	print("\n=== Benchmark Summary ===")
	print("Top slowest:")
	for item in summary["slowest"]:
		print(f"  - {item['database']:>5} / {item['case']:<7} : {item['elapsed_s']:.3f}s, peak {item['peak_rss_mb']:.1f}MB")

	print("Top highest peak RSS:")
	for item in summary["highest_peak_rss"]:
		print(f"  - {item['database']:>5} / {item['case']:<7} : peak {item['peak_rss_mb']:.1f}MB, {item['elapsed_s']:.3f}s")

	print("Top throughput:")
	for item in summary["fastest"]:
		print(f"  - {item['database']:>5} / {item['case']:<7} : {item['throughput_rows_per_s']:.0f} rows/s")


def main() -> int:
	parser = argparse.ArgumentParser(description="Benchmark easyicu real database extraction")
	parser.add_argument("--db-path", action="append", default=[], help="数据库路径，格式 db=/path")
	parser.add_argument("--patients", type=int, default=2000, help="每个 case 的患者数")
	parser.add_argument("--cases", nargs="+", default=["vitals", "sofa"], choices=sorted(CASE_PRESETS), help="要运行的 case 预设")
	parser.add_argument("--repeats", type=int, default=1, help="每个 case 重复次数")
	parser.add_argument("--memory-efficient", action="store_true", help="强制启用 memory_efficient 落盘模式")
	parser.add_argument("--parallel-workers", type=int, default=None, help="覆盖 parallel_workers")
	parser.add_argument("--concept-workers", type=int, default=None, help="覆盖 concept_workers")
	parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 chunk_size")
	parser.add_argument("--output", type=str, default="", help="输出 JSON 报告路径")
	parser.add_argument("--worker", type=str, default="", help=argparse.SUPPRESS)
	args = parser.parse_args()

	if args.worker:
		payload = json.loads(args.worker)
		print(json.dumps(_run_worker(payload), ensure_ascii=False))
		return 0

	db_paths = _discover_env_db_paths()
	db_paths.update(_parse_db_path_items(args.db_path))
	db_paths = {db: path for db, path in db_paths.items() if Path(path).exists()}
	if not db_paths:
		print("未找到可用数据库路径。请传 --db-path db=/path 或设置 EASYICU_<DB>_PATH 环境变量。", file=sys.stderr)
		return 2

	results: List[Dict[str, object]] = []
	for db, path in db_paths.items():
		for case in args.cases:
			preset = CASE_PRESETS[case]
			for repeat_idx in range(args.repeats):
				payload = {
					"database": db,
					"data_path": path,
					"patients": args.patients,
					"case": case,
					"concepts": preset["concepts"],
					"kwargs": preset["kwargs"],
					"memory_efficient": bool(args.memory_efficient),
					"parallel_workers": args.parallel_workers,
					"concept_workers": args.concept_workers,
					"chunk_size": args.chunk_size,
					"repeat": repeat_idx + 1,
				}
				print(f"[RUN] db={db} case={case} repeat={repeat_idx + 1}/{args.repeats} patients={args.patients}")
				result = _spawn_case(payload)
				results.append(result)
				print(
					f"      elapsed={result['elapsed_s']:.3f}s peak={result['peak_rss_mb']:.1f}MB "
					f"rows={result['rows']} throughput={result['throughput_rows_per_s']:.0f} rows/s"
				)

	summary = _summarize(results)
	_print_console_summary(summary)

	if args.output:
		output_path = Path(args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"\n报告已写入: {output_path}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
