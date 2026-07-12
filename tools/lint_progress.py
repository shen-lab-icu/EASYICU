#!/usr/bin/env python3
"""Lint the `项目进度/` middle-layer handoff pages (CURRENT.md).

Companion to ``lint_main_plan.py``: that one guards the top-level dashboard's
row count; this one guards the middle layer so it does not rot —

- every ``<module>/CURRENT.md`` carries a top ``更新:YYYY-MM-DD`` timestamp and
  is not staler than ``--stale-days`` (warning: it may no longer be the truth);
- no page references the per-session ``scratchpad/`` (a dead link the moment
  another agent picks up — this is the exact gap that bit us on 2026-07-10);
- every page keeps the 7 required sections (🎯 📍 🔨 ✅ ⏭️ ⚠️ 📚);
- the README index lists every module directory that actually has a CURRENT.md.

Run before and after editing any CURRENT.md, like the main-plan lint.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "项目进度"
REQUIRED_SECTIONS = ["🎯", "📍", "🔨", "✅", "⏭️", "⚠️", "📚"]
DATE_RE = re.compile(r"更新[:：]\s*(\d{4})-(\d{2})-(\d{2})")
SCRATCHPAD_RE = re.compile(r"scratchpad/")


def lint_current(path: Path, today: date, stale_days: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8")
    rel = f"{path.parent.name}/CURRENT.md"

    # 1. top timestamp + freshness
    m = DATE_RE.search(text)
    if not m:
        errors.append(f"{rel}: 顶部缺 `更新:YYYY-MM-DD` 时间戳")
    else:
        stamped = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        age = (today - stamped).days
        if age > stale_days:
            warnings.append(
                f"{rel}: 已 {age} 天未更新（阈值 {stale_days}）——确认是否仍是当前真相"
            )

    # 2. no session-scratchpad references (dead link for the next agent)
    for i, line in enumerate(text.splitlines(), start=1):
        if SCRATCHPAD_RE.search(line):
            errors.append(
                f"{rel}:{i}: 引用了 session `scratchpad/`（换 agent 即失效）——迁到持久路径"
            )

    # 3. required sections present
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    if missing:
        errors.append(f"{rel}: 缺必需小节 {' '.join(missing)}")

    return errors, warnings


def lint_root(root: Path, *, today: date, stale_days: int) -> tuple[list[str], list[str]]:
    currents = sorted(root.glob("*/CURRENT.md"))
    if not currents:
        return [f"{root} 下没有任何 <模块>/CURRENT.md"], []

    errors: list[str] = []
    warnings: list[str] = []
    for path in currents:
        e, w = lint_current(path, today, stale_days)
        errors += e
        warnings += w

    # README index must link every module that has a CURRENT.md
    readme = root / "README.md"
    if readme.exists():
        rtext = readme.read_text(encoding="utf-8")
        for path in currents:
            name = path.parent.name
            if f"{name}/CURRENT.md" not in rtext:
                errors.append(f"README.md: 索引缺模块 `{name}`（目录存在但 README 未链接）")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stale-days", type=int, default=21)
    parser.add_argument("--strict", action="store_true", help="把 warning 也当失败")
    args = parser.parse_args()

    errors, warnings = lint_root(args.root, today=date.today(), stale_days=args.stale_days)

    for w in warnings:
        print(f"WARN: {w}", file=sys.stderr)
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)

    if errors or (args.strict and warnings):
        return 1

    n = len(sorted(args.root.glob("*/CURRENT.md")))
    print(f"OK: {n} 个 CURRENT.md 通过（{len(warnings)} warning）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
