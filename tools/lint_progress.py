#!/usr/bin/env python3
"""Lint the `项目进度/` middle-layer handoff pages (CURRENT.md).

Companion to ``lint_main_plan.py``: that one guards the top-level dashboard's
row count; this one guards the middle layer so it does not rot —

- every ``<module>/CURRENT.md`` carries a top ``更新:YYYY-MM-DD`` timestamp and
  is not staler than ``--stale-days`` (warning: it may no longer be the truth);
- no page references the per-session ``scratchpad/`` (a dead link the moment
  another agent picks up — this is the exact gap that bit us on 2026-07-10);
- every page keeps the 7 required sections (🎯 📍 🔨 ✅ ⏭️ ⚠️ 📚);
- **no page outgrows the one thing it exists to do** (see below);
- the README index lists every module directory that actually has a CURRENT.md.

Why the size budget (added 2026-07-27)
--------------------------------------
The checks above guard freshness, links and structure — none of which actually
degrade over time. What degrades is **length**, and nothing was watching it.
By 2026-07-27 ``agent/CURRENT.md`` had reached 225 KB with 22 accumulated
paragraphs inside a section literally titled 「当前真相（一句话）」, and a fresh
session could no longer open the file at all: the read returned "output too
large" and the agent had to grep around its own handoff page. A cockpit that a
new pilot cannot read has inverted its purpose.

Crucially this was **not** a discipline lapse. Measured across the six modules,
bloat tracked how many rounds a module had seen (web 407 B / 1 round, 数据底座
11.7 KB / 7, agent 66 KB / 22) — so any module worked on long enough would rot.
That makes it a missing guard, not a missing habit. The top layer already had
one (``lint_main_plan.py`` caps the dashboard at 5-8 rows); the middle layer
simply never got the equivalent.

The remedy is relocation, never deletion: overflow moves verbatim into the
module's ``HISTORY.md`` (「当时做了什么」), with the detail staying in
``EASYICU/task_logs/`` (「怎么做的」). CURRENT.md keeps only 「现在是什么」.
Budgets are review triggers in the same spirit as the frontend file-size rule
in ``CLAUDE.md`` — a page slightly over is a nudge, not a crisis.

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

#: Whole-page budgets, in bytes of UTF-8, anchored on a measured number rather
#: than on wherever the pages happened to land after the 2026-07-27 cleanup.
#:
#: The observed truncation point for a single agent file-read was ~64 KB (that
#: is where reading the old page failed). FAIL therefore sits just under it: a
#: page past this is provably unreadable in one pass, which is the page's whole
#: job. WARN at 40 KB leaves genuine headroom to notice and prune *before*
#: hitting that wall, instead of discovering it mid-handoff.
#:
#: After the cleanup: 数据底座 15 KB, web 11 KB, idea-mining 12 KB,
#: 论文图件 36 KB, benchmark实验 32 KB — all silent. agent is 60 KB and still
#: warns, correctly: its ⏭️/🔨 sections still hold finished work that needs a
#: human call on what is still live. A warning that names real remaining work
#: is not noise.
PAGE_WARN_BYTES = 40_000
PAGE_FAIL_BYTES = 64_000

#: 「当前真相（一句话）」 is the section that rots first, because appending one
#: round's paragraph is always the path of least resistance. It is also the
#: cheapest to keep honest, so this one is a hard failure rather than a nudge.
#: 2,500 bytes is ~1,200 Chinese characters — already generous for "one sentence".
TRUTH_FAIL_BYTES = 2_500
TRUTH_HEADING = "## 🎯"

#: A README index row exists to be *scanned*, so it has to stay one glance long.
#: The 2026-07-27 audit found one at 295 characters — a paragraph pretending to
#: be a table cell, in the one table a human is told is the only page they need.
INDEX_ROW_MAX_CHARS = 150
INDEX_ROW_RE = re.compile(r"^\| \[([^\]]+)\]\([^)]*CURRENT\.md\) \|([^|]*)\|")


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

    # 4. size budgets — the axis that actually degrades. The remedy named in
    #    every message is relocation into HISTORY.md, never deletion: a fact
    #    that stops being current does not stop being true.
    remedy = f"下沉到 {path.parent.name}/HISTORY.md（逐字搬运，不删内容）"
    page_bytes = len(text.encode("utf-8"))
    if page_bytes > PAGE_FAIL_BYTES:
        errors.append(
            f"{rel}: 整页 {page_bytes:,} 字节 > {PAGE_FAIL_BYTES:,} —— "
            f"新会话已无法一次读完，驾驶舱失去意义。{remedy}"
        )
    elif page_bytes > PAGE_WARN_BYTES:
        warnings.append(
            f"{rel}: 整页 {page_bytes:,} 字节 > {PAGE_WARN_BYTES:,} —— "
            f"接近单次读取上限，下一次结构性改动请{remedy}"
        )

    truth_bytes = len(_section_text(text, TRUTH_HEADING).encode("utf-8"))
    if truth_bytes > TRUTH_FAIL_BYTES:
        errors.append(
            f"{rel}: 「🎯 当前真相（一句话）」{truth_bytes:,} 字节 > "
            f"{TRUTH_FAIL_BYTES:,} —— 它是「现在是什么」，不是逐轮流水账。"
            f"只留当前一段，往期{remedy}"
        )

    return errors, warnings


def _section_text(text: str, heading: str) -> str:
    """Return one ``## `` section's body, empty when the heading is absent."""

    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line.startswith(heading)
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


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

    # README index must link every module that has a CURRENT.md, and each row
    # must stay scannable — the index is the one page the human is told to open.
    readme = root / "README.md"
    if readme.exists():
        rtext = readme.read_text(encoding="utf-8")
        for path in currents:
            name = path.parent.name
            if f"{name}/CURRENT.md" not in rtext:
                errors.append(f"README.md: 索引缺模块 `{name}`（目录存在但 README 未链接）")
        for line in rtext.splitlines():
            m = INDEX_ROW_RE.match(line)
            if m and len(m.group(2).strip()) > INDEX_ROW_MAX_CHARS:
                errors.append(
                    f"README.md: 索引行 `{m.group(1)}` {len(m.group(2).strip())} 字 > "
                    f"{INDEX_ROW_MAX_CHARS} —— 索引是扫读用的，一句话；"
                    f"细节属于该模块的 CURRENT.md"
                )

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
