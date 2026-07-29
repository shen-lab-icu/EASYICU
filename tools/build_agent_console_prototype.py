"""Render a read-only three-pane console prototype from real research-agent runs.

This is a DESIGN PROTOTYPE, not a product surface. It reads artifacts that the
pipeline already writes (audit_log.jsonl, claim_ledger.csv, analysis_plan.json,
steps/*/outputs/step_summary.json) and renders them into one self-contained HTML
page so the proposed task-console layout can be judged against real data.

It writes nothing back into a run directory and imports nothing from easyicu.

Usage:
    python tools/build_agent_console_prototype.py RUN_DIR [RUN_DIR ...] -o OUT.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Events whose only job is to restate that a prior run already did the work.
# They repeat verbatim dozens of times on resume and would otherwise dominate
# the transcript; the console shows them once with a repeat count.
RESUME_NOISE = ("Skipped completed step from prior run",)


def parse_ts(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}分"
    return f"{seconds / 3600:.1f}小时"


@dataclass
class Event:
    ts: datetime | None
    phase: str
    event: str
    status: str
    step_id: str
    detail: dict
    repeats: int = 1


@dataclass
class StepGroup:
    step_id: str
    is_step: bool = False
    events: list[Event] = field(default_factory=list)
    # Time actually spent inside this step, summed over globally-adjacent event
    # pairs. Taking max(ts) - min(ts) instead would span every resume replay of
    # the same step and report hours for work that took seconds.
    seconds: float = 0.0

    @property
    def label(self) -> str:
        name = self.step_id
        if name[:2].isdigit() and "_" in name:
            name = name.split("_", 1)[1]
        return name.replace("_", " ")

    @property
    def raw_count(self) -> int:
        return sum(e.repeats for e in self.events)

    @property
    def status(self) -> str:
        kinds = {e.status for e in self.events}
        for candidate in ("error", "paused", "warning"):
            if candidate in kinds:
                return candidate
        return "complete" if "complete" in kinds else "running"


@dataclass
class Claim:
    claim_id: str
    text: str
    evidence_refs: str
    status: str
    note: str

    @property
    def bound(self) -> bool:
        return bool(self.evidence_refs.strip())


@dataclass
class Run:
    path: Path
    question: str
    groups: list[StepGroup]
    claims: list[Claim]
    artifacts: list[dict]
    seconds: float

    @property
    def slug(self) -> str:
        return self.path.name

    @property
    def case(self) -> str:
        """Benchmark case id, e.g. E2_lactate_mortality — runs/<case>/<arm>/<run>."""
        parts = self.path.parts
        return parts[-3] if len(parts) >= 3 else self.path.name

    @property
    def unbound_claims(self) -> int:
        return sum(1 for c in self.claims if not c.bound)

    @property
    def console_status(self) -> str:
        """Three states the task rail must distinguish.

        `blocked` is the state a fire-and-forget console does not have and the
        one this project cannot do without: the run stopped and is waiting on a
        person, so it must not read as either running or done.
        """
        if self.unbound_claims or not self.claims:
            return "blocked"
        if any(g.status == "paused" for g in self.groups):
            return "waiting"
        return "done"


def load_events(run: Path) -> list[Event]:
    log = run / "audit_log.jsonl"
    if not log.exists():
        return []
    events: list[Event] = []
    for line in log.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(
            Event(
                ts=parse_ts(row.get("timestamp", "")),
                phase=row.get("phase") or "",
                event=row.get("event") or "",
                status=row.get("status") or "",
                step_id=row.get("step_id") or "",
                detail=row.get("detail") or {},
            )
        )
    return events


def dedupe(events: list[Event]) -> list[Event]:
    """Collapse consecutive identical events into one row with a repeat count."""
    out: list[Event] = []
    for ev in events:
        if out and out[-1].event == ev.event and out[-1].status == ev.status:
            out[-1].repeats += 1
            continue
        out.append(ev)
    return out


def step_key(ev: Event) -> str:
    return ev.step_id or ev.phase or "run"


def group_by_step(events: list[Event]) -> list[StepGroup]:
    groups: dict[str, StepGroup] = {}
    order: list[str] = []
    for ev in events:
        key = step_key(ev)
        if key not in groups:
            groups[key] = StepGroup(step_id=key, is_step=bool(ev.step_id))
            order.append(key)
        groups[key].events.append(ev)

    for earlier, later in zip(events, events[1:]):
        if step_key(earlier) != step_key(later) or not earlier.ts or not later.ts:
            continue
        groups[step_key(earlier)].seconds += (later.ts - earlier.ts).total_seconds()

    for key in order:
        groups[key].events = dedupe(groups[key].events)
    return [groups[k] for k in order]


def load_claims(run: Path) -> list[Claim]:
    ledger = run / "claim_ledger.csv"
    if not ledger.exists():
        return []
    with ledger.open(newline="", errors="replace") as handle:
        return [
            Claim(
                claim_id=row.get("claim_id", ""),
                text=row.get("claim_text", ""),
                evidence_refs=row.get("evidence_refs", "") or "",
                status=row.get("status", ""),
                note=row.get("note", "") or "",
            )
            for row in csv.DictReader(handle)
        ]


def load_artifacts(run: Path) -> list[dict]:
    found: list[dict] = []
    for summary in sorted((run / "steps").glob("*/outputs/step_summary.json")):
        try:
            data = json.loads(summary.read_text(errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        for key, name in (data.get("output_files") or {}).items():
            binding = next(
                (
                    b
                    for b in (data.get("input_bindings") or [])
                    if b.get("evidence_id")
                ),
                {},
            )
            found.append(
                {
                    "step": data.get("step", summary.parent.parent.name),
                    "kind": key.split(":", 1)[0],
                    "name": name,
                    "evidence_id": binding.get("evidence_id", ""),
                    "sha256": (binding.get("sha256") or "")[:12],
                }
            )
    return found


def load_run(run: Path) -> Run:
    events = load_events(run)
    stamps = [e.ts for e in events if e.ts]
    question = ""
    plan = run / "analysis_plan.json"
    if plan.exists():
        try:
            question = json.loads(plan.read_text(errors="replace")).get(
                "research_question", ""
            )
        except json.JSONDecodeError:
            question = ""
    return Run(
        path=run,
        question=question or run.name,
        groups=group_by_step(events),
        claims=load_claims(run),
        artifacts=load_artifacts(run),
        seconds=(max(stamps) - min(stamps)).total_seconds() if len(stamps) > 1 else 0.0,
    )


STATUS_TEXT = {
    "blocked": ("已拦截 · 等你确认", "s-blocked"),
    "waiting": ("暂停 · 等你确认", "s-blocked"),
    "done": ("完成", "s-done"),
}
STEP_DOT = {
    "complete": "s-done",
    "error": "s-blocked",
    "paused": "s-wait",
    "warning": "s-wait",
    "running": "s-run",
}

CSS = """
*{box-sizing:border-box}
body{margin:0;font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;
 color:#1c1c1a;background:#f6f5f2}
.tabs>input{position:absolute;opacity:0;pointer-events:none}
.shell{display:grid;grid-template-columns:250px minmax(0,1fr) 360px;height:100vh}
.col{overflow:auto;background:#fff}
.rail{border-right:1px solid #e5e3dd;background:#faf9f7}
.mid{border-right:1px solid #e5e3dd}
.hd{padding:12px 14px;border-bottom:1px solid #e5e3dd;font-size:12px;color:#6b6a65;
 position:sticky;top:0;background:inherit;z-index:2}
.taskrow{display:block;padding:9px 14px;border-bottom:1px solid #f0eee9;cursor:pointer}
.taskrow:hover{background:#f2f0ec}
.taskrow .tt{font-size:13px;line-height:1.4;overflow:hidden;text-overflow:ellipsis;
 white-space:nowrap}
.taskrow .tq{font-size:11px;color:#6b6a65;line-height:1.4;margin-top:2px;
 display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.taskrow .tm{font-size:11px;color:#8a8983;margin-top:3px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
 vertical-align:1px}
.s-done{background:#3b6d11}.s-blocked{background:#a32d2d}.s-wait{background:#854f0b}
.s-run{background:#185fa5}
.pane{display:none;padding:0 0 40px}
.qhead{padding:14px 18px;border-bottom:1px solid #e5e3dd;position:sticky;top:0;
 background:#fff;z-index:2}
.qhead h1{font:500 15px/1.5 inherit;margin:0 0 4px}
.qhead .meta{font-size:12px;color:#6b6a65}
.chip{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;margin-left:6px}
.chip-blocked{background:#fceaea;color:#a32d2d}
.chip-done{background:#eaf3de;color:#3b6d11}
details{border-bottom:1px solid #f0eee9}
summary{padding:9px 18px;cursor:pointer;font-size:13px;list-style:none;display:flex;
 gap:8px;align-items:baseline}
summary::-webkit-details-marker{display:none}
summary:hover{background:#faf9f7}
summary .grow{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
summary .t{font-size:11px;color:#8a8983;font-variant-numeric:tabular-nums}
.ev{display:flex;gap:8px;padding:4px 18px 4px 34px;font-size:12px;color:#4a4945;
 align-items:baseline}
.ev:hover{background:#faf9f7}
.ev .grow{flex:1;min-width:0}
.ev .t{font-size:11px;color:#a3a29c;font-variant-numeric:tabular-nums;white-space:nowrap}
.rep{font-size:10px;background:#eeece7;color:#6b6a65;border-radius:99px;padding:0 6px;
 margin-left:6px}
.det{font-size:11px;color:#8a8983;padding:0 18px 6px 34px;font-family:ui-monospace,monospace;
 word-break:break-all}
.claim{padding:10px 16px;border-bottom:1px solid #f0eee9;font-size:12.5px}
.claim.unbound{background:#fdf3f3;border-left:3px solid #a32d2d}
.claim .ev-id{font:11px ui-monospace,monospace;color:#3b6d11;margin-top:4px;
 word-break:break-all}
.claim.unbound .ev-id{color:#a32d2d}
.art{display:flex;gap:8px;padding:7px 16px;border-bottom:1px solid #f0eee9;font-size:12px;
 align-items:baseline}
.art .grow{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.art .sha{font:10px ui-monospace,monospace;color:#8a8983}
.note{padding:10px 16px;font-size:11.5px;color:#8a8983;border-bottom:1px solid #f0eee9}
"""


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def plain(text: str) -> str:
    """Claim text carries manuscript markdown; the ledger view wants the words."""
    return text.replace("**", "").replace("[^", " [").strip()


def render_group(group: StepGroup) -> str:
    dot = STEP_DOT.get(group.status, "s-run")
    shown = len(group.events)
    raw = group.raw_count
    hidden = raw - shown
    tail = f"，去重折叠 {hidden} 条重复" if hidden else ""
    # A step_id group is real analysis work; a phase-only group is run-level
    # bookkeeping that repeats once per resume. Calling both "steps" would
    # inflate a 1-event phase into "32 steps".
    unit = f"执行了 {raw} 个步骤" if group.is_step else f"{raw} 条运行记录"
    rows = []
    for ev in group.events:
        rep = f'<span class="rep">×{ev.repeats}</span>' if ev.repeats > 1 else ""
        stamp = ev.ts.strftime("%H:%M:%S") if ev.ts else ""
        rows.append(
            f'<div class="ev"><span class="grow">{esc(ev.event)}{rep}</span>'
            f'<span class="t">{stamp}</span></div>'
        )
        keep = {
            k: v
            for k, v in ev.detail.items()
            if k not in ("run_id",) and not isinstance(v, (list, dict))
        }
        if keep:
            body = " · ".join(f"{k}={v}" for k, v in list(keep.items())[:5])
            rows.append(f'<div class="det">{esc(body)}</div>')
    return (
        f"<details><summary><span class=\"dot {dot}\"></span>"
        f'<span class="grow">{esc(group.label)} — {unit}{tail}</span>'
        f'<span class="t">{human_duration(group.seconds)}</span></summary>'
        f'{"".join(rows)}</details>'
    )


def render_pane(index: int, run: Run) -> tuple[str, str]:
    label, _ = STATUS_TEXT[run.console_status]
    chip = "chip-blocked" if run.console_status != "done" else "chip-done"
    groups = "".join(render_group(g) for g in run.groups)

    if run.claims:
        claims = "".join(
            f'<div class="claim{"" if c.bound else " unbound"}">{esc(plain(c.text)[:240])}'
            f'<div class="ev-id">'
            + (
                f"证据 {esc(c.evidence_refs[:60])} ✓ 已绑定"
                if c.bound
                else f"未绑定证据 · {esc(c.status)} · {esc(c.note[:80])}"
            )
            + "</div></div>"
            for c in run.claims
        )
    else:
        claims = '<div class="claim unbound">没有产出任何 claim<div class="ev-id">'
        claims += "写作阶段被 fail-closed 门拦下</div></div>"

    arts = "".join(
        f'<div class="art"><span class="grow">{esc(a["name"])}</span>'
        f'<span class="sha">{esc(a["sha256"])}</span></div>'
        for a in run.artifacts[:24]
    ) or '<div class="note">该 run 没有登记产物文件</div>'

    bound = sum(1 for c in run.claims if c.bound)
    transcript = f"""
<div class="pane pane-{index}">
  <div class="qhead">
    <h1>{esc(run.question)}</h1>
    <div class="meta">{len(run.groups)} 个阶段 · 用时 {human_duration(run.seconds)}
      <span class="chip {chip}">{label}</span></div>
  </div>
  {groups}
</div>"""
    artifacts = f"""
<div class="pane pane-{index}">
  <div class="hd">产物 · 证据绑定　{bound}/{len(run.claims)} 条 claim 已绑定</div>
  {claims}
  <div class="hd">文件</div>
  {arts}
  <div class="note">{esc(run.path.name)}</div>
</div>"""
    return transcript, artifacts


def build(runs: list[Run]) -> str:
    inputs = "".join(
        f'<input type="radio" name="task" id="t{i}"{" checked" if not i else ""}>'
        for i in range(len(runs))
    )
    rows = []
    for i, run in enumerate(runs):
        label, dot = STATUS_TEXT[run.console_status]
        rows.append(
            f'<label class="taskrow" for="t{i}">'
            f'<div class="tt"><span class="dot {dot}"></span>{esc(run.case)}</div>'
            f'<div class="tq">{esc(run.question[:70])}</div>'
            f'<div class="tm">{label} · {human_duration(run.seconds)}</div></label>'
        )
    rendered = [render_pane(i, run) for i, run in enumerate(runs)]
    transcripts = "".join(pair[0] for pair in rendered)
    artifacts = "".join(pair[1] for pair in rendered)
    rules = "".join(
        f"#t{i}:checked ~ .shell .pane-{i}{{display:block}}" for i in range(len(runs))
    )
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>EasyICU 任务控制台 — 设计原型</title>
<style>{CSS}{rules}</style></head>
<body class="tabs">
{inputs}
<div class="shell">
  <div class="col rail">
    <div class="hd">任务</div>
    {"".join(rows)}
  </div>
  <div class="col mid">{transcripts}</div>
  <div class="col">{artifacts}</div>
</div>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    loaded = [load_run(p) for p in args.runs if p.is_dir()]
    if not loaded:
        raise SystemExit("no readable run directories")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(loaded), encoding="utf-8")
    for run in loaded:
        raw = sum(g.raw_count for g in run.groups)
        shown = sum(len(g.events) for g in run.groups)
        print(
            f"{run.slug}: {raw} 事件 → {shown} 行（折叠 {raw - shown}）· "
            f"{len(run.claims)} claim（{run.unbound_claims} 未绑定）· "
            f"{len(run.artifacts)} 产物 · {run.console_status}"
        )
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
