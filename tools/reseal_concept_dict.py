#!/usr/bin/env python3
"""Reseal the clinical dictionaries: the LOCK and every current profile pin in one step.

A dictionary reseal moves two pins that must always agree:

* ``src/easyicu/data/concept-dict.LOCK.json``: ``concept_dict_sha256``,
  ``sofa2_dict_sha256``, ``total_concepts`` and the other foundation resource
  digests that ``scripts/releases/EX-A01_seal_full6_release.py`` validates;
* ``src/easyicu/research_agent/orchestration/profiles.py``: the next dated
  profile generation (``…_AUDITn``). Its one literal pin is the planner-canary
  profile's ``expected_concept_dict_sha``/``expected_sofa2_dict_sha``. Every other
  current profile derives from it: the reviewed/live pairs, the CURRENT_* refs,
  DESKTOP_FIRST_RESULT*, PUBLIC_DEMO_ACCEPTANCE and the SICdb overlays. The
  previous generation stays registered as history.

``client_trust.ACCEPTANCE_PROFILE_REF`` follows PUBLIC_DEMO_ACCEPTANCE.

The default run is a dry run that reports what would change (paths and digests
only, never dictionary content). ``--apply`` needs the user's consent text. All
files are written together. The pins are then verified in a fresh interpreter,
and every file is restored if any pin disagrees.

Usage::

    python tools/reseal_concept_dict.py
    python tools/reseal_concept_dict.py --apply --consent "<user's words>" \
        --label "twelfth reseal: lead-r13 final review" \
        --locked-at 2026-11-07T12:00:00Z [--source-commit <sha>] [--root <repo>]

The changelog, the reseal table, the v4 scorer tree digest and the research-agent
resource baseline are separate steps; the tool prints a reminder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK = "src/easyicu/data/concept-dict.LOCK.json"
PROFILES = "src/easyicu/research_agent/orchestration/profiles.py"
CLIENT_TRUST = "src/easyicu/research_agent/providers/client_trust.py"
CONCEPT_DICT = "src/easyicu/data/concept-dict.json"
SOFA2_DICT = "src/easyicu/data/sofa2-dict.json"
# Same fields and files as EX-A01_seal_full6_release.FOUNDATION_RESOURCE_PATHS.
LOCK_RESOURCES = {
    "concept_dict_sha256": CONCEPT_DICT,
    "sofa2_dict_sha256": SOFA2_DICT,
    "clinical_contracts_sha256": "src/easyicu/data/clinical-contracts.json",
    "clinical_contract_validator_sha256": "src/easyicu/clinical_contracts.py",
    "data_sources_sha256": "src/easyicu/data/data-sources.json",
}
_CANARY = re.compile(
    r"^E1_PROGRESSIVE_PLANNER_CANARY_(\d{4}_\d{2}_\d{2})_AUDIT(\d+) = replace\(", re.M
)
_OVERLAY_VERSION = re.compile(r'"version": "\d{8}-sic-overlay-v(\d+)"')
_ACCEPTANCE = re.compile(r'^ACCEPTANCE_PROFILE_REF = "public_demo_acceptance_dev/[^"]+"$', re.M)


class ResealError(RuntimeError):
    """A reseal precondition failed; nothing was written."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Generation:
    prev: str          # e.g. 2026_11_05_AUDIT11
    new: str           # e.g. 2026_11_07_AUDIT12
    version: str       # e.g. 20261107-audit12
    overlay_version: str
    locked_at: str


def next_generation(profiles_text: str, locked_at: str) -> Generation:
    found = [(int(n), date) for date, n in _CANARY.findall(profiles_text)]
    if not found:
        raise ResealError("no E1_PROGRESSIVE_PLANNER_CANARY_<date>_AUDITn generation in profiles.py")
    number, date = max(found)
    overlays = [int(v) for v in _OVERLAY_VERSION.findall(profiles_text)]
    if not overlays:
        raise ResealError("no SICdb overlay generation in profiles.py")
    when = datetime.fromisoformat(locked_at.replace("Z", "+00:00"))
    day = when.strftime("%Y%m%d")
    return Generation(
        prev=f"{date}_AUDIT{number}",
        new=f"{when.strftime('%Y_%m_%d')}_AUDIT{number + 1}",
        version=f"{day}-audit{number + 1}",
        overlay_version=f"{day}-sic-overlay-v{max(overlays) + 1}",
        locked_at=locked_at,
    )


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ResealError(f"profiles.py anchor for {what} found {text.count(old)} times, expected 1")
    return text.replace(old, new)


def render_profiles(text: str, gen: Generation, concept_sha: str, sofa2_sha: str, label: str) -> str:
    p, n = gen.prev, gen.new
    head = f'    version="{gen.version}", locked_at="{gen.locked_at}",\n'
    derive = (
        f"    expected_concept_dict_sha=E1_PROGRESSIVE_PLANNER_CANARY_{n}.expected_concept_dict_sha,\n"
        f"    expected_sofa2_dict_sha=E1_PROGRESSIVE_PLANNER_CANARY_{n}.expected_sofa2_dict_sha,\n"
    )
    comment = "".join(f"    # {line}\n" for line in _wrap(label))
    # 1. canary (the one literal pin) and reviewed pair; CURRENT refs move.
    text = _replace_once(
        text,
        f"CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF = (\n    E1_PROGRESSIVE_PLANNER_CANARY_{p}.ref\n)\n"
        f"CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF = E1_REVIEWED_DEMO_{p}.ref\n",
        f"E1_PROGRESSIVE_PLANNER_CANARY_{n} = replace(\n    E1_PROGRESSIVE_PLANNER_CANARY_{p},\n{head}{comment}"
        f'    expected_concept_dict_sha="{concept_sha}",\n    expected_sofa2_dict_sha="{sofa2_sha}",\n)\n\n'
        f"E1_REVIEWED_DEMO_{n} = replace(\n    E1_REVIEWED_DEMO_{p},\n{head}{derive})\n\n"
        f"CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF = (\n    E1_PROGRESSIVE_PLANNER_CANARY_{n}.ref\n)\n"
        f"CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF = E1_REVIEWED_DEMO_{n}.ref\n",
        "CURRENT_E1 refs",
    )
    # 2. live-PubMed pair.
    text = _replace_once(
        text,
        f"CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF = (\n    E1_PROGRESSIVE_PLANNER_CANARY_LIVE_PUBMED_{p}.ref\n)\n"
        f"CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF = (\n    E1_REVIEWED_DEMO_LIVE_PUBMED_{p}.ref\n)\n",
        f"E1_PROGRESSIVE_PLANNER_CANARY_LIVE_PUBMED_{n} = replace(\n    E1_PROGRESSIVE_PLANNER_CANARY_LIVE_PUBMED_{p},\n{head}{derive})\n\n"
        f"E1_REVIEWED_DEMO_LIVE_PUBMED_{n} = replace(\n    E1_REVIEWED_DEMO_LIVE_PUBMED_{p},\n{head}{derive})\n\n"
        f"CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF = (\n    E1_PROGRESSIVE_PLANNER_CANARY_LIVE_PUBMED_{n}.ref\n)\n"
        f"CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF = (\n    E1_REVIEWED_DEMO_LIVE_PUBMED_{n}.ref\n)\n",
        "CURRENT_E1 live refs",
    )
    # 3. undated current names move; the previous values stay as history.
    moved = ["DESKTOP_FIRST_RESULT", "DESKTOP_FIRST_RESULT_LIVE_PUBMED", "PUBLIC_DEMO_ACCEPTANCE"]
    overlays = [
        "E1_PROGRESSIVE_PLANNER_CANARY_SIC_OVERLAY", "E1_REVIEWED_DEMO_SIC_OVERLAY",
        "E1_PROGRESSIVE_PLANNER_CANARY_LIVE_PUBMED_SIC_OVERLAY", "E1_REVIEWED_DEMO_LIVE_PUBMED_SIC_OVERLAY",
        "WEB_DESKTOP_FIRST_RESULT_SIC_OVERLAY", "WEB_DESKTOP_FIRST_RESULT_LIVE_PUBMED_SIC_OVERLAY",
    ]
    block = [f"# Reseal {gen.version}: {label}", f"# moves the current names; {p} stays history."]
    for name in moved:
        block += [f"{name}_{p} = {name}", f"{name} = replace({name}_{p},", head.rstrip("\n"), derive.rstrip("\n"), ")"]
    block += [
        f"_SIC_OVERLAY_PROFILE_COORDINATES_{n} = {{",
        f"    **_SIC_OVERLAY_PROFILE_COORDINATES_{p},",
        f'    "version": "{gen.overlay_version}",',
        f'    "locked_at": "{gen.locked_at}",',
        f'    "expected_concept_dict_sha": E1_PROGRESSIVE_PLANNER_CANARY_{n}.expected_concept_dict_sha,',
        f'    "expected_sofa2_dict_sha": E1_PROGRESSIVE_PLANNER_CANARY_{n}.expected_sofa2_dict_sha,',
        "}",
    ]
    for name in overlays:
        block += [f"{name}_{p} = {name}", f"{name} = replace({name}_{p}, **_SIC_OVERLAY_PROFILE_COORDINATES_{n})"]
    anchor = "CURRENT_WEB_E1_PLANNER_CANARY_DEV_PROFILE_REF = E1_PROGRESSIVE_PLANNER_CANARY_SIC_OVERLAY.ref\n"
    text = _replace_once(text, anchor, "\n".join(block) + "\n" + anchor, "overlay block")
    # 4. registry: the new generation and the history aliases.
    entries = [f"E1_PROGRESSIVE_PLANNER_CANARY_{n}", f"E1_REVIEWED_DEMO_{n}",
               f"E1_PROGRESSIVE_PLANNER_CANARY_LIVE_PUBMED_{n}", f"E1_REVIEWED_DEMO_LIVE_PUBMED_{n}"]
    entries += [f"{name}_{p}" for name in moved + overlays]
    registry_anchor = f"    E1_PROGRESSIVE_PLANNER_CANARY_{p}.ref: E1_PROGRESSIVE_PLANNER_CANARY_{p},\n"
    text = _replace_once(
        text, registry_anchor,
        "".join(f"    {e}.ref: {e},\n" for e in entries) + registry_anchor, "registry",
    )
    return text


def _wrap(label: str, width: int = 84) -> list[str]:
    words, lines, line = label.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def render_lock(lock: dict, digests: dict, total: int, change: dict) -> str:
    out = dict(lock)
    out.update(digests)
    out["total_concepts"] = total
    out["finalized_foundation_changes"] = [*lock.get("finalized_foundation_changes", []), change]
    return json.dumps(out, indent=2, ensure_ascii=False) + "\n"


def plan(root: Path) -> dict:
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    digests = {field: sha256_file(root / path) for field, path in LOCK_RESOURCES.items()}
    changed = {f: (lock.get(f), d) for f, d in digests.items() if lock.get(f) != d}
    return {"lock": lock, "digests": digests, "changed": changed,
            "total": len(json.loads((root / CONCEPT_DICT).read_text(encoding="utf-8")))}


VERIFY = r"""
import json, hashlib, re, sys
from pathlib import Path
root = Path(sys.argv[1])
from easyicu.research_agent.orchestration import profiles
from easyicu.research_agent.providers import client_trust
lock = json.loads((root / "src/easyicu/data/concept-dict.LOCK.json").read_text())
problems = []
for field, path in (("concept_dict_sha256", "src/easyicu/data/concept-dict.json"),
                    ("sofa2_dict_sha256", "src/easyicu/data/sofa2-dict.json")):
    if lock[field] != hashlib.sha256((root / path).read_bytes()).hexdigest():
        problems.append(f"LOCK {field} != {path}")
reg = profiles.SUBMISSION_PROFILE_REGISTRY
audits = [int(m.group(1)) for v in reg.values() if (m := re.fullmatch(r"\d{8}-audit(\d+)", v.version))]
overlays = [int(m.group(1)) for v in reg.values() if (m := re.fullmatch(r"\d{8}-sic-overlay-v(\d+)", v.version))]
current = [v for v in reg.values()
           if re.fullmatch(rf"\d{{8}}-audit{max(audits)}", v.version)
           or re.fullmatch(rf"\d{{8}}-sic-overlay-v{max(overlays)}", v.version)]
for v in current:
    if v.expected_concept_dict_sha != lock["concept_dict_sha256"] or v.expected_sofa2_dict_sha != lock["sofa2_dict_sha256"]:
        problems.append(f"profile {v.ref} pins differ from LOCK")
for name in dir(profiles):
    if name.startswith(("CURRENT_E1", "CURRENT_WEB_E1")) and name.endswith("_REF"):
        if profiles.get_submission_profile(getattr(profiles, name)) not in current:
            problems.append(f"{name} is not in the current generation")
if client_trust.ACCEPTANCE_PROFILE_REF != profiles.PUBLIC_DEMO_ACCEPTANCE.ref:
    problems.append("client_trust.ACCEPTANCE_PROFILE_REF != PUBLIC_DEMO_ACCEPTANCE.ref")
print(json.dumps({"current_profiles": len(current), "problems": problems}))
"""


def verify_pins(root: Path) -> dict:
    """Check every current pin against the LOCK in a fresh interpreter (no module cache)."""
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    proc = subprocess.run([sys.executable, "-c", VERIFY, str(root)], capture_output=True, text=True, env=env, cwd=root)
    if proc.returncode != 0:
        return {"current_profiles": 0, "problems": [f"verifier failed: {proc.stderr.strip().splitlines()[-1:]}"]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def apply(root: Path, *, consent: str, label: str, locked_at: str, source_commit: str) -> dict:
    if not consent.strip():
        raise ResealError("--consent with the user's own words is required")
    state = plan(root)
    concept, sofa2 = state["digests"]["concept_dict_sha256"], state["digests"]["sofa2_dict_sha256"]
    if concept == state["lock"]["concept_dict_sha256"] and sofa2 == state["lock"]["sofa2_dict_sha256"]:
        raise ResealError("both dictionaries already match the LOCK; nothing to reseal")
    profiles_text = (root / PROFILES).read_text(encoding="utf-8")
    gen = next_generation(profiles_text, locked_at)
    new_profiles = render_profiles(profiles_text, gen, concept, sofa2, label)
    compile(new_profiles, PROFILES, "exec")
    trust_text = (root / CLIENT_TRUST).read_text(encoding="utf-8")
    if len(_ACCEPTANCE.findall(trust_text)) != 1:
        raise ResealError("client_trust.ACCEPTANCE_PROFILE_REF anchor not found exactly once")
    new_trust = _ACCEPTANCE.sub(
        f'ACCEPTANCE_PROFILE_REF = "public_demo_acceptance_dev/{gen.version}"', trust_text
    )
    change = {"git_commit": source_commit, "change": f"User-approved {gen.version} reseal: {label} Consent: {consent}"}
    new_lock = render_lock(state["lock"], state["digests"], state["total"], change)
    originals = {rel: (root / rel).read_bytes() for rel in (LOCK, PROFILES, CLIENT_TRUST)}
    try:
        for rel, content in ((PROFILES, new_profiles), (CLIENT_TRUST, new_trust), (LOCK, new_lock)):
            tmp = root / f"{rel}.reseal-tmp"
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, root / rel)
        result = verify_pins(root)
        if result["problems"]:
            raise ResealError("pins disagree after writing: " + "; ".join(result["problems"]))
    except BaseException:
        for rel, raw in originals.items():
            (root / rel).write_bytes(raw)
        raise
    return {"generation": gen, "changed": state["changed"], "verify": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--consent", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--locked-at", default="")
    parser.add_argument("--source-commit", default="")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    state = plan(root)
    print(f"root: {root}")
    if not state["changed"]:
        print("LOCK matches every foundation resource; no reseal pending.")
    for field, (old, new) in state["changed"].items():
        print(f"pending: {field} {old} -> {new}")
    print(f"total_concepts: {state['lock'].get('total_concepts')} -> {state['total']}")
    if not args.apply:
        print("dry run: nothing written (pass --apply with --consent, --label and --locked-at).")
        return 0
    if not (args.label and args.locked_at):
        raise SystemExit("--apply needs --label and --locked-at")
    commit = args.source_commit or subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    try:
        out = apply(root, consent=args.consent, label=args.label, locked_at=args.locked_at, source_commit=commit)
    except ResealError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    gen = out["generation"]
    print(f"wrote {LOCK}, {PROFILES} ({gen.prev} -> {gen.new}, {gen.version}, {gen.overlay_version}), {CLIENT_TRUST}")
    print(f"verified: {out['verify']['current_profiles']} current profiles pin the LOCK digests")
    print("next: changelog, reseal table, v4 scorer tree digest rebind, research-agent resource baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
