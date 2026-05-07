#!/usr/bin/env python3
"""Adversarial evaluation harness for ConceptUsageAuditor.

This script builds a synthetic ICU-aware ResearchContext, generates a
large set of safe and unsafe code snippets, and measures how often the
auditor catches the intended violations. It is designed as the seed of
the reviewer-facing red-team evaluation requested in the methods review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from easyicu.research_agent import ConceptUsageAuditor, build_research_context


def make_context():
    df = pd.DataFrame(
        {
            "stay_id": range(1, 21),
            "sofa2": [i % 6 for i in range(20)],
            "gcs": [15 - (i % 4) for i in range(20)],
            "lact": [1.0 + (i % 5) for i in range(20)],
            "death": [1 if i % 7 == 0 else 0 for i in range(20)],
        }
    )
    return build_research_context(
        research_question="red-team auditor eval",
        cohort=df,
        cohort_name="auditor_eval",
        database="synthetic",
        target_outcome="death",
    )


def make_cases() -> List[Dict[str, object]]:
    unsafe = []
    safe = []
    agg_templates = [
        ('df["{col}"].mean()', "error"),
        ('df.{col}.mean()', "error"),
        ('df["{col}"].std()', "error"),
        ('df["{col}"].agg("mean")', "error"),
        ('df["{col}"].rolling(3).mean()', "error"),
        ('np.mean(df["{col}"])', "error"),
    ]
    for col in ["sofa2", "gcs"]:
        for template, severity in agg_templates:
            unsafe.append(
                {
                    "name": f"{col}_{template.split('(')[0].replace('[', '_').replace('\"', '').replace('.', '_')}",
                    "code": "import numpy as np\nx = " + template.format(col=col),
                    "expected": severity,
                }
            )
    for i in range(40):
        unsafe.append(
            {
                "name": f"fillna_zero_{i}",
                "code": f'df["lact"] = df["lact"].fillna(0)\nmask = df["death"] == {i % 2}',
                "expected": "warning",
            }
        )
    for i in range(30):
        unsafe.append(
            {
                "name": f"lact_mean_no_median_{i}",
                "code": f'x = df["lact"].mean()\ny = {i}',
                "expected": "warning",
            }
        )
    safe_templates = [
        'x = df["sofa2"].max()',
        'x = df["gcs"].min()',
        'x = df["lact"].median()',
        'x = df["lact"].mean(); y = df["lact"].median()',
        'x = df["death"].mean()',
        'x = df["sofa2"].value_counts()',
    ]
    for i in range(40):
        safe.append(
            {
                "name": f"safe_{i}",
                "code": safe_templates[i % len(safe_templates)],
                "expected": "none",
            }
        )
    return unsafe + safe


def evaluate() -> Dict[str, object]:
    auditor = ConceptUsageAuditor()
    ctx = make_context()
    cases = make_cases()
    rows = []
    tp = fp = tn = fn = 0
    for case in cases:
        findings = auditor.audit(context=ctx, script_text=str(case["code"]))
        severities = {f.severity for f in findings}
        predicted = "none"
        if "error" in severities:
            predicted = "error"
        elif "warning" in severities:
            predicted = "warning"
        expected = str(case["expected"])
        if expected == "none":
            if predicted == "none":
                tn += 1
            else:
                fp += 1
        else:
            if predicted == "none":
                fn += 1
            else:
                tp += 1
        rows.append(
            {
                "name": case["name"],
                "expected": expected,
                "predicted": predicted,
                "n_findings": len(findings),
            }
        )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n_cases": len(cases),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate()
    (out_dir / "auditor_redteam_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(result["rows"]).to_csv(out_dir / "auditor_redteam_cases.csv", index=False)
    (out_dir / "auditor_redteam_summary.md").write_text(
        (
            "# ConceptUsageAuditor red-team evaluation\n\n"
            f"- cases: {result['n_cases']}\n"
            f"- precision: {result['precision']:.3f}\n"
            f"- recall: {result['recall']:.3f}\n"
            f"- tp/fp/tn/fn: {result['tp']}/{result['fp']}/{result['tn']}/{result['fn']}\n"
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
