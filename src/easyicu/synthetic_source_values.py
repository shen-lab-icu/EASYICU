"""Small, invented joint distributions; no fitted parameters or patient inputs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SEED = 20261026
CORE = frozenset({"map", "lact", "hgb", "crea", "k", "urine", "weight"})


def population(stays: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    index = np.arange(stays)
    patient = index - index // 5
    age = np.clip(rng.normal(64, 16, stays), 18, 95).round()[patient]
    severity = np.clip(rng.gamma(2, .65, stays) + (age - 60) / 100, .1, 4)
    weight = np.clip(rng.normal(78, 15, stays), 45, 125)[patient]
    male = rng.random(stays)[patient] > .45
    length = (48 + np.round(severity * 24 + rng.uniform(0, 48, stays))).astype(int)
    dead = rng.random(stays) < (.04 + severity * .06)
    late = ~dead & (rng.random(stays) < .09)
    dead[:3], late[:3] = [True, False, False], [False, True, False]
    # Repeated synthetic patients cannot die before their next admission.
    repeat = np.r_[patient[:-1] == patient[1:], False]
    dead[repeat] = False
    late[repeat] = False
    return dict(stay=index + 1, patient=patient + 1, age=age, severity=severity,
                weight=weight, male=male, length=length, dead=dead, late=late,
                origin=(index * 240 + 6) * 3_600_000)


def numeric_values(name, definition, pop, index, hour, rng):
    """Generate canonical-unit values with shared severity and recovery over time."""
    s = pop["severity"][index] * np.exp(-hour / 100)
    centres = {
        "map": 87 - 11 * s, "lact": 1 + 1.1 * s, "hgb": 13 - .8 * s,
        "crea": .7 + .65 * s, "k": 4 + .22 * s,
        "urine": 70 / (1 + .5 * s), "weight": pop["weight"][index],
        "height": np.full(len(index), 174.), "hr": 73 + 8 * s,
        "sbp": 120 - 10 * s, "dbp": 65 - 5 * s, "ph": 7.4 - .02 * s,
        "temp": 37 + .25 * s, "fio2": 30 + 5 * s,
        "spo2": 97 - s, "sao2": 97 - s, "o2sat": 97 - s,
        "egcs": np.full(len(index), 4.), "mgcs": np.full(len(index), 6.),
        "vgcs": np.full(len(index), 5.), "rass": np.zeros(len(index)),
    }
    lower = definition.get("min", 0)
    upper = definition.get("max", max(lower + 10, 20))
    centre = centres.get(name, np.full(len(index), lower + (upper - lower) * .25))
    noise = .02 * np.maximum(np.abs(centre), .1) * rng.normal(size=len(index))
    values = np.clip(centre + noise, lower, upper).astype(float)
    if name in {"egcs", "mgcs", "vgcs", "rass"}:
        values = np.round(values)
    if name in CORE:
        values[(index % 11 == 3) & (hour == 6)] = np.nan
        # Deliberate data-quality sentinels, not extreme clinical observations.
        values[(index == 0) & (hour == 0)] = {
            "map": 400, "lact": 70, "hgb": 70, "crea": 100,
            "k": 30, "urine": 90000, "weight": 900,
        }[name]
    return values


def raw_values(name, definition, source, pop, index, hour, rng):
    """Invert simple dictionary unit conversions so raw values use source units."""
    values = numeric_values(name, definition, pop, index, hour, rng)
    callback = source.get("callback", "")
    match = re.search(r"binary_op\(`([*/])`,\s*([0-9.]+)\)", callback)
    if match:
        factor = float(match[2])
        values = values / factor if match[1] == "*" else values * factor
    return values


def category_values(name, source, count, database):
    callback = source.get("callback", "")
    if name in {"vent_mode", "vent_breath_seq"}:
        return np.full(count, "1" if database == "hirid" else "PC ")
    if name == "rass":
        return np.full(count, "0")
    if name == "ett_gcs":
        return np.full(count, "Geïntubeerd")
    keys = re.findall(r"`([^`]+)`\s*=", callback)
    return np.resize(keys or ["synthetic"], count)


def binned(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(object)
