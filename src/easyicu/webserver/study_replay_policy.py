"""Fail-closed disclosure rules for a public, aggregate-only study replay.

The external figure-egress floor applies to every source, including fixtures.
An unsafe cell suppresses the entire export: replacing one cell can disclose it
through a total, percentage, prose or an already rendered figure.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
import re
from typing import Any

from easyicu.research_agent.gates.figure_privacy import (
    IDENTIFIER_COLUMNS,
    MIN_DISCLOSED_GROUP_SIZE,
    TIME_COLUMNS,
)
from easyicu.research_agent.gates.publication_disclosure import is_subject_count_name


class ReplayExportError(ValueError):
    """A stable code only; never include the rejected content in diagnostics."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def key_name(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


_SAFE_IDS = {
    "evidence_id",
    "figure_id",
    "step_id",
    "model_id",
    "panel_id",
    "claim_id",
    "run_id",
    "study_id",
    "analysis_id",
    "contract_id",
    "schema_id",
}
_PRIVATE_KEYS = re.compile(
    r"(?:provider|credential|password|secret|token|authorization|(?:^|_)auth(?:$|_)|api_key|apikey|"
    r"private_key|cookie|environment|environ|headers|data_url|(?:^|_)path(?:$|_)|"
    r"code|sql|raw_output|stdout|stderr)",
    re.I,
)
_PATH = re.compile(
    r'(?:file://[^\s<>"\']+|[A-Za-z]:[\\/][^\s<>"\']+|\\\\[^\s<>"\']+|'
    r'(?<![\w:/])(?:~[^/\s]*)?/[^\s<>"\']+)',
)
_SECRET = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----|"
    r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+|"
    r"\b(?:sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9]+|AKIA[A-Z0-9]{16})|"
    r"\b(?:[A-Za-z0-9_]*(?:api[_-]?key|password|secret|token|credential))\s*[:=]\s*"
    r'(?:"[^"]*"|\'[^\']*\'|[^\s,;]+)',
    re.I | re.S,
)
_URL = re.compile(r'\b[a-z][a-z0-9+.-]*://[^\s<>"\']+', re.I)
_IDENTITY_TEXT = re.compile(
    r"\b(?:subject|patient|stay|hadm|icustay|mrn|admission|person|record)"
    r'(?:[ _-]*(?:id|number|identifier|no))?[\s:=_-]+["\']?[A-Za-z0-9-]*\d[A-Za-z0-9-]*|'
    r"\b(?:subject_id|patient_id|stay_id|hadm_id|mrn)\s*[:=]\s*\S+|"
    r"(?<![\d.])\d{6,}(?![\d.])",
    re.I,
)
_COUNT_TEXT = re.compile(
    r"\b(?:n|count|events|patients|subjects|stays|participants|deaths)\s*[:=]\s*([\d,.]+)|"
    r"(?<![\w.])([\d,.]+)\s*(?:patients|subjects|stays|participants|deaths|events|例|人)\b",
    re.I,
)
_SMALL_WORD_COUNT = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\s+"
    r"(?:patients?|subjects?|stays?|participants?|deaths?|events?)\b|"
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:例|人)",
    re.I,
)
_ROW_KEYS = {
    "rows",
    "table_rows",
    "tablerows",
    "records",
    "raw_data",
    "patient_rows",
    "observations",
}
_SECRET_KEY = re.compile(
    r"api_?key|password|secret|token|credential|private_key|provider_key"
)


def secret_values(value: Any, *, private: bool = False, depth: int = 0) -> set[str]:
    """Find operational secrets already in a record, without opening key files."""
    if depth > 24:
        raise ReplayExportError("replay_size_limit")
    if isinstance(value, dict):
        return set().union(
            *(
                secret_values(
                    item,
                    private=private or bool(_SECRET_KEY.search(key_name(key))),
                    depth=depth + 1,
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, list):
        return set().union(
            *(secret_values(item, private=private, depth=depth + 1) for item in value)
        )
    return {value} if private and isinstance(value, str) and value else set()


def scrub_secret_values(value: Any, secrets: set[str]) -> Any:
    """Scrub even opaque keys echoed into an otherwise scientific string."""
    if isinstance(value, str):
        for secret in sorted(secrets, key=lambda item: (-len(item), item)):
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, list):
        return [scrub_secret_values(item, secrets) for item in value]
    if isinstance(value, dict):
        if any(scrub_secret_values(key, secrets) != key for key in value):
            raise ReplayExportError("replay_credential_in_identity")
        return {key: scrub_secret_values(item, secrets) for key, item in value.items()}
    return value


def identifier_name(name: str) -> bool:
    key = key_name(name)
    return (
        key in IDENTIFIER_COLUMNS
        or key.replace("_", "")
        in {item.replace("_", "") for item in IDENTIFIER_COLUMNS}
        or key
        in {"id", "ids", "patient", "patients", "patient_rows", "patientrows", "series"}
        or (key.endswith(("_id", "_ids")) and key not in _SAFE_IDS)
        or bool(
            re.search(
                r"(patient|subject|stay|person|case|record|admission)"
                r".*_(number|no|num|code|key|identifier)$",
                key,
            )
        )
    )


def subject_count(value: Any) -> int:
    try:
        number = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, ValueError):
        raise ReplayExportError("replay_aggregate_unproven") from None
    if (
        not number.is_finite()
        or number < 0
        or number > 10**12
        or number != number.to_integral_value()
    ):
        raise ReplayExportError("replay_aggregate_unproven")
    if 0 < number < MIN_DISCLOSED_GROUP_SIZE:
        raise ReplayExportError("replay_small_cell_forbidden")
    return int(number)


def public_text(value: Any, *, magnitude: bool = False) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ReplayExportError("replay_record_invalid")
    text = str(value)
    if len(text) > 100_000:
        raise ReplayExportError("replay_size_limit")
    text = _SECRET.sub("[redacted]", text)
    # URLs can carry credentials in a path, fragment or query; omit them whole.
    text = _URL.sub("[link omitted]", text)
    text = _PATH.sub("[path omitted]", text)
    if not magnitude and _IDENTITY_TEXT.search(text):
        raise ReplayExportError("replay_patient_level_forbidden")
    for match in _COUNT_TEXT.finditer(text):
        subject_count(match.group(1) or match.group(2))
    for match in _SMALL_WORD_COUNT.finditer(text):
        if match.group(1) is None:
            raise ReplayExportError("replay_small_cell_forbidden")
        subject_count(match.group(1))
    return text


def project_tree(value: Any, *, depth: int = 0, name: str = "") -> Any:
    """Project scientific JSON, dropping operational keys at every level."""
    if depth > 24:
        raise ReplayExportError("replay_size_limit")
    if isinstance(value, dict):
        if len(value) > 256:
            raise ReplayExportError("replay_size_limit")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReplayExportError("replay_record_invalid")
            normalized = key_name(key)
            if _PRIVATE_KEYS.search(normalized) and normalized not in {
                "code",
                "finding_codes",
                "action_code",
                "requires_user_authorization",
                "authorization_question",
            }:
                continue
            if identifier_name(key) or normalized in _ROW_KEYS | TIME_COLUMNS:
                raise ReplayExportError("replay_patient_level_forbidden")
            clean_key = public_text(key)
            if clean_key != key:
                raise ReplayExportError("replay_record_invalid")
            result[key] = project_tree(item, depth=depth + 1, name=normalized)
        return result
    if isinstance(value, list):
        if len(value) > 1000:
            raise ReplayExportError("replay_size_limit")
        return [project_tree(item, depth=depth + 1, name=name) for item in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        raise ReplayExportError("replay_record_invalid")
    if is_subject_count_name(name):
        subject_count(value)
    if isinstance(value, str):
        if (
            name in _SAFE_IDS
            and re.fullmatch(r"[A-Za-z0-9_.:-]{0,160}", value)
            and not _SECRET.search(value)
        ):
            return value
        if name.endswith("_sha256") and re.fullmatch(r"[a-f0-9]{64}", value):
            return value
        return public_text(value, magnitude=is_subject_count_name(name))
    if isinstance(value, (int, float)):
        public_text(value, magnitude=is_subject_count_name(name))
        return value
    raise ReplayExportError("replay_record_invalid")


def aggregate_table(headers: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    if (
        not headers
        or len(headers) > 64
        or len(set(headers)) != len(headers)
        or len(rows) > 1000
        or not rows
        or any(len(row) != len(headers) for row in rows)
    ):
        raise ReplayExportError("replay_table_invalid")
    names = [key_name(name) for name in headers]
    if any(
        identifier_name(name)
        or name in TIME_COLUMNS
        or name.endswith(("_time", "_datetime"))
        for name in names
    ):
        raise ReplayExportError("replay_patient_level_forbidden")
    counts = [index for index, name in enumerate(names) if is_subject_count_name(name)]
    if not counts:
        raise ReplayExportError("replay_aggregate_unproven")
    clean_rows = []
    for row in rows:
        for index in counts:
            subject_count(row[index])
        clean = []
        for index, value in enumerate(row):
            if isinstance(value, (list, dict)) or value is None:
                raise ReplayExportError("replay_table_invalid")
            text = public_text(value, magnitude=index in counts)
            if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text):
                raise ReplayExportError("replay_patient_level_forbidden")
            clean.append(text)
        clean_rows.append(clean)
    return {
        "renderer": "table",
        "headers": [public_text(name) for name in headers],
        "rows": clean_rows,
        "rows_truncated": False,
        "columns_truncated": False,
    }
