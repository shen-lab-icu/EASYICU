"""Bounded UI-consistency checks for an isolated workspace page.

These receipts describe declared browser checks. They are neither scientific
verification nor proof that an operating system saved a download.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import secrets
from threading import RLock
import time
from typing import Any, NoReturn

SCHEMA = "easyicu.web-runtime-qa/1"
MAX_REPORT_BYTES = 24_576
MAX_CONTRACT_BYTES = 16_384
MAX_CHECKS = 64
KINDS = {"text_number", "ratio", "bar_geometry", "select_cases", "download_control"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class RuntimeQaError(ValueError):
    """A contract or current-preview receipt did not meet the bounded schema."""

    def __init__(self, code: str, *, issues: list[dict[str, Any]] | None = None):
        super().__init__(code)
        self.code = code
        self.details = {"issues": deepcopy(issues or []), "schema": SCHEMA}


def _number(value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeQaError("runtime_qa_number_invalid")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise RuntimeQaError("runtime_qa_number_invalid")
    if minimum is not None and value < minimum:
        raise RuntimeQaError("runtime_qa_number_invalid")
    return float(value)


def _selector(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(ord(c) < 32 for c in value):
        raise RuntimeQaError("runtime_qa_selector_invalid")
    return value


def validate_contract(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise RuntimeQaError("runtime_qa_contract_invalid") from exc
    if len(encoded) > MAX_CONTRACT_BYTES or not isinstance(value, dict):
        raise RuntimeQaError("runtime_qa_contract_invalid")
    if set(value) != {"schema", "checks"} or value["schema"] != SCHEMA:
        raise RuntimeQaError("runtime_qa_contract_invalid")
    ids: set[str] = set()
    issues: list[dict[str, Any]] = []

    def fail(code: str, path: str, expected: Any) -> NoReturn:
        raise RuntimeQaError(code, issues=[{"code": code, "path": path, "expected": expected}])

    def selector(value: Any, path: str) -> str:
        try:
            return _selector(value)
        except RuntimeQaError as exc:
            fail(exc.code, path, "one CSS selector string, 1-256 characters")

    def number(value: Any, path: str, *, minimum: float | None = None) -> float:
        try:
            return _number(value, minimum=minimum)
        except RuntimeQaError as exc:
            fail(exc.code, path, "finite JSON number" + (f" >= {minimum:g}" if minimum is not None else "") + "; never a formatted percentage string")

    def checks(rows: Any, path: str, *, nested: bool = False) -> list[dict[str, Any]]:
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            fail("runtime_qa_checks_invalid", path, "1-32 checks; at most 64 including cases")
        result = []
        for index, row in enumerate(rows):
            row_path = f"{path}/{index}"
            try:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not _ID.fullmatch(row["id"]):
                    fail("runtime_qa_check_id_invalid", row_path + "/id", "required unique id on every check, including select-case children; 1-96 letters/digits/_.:-")
                identifier = row["id"]
                if identifier in ids or len(ids) >= MAX_CHECKS:
                    fail("runtime_qa_check_id_invalid", row_path + "/id", "unique id across the whole contract; at most 64 checks")
                ids.add(identifier)
                kind = row.get("kind")
                if not isinstance(kind, str) or kind not in KINDS or (nested and kind == "select_cases"):
                    fail("runtime_qa_check_kind_invalid", row_path + "/kind", sorted(KINDS - ({"select_cases"} if nested else set())))
                clean = {"id": identifier, "kind": kind}
                if kind == "text_number":
                    allowed = {"id", "kind", "selector", "expected", "tolerance"}
                    clean.update(selector=selector(row.get("selector"), row_path + "/selector"), expected=number(row.get("expected"), row_path + "/expected"),
                                 tolerance=number(row.get("tolerance", 0.000001), row_path + "/tolerance", minimum=0))
                elif kind == "ratio":
                    allowed = {"id", "kind", "numerator", "denominator", "output", "scale", "tolerance"}
                    clean.update(**{name: selector(row.get(name), row_path + "/" + name) for name in ("numerator", "denominator", "output")},
                                 scale=number(row.get("scale", 100), row_path + "/scale"),
                                 tolerance=number(row.get("tolerance", 0.051), row_path + "/tolerance", minimum=0))
                elif kind == "bar_geometry":
                    allowed = {"id", "kind", "selector", "value_selector", "axis", "domain_max", "extent", "min_extent", "tolerance_px"}
                    axis = row.get("axis", "height")
                    domain = number(row.get("domain_max"), row_path + "/domain_max", minimum=0)
                    if not isinstance(axis, str) or axis not in {"height", "width"} or domain == 0:
                        fail("runtime_qa_bar_geometry_invalid", row_path, "axis height|width and domain_max > 0")
                    clean.update(selector=selector(row.get("selector"), row_path + "/selector"), value_selector=selector(row.get("value_selector"), row_path + "/value_selector"),
                                 axis=axis, domain_max=domain, extent=number(row.get("extent"), row_path + "/extent", minimum=0),
                                 min_extent=number(row.get("min_extent", 0), row_path + "/min_extent", minimum=0),
                                 tolerance_px=number(row.get("tolerance_px", 0.5), row_path + "/tolerance_px", minimum=0))
                elif kind == "select_cases":
                    allowed = {"id", "kind", "selector", "cases"}
                    cases = row.get("cases")
                    if not isinstance(cases, list) or not 1 <= len(cases) <= 6:
                        fail("runtime_qa_select_cases_invalid", row_path + "/cases", "1-6 cases, each with value and checks")
                    clean.update(selector=selector(row.get("selector"), row_path + "/selector"), cases=[])
                    values = set()
                    for case_index, case in enumerate(cases):
                        case_path = f"{row_path}/cases/{case_index}"
                        if not isinstance(case, dict) or set(case) != {"value", "checks"}:
                            fail("runtime_qa_select_cases_invalid", case_path, "exact fields value (string) and checks (leaf checks)")
                        value = case["value"]
                        if not isinstance(value, str) or len(value) > 80 or value in values:
                            fail("runtime_qa_select_cases_invalid", case_path + "/value", "unique option value string, at most 80 characters")
                        values.add(value)
                        clean["cases"].append({"value": value, "checks": checks(case["checks"], case_path + "/checks", nested=True)})
                else:
                    allowed = {"id", "kind", "selector"}
                    clean["selector"] = selector(row.get("selector"), row_path + "/selector")
                if set(row) - allowed:
                    fail("runtime_qa_check_fields_invalid", row_path, sorted(allowed))
                result.append(clean)
            except RuntimeQaError as exc:
                issues.extend(exc.details["issues"][:max(0, 16 - len(issues))])
        return result

    normalized = {"schema": SCHEMA, "checks": checks(value["checks"], "/checks")}
    if issues:
        raise RuntimeQaError("runtime_qa_contract_invalid", issues=issues)
    if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > MAX_CONTRACT_BYTES:
        raise RuntimeQaError("runtime_qa_contract_too_large")
    return normalized


class _ContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self.parts: list[str] = []
        self.found = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and "data-easyicu-runtime-qa" in values:
            if values.get("type", "").lower() != "application/json" or self.found:
                raise RuntimeQaError("runtime_qa_contract_duplicate_or_executable")
            self.found += 1
            self.active = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.active = False

    def handle_data(self, data: str) -> None:
        if self.active:
            self.parts.append(data)
            if sum(len(part.encode("utf-8")) for part in self.parts) > MAX_CONTRACT_BYTES:
                raise RuntimeQaError("runtime_qa_contract_too_large")


def contract_from_html(source: str) -> dict[str, Any] | None:
    parser = _ContractParser()
    parser.feed(source)
    if not parser.found:
        return None
    try:
        parsed = json.loads("".join(parser.parts))
    except (ValueError, TypeError, RecursionError) as exc:
        raise RuntimeQaError("runtime_qa_contract_invalid", issues=[{
            "code": "runtime_qa_contract_json_invalid", "path": "",
            "expected": "one valid application/json contract object",
        }]) from exc
    if parsed is None:
        raise RuntimeQaError("runtime_qa_contract_invalid", issues=[{
            "code": "runtime_qa_contract_invalid", "path": "", "expected": "contract object, not null",
        }])
    return validate_contract(parsed)


def contract_diagnostics(source: str) -> dict[str, Any]:
    """Static schema diagnostics only; no generated script or data execution."""
    guidance = {
        "schema": SCHEMA, "allowed_kinds": sorted(KINDS), "polling_allowed": False,
        "scope": "static_contract_schema_only",
        "field_rules": {
            "all_checks": "unique id required, including every select-case child",
            "text_number.expected": "finite JSON number; 25.0 checks displayed 25.0%, never use the string 25.0%",
            "select_cases.cases": "each case has value and checks; no nested select_cases",
        },
    }
    try:
        contract = contract_from_html(source)
    except RuntimeQaError as exc:
        issues = exc.details["issues"] or [{
            "code": exc.code, "path": "", "expected": f"exact root fields schema={SCHEMA} and checks; at most 16 KiB",
        }]
        return {**guidance, "status": "invalid", "code": "runtime_qa_contract_invalid", "issues": issues,
                "next_action": "repair_contract_schema_then_check_once",
                "instruction": "Fix the listed schema paths after an exact file read. Do not poll runtime status, change data/answers to silence checks, or claim browser validation."}
    if contract is None:
        return {**guidance, "status": "not_declared", "code": "runtime_qa_contract_missing", "issues": [],
                "next_action": "declare_contract_or_report_unverified",
                "instruction": "No contract is declared. Do not poll or claim charts, filters, or downloads verified."}
    return {"status": "valid", "schema": SCHEMA, "scope": "static_contract_schema_only",
            "declared_check_count": len(_check_ids(contract)), "issues": []}


def contract_preflight_receipt(source: str, source_sha256: str, *, observed: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Readback for old checked pages whose contract cannot start in a browser."""
    diagnostic = contract_diagnostics(source)
    if diagnostic["status"] == "valid":
        return None
    result = {"schema": SCHEMA, "source_sha256": source_sha256,
            "status": "failed" if diagnostic["status"] == "invalid" else "partial",
            "code": diagnostic["code"], "scope": "static_contract_schema_only", "browser_observed": False,
            "checks": [], "coverage": [], "contract_diagnostics": diagnostic,
            "next_action": diagnostic["next_action"], "polling_allowed": False,
            "csv_export_ready": False, "scientific_verified": False, "browser_download_verified": False}
    if diagnostic["status"] == "not_declared" and observed:
        # Missing declarations do not erase genuine bounded browser warnings
        # or the protected CSV owner's readiness observation for this version.
        result = {**result, **deepcopy(observed), "code": diagnostic["code"],
                  "contract_diagnostics": diagnostic, "polling_allowed": False,
                  "next_action": diagnostic["next_action"], "browser_observed": "runtime_errors" in observed}
        if result["status"] != "failed":
            result["status"] = "partial"
    return result


def _check_ids(contract: dict[str, Any] | None) -> dict[str, str]:
    result = {}
    for check in (contract or {}).get("checks", []):
        result[check["id"]] = check["kind"]
        for case in check.get("cases", []):
            for child in case["checks"]:
                result[child["id"]] = child["kind"]
    return result


def validate_report(value: Any, contract: dict[str, Any] | None, source_sha256: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise RuntimeQaError("runtime_qa_report_invalid") from exc
    if len(encoded) > MAX_REPORT_BYTES or not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise RuntimeQaError("runtime_qa_report_invalid")
    if value.get("source_sha256") != source_sha256:
        raise RuntimeQaError("runtime_qa_source_changed")
    expected = _check_ids(contract)
    rows = value.get("checks")
    if not isinstance(rows, list) or len(rows) > MAX_CHECKS:
        raise RuntimeQaError("runtime_qa_report_invalid")
    found = set()
    clean = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] not in expected or row["id"] in found:
            raise RuntimeQaError("runtime_qa_report_check_invalid")
        if row.get("kind") != expected[row["id"]] or not isinstance(row.get("status"), str) or row["status"] not in {"passed", "failed", "pending"}:
            raise RuntimeQaError("runtime_qa_report_check_invalid")
        code = row.get("code", "")
        if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", code):
            raise RuntimeQaError("runtime_qa_report_check_invalid")
        item = {name: row[name] for name in ("id", "kind", "status", "code")}
        if item["kind"] == "download_control" and item["status"] == "passed":
            # A document can observe a button, not prove that the protected
            # Host transport accepted CSV bytes. Only note_csv_ready may do so.
            item.update(status="pending", code="csv_export_not_yet_observed")
        for key in ("observed", "expected"):
            if key in row:
                item[key] = _number(row[key])
        found.add(row["id"])
        clean.append(item)
    errors = value.get("runtime_errors", 0)
    if isinstance(errors, bool) or not isinstance(errors, int) or not 0 <= errors <= 100:
        raise RuntimeQaError("runtime_qa_report_invalid")
    warnings = value.get("warnings", [])
    if not isinstance(warnings, list) or len(warnings) > 16:
        raise RuntimeQaError("runtime_qa_report_invalid")
    if any(not isinstance(code, str) or code not in {"suspicious_percent_bar", "non_finite_display"} for code in warnings):
        raise RuntimeQaError("runtime_qa_report_invalid")
    status = "failed" if errors or any(row["status"] == "failed" for row in clean) else "complete"
    if status != "failed" and (not expected or found != set(expected) or warnings or any(row["status"] == "pending" for row in clean)):
        status = "partial"
    return {"schema": SCHEMA, "source_sha256": source_sha256, "status": status,
            "scope": "declared_ui_consistency_checks", "checks": clean,
            "declared_check_count": len(expected), "observed_check_count": len(found),
            "missing_check_ids": sorted(set(expected) - found),
            "coverage": sorted({row["kind"] for row in clean}), "runtime_errors": errors,
            "warnings": warnings, "csv_export_ready": False, "browser_download_verified": False,
            "scientific_verified": False}


class RuntimeQaStore:
    """Ephemeral, version-bound UI receipts; callers recheck workspace bytes."""

    def __init__(self, *, max_entries: int = 64, ttl_seconds: float = 1_800, clock=time.monotonic):
        self._rows: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
        self._lock = RLock()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._clock = clock

    def _prune(self) -> None:
        now = self._clock()
        for key in list(self._rows):
            if self._rows[key]["expires"] <= now:
                del self._rows[key]

    def begin(self, project_id: str, file: str, sha256: str, contract: Any = None) -> dict[str, Any]:
        if (not isinstance(project_id, str) or not 1 <= len(project_id) <= 160
                or not isinstance(file, str) or not 1 <= len(file) <= 240
                or not isinstance(sha256, str) or not _SHA.fullmatch(sha256)):
            raise RuntimeQaError("runtime_qa_source_invalid")
        normalized = validate_contract(contract)
        with self._lock:
            self._prune()
            key = (project_id, file, sha256)
            previous = self._rows.get(key)
            history = list(previous.get("failure_history", [])) if previous else []
            if previous and previous["report"] and previous["report"]["status"] == "failed":
                history.append(previous["report"])
            row = {"nonce": secrets.token_hex(32), "contract": normalized,
                   "expires": self._clock() + self._ttl, "report": None, "csv_ready": False,
                   "failure_history": history[-4:]}
            self._rows[key] = row
            self._rows.move_to_end(key)
            while len(self._rows) > self._max_entries:
                self._rows.popitem(last=False)
            return {"nonce": row["nonce"], "file": file, "source_sha256": sha256,
                    "status": "partial", "code": "browser_runtime_pending", "contract": deepcopy(normalized),
                    "scientific_verified": False, "browser_download_verified": False}

    def request(self, project_id: str, file: str, sha256: str, nonce: str) -> dict[str, Any] | None:
        """Authorize injection for exactly this current opaque preview request."""
        with self._lock:
            self._prune()
            row = self._rows.get((project_id, file, sha256))
            if not row or not isinstance(nonce, str) or not _SHA.fullmatch(nonce) or not secrets.compare_digest(row["nonce"], nonce):
                raise RuntimeQaError("runtime_qa_request_stale")
            return deepcopy(row["contract"])

    def record(self, project_id: str, file: str, sha256: str, nonce: str, report: Any) -> dict[str, Any]:
        with self._lock:
            self._prune()
            row = self._rows.get((project_id, file, sha256))
            if not row or not isinstance(nonce, str) or not _SHA.fullmatch(nonce) or not secrets.compare_digest(row["nonce"], nonce):
                raise RuntimeQaError("runtime_qa_request_stale")
            # A controlled frame can reload with the same preview nonce. Keep
            # failures seen during this request even if a later replay passes.
            normalized = validate_report(report, row["contract"], sha256)
            previous = row["report"]
            if previous and previous["status"] == "failed":
                row["failure_history"] = row.get("failure_history", [])[-3:] + [previous]
            row["report"] = normalized
            return self._public(row)

    def note_csv_ready(self, project_id: str, file: str, sha256: str) -> None:
        """Call only after the checked Host CSV endpoint accepted exact bytes."""
        with self._lock:
            self._prune()
            row = self._rows.get((project_id, file, sha256))
            if row:
                row["csv_ready"] = True

    def _public(self, row: dict[str, Any]) -> dict[str, Any]:
        report = deepcopy(row["report"])
        if report is None:
            return {"status": "partial", "code": "browser_runtime_pending", "scope": "declared_ui_consistency_checks",
                    "checks": [], "coverage": [], "scientific_verified": False,
                    "browser_download_verified": False, "csv_export_ready": row["csv_ready"],
                    "previous_failed_runs": len(row.get("failure_history", [])),
                    "polling_allowed": False, "next_action": "open_preview_once_then_report_browser_pending",
                    "instruction": "Do not repeat status reads or edit expected values while browser evidence is pending."}
        report["previous_failed_runs"] = len(row.get("failure_history", []))
        report["previous_failures"] = [
            {"runtime_errors": item["runtime_errors"],
             "checks": [check for check in item["checks"] if check["status"] == "failed"]}
            for item in row.get("failure_history", [])
        ]
        if row["csv_ready"]:
            report["csv_export_ready"] = True
            for check in report["checks"]:
                if check["kind"] == "download_control" and check["status"] == "pending":
                    check.update(status="passed", code="csv_ready_to_save")
            if report["status"] != "failed" and row["contract"] and not report["warnings"]:
                if len(report["checks"]) == len(_check_ids(row["contract"])) and all(check["status"] == "passed" for check in report["checks"]):
                    report["status"] = "complete"
        if report["status"] == "partial":
            report.update(polling_allowed=False, next_action="report_unverified_coverage_or_wait_for_user_browser_action")
        return report

    def read(self, project_id: str, file: str, sha256: str) -> dict[str, Any]:
        with self._lock:
            self._prune()
            row = self._rows.get((project_id, file, sha256))
            return self._public(row) if row else {"status": "partial", "code": "browser_runtime_pending",
                   "scope": "declared_ui_consistency_checks", "checks": [], "coverage": [], "scientific_verified": False,
                   "browser_download_verified": False, "csv_export_ready": False,
                   "polling_allowed": False, "next_action": "open_preview_once_then_report_browser_pending",
                   "instruction": "Do not repeat status reads or edit expected values while browser evidence is pending."}


def runtime_qa_bridge(nonce: str, source_sha256: str, contract: Any = None) -> tuple[str, str]:
    if not isinstance(nonce, str) or not isinstance(source_sha256, str) or not _SHA.fullmatch(nonce) or not _SHA.fullmatch(source_sha256):
        raise RuntimeQaError("runtime_qa_source_invalid")
    config = {"nonce": nonce, "source_sha256": source_sha256, "contract": validate_contract(contract)}
    encoded = json.dumps(config, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    probe = Path(__file__).parents[1] / "static" / "js" / "workspace-runtime-probe.js"
    child = "<script>" + probe.read_text(encoding="utf-8").replace("/*EASYICU_RUNTIME_QA_CONFIG*/null", encoded) + "</script>"
    wrapper = """<script>(function(){'use strict';
      const config=CONFIG, frame=document.getElementById('easyicu-workspace-preview-content');
      let delivered=false;
      addEventListener('message',function(event){const data=event.data;
        if(delivered || event.source!==frame.contentWindow || event.origin!=='null' || !data || data.type!=='easyicu:runtime-qa-child' || data.nonce!==config.nonce || !data.report || data.report.source_sha256!==config.source_sha256) return;
        try{if(new TextEncoder().encode(JSON.stringify(data.report)).length>24576)return;}catch(_){return;}
        delivered=true;parent.postMessage({type:'easyicu:runtime-qa',nonce:config.nonce,report:data.report},'*');
      });})();</script>""".replace("CONFIG", json.dumps({"nonce": nonce, "source_sha256": source_sha256}))
    return child, wrapper
