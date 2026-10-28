"""Sequential walk orchestration; all product operations cross the HTTP API."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import PHASES, ItemResult, Status, WalkStop
from .conversations import ConversationWalk
from .http import HTTPFailure, HttpAPI, normalize_base
from .phases import MetadataWalk
from .privacy import token
from .report import ReceiptStore
from .scope import check_request


@dataclass(frozen=True)
class Options:
    source: str
    out: Path
    base: str = "http://127.0.0.1:8780"
    only: tuple[str, ...] = PHASES
    max_workflows: int | None = None
    resume: bool = False
    approve_test_runs: bool = False
    timeouts: dict[str, float] = field(default_factory=lambda: {
        "environment": 60, "mapping": 180, "catalogue": 120,
        "literature": 600, "ideas": 120, "workflows": 1800, "interactions": 300,
    })
    request_timeout: float = 30
    poll_interval: float = 1

    def __post_init__(self):
        normalize_base(self.base)
        if not self.source.strip() or not self.only or set(self.only) - set(PHASES):
            raise ValueError("source_and_known_phases_required")
        if self.max_workflows is not None and self.max_workflows < 1:
            raise ValueError("max_workflows_must_be_positive")
        for number in [*self.timeouts.values(), self.request_timeout, self.poll_interval]:
            if not math.isfinite(number) or number <= 0:
                raise ValueError("timeouts_must_be_positive_finite")


class WalkRunner(MetadataWalk, ConversationWalk):
    def __init__(self, options: Options, *, api=None):
        self.options = options
        root = Path(__file__).resolve().parents[2]
        out = options.out.expanduser().resolve()
        if out == root or root in out.parents:
            raise ValueError("output_must_be_outside_repository")
        self.api = api or HttpAPI(options.base)
        self.store = ReceiptStore(out, f"{normalize_base(options.base)}/{options.source}", resume=options.resume)
        self.results: list[ItemResult] = []
        self._source = None
        self._catalogue = None
        self._workflows = None
        self._deadline = time.monotonic() + options.request_timeout

    def request(self, method, path, body=None, *, params=None):
        if method != "GET":
            # A second guard at the network boundary also covers future callers.
            if path == "/api/jobs/agent-run-review" and not self.options.approve_test_runs:
                raise WalkStop(Status.REFUSED, "approval_flag_required")
            if path == "/api/jobs/agent-run" and (body or {}).get("candidate_plan_only") is not True:
                raise WalkStop(Status.REFUSED, "candidate_plan_only_required")
        check_request(method, path, body, params, self.store.state)
        timeout = min(self.options.request_timeout, self._deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("phase_timeout")
        payload = self.api.request(method, path, body, params=params, timeout=timeout)
        if payload.get("ok") is False:
            raise WalkStop(Status.REFUSED if payload.get("blocked") else Status.FAILED,
                           token(payload.get("error"), "api_operation_refused"))
        return payload

    def pause(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("phase_timeout")
        time.sleep(min(self.options.poll_interval, remaining, 5))

    def item(self, phase, identity, action, *, execution_policy=False, reuse=True):
        policy = "approved" if execution_policy and self.options.approve_test_runs else "plan-only"
        if phase in {"workflows", "literature"}:
            policy += f":limit={self.options.max_workflows}"
        previous = self.store.finished(phase, identity, policy=policy) if self.options.resume and reuse else None
        if previous is not None:
            self.results.append(previous)
            self.store.write_report(self.results)
            return previous
        started, tick = time.time(), time.monotonic()
        self._deadline = tick + self.options.timeouts.get(phase, 120)
        status, reason, metrics = Status.PASSED, "", {}
        try:
            metrics = action() or {}
        except WalkStop as exc:
            status, reason, metrics = exc.status, exc.code, exc.metrics
        except TimeoutError:
            status, reason = Status.TIMED_OUT, "phase_timeout"
        except HTTPFailure as exc:
            status = Status.REFUSED if exc.status in {401, 403, 409, 422, 429, 503} else Status.FAILED
            reason = exc.code
        except Exception:
            # Exceptions may contain paths, identifiers, model replies or rows.
            # Retain a local reason code, never the exception's text/traceback.
            status, reason = Status.FAILED, "walk_protocol_error"
        result = ItemResult(phase, identity, status, reason, started, round(time.monotonic() - tick, 4), metrics)
        self.results.append(result)
        self.store.save(result, policy=policy)
        self.store.write_report(self.results)
        return result

    def source(self):
        if self._source is None:
            registry = self.request("GET", "/api/workspaces/registry")
            selector = self.options.source.casefold()
            rows = [row for row in registry.get("sources", []) if isinstance(row, dict)
                    and row.get("ok") is True and selector in {
                        str(row.get(key, "")).casefold() for key in ("id", "label", "name")}]
            if len(rows) != 1:
                raise WalkStop(Status.REFUSED, "source_ambiguous" if rows else "configured_source_not_found")
            self._source = {key: rows[0].get(key) for key in ("id", "database")}
            if any(token(v) == "not_reported" for v in self._source.values()):
                raise WalkStop(Status.REFUSED, "source_identity_invalid")
            if self.store.state.get("source_id") and (
                    self.store.state["source_id"] != self._source["id"]
                    or self.store.state.get("database") != self._source["database"]):
                raise WalkStop(Status.REFUSED, "resume_source_binding_changed")
            self.store.state.update(source_id=self._source["id"], database=self._source["database"])
            self.store.flush()
        return self._source

    def catalogues(self):
        if self._catalogue is None:
            payload = self.request("GET", "/api/capabilities")
            self._catalogue = (payload.get("capabilities") or {}).get("method_skills") or {}
            self._workflows = self.request("GET", "/api/capabilities/research-workflows").get("items", [])
            if not isinstance(self._catalogue.get("items"), list) or not isinstance(self._workflows, list):
                raise WalkStop(Status.FAILED, "catalogue_contract_missing")
        return self._catalogue, self._workflows

    def run(self) -> dict:
        # A configured name can be reassigned. Revalidate it before trusting receipts.
        self._deadline = time.monotonic() + self.options.timeouts.get("environment", 60)
        source_failure = None
        try:
            self.source()
        except (HTTPFailure, WalkStop, TimeoutError) as exc:
            source_failure = exc
        if source_failure is not None:
            def failed_source():
                raise source_failure
            self.item("environment", "selected-source", failed_source, reuse=False)
            return self.store.write_report(self.results)
        source_result = self.item("environment", "selected-source", lambda: {
            "source": self.source()["id"], "database": self.source()["database"],
        })
        if source_result.status != Status.PASSED:
            return self.store.write_report(self.results)
        phases = set(self.options.only)
        if "environment" in phases:
            self.environment_items()
        if "mapping" in phases:
            self.item("mapping", "dictionary-availability", self.mapping)
            self.item("mapping", "extraction-smoke", self.extraction_smoke)
        if "catalogue" in phases:
            self.item("catalogue", "inventory", self.catalogue_inventory)
        examples = []
        if phases & {"literature", "workflows"}:
            def enumerate_examples():
                catalogue, workflows = self.catalogues()
                seen = set()
                for row in [*workflows, *catalogue.get("components", [])]:
                    identity = token(row.get("id"))
                    if identity == "not_reported" or identity in seen:
                        raise WalkStop(Status.FAILED, "catalogue_identity_invalid")
                    seen.add(identity)
                    examples.append({"id": identity, "workflow": row in workflows,
                                     "example_question": row.get("example_question")})
                return {"item_count": len(examples)}
            # Enumeration is a dependency and is always refreshed, even on resume.
            discovery = self.item("catalogue", "example-discovery", enumerate_examples, reuse=False)
            if discovery.status != Status.PASSED:
                examples.clear()
        if "ideas" in phases:
            self.item("ideas", "synthetic-abstract", self.ideas)
        accepted = 0
        for example in examples:
            has_example = example["workflow"] or bool(example.get("example_question"))
            limited = has_example and self.options.max_workflows is not None and accepted >= self.options.max_workflows
            if has_example and not limited:
                accepted += 1
            for phase in ("literature", "workflows"):
                if phase not in phases or phase == "literature" and not example["workflow"]:
                    continue
                def act(example=example, phase=phase, limited=limited, has_example=has_example):
                    if limited or not has_example:
                        raise WalkStop(Status.SKIPPED, "max_workflows_reached" if limited else "runnable_example_not_exposed")
                    return self.walk_example(example, phase=phase)
                self.item(phase, example["id"], act, execution_policy=phase == "workflows")
        if "interactions" in phases:
            self.interaction_items()
        return self.store.write_report(self.results)
