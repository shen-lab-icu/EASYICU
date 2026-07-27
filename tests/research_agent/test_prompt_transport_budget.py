"""The prompt envelope must bind every consumer of a shared role transport.

Historically each agent class checked its own prompt before calling, so a
consumer that was not one of those classes sent unmeasured payloads. Real E1
transport receipts recorded ``analyzer`` prompts of 53,393 and 78,401 bytes
against a declared 48,000-byte ceiling and a ``planner`` prompt of 101,878
bytes against a declared 80,000 -- all delivered. These tests lock the fix at
both levels: the wrapper enforces, and no call site may quietly skip it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from easyicu.research_agent.providers.prompt_budget import (
    BUDGETED_ROLES,
    DEFAULT_MAX_PROMPT_TOKENS,
    CONSERVATIVE_BYTES_PER_TOKEN,
    OBSERVED_BYTES_PER_TOKEN,
    PROMPT_TRANSPORT_BUDGETS,
    PromptBudgetClient,
    PromptTransportBudgetError,
    UndeclaredPromptConsumerError,
    active_prompt_consumer,
    budgeted_role_client,
    declared_consumers_for_role,
    estimate_prompt_tokens,
    prompt_payload_bytes,
)


class _Message:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class _RecordingClient:
    """Stand-in transport that records what it was actually handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.seen_consumer: list[object] = []

    def complete(self, messages, **kwargs):
        self.calls.append(("complete", messages))
        self.seen_consumer.append(active_prompt_consumer.get())
        return "ok"

    def complete_with_usage(self, messages, **kwargs):
        self.calls.append(("complete_with_usage", messages))
        self.seen_consumer.append(active_prompt_consumer.get())
        return "ok", {"prompt_tokens": 1}

    def complete_with_images(self, *, prompt, image_paths, **kwargs):
        self.calls.append(("complete_with_images", prompt))
        self.seen_consumer.append(active_prompt_consumer.get())
        return "{}"


def _messages(total_bytes: int) -> list[_Message]:
    return [_Message("user", "x" * total_bytes)]


def _bytes_for_tokens(tokens: int) -> int:
    """Smallest byte payload whose estimate reaches ``tokens``."""

    size = int(tokens * CONSERVATIVE_BYTES_PER_TOKEN)
    while estimate_prompt_tokens(size) < tokens:
        size += 1
    while estimate_prompt_tokens(size - 1) >= tokens:
        size -= 1
    return size


def _over_budget(budget) -> list[_Message]:
    return _messages(_bytes_for_tokens(budget.limit_tokens + 1))


def _at_budget(budget) -> list[_Message]:
    return _messages(_bytes_for_tokens(budget.limit_tokens))


def _resolver(client):
    return lambda role: client


# ---------------------------------------------------------------------------
# The wrapper enforces, for every transport entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("consumer", sorted(PROMPT_TRANSPORT_BUDGETS))
def test_every_declared_consumer_is_enforced_on_complete(consumer: str) -> None:
    budget = PROMPT_TRANSPORT_BUDGETS[consumer]
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), budget.role, consumer)

    with pytest.raises(PromptTransportBudgetError) as excinfo:
        client.complete(_over_budget(budget))

    assert excinfo.value.consumer == consumer
    assert excinfo.value.limit_tokens == budget.limit_tokens
    assert excinfo.value.actual_tokens == budget.limit_tokens + 1
    # Fail closed means the payload never reached the provider.
    assert inner.calls == []


@pytest.mark.parametrize("consumer", sorted(PROMPT_TRANSPORT_BUDGETS))
def test_every_declared_consumer_is_enforced_on_complete_with_usage(
    consumer: str,
) -> None:
    budget = PROMPT_TRANSPORT_BUDGETS[consumer]
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), budget.role, consumer)

    with pytest.raises(PromptTransportBudgetError):
        client.complete_with_usage(_over_budget(budget))

    assert inner.calls == []


def test_a_prompt_at_the_limit_is_delivered() -> None:
    budget = PROMPT_TRANSPORT_BUDGETS["concept_audit"]
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), budget.role, "concept_audit")

    assert client.complete(_at_budget(budget)) == "ok"
    assert [name for name, _ in inner.calls] == ["complete"]


def test_the_error_names_the_consumer_not_only_the_role() -> None:
    """The role alone is what made the original breach unattributable."""

    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), "analyzer", "concept_audit")
    with pytest.raises(PromptTransportBudgetError) as excinfo:
        client.complete(_over_budget(PROMPT_TRANSPORT_BUDGETS["concept_audit"]))

    rendered = str(excinfo.value)
    assert "concept_audit" in rendered
    assert "analyzer" in rendered


# ---------------------------------------------------------------------------
# Sized against real traffic, not against a hopeful number
# ---------------------------------------------------------------------------
#
# The old ceilings sat *below* what this system normally produces, which is why
# they kept tripping. The guard exists to catch a projection that has run away,
# so real observed prompts must pass and only genuine runaway must fail.


@pytest.mark.parametrize(
    "payload_bytes,provider_tokens,role,consumer",
    [
        # Every one of these is a real completed transport receipt from the
        # 2026-07-23 E1 replay, with the provider's own prompt_tokens count.
        (101_878, 26_040, "planner", "legacy_model_roster_migration"),
        (78_401, 20_804, "analyzer", "analyzer_interpretation"),
        (66_119, 17_088, "planner", "cohort_extraction"),
        (53_393, 13_988, "analyzer", "vlm_visual_qa"),
    ],
)
def test_real_observed_prompts_are_not_refused(
    payload_bytes: int, provider_tokens: int, role: str, consumer: str
) -> None:
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), role, consumer)

    assert client.complete(_messages(payload_bytes)) == "ok"
    # And the estimate must not have under-counted what the provider metered.
    assert estimate_prompt_tokens(payload_bytes) >= provider_tokens


def test_the_default_ceiling_clears_the_largest_prompt_ever_produced() -> None:
    """26,040 provider-counted tokens is the high-water mark on record."""

    assert DEFAULT_MAX_PROMPT_TOKENS > 26_040


def test_a_genuinely_runaway_projection_is_still_refused() -> None:
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), "planner", "cohort_extraction")

    with pytest.raises(PromptTransportBudgetError):
        client.complete(_messages(_bytes_for_tokens(DEFAULT_MAX_PROMPT_TOKENS + 1)))
    assert inner.calls == []


# ---------------------------------------------------------------------------
# The ceiling is configuration, not a constant welded into the code
# ---------------------------------------------------------------------------


def test_an_operator_supplied_ceiling_replaces_the_default() -> None:
    inner = _RecordingClient()
    client = budgeted_role_client(
        _resolver(inner), "analyzer", "concept_audit", limit_tokens=1_000
    )

    assert client.limit_tokens == 1_000
    with pytest.raises(PromptTransportBudgetError):
        client.complete(_messages(_bytes_for_tokens(1_001)))


def test_raising_the_ceiling_admits_a_previously_refused_payload() -> None:
    payload = _messages(_bytes_for_tokens(DEFAULT_MAX_PROMPT_TOKENS + 5_000))

    tight = budgeted_role_client(
        _resolver(_RecordingClient()), "analyzer", "concept_audit"
    )
    with pytest.raises(PromptTransportBudgetError):
        tight.complete(payload)

    roomy = budgeted_role_client(
        _resolver(_RecordingClient()),
        "analyzer",
        "concept_audit",
        limit_tokens=DEFAULT_MAX_PROMPT_TOKENS + 10_000,
    )
    assert roomy.complete(payload) == "ok"


def test_the_pipeline_config_exposes_the_ceiling() -> None:
    from easyicu.research_agent.orchestration.config import PipelineConfig

    assert PipelineConfig.max_prompt_tokens_per_call == DEFAULT_MAX_PROMPT_TOKENS
    raised = PipelineConfig(workdir=".").with_overrides(
        max_prompt_tokens_per_call=120_000
    )
    assert raised.max_prompt_tokens_per_call == 120_000


def test_the_error_says_the_ceiling_is_not_the_model_window() -> None:
    """Blocking without saying whose limit it is is what wasted past effort."""

    client = budgeted_role_client(
        _resolver(_RecordingClient()), "analyzer", "concept_audit"
    )
    with pytest.raises(PromptTransportBudgetError) as excinfo:
        client.complete(_messages(_bytes_for_tokens(DEFAULT_MAX_PROMPT_TOKENS + 1)))

    rendered = str(excinfo.value)
    assert "not the model's context window" in rendered.lower().replace("NOT", "not")
    assert "max_prompt_tokens_per_call" in rendered


# ---------------------------------------------------------------------------
# The byte -> token estimate is calibrated, and never under-counts
# ---------------------------------------------------------------------------


def test_the_estimate_never_undercounts_any_observed_call() -> None:
    """Estimating high can refuse a prompt that fits; low would let one slip."""

    observed = [
        (53_393, 13_988),
        (101_878, 26_040),
        (24_901, 6_290),
        (78_401, 20_804),
        (27_125, 6_191),
        (26_064, 6_315),
        (66_119, 17_088),
        (33_762, 7_884),
    ]
    for payload_bytes, provider_tokens in observed:
        assert estimate_prompt_tokens(payload_bytes) >= provider_tokens


def test_the_calibration_constant_stays_under_every_observed_ratio() -> None:
    observed_min = min(
        53_393 / 13_988,
        101_878 / 26_040,
        24_901 / 6_290,
        78_401 / 20_804,
        27_125 / 6_191,
        26_064 / 6_315,
        66_119 / 17_088,
        33_762 / 7_884,
    )
    assert OBSERVED_BYTES_PER_TOKEN <= observed_min
    # And the estimator's own constant keeps margin below that, for content
    # types (CJK) the English/JSON sample does not represent.
    assert CONSERVATIVE_BYTES_PER_TOKEN < OBSERVED_BYTES_PER_TOKEN


# ---------------------------------------------------------------------------
# Images are attachments, not prose
# ---------------------------------------------------------------------------


def test_image_bytes_do_not_count_against_the_text_envelope() -> None:
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), "analyzer", "vlm_visual_qa")

    client.complete_with_images(
        prompt="review this figure",
        image_paths=["a.png"] * 400,
    )

    assert [name for name, _ in inner.calls] == ["complete_with_images"]


def test_an_oversized_vlm_text_prompt_is_still_refused() -> None:
    budget = PROMPT_TRANSPORT_BUDGETS["vlm_visual_qa"]
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), "analyzer", "vlm_visual_qa")

    with pytest.raises(PromptTransportBudgetError):
        client.complete_with_images(
            prompt="x" * _bytes_for_tokens(budget.limit_tokens + 1),
            image_paths=["a.png"],
        )
    assert inner.calls == []


# ---------------------------------------------------------------------------
# An undeclared consumer fails closed
# ---------------------------------------------------------------------------


def test_an_undeclared_consumer_cannot_obtain_a_budgeted_role_client() -> None:
    inner = _RecordingClient()
    with pytest.raises(UndeclaredPromptConsumerError) as excinfo:
        budgeted_role_client(_resolver(inner), "analyzer", "some_new_reviewer")

    assert "some_new_reviewer" in str(excinfo.value)
    # The message must point at the fix, not just report the refusal.
    assert "PROMPT_TRANSPORT_BUDGETS" in str(excinfo.value)


def test_a_consumer_declared_for_another_role_is_refused() -> None:
    """concept_audit is an analyzer consumer; it may not borrow the planner."""

    inner = _RecordingClient()
    with pytest.raises(UndeclaredPromptConsumerError):
        budgeted_role_client(_resolver(inner), "planner", "concept_audit")


def test_a_missing_role_client_stays_missing() -> None:
    assert budgeted_role_client(lambda role: None, "analyzer", "concept_audit") is None


def test_wrapping_is_idempotent() -> None:
    inner = _RecordingClient()
    once = budgeted_role_client(_resolver(inner), "analyzer", "concept_audit")
    twice = budgeted_role_client(_resolver(once), "analyzer", "concept_audit")
    assert twice is once


# ---------------------------------------------------------------------------
# Attribution: the receipt must be able to say which consumer called
# ---------------------------------------------------------------------------


def test_the_consumer_is_visible_to_the_transport_during_the_call() -> None:
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), "analyzer", "concept_audit")

    client.complete(_messages(10))

    assert inner.seen_consumer == ["concept_audit"]


def test_the_consumer_context_does_not_leak_after_the_call() -> None:
    inner = _RecordingClient()
    client = budgeted_role_client(_resolver(inner), "analyzer", "concept_audit")

    client.complete(_messages(10))
    assert active_prompt_consumer.get() is None


def test_the_consumer_context_does_not_leak_when_the_call_raises() -> None:
    class _Failing(_RecordingClient):
        def complete(self, messages, **kwargs):
            raise RuntimeError("provider down")

    client = budgeted_role_client(_resolver(_Failing()), "analyzer", "concept_audit")
    with pytest.raises(RuntimeError):
        client.complete(_messages(10))
    assert active_prompt_consumer.get() is None


def test_the_transport_receipt_records_the_consumer(tmp_path) -> None:
    import json

    from easyicu.research_agent.providers.cost import CostMeter, MeteredClient

    meter = CostMeter(runtime_dir=tmp_path)
    metered = MeteredClient(_RecordingClient(), role="analyzer", meter=meter)
    client = PromptBudgetClient(
        metered, budget=PROMPT_TRANSPORT_BUDGETS["concept_audit"]
    )

    client.complete(_messages(10))

    receipts = sorted((tmp_path / "provider_transport_receipts").glob("*.json"))
    assert receipts, "the metered call wrote no transport receipt"
    payload = json.loads(receipts[0].read_text())
    assert payload["role"] == "analyzer"
    assert payload["consumer"] == "concept_audit"


# ---------------------------------------------------------------------------
# The wrapper must not hide the client it wraps
# ---------------------------------------------------------------------------


def test_a_wrapped_mock_is_still_discoverable_as_a_mock() -> None:
    """An opaque proxy silently breaks the pipeline's mock context binding.

    The first version of this wrapper did not publish its child graph, so
    ``iter_mock_clients`` stopped finding the mock behind it, the scripted
    responses lost their context, and two provider-budget tests failed with an
    unrelated-looking repair-authority error.
    """

    from easyicu.research_agent.authority.pipeline_cache import iter_mock_clients
    from easyicu.research_agent.providers.mocks import MockLLMClient

    mock = MockLLMClient()
    assert list(iter_mock_clients(mock)) == [mock]

    wrapped = budgeted_role_client(_resolver(mock), "analyzer", "concept_audit")
    assert list(iter_mock_clients(wrapped)) == [mock]


def test_the_wrapper_registers_its_child_graph() -> None:
    from easyicu.research_agent.providers.factory import provider_client_is_mockish
    from easyicu.research_agent.providers.mocks import MockLLMClient

    mock = MockLLMClient()
    wrapped = budgeted_role_client(_resolver(mock), "analyzer", "concept_audit")
    assert provider_client_is_mockish(mock)
    assert provider_client_is_mockish(wrapped)


def test_the_wrapper_delegates_unknown_attributes() -> None:
    class _WithExtra(_RecordingClient):
        model = "gpt-5.6-luna"

    wrapped = budgeted_role_client(_resolver(_WithExtra()), "analyzer", "concept_audit")
    assert wrapped.model == "gpt-5.6-luna"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_payload_bytes_matches_the_receipt_measure() -> None:
    messages = [_Message("system", "abc"), _Message("user", "de")]
    assert prompt_payload_bytes(messages) == 5


def test_payload_bytes_counts_utf8_not_characters() -> None:
    assert prompt_payload_bytes([_Message("user", "血压")]) == 6


def test_payload_bytes_reads_mapping_messages() -> None:
    assert prompt_payload_bytes([{"role": "user", "content": "abcd"}]) == 4


# ---------------------------------------------------------------------------
# Source contract: no call site may quietly skip the budget
# ---------------------------------------------------------------------------


_AGENT_ROOT = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "easyicu" / "research_agent"
)

# Call sites allowed to take a budgeted role client raw, each with the reason.
# Anything not listed here must go through ``budgeted_role_client``.
_DECLARED_RAW_RESOLVER_SITES = {
    # CriticAgent stores the client but never issues a provider call; its
    # review_step / review_manuscript paths are deterministic.
    ("execution/phase.py", "analyzer", "CriticAgent"),
    ("reporting/write_phase.py", "analyzer", "CriticAgent"),
    # PlannerAgent and ReplannerAgent already enforce _PLANNER_PROMPT_BYTE_LIMIT
    # on their own constructed prompt. That metric measures the projection they
    # build, which is not byte-identical to the transport payload, so re-basing
    # them onto the transport measure is a separate, evidence-backed change.
    ("pipeline.py", "planner", "PlannerAgent"),
    ("execution/phase.py", "planner", "ReplannerAgent"),
}


def _raw_role_resolver_sites() -> set[tuple[str, str, int]]:
    """Every ``role_resolver("<budgeted role>")`` call that is not wrapped.

    ``budgeted_role_client(role_resolver, "analyzer", ...)`` passes the
    resolver, it does not call it, so a wrapped site produces no such node.
    """

    found: set[tuple[str, str, int]] = set()
    for path in sorted(_AGENT_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "role_resolver" or len(node.args) != 1:
                continue
            arg = node.args[0]
            if not isinstance(arg, ast.Constant) or arg.value not in BUDGETED_ROLES:
                continue
            rel = path.relative_to(_AGENT_ROOT).as_posix()
            found.add((rel, str(arg.value), node.lineno))
    return found


def test_every_budgeted_role_consumer_is_wrapped_or_declared() -> None:
    """A new consumer of a shared role transport must not slip through.

    This is the check that would have caught the original defect: the concept
    auditor, the VLM fallback and cohort extraction all took a budgeted role
    client raw and no test noticed.
    """

    lines = {(path, line) for path, _role, line in _raw_role_resolver_sites()}
    source_by_path = {
        path: (_AGENT_ROOT / path).read_text().splitlines() for path, _line in lines
    }

    undeclared = []
    for path, role, lineno in sorted(_raw_role_resolver_sites()):
        # Attribute the site to its consumer by the construction it feeds.
        window = "\n".join(source_by_path[path][max(0, lineno - 3) : lineno + 2])
        matched = [
            declared
            for declared in _DECLARED_RAW_RESOLVER_SITES
            if declared[0] == path and declared[1] == role and declared[2] in window
        ]
        if not matched:
            undeclared.append(f"{path}:{lineno} role={role}")

    assert not undeclared, (
        "these call sites take a budgeted role client without a declared "
        "prompt transport budget: " + ", ".join(undeclared)
    )


def test_the_declared_raw_sites_all_still_exist() -> None:
    """A stale exemption must not silently keep protecting nothing."""

    sites = _raw_role_resolver_sites()
    present = set()
    for path, role, lineno in sites:
        source = (_AGENT_ROOT / path).read_text().splitlines()
        window = "\n".join(source[max(0, lineno - 3) : lineno + 2])
        for declared in _DECLARED_RAW_RESOLVER_SITES:
            if declared[0] == path and declared[1] == role and declared[2] in window:
                present.add(declared)

    assert (
        present == _DECLARED_RAW_RESOLVER_SITES
    ), "declared raw resolver sites that no longer exist: " + ", ".join(
        sorted(str(s) for s in _DECLARED_RAW_RESOLVER_SITES - present)
    )


def test_the_live_analyzer_consumers_are_all_declared() -> None:
    consumers = declared_consumers_for_role("analyzer")
    assert set(consumers) == {
        "analyzer_interpretation",
        "concept_audit",
        "vlm_visual_qa",
    }


def test_every_budget_declares_a_rationale() -> None:
    for consumer, budget in PROMPT_TRANSPORT_BUDGETS.items():
        assert budget.rationale.strip(), f"{consumer} declares no rationale"
        assert budget.limit_tokens > 0
