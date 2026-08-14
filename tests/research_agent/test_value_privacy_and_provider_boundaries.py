"""Value-level privacy, provider-boundary and retry-attempt regressions.

Origin: 2026-07-29 external review (eighth pass).

Round 7 made the privacy audit read values instead of column names. This round
is about the *types* those values arrive as. Every bypass below is a case where
the leak is present in the file and the scanner walked past it because of how
the value was spelled: an identifier that came back as ``30042318.0`` because
the column is nullable, a suppressed cell written ``3.0`` instead of ``3``, an
admission timestamp sitting in a column called ``label``, and a column called
``patient_number`` that the magnitude exemption read as a count.

The audit is run for real against real registered artefacts throughout — the
point of the round is that a hand-made verdict proves nothing about the
scanner.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pandas as pd
import pytest


def _pin_provider_test_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make URL-policy tests independent of VPN/proxy DNS rewriting."""

    addresses = {
        "169.254.169.254": "169.254.169.254",
        "10.0.0.5": "10.0.0.5",
        "example.com": "93.184.216.34",
        "api.openai.com": "104.18.33.45",
        "127.0.0.1": "127.0.0.1",
    }

    def getaddrinfo(host, port, *args, **kwargs):
        address = addresses[str(host)]
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, int(port or 0)),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def _store(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    return EvidenceStore(tmp_path / "run")


def _contract(figure_id="Figure2", roles=("primary_estimand",), sources=()):
    return SimpleNamespace(
        figure_id=figure_id,
        core_claim="SOFA-2 predicts in-hospital mortality.",
        statistics_note=None,
        image_integrity_note=None,
        panels=[
            SimpleNamespace(role=role, title=f"Panel {i}", claim="A claim.")
            for i, role in enumerate(roles)
        ],
        source_data=list(sources),
    )


def _audit(tmp_path, filename, writer):
    """Register a real source file and run the real audit over it."""

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    store = _store(tmp_path)
    run_dir = Path(store.root)
    staged = tmp_path / filename
    writer(staged)
    store.register_file(
        kind="table",
        description="Figure source.",
        source_path=staged,
        evidence_id="src",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    return audit_figure_privacy(
        contract=_contract(sources=("src",)),
        evidence=store,
        run_dir=run_dir,
        source_evidence_ids=["src"],
    )


# ---------------------------------------------------------------------------
# P0-4  the value scan skipped whole classes of value
# ---------------------------------------------------------------------------


def test_a_nullable_parquet_id_column_is_caught(tmp_path):
    """One missing value turns an id column into float64, and floats were skipped.

    This is not a corner case: any Parquet id column with a gap in it arrives
    as ``30042318.0``. The previous scanner returned early on every float, so
    the single most likely way for identifiers to reach a figure source was
    also the one type it never looked at.
    """

    def write(path):
        pd.DataFrame(
            {
                "label": [30042318.0, 30042319.0, None],
                "predicted": [0.82, 0.44, 0.61],
            }
        ).to_parquet(path)

    audit = _audit(tmp_path, "scores.parquet", write)

    assert audit.aggregate_only is False
    assert any("identifier-shaped value" in reason for reason in audit.reasons)


def test_a_float_valued_measurement_column_is_still_cleared(tmp_path):
    """The other side of the same rule: precision is not an identifier.

    A gate that refuses ordinary numbers gets switched off, so the fractional
    floats a results table is made of must survive the new float handling.
    """

    def write(path):
        pd.DataFrame(
            {
                "stratum": ["high", "low"],
                "auroc": [0.812345, 0.774321],
                "p_value": [0.000001, 0.000123],
                "n": [4200, 5221],
            }
        ).to_parquet(path)

    audit = _audit(tmp_path, "summary.parquet", write)

    assert audit.aggregate_only is True, audit.reasons


def test_a_small_cell_written_as_a_float_is_caught(tmp_path):
    """``int("3.0")`` raises, so the count never reached the threshold check."""

    def write(path):
        path.write_text(
            "subgroup,n\nrare phenotype,3.0\ncommon,5221.0\n", encoding="utf-8"
        )

    audit = _audit(tmp_path, "strata.csv", write)

    assert audit.aggregate_only is False
    assert any("below" in reason for reason in audit.reasons)


def test_an_event_timestamp_in_a_generic_column_is_caught(tmp_path):
    """A time of day is an event, whatever the column is called.

    The name-level check only knows ``charttime``-style names. An admission
    time parked in a column called ``label`` carries no six-digit run either,
    so nothing looked at it.
    """

    def write(path):
        path.write_text(
            "label,rate\n2180-01-01T12:30:00,0.4\n2180-01-02T03:15:00,0.6\n",
            encoding="utf-8",
        )

    audit = _audit(tmp_path, "events.csv", write)

    assert audit.aggregate_only is False
    assert any("event timestamp value" in reason for reason in audit.reasons)


def test_a_date_only_axis_is_not_treated_as_an_event_time(tmp_path):
    """The deliberate boundary: a study-period axis is aggregate content.

    Only a time-of-day component makes a cell an event time. A reviewer who
    disagrees with that line should see it stated rather than discover it.
    """

    def write(path):
        path.write_text(
            "month,admissions\n2180-01-01,412\n2180-02-01,388\n", encoding="utf-8"
        )

    audit = _audit(tmp_path, "trend.csv", write)

    assert audit.aggregate_only is True, audit.reasons


def test_patient_number_is_an_identity_not_a_magnitude(tmp_path):
    """``_number`` made the column magnitude-named, which exempted its values."""

    def write(path):
        path.write_text(
            "patient_number,outcome\n30042318,1\n30042319,0\n", encoding="utf-8"
        )

    audit = _audit(tmp_path, "cohort.csv", write)

    assert audit.aggregate_only is False


def test_a_genuine_total_keeps_its_magnitude_exemption(tmp_path):
    """``total_number`` has no subject in it, so a seven-digit total is fine."""

    def write(path):
        path.write_text(
            "stratum,total_number,row_count\nhigh,1234567,4200\nlow,7654321,5221\n",
            encoding="utf-8",
        )

    audit = _audit(tmp_path, "totals.csv", write)

    assert audit.aggregate_only is True, audit.reasons


def test_the_audit_still_does_not_quote_the_value_it_caught(tmp_path):
    """Round 7's masking rule has to survive the new float path.

    ``30042318.0`` is converted to ``30042318`` before matching, so the raw
    digits are one careless f-string away from the receipt a reviewer reads.
    """

    def write(path):
        pd.DataFrame({"label": [30042318.0, None], "p": [0.1, 0.2]}).to_parquet(path)

    audit = _audit(tmp_path, "scores.parquet", write)

    blob = json.dumps(audit.as_receipt())
    assert "30042318" not in blob
    assert audit.aggregate_only is False


# ---------------------------------------------------------------------------
# The version bump: a clearance from the leaky scanner must not still authorize
# ---------------------------------------------------------------------------


def test_the_previous_audit_version_no_longer_clears_a_figure(tmp_path):
    """1.1.0 receipts were produced by the scanner that had these holes.

    The egress gate refuses an audit version it does not trust, which is what
    makes bumping the version the mechanism that retires those clearances
    rather than a comment.
    """

    from easyicu.research_agent.gates.figure_privacy import (
        FIGURE_PRIVACY_AUDIT_VERSION,
        TRUSTED_AUDIT_VERSIONS,
    )

    assert FIGURE_PRIVACY_AUDIT_VERSION == "1.2.0"
    assert "1.1.0" not in TRUSTED_AUDIT_VERSIONS


# ---------------------------------------------------------------------------
# P1-6  the interval the manuscript printed, not just that an interval exists
# ---------------------------------------------------------------------------


def _records(*summaries):
    return [
        {"step_id": step_id, "status": "ok", "step_summary": summary}
        for step_id, summary in summaries
    ]


def test_a_confidence_interval_that_matches_nothing_registered_is_caught():
    """The old check only asked whether *some* CI existed anywhere.

    ``0.71-0.84`` and the registered ``0.83-0.90`` are different intervals.
    Nothing compared them, so a writer could print whichever interval it
    remembered and the audit stayed silent as long as the run had registered
    some pair of bounds somewhere.
    """

    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    findings = _audit_manuscript_numeric_claims(
        "The model reached an AUROC of 0.868 (95% CI 0.71-0.84)[^claim_1].\n\n"
        "[^claim_1]: value=0.868; step=01_model_training; evidence=e1\n",
        per_step_records=_records(
            (
                "01_model_training",
                {
                    "statistic:auroc": 0.868,
                    "statistic:auroc_ci_lower": 0.83,
                    "statistic:auroc_ci_upper": 0.90,
                },
            )
        ),
    )

    assert any(
        finding.detail.get("reason") == "ci_bounds_do_not_match_registered"
        for finding in findings
    ), [f.message for f in findings]


def test_the_registered_confidence_interval_passes():
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    findings = _audit_manuscript_numeric_claims(
        "The model reached an AUROC of 0.868 (95% CI 0.83-0.90)[^claim_1].\n\n"
        "[^claim_1]: value=0.868; step=01_model_training; evidence=e1\n",
        per_step_records=_records(
            (
                "01_model_training",
                {
                    "statistic:auroc": 0.868,
                    "statistic:auroc_ci_lower": 0.83,
                    "statistic:auroc_ci_upper": 0.90,
                },
            )
        ),
    )

    assert not [
        finding for finding in findings if finding.detail.get("metric") == "auroc_ci"
    ], [f.message for f in findings]


def test_bounds_cannot_be_assembled_from_two_different_steps():
    """A lower bound from the primary model and an upper from the sensitivity.

    Each bound existed somewhere in the run, which is exactly what the old
    check tested for. Resolving the pair from one summary is what makes the
    combination impossible.
    """

    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    findings = _audit_manuscript_numeric_claims(
        "The model reached an AUROC of 0.868 (95% CI 0.83-0.95)[^claim_1].\n\n"
        "[^claim_1]: value=0.868; step=01_model_training; evidence=e1\n",
        per_step_records=_records(
            (
                "01_model_training",
                {
                    "statistic:auroc": 0.868,
                    "statistic:auroc_ci_lower": 0.83,
                    "statistic:auroc_ci_upper": 0.90,
                },
            ),
            (
                "02_sensitivity",
                {
                    "statistic:auroc": 0.871,
                    "statistic:auroc_ci_lower": 0.79,
                    "statistic:auroc_ci_upper": 0.95,
                },
            ),
        ),
    )

    assert any(
        finding.detail.get("reason") == "ci_bounds_do_not_match_registered"
        for finding in findings
    ), [f.message for f in findings]


# ---------------------------------------------------------------------------
# P1-2  the provider base URL decides where an API key is sent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://169.254.169.254/latest/meta-data", "link_local_or_reserved_address"),
        ("http://10.0.0.5/v1", "private_address"),
        ("https://user:secret@example.com/v1", "credentials_in_url"),
        ("file:///etc/passwd", "scheme_not_http"),
        ("http://example.com/v1", "plaintext_to_non_loopback"),
        ("https://metadata.google.internal/v1", "metadata_host"),
    ],
)
def test_a_provider_url_that_should_not_receive_a_key_is_refused(
    url, reason, monkeypatch
):
    """The server sends ``Authorization: Bearer <key>`` to this address.

    An unchecked value is both an SSRF probe from inside the host's network
    and a delivery mechanism for the operator's key.
    """

    _pin_provider_test_dns(monkeypatch)
    from easyicu.webserver.provider_adapter import (
        ProviderAdapterError,
        validate_provider_base_url,
    )

    with pytest.raises(ProviderAdapterError) as excinfo:
        validate_provider_base_url(url)

    assert excinfo.value.detail["reason"] == reason


def test_official_https_provider_allows_local_proxy_fake_ip(monkeypatch) -> None:
    from easyicu.webserver.provider_url_security import (
        ProviderUrlSecurityError,
        validate_credential_endpoint,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("198.18.0.50", 443),
            )
        ],
    )

    assert (
        validate_credential_endpoint("https://api.anthropic.com/v1")
        == "https://api.anthropic.com/v1"
    )
    with pytest.raises(ProviderUrlSecurityError) as caught:
        validate_credential_endpoint("https://untrusted.example/v1")
    assert caught.value.reason == "private_address"


def test_provider_hostname_with_public_and_loopback_answers_is_refused(
    monkeypatch,
) -> None:
    from easyicu.webserver.provider_url_security import (
        ProviderUrlSecurityError,
        validate_credential_endpoint,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 443),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 443),
            ),
        ],
    )

    with pytest.raises(ProviderUrlSecurityError) as caught:
        validate_credential_endpoint("https://mixed.example/v1")
    assert caught.value.reason == "mixed_address_scope"


@pytest.mark.parametrize(
    "url",
    ["https://api.openai.com/v1", "http://127.0.0.1:8787/v1"],
)
def test_the_real_provider_and_the_local_proxy_still_work(url, monkeypatch):
    """Plaintext to loopback stays allowed: that is the local model proxy."""

    _pin_provider_test_dns(monkeypatch)
    from easyicu.webserver.provider_adapter import validate_provider_base_url

    assert validate_provider_base_url(url) == url


# ---------------------------------------------------------------------------
# P1-5  an error message is not a place to put a patient record
# ---------------------------------------------------------------------------


def test_a_window_expansion_error_describes_the_row_without_quoting_it():
    """This message reaches web job errors, CI logs and agent findings."""

    from easyicu.api import WindowExpansionError, _guard_window_expansion

    with pytest.raises(WindowExpansionError) as excinfo:
        _guard_window_expansion(
            10_001,
            concept_name="vent_ind",
            duration=525600.0,
            unit="minutes",
            row={
                "stay_id": 30042318,
                "charttime": "2180-01-01T12:30:00",
                "vent_ind": 1,
            },
        )

    message = str(excinfo.value)
    assert "30042318" not in message
    assert "2180-01-01" not in message
    assert "vent_ind" in message
    assert "sha256" in message


# ---------------------------------------------------------------------------
# P1-7  two sends of the same figure are two attempts
# ---------------------------------------------------------------------------


def test_a_retry_does_not_close_the_earlier_attempt(tmp_path):
    """Matching on the image digest closed every open attempt for that image.

    Send the panel once, succeed; send it again, fail. Digest matching wrote
    the failure onto both rows, and the receipt then claimed a send that had
    in fact completed had failed.
    """

    from easyicu.research_agent.gates.figure_egress import (
        TRANSPORT_COMPLETED,
        TRANSPORT_FAILED,
        FigureEgressPolicy,
    )

    policy = FigureEgressPolicy(allow_external_upload=True)
    first = policy.record_upload([{"path": "a.png", "sha256": "a" * 64}])
    policy.record_transport_outcome(first, TRANSPORT_COMPLETED)

    second = policy.record_upload([{"path": "a.png", "sha256": "a" * 64}])
    policy.record_transport_outcome(second, TRANSPORT_FAILED)

    assert policy.transport_summary() == {
        "transport_completed": 1,
        "transport_failed": 1,
    }
    assert first[0]["attempt_id"] != second[0]["attempt_id"]


# ---------------------------------------------------------------------------
# P1-1 (partial)  a loopback peer behind a proxy is not a local user
# ---------------------------------------------------------------------------


def test_a_forwarded_request_is_refused_even_though_the_peer_is_loopback(monkeypatch):
    """Behind nginx/caddy/an SSH forward, every remote request looks local.

    The filesystem APIs are gated on the socket peer being loopback, and a
    proxy makes that true for the whole internet. The forwarding headers are
    the evidence a proxy is there.
    """

    from fastapi.testclient import TestClient

    from easyicu.webserver.app import app

    monkeypatch.delenv("EASYICU_WEB_TRUST_PROXY", raising=False)
    client = TestClient(app)

    direct = client.get("/api/settings")
    assert direct.status_code == 200

    forwarded = client.get("/api/settings", headers={"X-Forwarded-For": "203.0.113.7"})
    assert forwarded.status_code == 403
    assert "proxy" in forwarded.json()["detail"].lower()


def test_an_operator_can_declare_the_proxy_authenticates(monkeypatch):
    from fastapi.testclient import TestClient

    from easyicu.webserver.app import app

    monkeypatch.setenv("EASYICU_WEB_TRUST_PROXY", "1")
    client = TestClient(app)

    forwarded = client.get("/api/settings", headers={"X-Forwarded-For": "203.0.113.7"})
    assert forwarded.status_code == 200
