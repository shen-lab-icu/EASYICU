"""Shared R rails: probe, fingerprint, hardened runner."""

from __future__ import annotations

import re
import shutil

import pytest

from easyicu.research_agent import r_runtime as rr


def _need_rscript():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not available")


def test_probe_reports_version_and_declared_packages():
    _need_rscript()
    receipt = rr.probe_r_runtime(refresh=True)
    assert receipt.available is True
    assert re.fullmatch(r"\d+\.\d+\.\d+", receipt.r_version or "")
    assert receipt.packages.get("survival")
    assert set(receipt.packages) == set(rr.DECLARED_R_PACKAGES)


def test_probe_absent_package_reports_none():
    _need_rscript()
    assert "MatchIt" not in rr.DECLARED_R_PACKAGES
    assert "MatchIt" in rr.ABSENT_R_PACKAGES


def test_fingerprint_stable_and_version_bound():
    _need_rscript()
    first = rr.r_runtime_fingerprint()
    second = rr.r_runtime_fingerprint()
    assert first == second
    assert first.startswith("R/")
    assert "survival/" in first


def test_runner_executes_trivial_script():
    _need_rscript()
    rscript = shutil.which("Rscript")
    assert rscript
    result = rr.run_rscript(
        rscript=rscript, script='cat("easyicu-ok\\n")', timeout_s=60
    )
    assert result.timed_out is False
    assert result.returncode == 0
    assert "easyicu-ok" in result.stdout


def test_runner_surfaces_failure_and_timeout():
    _need_rscript()
    rscript = shutil.which("Rscript")
    assert rscript
    failed = rr.run_rscript(
        rscript=rscript, script='stop("boom")', timeout_s=60
    )
    assert failed.timed_out is False
    assert failed.returncode != 0
    assert "boom" in failed.stderr
    slow = rr.run_rscript(
        rscript=rscript, script="Sys.sleep(30)", timeout_s=1
    )
    assert slow.timed_out is True
    assert slow.returncode == -1


def test_runner_validates_inputs():
    with pytest.raises(rr.RRuntimeError):
        rr.run_rscript(rscript="", script="x <- 1")
    with pytest.raises(rr.RRuntimeError):
        rr.run_rscript(rscript="Rscript", script="   ")
    with pytest.raises(rr.RRuntimeError):
        rr.run_rscript(rscript="Rscript", script="x <- 1", timeout_s=0)
