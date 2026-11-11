"""S856: end-user CLI behaviour of ``easyicu`` (extract/import/setup/download).

User-input errors print one actionable line, no traceback, and exit 2; the
credentialed databases are refused before any network call.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from easyicu.scripts import extract_features

ROOT = Path(__file__).resolve().parents[2]
CJK = re.compile("[一-鿿]")


def _cli(*argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    code = (
        "import sys; from easyicu.scripts.extract_features import main; "
        "sys.exit(main(sys.argv[1:]))"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=300,
    )


def _lines(proc: subprocess.CompletedProcess, needle: str) -> list[str]:
    return [line for line in proc.stderr.splitlines() if needle in line]


def test_extract_on_an_empty_directory_prints_one_actionable_line(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _cli("extract", "--sources", "miiv", "--data-dirs", str(empty), "--output", str(tmp_path / "out"))

    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr + proc.stdout
    assert "ConceptExtractionUnavailable" not in proc.stderr
    assert len(_lines(proc, "No prepared database found")) == 1
    assert "run the converter first" in proc.stderr
    assert "Loading demo concepts" not in proc.stderr
    assert not CJK.search((proc.stderr + proc.stdout).replace(str(empty), "<dir>"))
    assert not list((tmp_path / "out").glob("*.csv"))


def test_a_refused_concept_extraction_is_one_line_without_traceback(tmp_path, monkeypatch, caplog):
    from easyicu.concept.errors import ConceptExtractionUnavailable

    def refuse(*_args, **_kwargs):
        raise ConceptExtractionUnavailable(
            concept_id="death", database="miiv", stage="hospital_status", detail="no admissions"
        )

    (tmp_path / "patients.parquet").write_bytes(b"")
    monkeypatch.setattr(extract_features, "resolve_datasource", lambda *a: object())
    monkeypatch.setattr(extract_features, "extract_groups", refuse)
    with caplog.at_level("INFO"):
        code = extract_features.main(
            ["extract", "--sources", "miiv", "--data-dirs", str(tmp_path), "--output", str(tmp_path / "o")]
        )

    assert code == 2
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and errors[0].exc_info is None
    assert "hospital_status" in errors[0].getMessage()


def test_import_on_a_bad_path_summarises_the_missing_tables_once(tmp_path):
    bad = tmp_path / "not_prepared"
    bad.mkdir()
    proc = _cli("import", "--sources", "miiv", "--data-dirs", str(bad))

    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert len(_lines(proc, "tables missing under")) == 1
    assert "is this the prepared directory?" in proc.stderr
    assert "Missing source files" not in proc.stderr
    assert len(proc.stderr) < 1500  # the full list is no longer dumped


def test_the_full_missing_file_list_is_kept_at_debug(tmp_path, caplog):
    with caplog.at_level("DEBUG"):
        code = extract_features.main(["import", "--sources", "miiv", "--data-dirs", str(tmp_path)])
    assert code == 2
    debug = [r for r in caplog.records if r.levelname == "DEBUG" and "Missing source files" in r.getMessage()]
    assert debug
    assert not [r for r in caplog.records if r.levelno >= 30 and "Missing source files" in r.getMessage()]


@pytest.fixture
def no_network(monkeypatch):
    import requests

    from easyicu.io import download

    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("network used")

    monkeypatch.setattr(requests.Session, "request", boom)
    monkeypatch.setattr(requests.Session, "get", boom)
    monkeypatch.setattr(download.PhysioNetDownloader, "download_file", boom)
    return calls


@pytest.mark.parametrize("source", ["miiv", "mimic", "eicu", "hirid", "aumc", "sic"])
def test_download_of_a_credentialed_source_refuses_without_network(tmp_path, no_network, caplog, source):
    code = extract_features.main(["download", "--sources", source, "--data-dirs", str(tmp_path / "d")])

    assert code == 2
    assert no_network == []
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "requires credentialed access" in errors[0]
    assert "DataConverter" in errors[0]
    assert not (tmp_path / "d").exists()


def test_setup_of_a_credentialed_source_without_local_data_refuses_without_network(tmp_path, no_network, caplog):
    code = extract_features.main(["setup", "--sources", "miiv", "--data-dirs", str(tmp_path)])

    assert code == 2
    assert no_network == []
    assert any("requires credentialed access" in r.getMessage() for r in caplog.records)


def test_an_unsupported_source_is_refused_up_front(tmp_path, no_network):
    assert extract_features.main(["download", "--sources", "nwicu", "--data-dirs", str(tmp_path)]) == 2
    assert no_network == []


def test_a_public_demo_download_still_reaches_the_downloader(tmp_path, monkeypatch):
    from easyicu.io import download

    fetched = []
    monkeypatch.setattr(
        download.PhysioNetDownloader, "download_file", lambda self, url, dest, **kw: fetched.append(url)
    )
    code = extract_features.main(["download", "--sources", "mimic_demo", "--data-dirs", str(tmp_path)])

    assert code == 0
    assert fetched and all(url.startswith("https://physionet.org/files/mimiciii-demo/") for url in fetched)


@pytest.mark.parametrize("command", ["extract", "import", "setup", "download"])
def test_user_input_errors_share_one_exit_code(tmp_path, command):
    args = [command, "--sources", "miiv", "mimic", "--data-dirs", str(tmp_path)]
    if command == "extract":
        args += ["--output", str(tmp_path / "o")]
    assert extract_features.main(args) == extract_features.EXIT_USAGE == 2


def test_the_help_text_has_no_internal_names(capsys):
    for argv in ([], ["extract"], ["download"], ["import"], ["setup"]):
        with pytest.raises(SystemExit):
            extract_features.main([*argv, "--help"])
    text = capsys.readouterr().out
    assert "ricu.R" not in text
    assert "bench7v2" not in text
    assert "usage:" in text
