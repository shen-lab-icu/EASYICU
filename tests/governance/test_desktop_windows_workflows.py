"""Offline event, artifact provenance and draft-only Windows delivery contracts."""

import json

import yaml

from tests.support.desktop_bundle import ROOT


def workflow(name):
    return yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())


def test_acceptance_follows_only_successful_same_repository_dev9_builds():
    build = workflow("desktop_windows_build")
    acceptance = workflow("windows_laptop_acceptance")
    triggers = acceptance.get("on", acceptance.get(True))
    assert set(triggers) == {"workflow_dispatch", "workflow_run", "workflow_call"}
    assert triggers["workflow_run"] == {
        "workflows": [build["name"]], "types": ["completed"],
        "branches": ["codex/dev9-expansion-20260917"],
    }
    guard = acceptance["jobs"]["acceptance"]["if"]
    assert " ".join(guard.split()) == (
        "github.event_name != 'workflow_run' || "
        "(github.event.workflow_run.conclusion == 'success' && "
        "github.event.workflow_run.head_branch == 'codex/dev9-expansion-20260917' && "
        "github.event.workflow_run.head_repository.full_name == github.repository)"
    )
    assert acceptance["permissions"] == {"contents": "read", "actions": "read"}


def test_dev9_build_can_call_acceptance_without_a_default_branch_workflow():
    caller = workflow("desktop_windows_build")["jobs"]["acceptance"]
    assert caller["needs"] == "installer"
    assert caller["if"] == "github.ref == 'refs/heads/codex/dev9-expansion-20260917'"
    assert caller["uses"] == "./.github/workflows/windows_laptop_acceptance.yml"
    assert caller["with"]["installer_run_id"] == "${{ format('{0}', github.run_id) }}"
    assert caller["with"]["ref"] == "${{ github.sha }}"
    assert "secrets" not in caller


def test_acceptance_downloads_the_completed_runs_artifact_and_matching_sources():
    job = workflow("windows_laptop_acceptance")["jobs"]["acceptance"]
    steps = job["steps"]
    checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ github.event.workflow_run.head_sha || inputs.ref || github.sha }}"
    download = next(s for s in steps if s.get("uses", "").startswith("actions/download-artifact@"))
    assert download["with"]["name"] == "EasyICU-windows-x64"
    assert download["with"]["run-id"] == "${{ github.event.workflow_run.id || inputs.installer_run_id }}"
    assert download["with"]["github-token"] == "${{ github.token }}"
    assert download["if"] == "github.event_name == 'workflow_run' || inputs.installer_run_id != ''"
    build = next(s for s in steps if s.get("name") == "Build the candidate desktop installer")
    assert build["if"] == "github.event_name != 'workflow_run' && inputs.installer_run_id == ''"
    runtime = next(s for s in steps if "--runtime-only" in s.get("run", ""))
    assert runtime["if"] == download["if"]
    install = next(s for s in steps if "Install and accept" in s.get("name", ""))
    assert "verify_desktop_artifact.py windows" in install["run"]
    # The driver runs the app's owners in-process, in the frozen app's UTF-8 mode.
    assert "& $runtime -I -X utf8 tools/windows_laptop_acceptance.py" in install["run"]
    assert "--source-commit" in install["run"]
    assert install["env"]["SOURCE_COMMIT"] == "${{ steps.candidate.outputs.sha }}"
    candidate = next(s for s in steps if s.get("id") == "candidate")
    assert "$sha = git rev-parse HEAD" in candidate["run"]
    assert install["env"]["MEMORY_PRESSURE_GB"] == "${{ github.event_name == 'workflow_run' && 8 || inputs.memory_pressure_gb }}"
    assert job["env"]["EASYICU_DISABLE_PROVIDER_ENV_FILE"] == "1"
    assert "secrets." not in json.dumps(job)


def test_release_is_tag_push_only_draft_and_uses_the_smoked_installers():
    build = workflow("desktop_windows_build")
    release = build["jobs"]["release"]
    assert release["needs"] == ["installer", "macos"]
    assert release["if"] == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    assert release["permissions"] == {"contents": "write"}
    assert build["permissions"] == {"contents": "read"}
    steps = release["steps"]
    downloads = [s for s in steps if s.get("uses", "").startswith("actions/download-artifact@")]
    assert [d["with"]["name"] for d in downloads] == ["EasyICU-windows-x64", "EasyICU-macos-arm64"]
    for download in downloads:
        assert "run-id" not in download["with"]  # This exact tag build, never a latest-branch lookup.
    create = next(s for s in steps if "gh release create" in s.get("run", ""))
    assert create["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert create["env"]["RELEASE_TAG"] == "${{ github.ref_name }}"
    for flag in ("--draft", "--verify-tag", "--generate-notes", '--target "$env:GITHUB_SHA"'):
        assert flag in create["run"]
    assert '"$installer" "$dmg" "$releaseDir\\SHA256SUMS.txt"' in create["run"]
    for platform in ("windows", "macos"):
        assert f"verify_desktop_artifact.py {platform}" in create["run"]
    assert create["run"].count('--source-commit "$env:GITHUB_SHA"') == 2
    # A 0.x preview is labelled Pre-release and never becomes Latest.
    assert "if ($env:RELEASE_TAG -like 'v0.*') { $stage = @('--prerelease') }" in create["run"]
    assert "build_windows.py" not in json.dumps(release)
    assert "build_macos.py" not in json.dumps(release)
    assert "secrets." not in json.dumps(build)
    assert "pages" not in json.dumps(build).lower()
    for job in ("installer", "macos"):
        signing = next(s for s in build["jobs"][job]["steps"] if "signing" in s.get("name", "").lower())
        assert signing["if"] == "${{ false }}"


def test_macos_dmg_is_smoked_from_the_installed_app_before_it_is_retained():
    build = workflow("desktop_windows_build")
    job = build["jobs"]["macos"]
    assert job["runs-on"] == "macos-15"
    steps = job["steps"]
    names = [step.get("name", "") for step in steps]
    smoke = next(step for step in steps if "smoke_macos.py" in step.get("run", ""))
    assert "python -I desktop/scripts/smoke_macos.py --dmg" in smoke["run"]
    assert "EasyICU 临床 profile" in smoke["run"]
    artifact = next(step for step in steps if step.get("with", {}).get("name") == "EasyICU-macos-arm64")
    assert artifact.get("if", "success()") == "success()"
    assert "dmg/*.dmg" in artifact["with"]["path"] and "build-inputs.json" in artifact["with"]["path"]
    assert names.index(smoke["name"]) < names.index(artifact["name"])
    assert any("build_macos.py" in step.get("run", "") for step in steps)
    # A newer dev9 push cancels an unfinished branch build; tag builds always finish.
    assert build["concurrency"] == {"group": "desktop-installers-${{ github.ref }}",
                                    "cancel-in-progress": "${{ github.ref_type == 'branch' }}"}
