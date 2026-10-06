"""Regression coverage for CI runtime and packaging warning fixes."""

import importlib.util
import io
import os
from pathlib import Path
import re
import stat
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
# These exact refs were audited against their published action manifests: all
# JavaScript entrypoints (including post actions) use the Node 24 runtime.
NODE24_ACTIONS = {
    "actions/checkout@v7",
    "actions/setup-python@v7",
    "actions/cache@v6",
    "actions/upload-artifact@v7",
    "actions/download-artifact@v8",
    "actions/setup-node@v7",
    "jwlawson/actions-setup-cmake@v2",
    "astral-sh/setup-uv@v8.2.0",
    "benjlevesque/short-sha@v4.0",
    "peaceiris/actions-gh-pages@v4.1.0",
}


def test_workflows_use_only_audited_node24_actions():
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        refs = re.findall(
            r"^[ \t]*(?:-\s*)?uses:\s*(\S+)\s*$",
            workflow.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        assert refs, f"No action declarations found in {workflow}"
        assert set(refs) <= NODE24_ACTIONS, f"Unaudited action runtime in {workflow}: {set(refs) - NODE24_ACTIONS}"


def test_packaging_does_not_request_disabled_pypy_or_redundant_ninja():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "pp*" not in configuration["tool"]["cibuildwheel"]["skip"]
    requirements = {
        re.match(r"[\w-]+", requirement).group(0).lower()
        for requirement in configuration["build-system"]["requires"]
    }
    assert "ninja" not in requirements
    assert "scikit-build-core" in requirements


def test_native_setup_replaces_deprecated_actions_without_losing_linker_pin():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert workflow.count("run: ./scripts/setup_msvc.ps1") == 2
    assert workflow.count("run: python scripts/setup_ninja.py") == 2
    assert workflow.count("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=$linker") == 2
    assert (ROOT / "scripts/setup_msvc.ps1").is_file()


@pytest.fixture
def ninja_setup(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("setup_ninja", ROOT / "scripts/setup_ninja.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    github_path = tmp_path / "github-path"
    github_path.write_text("existing-path\n", encoding="utf-8")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    monkeypatch.setenv("GITHUB_PATH", str(github_path))
    return module, github_path


def mock_ninja_archive(monkeypatch, module, platform, version):
    filename = "ninja.exe" if platform == "win32" else "ninja"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, b"mock Ninja binary")
    payload = buffer.getvalue()

    def download(url, timeout):
        assert url == f"https://github.com/ninja-build/ninja/releases/download/v1.11.1/{module.ASSETS[platform]}"
        assert timeout == 60
        return io.BytesIO(payload)

    def executable_version(command, text):
        assert Path(command[0]).name == filename
        assert command[1:] == ["--version"]
        assert text is True
        return f"{version}\n"

    monkeypatch.setattr(module.urllib.request, "urlopen", download)
    monkeypatch.setattr(module.subprocess, "check_output", executable_version)


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_ninja_installs_exact_upstream_binary_and_preserves_path(ninja_setup, monkeypatch, platform):
    module, github_path = ninja_setup
    mock_ninja_archive(monkeypatch, module, platform, module.VERSION)
    module.install_ninja(platform)
    directory = github_path.parent / "runner" / "ninja-1.11.1"
    filename = "ninja.exe" if platform == "win32" else "ninja"
    executable = directory / filename
    assert executable.read_bytes() == b"mock Ninja binary"
    assert github_path.read_text(encoding="utf-8") == f"existing-path\n{directory.resolve()}\n"
    if platform != "win32" and os.name != "nt":
        assert stat.S_IMODE(executable.stat().st_mode) == 0o755


def test_ninja_rejects_unsupported_host_before_downloading(ninja_setup):
    module, github_path = ninja_setup
    with pytest.raises(RuntimeError, match="Unsupported Ninja host"):
        module.install_ninja("unsupported")
    assert github_path.read_text(encoding="utf-8") == "existing-path\n"


def test_ninja_rejects_wrong_version_before_exporting_path(ninja_setup, monkeypatch):
    module, github_path = ninja_setup
    mock_ninja_archive(monkeypatch, module, "linux", "0.0.0")
    with pytest.raises(RuntimeError, match="Expected Ninja 1.11.1"):
        module.install_ninja("linux")
    assert github_path.read_text(encoding="utf-8") == "existing-path\n"
