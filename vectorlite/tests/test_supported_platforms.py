"""Keep release targets limited to Linux x64, Windows x64, and macOS arm64."""

from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = {
    "linux-x64": ("manylinux_2_28_x86_64", "vectorlite-linux-x64", "vectorlite.so"),
    "win32-x64": ("win_amd64", "vectorlite-win32-x64", "vectorlite.dll"),
    "darwin-arm64": ("macosx_11_0_arm64", "vectorlite-darwin-arm64", "vectorlite.dylib"),
}


def test_ci_native_and_wheel_matrices_cover_only_supported_targets():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    runners = re.findall(r"^\s+- os:\s+(\S+)\s*$", workflow, re.MULTILINE)
    targets = re.findall(r"^\s+rust_target:\s+(\S+)\s*$", workflow, re.MULTILINE)
    assert Counter(runners) == {"ubuntu-latest": 2, "windows-latest": 2, "macos-14": 2}
    assert Counter(targets) == {
        "x86_64-unknown-linux-gnu": 2,
        "x86_64-pc-windows-msvc": 2,
        "aarch64-apple-darwin": 2,
    }
    node_packages = re.findall(r"^\s+node_package:\s+(\S+)\s*$", workflow, re.MULTILINE)
    assert set(node_packages) == {package for _, package, _ in PLATFORMS.values()}


def test_cibuildwheel_excludes_intel_and_universal_macos_wheels():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheels = configuration["tool"]["cibuildwheel"]
    assert wheels["macos"]["archs"] == ["arm64"]
    assert wheels["macos"]["environment"]["MACOSX_DEPLOYMENT_TARGET"] == "11.0"
    for identifier in ["cp314-macosx_x86_64", "cp314-macosx_universal2"]:
        assert any(fnmatchcase(identifier, pattern) for pattern in wheels["skip"])
    assert not any(fnmatchcase("cp314-macosx_arm64", pattern) for pattern in wheels["skip"])


def write_wheel(root, tag, filename, payload):
    directory = root / "wheelhouse" / f"vectorlite-wheel-{tag}"
    directory.mkdir(parents=True)
    wheel = directory / f"vectorlite_py-0.3.0-py3-none-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"vectorlite_py/{filename}", payload)


@pytest.fixture
def release_staging(tmp_path):
    def create(missing=None, extra_tag=None):
        destinations = {}
        for platform, (tag, package, filename) in PLATFORMS.items():
            destination = tmp_path / "bindings/nodejs/packages" / package / "src" / filename
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"not staged")
            destinations[platform] = destination
            if platform != missing:
                write_wheel(tmp_path, tag, filename, platform.encode("ascii"))
        if extra_tag:
            write_wheel(tmp_path, extra_tag, "vectorlite.dylib", b"unsupported macOS wheel")
        return tmp_path, destinations

    return create


def extract_wheels(root):
    # Exercise the shell wrapper's embedded Python with the current interpreter
    # directly, so these packaging regressions also run on Windows without sh.
    script = (ROOT / "extract_wheels.sh").read_text(encoding="utf-8")
    match = re.search(r"<<'PY'\n(.*?)\nPY\s*$", script, re.DOTALL)
    assert match, "Cannot locate extract_wheels.sh's Python staging implementation"
    return subprocess.run(
        [sys.executable, "-c", match.group(1), str(root)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_extract_wheels_requires_only_the_three_supported_platforms(release_staging):
    root, destinations = release_staging()
    result = extract_wheels(root)
    assert result.returncode == 0, result.stdout + result.stderr
    for platform, destination in destinations.items():
        assert destination.read_bytes() == platform.encode("ascii")
    assert not (root / "bindings/nodejs/packages/vectorlite-darwin-x64").exists()


@pytest.mark.parametrize("tag", ["macosx_10_15_x86_64", "macosx_11_0_universal2"])
def test_extract_wheels_rejects_unsupported_macos_before_staging(release_staging, tag):
    root, destinations = release_staging(extra_tag=tag)
    result = extract_wheels(root)
    assert result.returncode != 0
    assert "Unsupported wheel platform" in result.stderr
    assert all(destination.read_bytes() == b"not staged" for destination in destinations.values())


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_extract_wheels_still_requires_each_supported_platform(release_staging, platform):
    root, destinations = release_staging(missing=platform)
    result = extract_wheels(root)
    assert result.returncode != 0
    assert f"Missing validated platform wheels: {platform}" in result.stderr
    assert all(destination.read_bytes() == b"not staged" for destination in destinations.values())
