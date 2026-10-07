"""Regression coverage for installed-wheel imports during pytest collection."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER = Path(__file__).resolve().parents[2] / "scripts/run_wheel_tests.py"
SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")


@pytest.fixture
def wheel_environment(tmp_path):
    def create(with_library=True):
        checkout = tmp_path / "checkout"
        installed = tmp_path / "site-packages"
        source_package = checkout / "bindings/python/vectorlite_py"
        installed_package = installed / "vectorlite_py"
        for package, origin in [(source_package, "checkout"), (installed_package, "wheel")]:
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                "from pathlib import Path\n"
                f"ORIGIN = {origin!r}\n"
                "def vectorlite_path():\n"
                "    return str(Path(__file__).with_name('vectorlite'))\n",
                encoding="utf-8",
            )
        metadata = installed / "vectorlite_py-0.3.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: vectorlite_py\nVersion: 0.3.0\n",
            encoding="utf-8",
        )
        # A stale checkout artifact must not make wheel-origin validation pass.
        (source_package / f"vectorlite{SUFFIX}").touch()
        if with_library:
            (installed_package / f"vectorlite{SUFFIX}").touch()

        binding_tests = source_package / "test"
        regression_tests = checkout / "vectorlite/tests"
        for directory in [binding_tests, regression_tests]:
            directory.mkdir(parents=True)
            (directory / "test_origin.py").write_text(
                "import vectorlite_py\n"
                "def test_imports_installed_wheel():\n"
                "    assert vectorlite_py.ORIGIN == 'wheel'\n",
                encoding="utf-8",
            )
        (binding_tests / "__init__.py").touch()
        runner = checkout / "scripts/run_wheel_tests.py"
        runner.parent.mkdir()
        runner.write_bytes(RUNNER.read_bytes())

        env = os.environ.copy()
        env["PYTHONPATH"] = str(installed)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env.pop("PYTEST_ADDOPTS", None)
        env.pop("VECTORLITE_RUST_EXTENSION", None)
        return checkout, installed, env, runner

    return create


def run_tests(environment):
    checkout, _, env, runner = environment
    return subprocess.run(
        [sys.executable, str(runner), "-q"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_preloads_wheel_before_collecting_nested_source_tests(wheel_environment):
    result = run_tests(wheel_environment())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Testing installed wheel:" in result.stdout
    assert "2 passed" in result.stdout


def test_rejects_source_package_even_when_checkout_library_exists(wheel_environment):
    environment = wheel_environment()
    checkout, installed, env, _ = environment
    env["PYTHONPATH"] = os.pathsep.join([str(checkout / "bindings/python"), str(installed)])
    result = run_tests(environment)
    assert result.returncode != 0
    assert "Installed wheel is shadowed" in result.stderr
    assert "Testing installed wheel:" not in result.stdout


def test_rejects_wheel_missing_native_library(wheel_environment):
    result = run_tests(wheel_environment(with_library=False))
    assert result.returncode != 0
    assert "native library is missing or misplaced" in result.stderr


def test_rejects_direct_cargo_extension_override(wheel_environment):
    environment = wheel_environment()
    environment[2]["VECTORLITE_RUST_EXTENSION"] = "some-other-extension"
    result = run_tests(environment)
    assert result.returncode != 0
    assert "must not use VECTORLITE_RUST_EXTENSION" in result.stderr


def test_propagates_pytest_failures(wheel_environment):
    environment = wheel_environment()
    checkout = environment[0]
    (checkout / "vectorlite/tests/test_failure.py").write_text(
        "def test_failure():\n    assert False, 'intentional regression failure'\n",
        encoding="utf-8",
    )
    result = run_tests(environment)
    assert result.returncode == 1
    assert "1 failed, 2 passed" in result.stdout
