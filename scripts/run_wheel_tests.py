#!/usr/bin/env python3
"""Run SQL integration suites against an installed wheel, never the checkout."""

from importlib.metadata import distribution
import os
from pathlib import Path
import sys


def main():
    if os.environ.get("VECTORLITE_RUST_EXTENSION"):
        raise SystemExit("Wheel tests must not use VECTORLITE_RUST_EXTENSION")

    # Preload the installed package before pytest imports package parents while
    # collecting bindings/python/vectorlite_py/test. importlib mode alone does
    # not prevent pytest from importing the checkout's vectorlite_py package.
    import vectorlite_py

    installed_package = Path(
        distribution("vectorlite_py").locate_file("vectorlite_py/__init__.py")
    ).resolve()
    imported_package = Path(vectorlite_py.__file__).resolve()
    if imported_package != installed_package:
        raise SystemExit(
            f"Installed wheel is shadowed by {imported_package}; "
            f"expected {installed_package}. Remove the source-tree PYTHONPATH."
        )

    suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    library = Path(vectorlite_py.vectorlite_path()).with_suffix(suffix).resolve()
    if library != installed_package.with_name(f"vectorlite{suffix}") or not library.is_file():
        raise SystemExit(f"Installed wheel's native library is missing or misplaced: {library}")
    print(f"Testing installed wheel: {imported_package}", flush=True)
    print(f"Native library: {library}", flush=True)

    import pytest

    repo_root = Path(__file__).resolve().parents[1]
    return pytest.main([
        "--import-mode=importlib",
        str(repo_root / "bindings/python/vectorlite_py/test"),
        str(repo_root / "vectorlite/tests"),
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
