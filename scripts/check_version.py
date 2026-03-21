#!/usr/bin/env python3
"""Verify all version strings match the VERSION file.

Usage: python scripts/check_version.py
Exit code 0 if all versions match, 1 if any mismatch.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_expected_version() -> str:
    return (ROOT / "VERSION").read_text().strip()


def check_file_regex(path: Path, pattern: str, expected: str, errors: list) -> None:
    text = path.read_text()
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        errors.append(f"  {path}: pattern not found")
    elif match.group(1) != expected:
        errors.append(f"  {path}: found '{match.group(1)}', expected '{expected}'")


def check_json_version(path: Path, expected: str, errors: list) -> None:
    data = json.loads(path.read_text())
    actual = data.get("version", "")
    if actual != expected:
        errors.append(f"  {path}: found '{actual}', expected '{expected}'")


def main() -> None:
    expected = read_expected_version()
    print(f"Expected version: {expected}")

    errors: list[str] = []

    # pyproject.toml
    check_file_regex(
        ROOT / "pyproject.toml",
        r'^version = "(.+)"',
        expected,
        errors,
    )

    # Python __init__.py
    check_file_regex(
        ROOT / "bindings" / "python" / "vectorlite_py" / "__init__.py",
        r"^__version__ = '(.+)'",
        expected,
        errors,
    )

    # vcpkg.json
    check_file_regex(
        ROOT / "vcpkg.json",
        r'"version-string": "(.+)"',
        expected,
        errors,
    )

    # Node.js package.json files
    nodejs_dir = ROOT / "bindings" / "nodejs" / "packages"
    for pkg_name in [
        "vectorlite",
        "vectorlite-darwin-arm64",
        "vectorlite-darwin-x64",
        "vectorlite-linux-x64",
        "vectorlite-win32-x64",
    ]:
        check_json_version(nodejs_dir / pkg_name / "package.json", expected, errors)

    # Main package optionalDependencies
    main_pkg = json.loads((nodejs_dir / "vectorlite" / "package.json").read_text())
    for dep, ver in main_pkg.get("optionalDependencies", {}).items():
        if ver != expected:
            errors.append(
                f"  {nodejs_dir / 'vectorlite' / 'package.json'}: "
                f"optionalDependencies[{dep}] = '{ver}', expected '{expected}'"
            )

    if errors:
        print("Version mismatches found:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("All versions match.")


if __name__ == "__main__":
    main()
