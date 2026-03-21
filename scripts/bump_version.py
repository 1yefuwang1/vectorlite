#!/usr/bin/env python3
"""Bump version across all vectorlite packages.

Usage: python scripts/bump_version.py <new_version>
Example: python scripts/bump_version.py 0.3.0
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VERSION_FILE = ROOT / "VERSION"
PYPROJECT_TOML = ROOT / "pyproject.toml"
PYTHON_INIT = ROOT / "bindings" / "python" / "vectorlite_py" / "__init__.py"
VCPKG_JSON = ROOT / "vcpkg.json"
NODEJS_PACKAGES_DIR = ROOT / "bindings" / "nodejs" / "packages"

PLATFORM_PACKAGES = [
    "vectorlite-darwin-arm64",
    "vectorlite-darwin-x64",
    "vectorlite-linux-x64",
    "vectorlite-win32-x64",
]


def update_file_regex(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count == 0:
        print(f"  WARNING: no match for pattern in {path}")
    path.write_text(new_text)


def update_json_version(path: Path, version: str) -> None:
    data = json.loads(path.read_text())
    data["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <new_version>")
        sys.exit(1)

    version = sys.argv[1]

    if not re.match(r"^\d+\.\d+\.\d+(-[\w.]+)?$", version):
        print(f"Error: '{version}' is not a valid version (expected X.Y.Z)")
        sys.exit(1)

    # 1. VERSION file
    print(f"Updating {VERSION_FILE}")
    VERSION_FILE.write_text(version + "\n")

    # 2. pyproject.toml
    print(f"Updating {PYPROJECT_TOML}")
    update_file_regex(
        PYPROJECT_TOML,
        r'^version = ".*"',
        f'version = "{version}"',
    )

    # 3. Python __init__.py
    print(f"Updating {PYTHON_INIT}")
    update_file_regex(
        PYTHON_INIT,
        r"^__version__ = '.*'",
        f"__version__ = '{version}'",
    )

    # 4. vcpkg.json
    print(f"Updating {VCPKG_JSON}")
    update_file_regex(
        VCPKG_JSON,
        r'"version-string": ".*"',
        f'"version-string": "{version}"',
    )

    # 5. Main Node.js package.json (version + optionalDependencies)
    main_pkg = NODEJS_PACKAGES_DIR / "vectorlite" / "package.json"
    print(f"Updating {main_pkg}")
    data = json.loads(main_pkg.read_text())
    data["version"] = version
    if "optionalDependencies" in data:
        for dep in data["optionalDependencies"]:
            data["optionalDependencies"][dep] = version
    main_pkg.write_text(json.dumps(data, indent=2) + "\n")

    # 6. Platform-specific Node.js package.json files
    for pkg_name in PLATFORM_PACKAGES:
        pkg_path = NODEJS_PACKAGES_DIR / pkg_name / "package.json"
        print(f"Updating {pkg_path}")
        update_json_version(pkg_path, version)

    print(f"\nVersion bumped to {version}")


if __name__ == "__main__":
    main()
