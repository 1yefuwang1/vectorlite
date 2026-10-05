#!/bin/sh
# npm platform packages ship the exact Rust extension from the validated wheels.
set -eu
repo_root=$(CDPATH= cd "$(dirname "$0")" && pwd)
"${PYTHON:-python3}" - "$repo_root" <<'PY'
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1]).resolve()
platforms = {
    "linux-x64": ("vectorlite-linux-x64", "vectorlite.so"),
    "win32-x64": ("vectorlite-win32-x64", "vectorlite.dll"),
    "darwin-arm64": ("vectorlite-darwin-arm64", "vectorlite.dylib"),
    "darwin-x64": ("vectorlite-darwin-x64", "vectorlite.dylib"),
}
selected = {}
for wheel in sorted((root / "wheelhouse").glob("vectorlite-wheel*/*.whl")):
    name = wheel.name
    if "linux" in name and name.endswith("x86_64.whl"):
        platform = "linux-x64"
    elif "win" in name and name.endswith("amd64.whl"):
        platform = "win32-x64"
    elif "macosx" in name and name.endswith("arm64.whl"):
        platform = "darwin-arm64"
    elif "macosx" in name and name.endswith("x86_64.whl"):
        platform = "darwin-x64"
    else:
        raise SystemExit(f"Unsupported wheel platform: {wheel}")
    if platform in selected:
        raise SystemExit(f"Multiple wheels for {platform}: {selected[platform]} and {wheel}")
    selected[platform] = wheel

missing = platforms.keys() - selected.keys()
if missing:
    raise SystemExit(f"Missing validated platform wheels: {', '.join(sorted(missing))}")

# Validate every archive before modifying the platform-package binaries. Private
# repair sidecars require preserving their relative loader paths; don't silently
# publish a bare binary with missing dependencies if that ever becomes necessary.
binaries = []
for platform, (package, filename) in platforms.items():
    wheel = selected[platform]
    with zipfile.ZipFile(wheel) as archive:
        sidecars = [
            member for member in archive.namelist()
            if ".libs/" in member or ".dylibs/" in member
        ]
        if sidecars:
            raise SystemExit(f"Wheel {wheel} has private runtime sidecars; npm staging must preserve them: {sidecars}")
        binary = archive.read(f"vectorlite_py/{filename}")
    destination = root / "bindings/nodejs/packages" / package / "src" / filename
    if not destination.parent.is_dir():
        raise SystemExit(f"Platform package is missing: {destination.parent}")
    binaries.append((destination, binary))

for destination, binary in binaries:
    destination.write_bytes(binary)
    print(f"Staged Rust extension: {destination}")
PY
