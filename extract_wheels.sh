#!/bin/sh
# npm platform packages ship the exact Rust extension from the validated wheels.
set -eu
repo_root=$(CDPATH= cd "$(dirname "$0")" && pwd)
"${PYTHON:-python3}" - "$repo_root" <<'PY'
import importlib.util
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("artifact_checks", root / "scripts/check_macos_artifact.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
platforms = {
    "linux-x64": ("vectorlite-linux-x64", "vectorlite.so"),
    "win32-x64": ("vectorlite-win32-x64", "vectorlite.dll"),
    "darwin-arm64": ("vectorlite-darwin-arm64", "vectorlite.dylib"),
}
selected = {}
for wheel in sorted((root / "wheelhouse").glob("vectorlite-wheel*/*.whl")):
    name = wheel.name
    if "linux" in name and name.endswith("x86_64.whl"):
        platform = "linux-x64"
    elif "win" in name and name.endswith("amd64.whl"):
        platform = "win32-x64"
    elif "macosx" in name and name.endswith("arm64.whl"):
        if not name.endswith("-macosx_11_0_arm64.whl"):
            raise SystemExit(f"Unsupported wheel platform: {wheel}; expected macosx_11_0_arm64")
        platform = "darwin-arm64"
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
        try:
            notices = checks.notice_payloads(archive)
            binary = archive.read(f"vectorlite_py/{filename}")
            if platform == "darwin-arm64":
                checks.validate_macho(binary, str(wheel))
        except (ValueError, KeyError) as error:
            raise SystemExit(f"Invalid wheel artifact {wheel}: {error}") from error
    destination = root / "bindings/nodejs/packages" / package / "src" / filename
    if not destination.parent.is_dir():
        raise SystemExit(f"Platform package is missing: {destination.parent}")
    binaries.append((destination, binary))
    binaries.append((destination.parent.parent / "LICENSE", notices["LICENSE.txt"]))
    for relative, payload in notices.items():
        binaries.append((destination.parent / "licenses" / relative, payload))

for destination, payload in binaries:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"Staged validated binary/notice: {destination}")
PY
