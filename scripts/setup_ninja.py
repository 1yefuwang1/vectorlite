#!/usr/bin/env python3
"""Install the pinned upstream Ninja release without a JavaScript setup action."""

import io
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile


VERSION = "1.11.1"
ASSETS = {"linux": "ninja-linux.zip", "darwin": "ninja-mac.zip", "win32": "ninja-win.zip"}


def install_ninja(platform):
    if platform not in ASSETS:
        raise RuntimeError(f"Unsupported Ninja host platform: {platform}")
    directory = Path(os.environ["RUNNER_TEMP"]).resolve() / f"ninja-{VERSION}"
    github_path = Path(os.environ["GITHUB_PATH"])
    filename = "ninja.exe" if platform == "win32" else "ninja"
    url = f"https://github.com/ninja-build/ninja/releases/download/v{VERSION}/{ASSETS[platform]}"
    with urllib.request.urlopen(url, timeout=60) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            binary = archive.read(filename)
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / filename
    executable.write_bytes(binary)
    if platform != "win32":
        executable.chmod(0o755)
    installed_version = subprocess.check_output([str(executable), "--version"], text=True).strip()
    if installed_version != VERSION:
        raise RuntimeError(f"Expected Ninja {VERSION}, got {installed_version!r}")
    with github_path.open("a", encoding="utf-8") as output:
        output.write(f"{directory}\n")
    print(f"Installed Ninja {installed_version}: {executable}")


if __name__ == "__main__":
    install_ninja(sys.platform)
