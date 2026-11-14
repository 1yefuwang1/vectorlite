#!/usr/bin/env python3
"""Build and run the separately linked native Tokio host against an extension.

Only the host executable links SQLite; the extension still uses the host API.
The test has a separate Cargo target directory to avoid stale-task/runtime claims
based only on the extension's unit-test linkage. Cargo's --locked preserves the
fixture's exact dependency versions; dependencies already used by the main crate
are normally present in its cache.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--sqlite-lib-dir", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    library = args.sqlite_lib_dir.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    target_dir = args.target_dir.resolve()
    if not library.is_dir() or not extension.is_file():
        raise ValueError("Invalid host library or extension path")
    environment = os.environ.copy()
    flags = ["-L", f"native={library}"]
    if sys.platform == "darwin":
        flags += ["-l", "dylib=iconv"]
    elif sys.platform.startswith("linux"):
        flags += ["-l", "dylib=dl", "-l", "dylib=pthread", "-l", "dylib=m"]
    # Encoded flags retain paths with spaces on all platforms and do not inherit
    # Cargo target-directory/RUSTFLAGS assumptions from the extension build.
    environment.pop("RUSTFLAGS", None)
    environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    command = [args.cargo, "build", "--locked", "--manifest-path", str(manifest),
               "--target", args.target, "--target-dir", str(target_dir)]
    if environment.get("CARGO_NET_OFFLINE") == "true":
        command.append("--offline")
    subprocess.run(command, env=environment, check=True)
    filename = "vectorlite-batch-tokio-host.exe" if sys.platform == "win32" else "vectorlite-batch-tokio-host"
    executable = target_dir / args.target / "debug" / filename
    return subprocess.run([str(executable), str(extension)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
