#!/usr/bin/env python3
"""Build the standalone driver against an explicitly supplied pinned upstream tree.

Does not modify vectorlite's Cargo manifest, lockfile, or production dependencies.
The output directory must be new. Source acquisition is intentionally explicit;
see the companion plan for the pinned codeload URL and archive fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time

REVISION = "97a828a500848018d8be28c3ea6d5a585b07362f"
INDEX_SHA256 = "26cd82192db87606a524aa2f4aa56f90da7ab3d00f588721bcd39dbdeb0b4cde"
ARCHIVE_SHA256 = "88ea51a9db42d483b330d1be1ea764d0fa8ac7e39876aa9bb02878bd0c1377c1"


def verify_source(upstream: Path, archive: Path) -> int:
    if fingerprint(archive)["sha256"] != ARCHIVE_SHA256:
        raise ValueError("source archive does not match the pinned measured revision")
    prefix = f"DiskANN-{REVISION}/"
    checked = 0
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if not member.isfile():
                continue
            if not member.name.startswith(prefix):
                raise ValueError("unexpected archive root")
            relative = member.name[len(prefix):]
            path = (upstream / relative).resolve(strict=True)
            if not path.is_relative_to(upstream):
                raise ValueError("source file escapes the upstream tree")
            payload = source.extractfile(member)
            if payload is None or hashlib.sha256(payload.read()).hexdigest() != fingerprint(path)["sha256"]:
                raise ValueError(f"upstream file changed: {relative}")
            checked += 1
    return checked


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--target-dir", type=Path, help="optional existing Cargo build cache")
    parser.add_argument("--cargo-home", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    upstream = args.upstream_source.resolve(strict=True)
    archive = args.source_archive.resolve(strict=True)
    verified_files = verify_source(upstream, archive)
    if fingerprint(upstream / "diskann/src/graph/index.rs")["sha256"] != INDEX_SHA256:
        raise ValueError("graph source does not match the pinned unmodified upstream revision")
    if args.build_dir.exists() or args.build_dir.is_symlink():
        raise ValueError("build directory must be new")
    args.build_dir.mkdir(parents=True)
    build = args.build_dir.resolve()
    manifest = build / "Cargo.toml"
    dependencies = []
    for name in ("diskann", "diskann-providers", "diskann-disk", "diskann-utils", "diskann-vector"):
        path = upstream / name
        if not (path / "Cargo.toml").is_file():
            raise ValueError(f"missing upstream crate {name}")
        dependencies.append(f'{name} = {{ path = {json.dumps(str(path))}, default-features = false }}')
    contents = "\n".join([
        "[package]", 'name = "vectorlite-upstream-diskann-benchmark"', 'version = "0.1.0"',
        'edition = "2021"', "", "[workspace]", "", "[[bin]]", 'name = "upstream_diskann_benchmark"',
        f'path = {json.dumps(str(root / "benchmark/upstream_diskann/src/main.rs"))}', "", "[dependencies]",
        *dependencies, 'anyhow = "=1.0.104"', 'serde_json = "=1.0.151"',
        'tokio = { version = "=1.53.2", features = ["rt", "rt-multi-thread"] }', "",
        "[profile.release]", "opt-level = 3", "codegen-units = 1", 'panic = "abort"', "",
    ])
    manifest.write_text(contents)
    # Prefer the distributed standalone lock if present; otherwise seed upstream's
    # lock and let Cargo resolve the small benchmark root once, recording the result.
    lock = root / "benchmark/upstream_diskann/Cargo.lock"
    if not lock.exists():
        lock = upstream / "Cargo.lock"
    shutil.copyfile(lock, build / "Cargo.lock")
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(args.target_dir.resolve() if args.target_dir else build / "target")
    env["RUSTUP_TOOLCHAIN"] = "stable"
    env["RUSTFLAGS"] = ""
    if args.cargo_home:
        env["CARGO_HOME"] = str(args.cargo_home.resolve())
    if platform.system() == "Darwin":
        env["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
    command = ["cargo", "build", "--release", "--manifest-path", str(manifest)]
    if (root / "benchmark/upstream_diskann/Cargo.lock").exists():
        command.insert(2, "--locked")
    start = time.monotonic()
    result = subprocess.run(command, env=env, check=False)
    evidence = {
        "revision": REVISION,
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "environment": {key: env.get(key) for key in (
            "CARGO_TARGET_DIR", "CARGO_HOME", "RUSTUP_TOOLCHAIN", "RUSTFLAGS", "MACOSX_DEPLOYMENT_TARGET")},
        "source": fingerprint(upstream / "diskann/src/graph/index.rs"),
        "source_archive": fingerprint(archive),
        "unchanged_upstream_files_before_build": verified_files,
        "unchanged_upstream_files_after_build": verify_source(upstream, archive),
        "manifest": fingerprint(manifest),
        "lockfile": fingerprint(build / "Cargo.lock"),
        "rustc_verbose": subprocess.check_output(["rustc", "--version", "--verbose"], env=env, text=True),
        "rustc_cfg": subprocess.check_output(["rustc", "--print", "cfg"], env=env, text=True),
        "cargo_version": subprocess.check_output(["cargo", "--version"], env=env, text=True),
        "host": platform.platform(),
        "driver_sources": [fingerprint(root / name) for name in (
            "benchmark/upstream_diskann/src/main.rs", "benchmark/upstream_diskann/src/memory.rs",
            "benchmark/upstream_diskann/src/ssd.rs", "benchmark/build_upstream_diskann.py",
            "benchmark/upstream_diskann_benchmark.py", "benchmark/upstream_diskann/Cargo.lock")],
    }
    suffix = ".exe" if os.name == "nt" else ""
    executable = Path(env["CARGO_TARGET_DIR"]) / "release" / f"upstream_diskann_benchmark{suffix}"
    if result.returncode == 0:
        evidence["executable"] = fingerprint(executable)
    (build / "build-result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
