#!/usr/bin/env python3
"""Run the native batch host with a fresh, workspace-owned SQLite file.

The native fixture exclusively reserves its file and leaves it for this managed
wrapper to clean up. All destructive targets are checked absolute direct children
of the exact tempfile directory; no caller-supplied deletion path is accepted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    work_root = args.work_root.resolve(strict=True)
    if not executable.is_file() or not extension.is_file() or not work_root.is_dir():
        raise ValueError("Native test executable, extension, or work root is invalid")
    directory = Path(tempfile.mkdtemp(prefix="batch-native-", dir=work_root)).resolve(strict=True)
    if directory.parent != work_root or not directory.name.startswith("batch-native-"):
        raise ValueError("Unexpected native test directory")
    database = directory / "batch.sqlite"
    try:
        result = subprocess.run([str(executable), str(extension), str(database)], check=False)
        return result.returncode
    finally:
        allowed = {"batch.sqlite", "batch.sqlite-wal", "batch.sqlite-shm", "batch.sqlite-journal"}
        paths = list(directory.iterdir())
        for path in paths:
            if (path.name not in allowed or path.is_symlink()
                    or path.resolve().parent != directory or not path.is_file()):
                raise ValueError(f"Refusing to clean unexpected native test artifact: {path}")
        for path in paths:
            path.unlink()
        if directory.resolve(strict=True).parent != work_root:
            raise ValueError("Native test cleanup directory changed")
        directory.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
