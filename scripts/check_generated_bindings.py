#!/usr/bin/env python3
"""Regenerate the canonical SQLite API bindings and fail if they drift.

Run this on the host that produces the committed artifact (macOS in CI). The
per-platform struct layout is verified separately by the abi-check C test, which
compiles against each host's real sqlite3 headers. Two normalizations keep the
generation stable:

* the generator parses a copy of the sqlite3 headers with `va_list` rewritten to
  `void *` (the three va_list slots are replaced with opaque pointers anyway), so
  the layout does not depend on the platform's `va_list` typedef; and
* bindgen's prettyplease output is re-formatted with rustfmt, which is the style
  the committed file uses.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "vectorlite" / "vectorlite-sqlite-sys" / "src" / "bindings.rs"
BINDINGS_WARNING = re.compile(r"Generated target bindings at (.+)")


def run(command, **kwargs):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        sys.stdout.write(result.stdout[-6000:])
        sys.stderr.write(result.stderr[-6000:])
    return result


def strip_placeholders(text: str, names) -> str:
    """Drop the canonical opaque va_list slot lines so platforms can be compared."""
    keep = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if any(stripped.startswith(f"pub(crate) {name}:") or stripped.startswith(f"pub {name}:")
               for name in names):
            continue
        keep.append(line)
    return "".join(keep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the temporary generated copy")
    args = parser.parse_args()

    if shutil.which("rustfmt") is None:
        print("rustfmt is required for this check", file=sys.stderr)
        return 1

    # Force the build script to run so the comparison is real rather than cached.
    clean = run(["cargo", "clean", "-p", "vectorlite-sqlite-sys"])
    if clean.returncode != 0:
        return clean.returncode

    build = run(
        ["cargo", "test", "--locked", "-p", "vectorlite-sqlite-sys",
         "--features", "regenerate,abi-check"],
        env=os.environ.copy(),
    )
    if build.returncode != 0:
        return build.returncode

    reported = BINDINGS_WARNING.findall(build.stdout + build.stderr)
    if not reported:
        print("the build script did not report a generated bindings path", file=sys.stderr)
        return 1
    generated = Path(reported[-1].strip())
    if not generated.is_file():
        print(f"generated bindings are missing: {generated}", file=sys.stderr)
        return 1

    directory = Path(tempfile.mkdtemp(prefix="vectorlite-bindings-"))
    candidate = directory / "bindings.rs"
    try:
        shutil.copyfile(generated, candidate)
        formatted = run(["rustfmt", "--edition", "2021", str(candidate)])
        if formatted.returncode != 0:
            return formatted.returncode
        # bindgen drops struct fields whose type it cannot represent, and which
        # types are representable differs per platform (Linux spells va_list as
        # __builtin_va_list). The committed file always carries the canonical opaque
        # placeholders for those three slots, so compare with those single-line
        # fields removed from both sides; the per-platform struct layout itself is
        # asserted by the abi-check C test that runs on every platform.
        placeholders = ("vmprintf", "xvsnprintf", "str_vappendf")
        committed_text = COMMITTED.read_text(encoding="utf-8")
        for name in placeholders:
            marker = f"    pub(crate) {name}:"
            if marker not in committed_text:
                print(f"committed bindings lost the canonical placeholder for {name}", file=sys.stderr)
                return 1
        stripped_committed = directory / "bindings-committed.rs"
        stripped_candidate = directory / "bindings-generated.rs"
        stripped_committed.write_text(
            strip_placeholders(committed_text, placeholders), encoding="utf-8"
        )
        stripped_candidate.write_text(
            strip_placeholders(candidate.read_text(encoding="utf-8"), placeholders),
            encoding="utf-8",
        )
        diff = run(["diff", "-u", str(stripped_candidate), str(stripped_committed)])
        if diff.returncode != 0:
            sys.stdout.write(diff.stdout[-12000:])
            print(
                "Committed SQLite bindings differ from regeneration; re-run this script, "
                "copy the generated file, and format it with rustfmt.",
                file=sys.stderr,
            )
            return 1
    finally:
        if args.keep:
            print(f"kept generated copy: {candidate}")
        else:
            shutil.rmtree(directory, ignore_errors=True)

    print("Committed SQLite bindings match regeneration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
