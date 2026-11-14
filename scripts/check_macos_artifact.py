#!/usr/bin/env python3
"""Audit packaged notices and macOS binary metadata, not macOS 11 runtime behavior.

Supports thin arm64 Mach-O dylibs, native static archives, and built wheels.
Archive members are checked individually: relinking a newer native object with a
lower final deployment target does not establish compatibility with that target.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MAXIMUM_MACOS = (11, 0, 0)
CPU_TYPE_ARM64 = 0x0100000C
NOTICE_PREFIX = "vectorlite_py/licenses/"
DISKANN_VERSION = "0.60.0"
DISKANN_REVISION = "97a828a500848018d8be28c3ea6d5a585b07362f"
UPSTREAM_BLOBS = {
    "LICENSE.txt": "b2f52a2bad4e27e2d9c68a755abb74cb8943f2fa",
    "NOTICE.txt": "faf70aa99f51f9bdd98c82b8949ba2826d408a23",
}
REQUIRED_NOTICES = {
    "LICENSE.txt",
    "diskann-0.60.0/LICENSE.txt",
    "diskann-0.60.0/NOTICE.txt",
    "diskann-0.60.0/PROVENANCE.json",
    "hnswlib/LICENSE.txt",
    "highway/LICENSE.txt",
    "rust-runtime/MANIFEST.json",
    "rust-runtime/README.md",
}


def _version(encoded):
    return encoded >> 16, (encoded >> 8) & 255, encoded & 255


def validate_macho(data, label="artifact", *, file_type=6):
    """Return the declared minimum OS; reject absent/unsupported binary metadata."""
    if len(data) < 32 or data[:4] != b"\xcf\xfa\xed\xfe":
        raise ValueError(f"{label}: expected a thin little-endian 64-bit Mach-O")
    _, cpu, _, kind, count, command_bytes, _, _ = struct.unpack_from("<IiiIIIII", data)
    if cpu != CPU_TYPE_ARM64:
        raise ValueError(f"{label}: expected arm64 architecture")
    if kind != file_type:
        raise ValueError(f"{label}: unexpected Mach-O file type {kind}")
    end = 32 + command_bytes
    if end > len(data) or count > command_bytes // 8:
        raise ValueError(f"{label}: truncated Mach-O load commands")
    position = 32
    versions = []
    for _ in range(count):
        if position + 8 > end:
            raise ValueError(f"{label}: truncated Mach-O command header")
        command, length = struct.unpack_from("<II", data, position)
        if length < 8 or length % 4 or position + length > end:
            raise ValueError(f"{label}: invalid Mach-O command length")
        if command == 0x32:  # LC_BUILD_VERSION
            if length < 24:
                raise ValueError(f"{label}: truncated LC_BUILD_VERSION")
            platform, minimum, _, tools = struct.unpack_from("<IIII", data, position + 8)
            if platform != 1 or tools > (length - 24) // 8:
                raise ValueError(f"{label}: invalid macOS build-version metadata")
            versions.append(_version(minimum))
        elif command == 0x24:  # LC_VERSION_MIN_MACOSX
            if length < 16:
                raise ValueError(f"{label}: truncated LC_VERSION_MIN_MACOSX")
            versions.append(_version(struct.unpack_from("<I", data, position + 8)[0]))
        position += length
    if position != end or len(versions) != 1:
        raise ValueError(f"{label}: missing or ambiguous macOS minimum-version metadata")
    minimum = versions[0]
    if minimum > MAXIMUM_MACOS:
        raise ValueError(f"{label}: macOS minimum {'.'.join(map(str, minimum))} exceeds 11.0")
    return minimum


def validate_archive(data, label="archive"):
    """Check every Mach-O object in a BSD/GNU ar archive; skip only symbol tables."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError(f"{label}: expected an ar static archive")
    position = 8
    strings = b""
    checked = 0
    while position < len(data):
        if position + 60 > len(data):
            raise ValueError(f"{label}: truncated archive member header")
        header = data[position:position + 60]
        if header[58:] != b"`\n":
            raise ValueError(f"{label}: invalid archive member header")
        try:
            name = header[:16].decode("ascii").rstrip()
            length = int(header[48:58].decode("ascii").strip())
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"{label}: invalid archive member fields") from error
        position += 60
        if length < 0 or position + length > len(data):
            raise ValueError(f"{label}: truncated archive member")
        payload = data[position:position + length]
        position += length
        if length % 2:
            if position >= len(data):
                raise ValueError(f"{label}: missing archive padding")
            position += 1
        if name == "//":
            strings = payload
            continue
        if name in {"/", "/SYM64/"}:
            continue
        if name.startswith("#1/"):
            try:
                name_length = int(name[3:])
                if name_length < 0 or name_length > len(payload):
                    raise ValueError("invalid extended name length")
                name = payload[:name_length].decode("utf-8").rstrip("\0")
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"{label}: invalid BSD archive member name") from error
            payload = payload[name_length:]
        elif re.fullmatch(r"/\d+", name):
            offset = int(name[1:])
            terminator = strings.find(b"\n", offset)
            if offset >= len(strings) or terminator < 0:
                raise ValueError(f"{label}: invalid GNU archive member name")
            name = strings[offset:terminator].decode("utf-8").rstrip("/")
        else:
            name = name.rstrip("/")
        if name.startswith("__.SYMDEF"):
            continue
        validate_macho(payload, f"{label}({name})", file_type=1)
        checked += 1
    if not checked:
        raise ValueError(f"{label}: archive contains no auditable Mach-O objects")
    return checked


def notice_payloads(archive):
    """Return bounded notice files, checking required texts and pinned provenance."""
    members = archive.namelist()
    if len(members) != len(set(members)):
        raise ValueError("Archive contains duplicate members")
    notices = {}
    for info in archive.infolist():
        if not info.filename.startswith(NOTICE_PREFIX) or info.is_dir():
            continue
        relative = info.filename[len(NOTICE_PREFIX):]
        if not relative or any(part in {"", ".", ".."} for part in relative.split("/")) or "\\" in relative:
            raise ValueError("Invalid bundled notice path")
        if info.file_size > 1024 * 1024:
            raise ValueError(f"Bundled notice is unexpectedly large: {relative}")
        notices[relative] = archive.read(info)
    missing = REQUIRED_NOTICES - notices.keys()
    if missing:
        raise ValueError(f"Missing bundled notice files: {', '.join(sorted(missing))}")
    if any(not notices[name].strip() for name in REQUIRED_NOTICES):
        raise ValueError("Bundled notice file is empty")
    if notices["LICENSE.txt"] != (ROOT / "LICENSE").read_bytes():
        raise ValueError("Bundled project license differs from the unchanged Apache license")
    for name in ("LICENSE.txt", "NOTICE.txt", "PROVENANCE.json"):
        key = f"diskann-{DISKANN_VERSION}/{name}"
        if notices[key] != (ROOT / "third_party" / f"diskann-{DISKANN_VERSION}" / name).read_bytes():
            raise ValueError(f"Bundled DiskANN notice differs from pinned source: {name}")
    provenance = json.loads(notices[f"diskann-{DISKANN_VERSION}/PROVENANCE.json"])
    if provenance.get("version") != DISKANN_VERSION or provenance.get("revision") != DISKANN_REVISION:
        raise ValueError("Unsupported DiskANN notice provenance")
    if provenance.get("source_git_blobs") != UPSTREAM_BLOBS:
        raise ValueError("Unsupported DiskANN notice blob provenance")
    for name, expected in UPSTREAM_BLOBS.items():
        payload = notices[f"diskann-{DISKANN_VERSION}/{name}"]
        git_blob = f"blob {len(payload)}\0".encode("ascii") + payload
        if hashlib.sha1(git_blob).hexdigest() != expected:
            raise ValueError(f"Bundled DiskANN notice differs from upstream Git object: {name}")
    validate_runtime_notices(notices)
    return notices


def validate_runtime_notices(notices):
    """Verify committed inventory/lock identity and every runtime notice payload."""
    bundle = ROOT / "third_party/rust-runtime"
    expected_manifest = (bundle / "MANIFEST.json").read_bytes()
    if notices["rust-runtime/MANIFEST.json"] != expected_manifest:
        raise ValueError("Bundled runtime license manifest differs from selected inventory")
    manifest = json.loads(expected_manifest)
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported runtime license manifest version")
    if hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest() != manifest["cargo_lock_sha256"]:
        raise ValueError("Runtime license inventory must be refreshed after Cargo.lock changes")
    if manifest["unresolved_notice_gaps"]:
        raise ValueError("Runtime license inventory reports unresolved notice gaps")
    expected = {"rust-runtime/MANIFEST.json", "rust-runtime/README.md"}
    if notices["rust-runtime/README.md"] != (bundle / "README.md").read_bytes():
        raise ValueError("Bundled runtime license scope documentation differs from inventory")
    for package in manifest["packages"]:
        for record in package["files"]:
            relative = record["path"]
            if any(part in {"", ".", ".."} for part in relative.split("/")) or "\\" in relative:
                raise ValueError("Invalid runtime license manifest path")
            key = f"rust-runtime/{relative}"
            expected.add(key)
            if key not in notices:
                raise ValueError(f"Missing bundled runtime notice: {key}")
            if hashlib.sha256(notices[key]).hexdigest() != record["sha256"]:
                raise ValueError(f"Bundled runtime notice checksum mismatch: {key}")
    actual = {name for name in notices if name.startswith("rust-runtime/")}
    if actual != expected:
        raise ValueError("Bundled runtime notice paths differ from selected inventory")


def validate_wheel(path):
    with zipfile.ZipFile(path) as archive:
        notice_payloads(archive)
        if "macosx" in path.name:
            if not path.name.endswith("-macosx_11_0_arm64.whl"):
                raise ValueError(f"{path}: supported macOS wheel tag is macosx_11_0_arm64")
            validate_macho(archive.read("vectorlite_py/vectorlite.dylib"), str(path))


def validate_path(path):
    if path.suffix == ".whl":
        validate_wheel(path)
    else:
        data = path.read_bytes()
        if data.startswith(b"!<arch>\n"):
            validate_archive(data, str(path))
        else:
            validate_macho(data, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.artifacts:
        try:
            validate_path(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            raise SystemExit(f"Artifact validation failed: {error}") from error
        print(f"Validated artifact metadata/notices: {path}")
    print("Metadata checks do not replace execution on macOS 11.")


if __name__ == "__main__":
    main()
