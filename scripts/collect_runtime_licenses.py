#!/usr/bin/env python3
"""Collect declared licenses and exact source notices for selected runtime crates.

This records upstream data, not legal advice or a choice among SPDX alternatives.
Normal dependency edges are followed per target; build/dev edges and proc-macro
subtrees are excluded. Some unused code may be eliminated by the linker; this
inventory conservatively covers all selected normal Rust dependency crates.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc", "aarch64-apple-darwin"]
LEGAL = re.compile(r"copyright|permission (?:is hereby|to use)|redistribution and use|SPDX-License-Identifier", re.I)
LEADING_COMMENTS = re.compile(r"(?:\s*//[^\n]*(?:\n|$)|\s*/\*[\s\S]*?\*/|\s+)*")
TEXT_PREFIXES = ("license", "licence", "copying", "copyright", "notice", "unlicense")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def selected(data):
    packages = {package["id"]: package for package in data["packages"]}
    nodes = {node["id"]: node for node in data["resolve"]["nodes"]}
    root = next(package["id"] for package in data["packages"] if package["name"] == "vectorlite")
    pending = [root]
    found = {}
    while pending:
        package_id = pending.pop()
        if package_id in found:
            continue
        package = packages[package_id]
        if any("proc-macro" in target["kind"] for target in package["targets"]):
            continue
        found[package_id] = package
        for dependency in nodes[package_id]["deps"]:
            if any(kind["kind"] is None for kind in dependency["dep_kinds"]):
                pending.append(dependency["pkg"])
    return {key: value for key, value in found.items() if value["source"] is not None}, nodes


def collect(metadata, output):
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing bundle: {output}")
    inventory = {}
    exclusions = set()
    for target, data in metadata.items():
        packages, nodes = selected(data)
        for package_id, package in packages.items():
            entry = inventory.setdefault(package_id, {"package": package, "targets": [], "features": {}})
            entry["targets"].append(target)
            entry["features"][target] = nodes[package_id]["features"]
        exclusions.update((package["name"], package["version"]) for package in data["packages"] if package["source"] and package["id"] not in packages)
    import tomllib
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    checksums = {(p["name"], p["version"]): p.get("checksum") for p in lock["package"]}
    records = []
    unresolved = []
    for entry in sorted(inventory.values(), key=lambda item: (item["package"]["name"], item["package"]["version"])):
        package = entry["package"]
        crate = f"{package['name']}-{package['version']}"
        source = Path(package["manifest_path"]).parent
        destination = output / "crates" / crate
        files = []
        def store(name, payload, provenance):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            files.append({"path": f"crates/{crate}/{name}", "sha256": digest(payload), **provenance})
        discovered = [path for path in source.rglob("*") if path.is_file() and path.name.lower().startswith(TEXT_PREFIXES)]
        for path in sorted(discovered):
            relative = path.relative_to(source).as_posix()
            store(relative, path.read_bytes(), {"origin": "published crate archive", "source_path": relative})
        if package["name"].startswith("diskann"):
            for name in ("LICENSE.txt", "NOTICE.txt"):
                canonical = ROOT / "third_party/diskann-0.60.0" / name
                store(name, canonical.read_bytes(), {
                    "origin": "pinned upstream repository, absent from published crate archive",
                    "source_path": name,
                    "source_url": f"https://raw.githubusercontent.com/microsoft/DiskANN/97a828a500848018d8be28c3ea6d5a585b07362f/{name}",
                })
        headers = []
        for path in sorted((source / "src").rglob("*")):
            if not path.is_file() or path.suffix not in {".rs", ".c", ".h"}:
                continue
            payload = path.read_bytes()
            try:
                text = payload.decode("utf-8")
            except UnicodeError:
                continue
            prefix = LEADING_COMMENTS.match(text).group(0)
            if not LEGAL.search(prefix):
                continue
            relative = path.relative_to(source).as_posix()
            excerpt = prefix.encode("utf-8")
            headers.append((relative, excerpt, digest(payload)))
        if headers:
            sections = []
            ranges = []
            for relative, excerpt, checksum in headers:
                sections.append(f"===== Unmodified leading source comments: {relative} =====\n".encode("utf-8") + excerpt + b"\n")
                ranges.append({"source_path": relative, "source_sha256": checksum, "notice_sha256": digest(excerpt), "bytes_from_start": len(excerpt)})
            store("SOURCE_NOTICES.txt", b"".join(sections), {"origin": "unmodified leading source comments", "source_excerpts": ranges})
        vcs_path = source / ".cargo_vcs_info.json"
        vcs = json.loads(vcs_path.read_text(encoding="utf-8")) if vcs_path.exists() else None
        if not files:
            unresolved.append({"crate": crate, "issue": "No included license/copyright notice text found; declaration alone is not redistribution text."})
        records.append({
            "name": package["name"], "version": package["version"],
            "declared_spdx": package["license"], "declared_license_file": package["license_file"],
            "repository": package["repository"], "registry_source": package["source"],
            "registry_archive_url": f"https://static.crates.io/crates/{package['name']}/{crate}.crate",
            "registry_archive_sha256": checksums[(package["name"], package["version"])],
            "vcs": vcs, "targets": sorted(entry["targets"]), "resolved_features": entry["features"], "files": files,
            "notes": ["All included license alternatives are retained; no SPDX OR alternative is selected.",
                      "Leading per-file license/copyright comment excerpts are retained conservatively, without claiming every function survives linking."],
        })
    chosen = {(record["name"], record["version"]) for record in records}
    manifest = {
        "format_version": 1, "cargo_lock_sha256": digest((ROOT / "Cargo.lock").read_bytes()),
        "target_triples": TARGETS, "collection_method": "cargo metadata --locked --format-version 1 --offline --filter-platform <target>; normal-edge traversal from vectorlite stopping before proc-macro subtrees",
        "scope": "Conservative union of selected external normal runtime dependencies for official target triples; build/dev/proc-macro dependencies and unused optional dependency branches excluded. Project/workspace license and native HNSW/Highway notices are packaged separately. Rust standard-library/toolchain licensing is outside this crate inventory. This manifest is provenance, not legal review or a redistribution-compliance certification.",
        "excluded_packages": [{"name": name, "version": version} for name, version in sorted(exclusions - chosen)],
        "packages": records, "unresolved_notice_gaps": unresolved,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="NEW destination directory")
    args = parser.parse_args()
    data = {}
    for target in TARGETS:
        result = subprocess.run(["cargo", "metadata", "--locked", "--format-version", "1", "--offline", "--filter-platform", target], cwd=ROOT, check=True, capture_output=True, text=True)
        data[target] = json.loads(result.stdout)
    manifest = collect(data, args.output.resolve())
    print(f"Collected {len(manifest['packages'])} runtime crates; {len(manifest['unresolved_notice_gaps'])} missing notice texts")


if __name__ == "__main__":
    main()
