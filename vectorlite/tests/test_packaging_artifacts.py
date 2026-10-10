"""Synthetic artifact/license checks; no extension or package build is involved."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("artifact_checks", ROOT / "scripts/check_macos_artifact.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


def macho(minimum=11, *, architecture=0x0100000C, file_type=6, legacy=False):
    if legacy:
        command = struct.pack("<4I", 0x24, 16, minimum << 16, minimum << 16)
    else:
        command = struct.pack("<6I", 0x32, 24, 1, minimum << 16, minimum << 16, 0)
    return struct.pack("<IiiIIIII", 0xFEEDFACF, architecture, 0, file_type, 1, len(command), 0, 0) + command


def member(name, payload):
    header = f"{name:<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(payload):<10}`\n".encode("ascii")
    assert len(header) == 60
    return header + payload + (b"\n" if len(payload) % 2 else b"")


def notices():
    payloads = {"LICENSE.txt": (ROOT / "LICENSE").read_bytes()}
    for component in ("hnswlib", "highway"):
        payloads[f"{component}/LICENSE.txt"] = (ROOT / "LICENSE").read_bytes()
    for name in ("LICENSE.txt", "NOTICE.txt", "PROVENANCE.json"):
        payloads[f"diskann-0.60.0/{name}"] = (ROOT / "third_party/diskann-0.60.0" / name).read_bytes()
    runtime = ROOT / "third_party/rust-runtime"
    for path in runtime.rglob("*"):
        if path.is_file():
            payloads[f"rust-runtime/{path.relative_to(runtime).as_posix()}"] = path.read_bytes()
    return payloads


def zip_payloads(payloads):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(CHECKS.NOTICE_PREFIX + name, payload)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("legacy", [False, True])
def test_macos_metadata_accepts_thin_arm64_at_declared_floor(legacy):
    assert CHECKS.validate_macho(macho(legacy=legacy)) == (11, 0, 0)


@pytest.mark.parametrize("payload,message", [
    (macho(27), "exceeds 11.0"),
    (macho(architecture=0x01000007), "expected arm64"),
    (macho(file_type=1), "file type"),
    (b"\xca\xfe\xba\xbe" + bytes(64), "thin"),
    (macho()[:-1], "truncated"),
    (struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 6, 0, 0, 0, 0), "missing"),
])
def test_macos_metadata_rejects_higher_floor_architecture_and_truncation(payload, message):
    with pytest.raises(ValueError, match=message):
        CHECKS.validate_macho(payload)


def test_macos_load_command_sizes_and_platform_are_validated():
    for word, value in [(9, 4), (10, 2), (13, 999)]:
        broken = bytearray(macho())
        struct.pack_into("<I", broken, word * 4, value)
        with pytest.raises(ValueError):
            CHECKS.validate_macho(broken)


def test_archive_audits_each_object_not_just_final_link_target():
    current = macho(file_type=1)
    symbols = member("__.SYMDEF/", b"not a Mach-O symbol index")
    assert CHECKS.validate_archive(b"!<arch>\n" + symbols + member("good.o/", current)) == 1
    assert CHECKS.validate_macho(macho()) == (11, 0, 0)
    with pytest.raises(ValueError, match="old.o.*exceeds 11.0"):
        CHECKS.validate_archive(b"!<arch>\n" + member("good.o/", current) + member("old.o/", macho(27, file_type=1)))


def test_archive_supports_bsd_and_gnu_long_names():
    payload = macho(file_type=1)
    name = b"long-bsd-object-name.o"
    bsd = member(f"#1/{len(name)}", name + payload)
    gnu = member("//", b"long-gnu-object-name.o/\n") + member("/0", payload)
    assert CHECKS.validate_archive(b"!<arch>\n" + bsd + gnu) == 2


@pytest.mark.parametrize("payload", [
    b"!<arch>\n", b"!<arch>\nshort", b"!<arch>\n" + member("empty.o/", b"bad"),
    b"!<arch>\n" + member("/0", macho(file_type=1)),
])
def test_archive_rejects_missing_or_unverifiable_members(payload):
    with pytest.raises(ValueError):
        CHECKS.validate_archive(payload)


def test_actual_zip_notice_members_match_pinned_text_and_provenance():
    with zipfile.ZipFile(zip_payloads(notices())) as archive:
        assert CHECKS.notice_payloads(archive) == notices()


@pytest.mark.parametrize("missing", sorted(CHECKS.REQUIRED_NOTICES))
def test_actual_zip_missing_notice_is_rejected(missing):
    payloads = notices()
    del payloads[missing]
    with zipfile.ZipFile(zip_payloads(payloads)) as archive:
        with pytest.raises(ValueError, match="Missing bundled notice"):
            CHECKS.notice_payloads(archive)


def test_actual_zip_changed_upstream_notice_is_rejected():
    payloads = notices()
    payloads["diskann-0.60.0/NOTICE.txt"] = b"MIT metadata without copyright text"
    with zipfile.ZipFile(zip_payloads(payloads)) as archive:
        with pytest.raises(ValueError, match="differs from pinned"):
            CHECKS.notice_payloads(archive)


@pytest.mark.parametrize("name,payload", [("../outside", b"bad"), ("huge.txt", b"x" * (1024 * 1024 + 1))])
def test_actual_zip_notice_paths_and_expanded_sizes_are_bounded(name, payload):
    payloads = notices()
    payloads[name] = payload
    with zipfile.ZipFile(zip_payloads(payloads)) as archive:
        with pytest.raises(ValueError):
            CHECKS.notice_payloads(archive)


def test_actual_wheel_binary_floor_is_checked_not_only_filename(tmp_path):
    wheel = tmp_path / "vectorlite_py-0.3.0-py3-none-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in notices().items():
            archive.writestr(CHECKS.NOTICE_PREFIX + name, payload)
        archive.writestr("vectorlite_py/vectorlite.dylib", macho(27))
    with pytest.raises(ValueError, match="exceeds 11.0"):
        CHECKS.validate_wheel(wheel)


def test_native_overlay_precedes_project_and_tracks_cache_abi():
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    overlay = (ROOT / "cmake/triplets/arm64-osx.cmake").read_text(encoding="utf-8")
    assert source.index('list(APPEND VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triplets")') < source.index("project(vectorlite")
    managed_append = 'list(APPEND VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triplets")'
    assert source.index("$ENV{VCPKG_OVERLAY_TRIPLETS}") < source.index(managed_append)
    assert source.index("list(REMOVE_ITEM VCPKG_OVERLAY_TRIPLETS") < source.index(managed_append)
    assert 'set(VCPKG_OSX_DEPLOYMENT_TARGET "11.0")' in overlay
    assert "CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS" in source
    native = (ROOT / "vectorlite/CMakeLists.txt").read_text(encoding="utf-8")
    assert "check_macos_artifact.py" in native
    assert "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}" in native
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert workflow.count("'cmake/triplets/*.cmake'") == 2
    assert "check_macos_artifact.py wheelhouse/*.whl" in workflow
    assert "check_macos_artifact.py build/release/vectorlite/vectorlite.dylib" in workflow


def test_sdist_explicitly_includes_new_build_and_notice_inputs():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = configuration["tool"]["scikit-build"]["sdist"]["include"]
    assert {
        "cmake/triplets/*.cmake", "scripts/check_macos_artifact.py",
        "scripts/collect_runtime_licenses.py", "scripts/run_batch_native_test.py",
        "scripts/run_batch_tokio_host.py", "third_party/**",
        "vectorlite/src/**/*.rs", "vectorlite/include/*.h", "vectorlite/tests/*.py",
        "vectorlite/tests/*.c", "vectorlite/tests/batch_tokio_host/Cargo.toml",
        "vectorlite/tests/batch_tokio_host/Cargo.lock", "vectorlite/tests/batch_tokio_host/src/*.rs",
        "doc/diskann.md", "benchmark/diskann_benchmark.py", "benchmark/test_diskann_benchmark_utils.py",
        "benchmark/diskann_batch_benchmark.c", "benchmark/diskann_batch_benchmark.py",
    } <= set(includes)
    excludes = configuration["tool"]["scikit-build"]["sdist"]["exclude"]
    assert {"build/**", "target/**", ".cache/**"} <= set(excludes)
    assert configuration["tool"]["cibuildwheel"]["macos"]["environment"]["MACOSX_DEPLOYMENT_TARGET"] == "11.0"


def test_upstream_provenance_and_own_npm_license_remain_distinct():
    provenance = json.loads((ROOT / "third_party/diskann-0.60.0/PROVENANCE.json").read_text())
    assert provenance["revision"] == CHECKS.DISKANN_REVISION
    assert provenance["version"] == "0.60.0"
    assert provenance["license"] == "MIT"
    assert provenance["source_git_blobs"] == CHECKS.UPSTREAM_BLOBS
    for name, expected in provenance["source_git_blobs"].items():
        payload = (ROOT / "third_party/diskann-0.60.0" / name).read_bytes()
        assert hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest() == expected
    assert set(provenance["crates"]) == {"diskann", "diskann-utils", "diskann-vector", "diskann-wide"}
    for package in ("vectorlite", "vectorlite-linux-x64", "vectorlite-win32-x64", "vectorlite-darwin-arm64"):
        directory = ROOT / "bindings/nodejs/packages" / package
        manifest = json.loads((directory / "package.json").read_text())
        assert manifest["license"] == "Apache-2.0"
        assert {"src", "LICENSE"} <= set(manifest["files"])
        assert (directory / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    manifest = json.loads((ROOT / "bindings/nodejs/packages/vectorlite/package.json").read_text())
    assert "notices.test.js" in manifest["scripts"]["test"]
    assert "node test/test.js" in manifest["scripts"]["test"]
    smoke = (ROOT / "bindings/nodejs/packages/vectorlite/test/test.js").read_text()
    assert "hnsw(max_elements=100)" in smoke
    assert "diskann()" in smoke
    assert "assert.throws(failed" in smoke
    assert "new sqlite3(databasePath)" in smoke


def test_runtime_inventory_retains_expressions_targets_and_every_exact_notice():
    manifest = json.loads((ROOT / "third_party/rust-runtime/MANIFEST.json").read_text())
    assert manifest["cargo_lock_sha256"] == hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest()
    assert manifest["unresolved_notice_gaps"] == []
    packages = {p["name"]: p for p in manifest["packages"]}
    assert len(packages) == 42
    assert packages["diskann"]["registry_source"] is None
    patch = packages["diskann"]["local_patch"]
    assert patch == json.loads((ROOT / "third_party/diskann-0.60.0/PROVENANCE.json").read_text())["local_patch"]
    assert {entry["path"] for entry in patch["modified_files"]} == {"src/graph/index.rs"}
    for entry in patch["modified_files"]:
        payload = (ROOT / patch["source_directory"] / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["local_sha256"]
    assert "cc" not in packages and "bytemuck_derive" not in packages
    assert "bindgen" not in packages and "rayon" not in packages
    assert packages["zerocopy"]["declared_spdx"] == "BSD-2-Clause OR Apache-2.0 OR MIT"
    assert packages["rustix"]["declared_spdx"] == "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT"
    assert packages["memchr"]["declared_spdx"] == "Unlicense OR MIT"
    assert packages["windows_x86_64_msvc"]["targets"] == ["x86_64-pc-windows-msvc"]
    assert packages["linux-raw-sys"]["targets"] == ["x86_64-unknown-linux-gnu"]
    assert packages["libm"]["declared_spdx"] == "MIT"
    libm_notice = (ROOT / "third_party/rust-runtime/crates/libm-0.2.16/SOURCE_NOTICES.txt").read_text()
    assert "Sun Microsystems" in libm_notice
    assert "David Schultz" in libm_notice
    assert "Alexei Sibidanov" in libm_notice
    payloads = notices()
    for package in packages.values():
        assert package["registry_archive_sha256"]
        assert package["files"]
        for file in package["files"]:
            payload = payloads[f"rust-runtime/{file['path']}"]
            assert hashlib.sha256(payload).hexdigest() == file["sha256"]
            assert len(payload) <= 1024 * 1024


@pytest.mark.parametrize("failure", ["missing", "changed", "manifest"])
def test_runtime_notice_payload_guards_missing_or_changed_crate_text(failure):
    payloads = notices()
    name = "rust-runtime/crates/libm-0.2.16/SOURCE_NOTICES.txt"
    if failure == "missing":
        del payloads[name]
    elif failure == "changed":
        payloads[name] = b"Missing upstream copyright notices"
    else:
        manifest = json.loads(payloads["rust-runtime/MANIFEST.json"])
        manifest["packages"][0]["declared_spdx"] = "MIT"
        payloads["rust-runtime/MANIFEST.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(zip_payloads(payloads)) as archive:
        with pytest.raises(ValueError, match="runtime"):
            CHECKS.notice_payloads(archive)


def test_runtime_collector_excludes_build_dev_and_proc_macro_subtrees():
    spec = importlib.util.spec_from_file_location("license_collector", ROOT / "scripts/collect_runtime_licenses.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    names = ["vectorlite", "runtime", "build", "dev", "macro", "macro-helper"]
    packages = [{"id": name, "name": name, "source": None if name == "vectorlite" else "registry", "targets": [{"kind": ["proc-macro" if name == "macro" else "lib"]}]} for name in names]
    def dependency(name, kind=None):
        return {"pkg": name, "dep_kinds": [{"kind": kind}]}
    nodes = [{"id": name, "deps": [], "features": []} for name in names]
    nodes[0]["deps"] = [dependency("runtime"), dependency("build", "build"), dependency("dev", "dev"), dependency("macro")]
    nodes[4]["deps"] = [dependency("macro-helper")]
    selected, _ = collector.selected({"packages": packages, "resolve": {"nodes": nodes}})
    assert set(selected) == {"runtime"}


def test_ci_bounded_diskann_reports_use_installed_wheel_not_source_library():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "Run streamed DiskANN production and storage smoke reports" in workflow
    assert "Path(vectorlite_py.vectorlite_path()).with_suffix(suffix).resolve()" in workflow
    assert "--count 128 --dim 16 --metric l2" in workflow
    assert "--count 2048 --dim 128 --metric l2" in workflow
    assert "--fixture storage-linear" in workflow
    assert "benchmark/diskann-production-smoke.json" in workflow
    assert "benchmark/diskann-storage-smoke.json" in workflow
