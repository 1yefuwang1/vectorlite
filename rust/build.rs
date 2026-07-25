use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Finds a vcpkg-installed triplet directory (produced by the CMake build) that
/// contains both the C++ headers and the requested static library. Triplet- and
/// platform-agnostic: it scans `build/<preset>/vcpkg_installed/<triplet>/` and
/// `vcpkg/installed/<triplet>/` for any triplet that satisfies both markers.
/// Returns `(include_dir, lib_dir)`.
fn find_vcpkg(repo_root: &Path, static_lib_marker: &str) -> Option<(PathBuf, PathBuf)> {
    let mut installed_roots: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = fs::read_dir(repo_root.join("build")) {
        for entry in rd.flatten() {
            installed_roots.push(entry.path().join("vcpkg_installed"));
        }
    }
    installed_roots.push(repo_root.join("vcpkg/installed"));

    for root in installed_roots {
        let rd = match fs::read_dir(&root) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let triplet = entry.path();
            let include = triplet.join("include");
            let lib = triplet.join("lib");
            if include.join("hnswlib/hnswlib.h").exists() && lib.join(static_lib_marker).exists() {
                return Some((include, lib));
            }
        }
    }
    None
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.parent().unwrap().to_path_buf();

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let msvc = target_env == "msvc";

    // Platform-specific static archive name for highway produced by vcpkg.
    let hwy_marker = if msvc { "hwy.lib" } else { "libhwy.a" };

    let (vcpkg_include, lib_dir) = find_vcpkg(&repo_root, hwy_marker)
        .expect("could not locate a vcpkg_installed triplet with hnswlib headers and the highway static lib; build the CMake project first");

    let vectorlite_src = repo_root.join("vectorlite");
    let ops_cpp = vectorlite_src.join("ops/ops.cpp");
    let shim_cpp = manifest_dir.join("cpp/core_shim.cpp");

    // Compile the C++ core: the existing (un-ported) ops SIMD kernels plus the
    // thin C ABI shim around hnswlib + vectorlite spaces.
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(&ops_cpp)
        .file(&shim_cpp)
        .include(&vectorlite_src)
        .include(vectorlite_src.join("ops"))
        .include(&vcpkg_include)
        .include(manifest_dir.join("cpp"))
        .warnings(false);
    if msvc {
        // Matches the C++ CMake build's MSVC flags; the SIMD baseline needs AVX.
        build.flag("/arch:AVX");
    } else {
        build.flag_if_supported("-fPIC");
    }
    build.compile("vectorlite_core");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=hwy");

    // SQLite is deliberately NOT linked in. A loadable extension never calls
    // SQLite directly: every call goes through the sqlite3_api_routines table
    // the host passes at load time (the loadable-extension contract), so the
    // library has no undefined SQLite symbols to resolve and needs no embedded
    // copy. This keeps the artifact small (~0.6 MB) and identical across
    // platforms; the host process provides SQLite at load time.

    // The C++ core (hnswlib's std::thread/std::mutex, libstdc++/highway) needs a
    // few platform system libraries on Linux. glibc >= 2.34 folds these into
    // libc, but link them explicitly so older toolchains resolve them too.
    match target_os.as_str() {
        "linux" | "android" => {
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=m");
        }
        // macOS provides pthread/dl/m via libSystem (linked automatically).
        // Windows (MSVC) needs no extra libs.
        _ => {}
    }

    // SQLite extension API bindings live in the vendored vectorlite-sqlite-sys
    // crate (committed, pre-generated), so no bindgen/libclang is needed here.

    println!("cargo:rerun-if-changed=cpp/core_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/core_shim.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", ops_cpp.display());
}
