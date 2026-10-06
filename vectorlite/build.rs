use std::env;
use std::path::PathBuf;

mod build_support;
use build_support::find_vcpkg;

fn main() {
    let repo_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source_dir = repo_root.join("vectorlite");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let msvc = target_env == "msvc";
    if msvc
        && env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .split(',')
            .any(|feature| feature == "crt-static")
    {
        panic!("Vectorlite's MSVC vcpkg triplets use the dynamic CRT; remove the crt-static target feature");
    }

    // Platform-specific static archive name for highway produced by vcpkg.
    let hwy_marker = if msvc { "hwy.lib" } else { "libhwy.a" };

    let target = env::var("TARGET").unwrap();
    let profile = env::var("PROFILE").unwrap();
    let override_dir = env::var_os("VECTORLITE_VCPKG_TRIPLET_DIR").map(PathBuf::from);
    println!("cargo:rerun-if-env-changed=VECTORLITE_VCPKG_TRIPLET_DIR");
    build_support::watch_discovery(&repo_root, override_dir.is_some());
    let lib_marker = format!("lib/{hwy_marker}");
    let triplet = find_vcpkg(
        &repo_root,
        &target,
        &profile,
        override_dir.as_deref(),
        &[
            "include/hnswlib/hnswlib.h",
            "include/hwy/highway.h",
            &lib_marker,
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let vcpkg_include = triplet.join("include");
    let lib_dir = triplet.join("lib");

    let ops_cpp = source_dir.join("ops/ops.cpp");
    let shim_cpp = source_dir.join("cpp/core_shim.cpp");

    // Compile the C++ core: the existing (un-ported) ops SIMD kernels plus the
    // thin C ABI shim around hnswlib + vectorlite spaces.
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(&ops_cpp)
        .file(&shim_cpp)
        .include(&source_dir)
        .include(source_dir.join("ops"))
        .include(&vcpkg_include)
        .include(source_dir.join("cpp"))
        .warnings(profile == "debug");
    if msvc {
        // hnswlib relies on RAII cleanup when its operations throw.
        build.flag("/EHsc");
        if matches!(
            env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
            Ok("x86" | "x86_64")
        ) {
            build.flag("/arch:AVX");
        }
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
    // copy. The host process provides SQLite at load time on every platform.

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

    println!("cargo:rerun-if-changed=vectorlite/cpp/core_shim.cpp");
    println!("cargo:rerun-if-changed=vectorlite/cpp/core_shim.h");
    println!("cargo:rerun-if-changed=vectorlite/build.rs");
    println!("cargo:rerun-if-changed=vectorlite/build_support.rs");
    println!("cargo:rerun-if-changed={}", ops_cpp.display());
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("ops/ops.h").display()
    );
    for path in [
        vcpkg_include.join("hnswlib"),
        vcpkg_include.join("hwy"),
        lib_dir.join(hwy_marker),
        triplet.parent().unwrap().join("vcpkg/status"),
    ] {
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
