#[cfg(any(feature = "regenerate", feature = "abi-check"))]
#[path = "../build_support.rs"]
mod build_support;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../build_support.rs");
    #[cfg(any(feature = "regenerate", feature = "abi-check"))]
    {
        use std::{env, path::PathBuf};
        println!("cargo:rerun-if-env-changed=VECTORLITE_VCPKG_TRIPLET_DIR");
        let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
        let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
        let override_dir = env::var_os("VECTORLITE_VCPKG_TRIPLET_DIR").map(PathBuf::from);
        build_support::watch_discovery(repo_root, override_dir.is_some());
        let triplet = build_support::find_vcpkg(
            repo_root,
            &env::var("TARGET").unwrap(),
            &env::var("PROFILE").unwrap(),
            override_dir.as_deref(),
            &["include/sqlite3.h", "include/sqlite3ext.h"],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let include = triplet.join("include");
        for header in ["sqlite3.h", "sqlite3ext.h"] {
            println!("cargo:rerun-if-changed={}", include.join(header).display());
        }
        #[cfg(feature = "regenerate")]
        regenerate(&include);
        #[cfg(feature = "abi-check")]
        {
            println!("cargo:rerun-if-changed=tests/abi_check.c");
            cc::Build::new()
                .include(include)
                .file("tests/abi_check.c")
                .compile("vectorlite_sqlite_abi_check");
        }
    }
}

#[cfg(feature = "regenerate")]
fn regenerate(include: &std::path::Path) {
    use std::{env, path::PathBuf};
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let wrapper = out_dir.join("wrapper.h");
    std::fs::write(&wrapper, "#include \"sqlite3ext.h\"\n").unwrap();
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_type("sqlite3.*")
        .allowlist_var("SQLITE_.*")
        .blocklist_item("SQLITE_OS_.*")
        .blocklist_type("va_list|__builtin_va_list|__va_list_tag")
        .blocklist_function(".*")
        .layout_tests(false)
        .formatter(bindgen::Formatter::Prettyplease)
        .generate()
        .expect("unable to generate SQLite bindings")
        .to_string();
    // The extension never uses these va_list APIs. Preserve their table slots,
    // but make them private opaque function pointers instead of publishing a
    // platform-dependent calling convention in the committed bindings.
    let mut bindings = bindings;
    for name in ["vmprintf", "xvsnprintf", "str_vappendf"] {
        let marker = format!("    pub {name}:");
        let start = bindings.find(&marker).expect(
            "SQLite API layout was not generated; use a libclang supported by bindgen (on macOS, the Command Line Tools libclang)",
        );
        let end =
            start + bindings[start + marker.len()..].find("\n    pub ").unwrap() + marker.len();
        bindings.replace_range(
            start..end,
            &format!("    pub(crate) {name}: ::std::option::Option<unsafe extern \"C\" fn()>,"),
        );
    }
    let output = out_dir.join("bindings.rs");
    std::fs::write(&output, bindings).expect("could not write generated bindings");
    println!(
        "cargo:warning=Generated target bindings at {}",
        output.display()
    );
}
