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
    // Feed bindgen a platform-normalized copy of the headers: the three
    // `va_list` slots in `sqlite3_api_routines` are replaced below with opaque
    // function pointers, but bindgen cannot represent `va_list` itself on every
    // platform (on Linux it is `__builtin_va_list`, which the blocklist drops, so
    // the fields would silently disappear and the generated struct layout would
    // be wrong). Rewriting the token to `void *` keeps generation identical on
    // every target; the canonical opaque form is what actually gets committed.
    let normalized = out_dir.join("normalized_headers");
    std::fs::create_dir_all(&normalized).unwrap();
    // Copy every sqlite3* header: the vcpkg build includes its own config header
    // next to sqlite3.h, so the normalized directory must be self-contained.
    for entry in std::fs::read_dir(include).unwrap() {
        let path = entry.unwrap().path();
        let name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) if name.starts_with("sqlite3") && name.ends_with(".h") => name.to_owned(),
            _ => continue,
        };
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(
            normalized.join(name),
            replace_identifier(&text, "va_list", "void *"),
        )
        .unwrap();
    }
    // Include the normalized header by absolute path: a bare quoted include would
    // be resolved next to the wrapper file and then through the toolchain search
    // path, where a system sqlite3ext.h of another version can win on Linux.
    let normalized_header = normalized.join("sqlite3ext.h");
    let wrapper = out_dir.join("wrapper.h");
    std::fs::write(
        &wrapper,
        format!("#include \"{}\"\n", normalized_header.display()),
    )
    .unwrap();
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", normalized.display()))
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

/// Replaces whole-identifier occurrences only, so `va_list` inside a longer
/// identifier or a comment word is untouched where it matters.
#[cfg(feature = "regenerate")]
fn replace_identifier(text: &str, needle: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut index = 0;
    while let Some(found) = text[index..].find(needle) {
        let start = index + found;
        let end = start + needle.len();
        let before = start
            .checked_sub(1)
            .map(|position| bytes[position] as char)
            .unwrap_or(' ');
        let after = text[end..].chars().next().unwrap_or(' ');
        output.push_str(&text[index..start]);
        if (before.is_ascii_alphanumeric() || before == '_')
            || (after.is_ascii_alphanumeric() || after == '_')
        {
            output.push_str(needle);
        } else {
            output.push_str(replacement);
        }
        index = end;
    }
    output.push_str(&text[index..]);
    output
}
