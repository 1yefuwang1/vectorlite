// By default this build script does nothing: the bindings are committed in
// src/bindings.rs. With `--features regenerate` it re-runs bindgen against the
// vcpkg sqlite3ext.h and rewrites src/bindings.rs (requires libclang).
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "regenerate")]
    regenerate::run();
}

#[cfg(feature = "regenerate")]
mod regenerate {
    use std::env;
    use std::path::{Path, PathBuf};

    fn find_dir(repo_root: &Path, candidates: &[&str], marker: &str) -> Option<PathBuf> {
        for cand in candidates {
            let dir = repo_root.join(cand);
            if dir.join(marker).exists() {
                return Some(dir);
            }
        }
        None
    }

    pub fn run() {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        // rust/vectorlite-sqlite-sys -> rust -> repo root
        let repo_root = manifest_dir.parent().unwrap().parent().unwrap().to_path_buf();

        let include_candidates = [
            "build/dev/vcpkg_installed/x64-linux/include",
            "build/release/vcpkg_installed/x64-linux/include",
            "vcpkg/packages/sqlite3_x64-linux/include",
        ];
        let sqlite_include = find_dir(&repo_root, &include_candidates, "sqlite3ext.h")
            .expect("could not locate sqlite3ext.h in vcpkg include dirs");

        let wrapper = PathBuf::from(env::var("OUT_DIR").unwrap()).join("wrapper.h");
        std::fs::write(&wrapper, "#include \"sqlite3ext.h\"\n").unwrap();

        let mut builder = bindgen::Builder::default()
            .header(wrapper.to_str().unwrap())
            .clang_arg(format!("-I{}", sqlite_include.display()));

        // libclang may not ship its builtin headers (stdarg.h, ...).
        for builtin in [
            "/usr/lib/llvm-18/lib/clang/18/include",
            "/usr/lib/gcc/x86_64-linux-gnu/13/include",
            "/usr/lib/gcc/x86_64-linux-gnu/12/include",
            "/usr/lib/gcc/x86_64-linux-gnu/11/include",
        ] {
            if Path::new(builtin).join("stdarg.h").exists() {
                builder = builder.clang_arg(format!("-I{}", builtin));
                break;
            }
        }

        let bindings = builder
            .allowlist_type("sqlite3.*")
            .allowlist_var("SQLITE_.*")
            .blocklist_function(".*")
            .layout_tests(false)
            .generate()
            .expect("unable to generate sqlite bindings");

        bindings
            .write_to_file(manifest_dir.join("src/bindings.rs"))
            .expect("couldn't write src/bindings.rs");
    }
}
