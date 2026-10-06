//! Shared, target-aware discovery for the native core and SQLite bindings.
use std::fs;
use std::path::{Path, PathBuf};

/// Cargo must reconsider selection when a preferred installation or another
/// candidate appears. Watch only native build roots, never Cargo's target tree.
pub fn watch_discovery(repo_root: &Path, explicit: bool) {
    if explicit {
        return;
    }
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("build").display()
    );
    let installed = repo_root.join("vcpkg/installed");
    let watch = if installed.exists() {
        installed
    } else {
        repo_root.join("vcpkg")
    };
    println!("cargo:rerun-if-changed={}", watch.display());
}

fn triplets(target: &str) -> Result<Vec<String>, String> {
    let arch = match target.split('-').next().unwrap_or("") {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "i686" => "x86",
        "armv7" => "arm",
        _ => return Err(format!("Unsupported Cargo target: {target}")),
    };
    let platform = if target.ends_with("-apple-darwin") {
        "osx"
    } else if target.ends_with("-windows-msvc") {
        "windows-static-md"
    } else if target.ends_with("-windows-gnu") {
        "mingw-static"
    } else if target.contains("-linux-musl") {
        "linux-musl"
    } else if target.contains("-linux-gnu") {
        "linux"
    } else if target.ends_with("-android") || target.ends_with("-androideabi") {
        "android"
    } else {
        return Err(format!("Unsupported Cargo target: {target}"));
    };
    let base = format!("{arch}-{platform}");
    Ok(vec![base.clone(), format!("{base}-release")])
}

fn candidates(root: &Path, names: &[String], markers: &[&str]) -> Vec<PathBuf> {
    names
        .iter()
        .map(|name| root.join(name))
        .filter(|dir| markers.iter().all(|marker| dir.join(marker).is_file()))
        .collect()
}

fn select(mut dirs: Vec<PathBuf>, target: &str) -> Result<PathBuf, String> {
    dirs.sort();
    dirs.dedup();
    match dirs.len() {
        1 => Ok(dirs.remove(0)),
        0 => Err(format!(
            "No vcpkg installation matches Cargo target {target}; configure CMake for that target or set VECTORLITE_VCPKG_TRIPLET_DIR"
        )),
        _ => Err(format!(
            "Ambiguous vcpkg installations for {target}: {}. Set VECTORLITE_VCPKG_TRIPLET_DIR to the intended triplet directory",
            dirs.iter().map(|dir| dir.display().to_string()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Prefer the matching CMake preset, then require a unique compatible fallback.
/// The explicit override is a full triplet directory, including its triplet name.
/// Custom triplets can use a supported name with a `-release` suffix; arbitrary
/// names are rejected because their target/CRT compatibility cannot be inferred.
pub fn find_vcpkg(
    repo_root: &Path,
    target: &str,
    profile: &str,
    override_dir: Option<&Path>,
    markers: &[&str],
) -> Result<PathBuf, String> {
    let names = triplets(target)?;
    if let Some(dir) = override_dir {
        if !names
            .iter()
            .any(|name| dir.file_name() == Some(name.as_ref()))
        {
            return Err(format!(
                "Vcpkg override {} is incompatible with {target}; expected one of {}",
                dir.display(),
                names.join(", ")
            ));
        }
        if !markers.iter().all(|marker| dir.join(marker).is_file()) {
            return Err(format!(
                "Vcpkg override {} is missing required headers or libraries",
                dir.display()
            ));
        }
        return Ok(dir.to_path_buf());
    }
    let preset = if profile == "debug" { "dev" } else { "release" };
    let preferred = repo_root.join("build").join(preset).join("vcpkg_installed");
    let preferred_dirs = candidates(&preferred, &names, markers);
    if !preferred_dirs.is_empty() {
        return select(preferred_dirs, target);
    }
    let mut dirs = Vec::new();
    if let Ok(entries) = fs::read_dir(repo_root.join("build")) {
        for entry in entries.flatten() {
            dirs.extend(candidates(
                &entry.path().join("vcpkg_installed"),
                &names,
                markers,
            ));
        }
    }
    dirs.extend(candidates(
        &repo_root.join("vcpkg/installed"),
        &names,
        markers,
    ));
    select(dirs, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    const MARKERS: &[&str] = &["include/hnswlib/hnswlib.h", "lib/libhwy.a"];
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "vectorlite-discovery-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn install(&self, preset: &str, triplet: &str) -> PathBuf {
            let dir = self
                .0
                .join("build")
                .join(preset)
                .join("vcpkg_installed")
                .join(triplet);
            for marker in MARKERS {
                let path = dir.join(marker);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, "").unwrap();
            }
            dir
        }
        fn find(&self, profile: &str, explicit: Option<&Path>) -> Result<PathBuf, String> {
            find_vcpkg(&self.0, "aarch64-apple-darwin", profile, explicit, MARKERS)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn rejects_foreign_architecture_and_incompatible_override() {
        let f = Fixture::new();
        let foreign = f.install("release", "x64-osx");
        assert!(f.find("release", None).is_err());
        assert!(f.find("release", Some(&foreign)).is_err());
    }
    #[test]
    fn picks_target_and_preset_independently_of_other_installations() {
        let f = Fixture::new();
        f.install("release", "x64-osx");
        let release = f.install("release", "arm64-osx");
        let debug = f.install("dev", "arm64-osx");
        assert_eq!(f.find("release", None).unwrap(), release);
        assert_eq!(f.find("debug", None).unwrap(), debug);
    }
    #[test]
    fn rejects_ambiguous_fallback_and_accepts_explicit_selection() {
        let f = Fixture::new();
        let first = f.install("cross-one", "arm64-osx");
        f.install("cross-two", "arm64-osx");
        assert!(f.find("release", None).unwrap_err().contains("Ambiguous"));
        assert_eq!(f.find("release", Some(&first)).unwrap(), first);
    }
    #[test]
    fn requires_all_markers_in_override() {
        let f = Fixture::new();
        let dir = f.install("release", "arm64-osx");
        fs::remove_file(dir.join(MARKERS[0])).unwrap();
        assert!(f.find("release", Some(&dir)).is_err());
    }
    #[test]
    fn distinguishes_c_runtime_and_windows_linkage() {
        assert_eq!(
            triplets("x86_64-unknown-linux-musl").unwrap()[0],
            "x64-linux-musl"
        );
        assert_eq!(
            triplets("x86_64-pc-windows-msvc").unwrap()[0],
            "x64-windows-static-md"
        );
        assert_eq!(
            triplets("x86_64-pc-windows-gnu").unwrap()[0],
            "x64-mingw-static"
        );
        assert!(triplets("wasm32-unknown-unknown").is_err());
    }
}
