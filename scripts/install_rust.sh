#!/bin/sh
# cibuildwheel's before-all hook runs inside the manylinux image. Rust installed
# on the GitHub Actions host is not available in that container.
set -eu

export CARGO_HOME="${CARGO_HOME:-/opt/rust/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rust/rustup}"
export PATH="$CARGO_HOME/bin:$PATH"

if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable
fi

rustup toolchain install stable --profile minimal --component rustfmt --component clippy
rustup default stable
cargo --version
rustc --version
