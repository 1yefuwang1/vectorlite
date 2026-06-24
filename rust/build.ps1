# Builds the Rust port of vectorlite and deploys it into the Python package
# (Windows / PowerShell). See build.sh for the Linux/macOS equivalent.
#
# Prerequisites: a Rust toolchain (MSVC) and the vcpkg headers/libs produced by
# the C++ CMake build. No libclang is needed for the default build.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

cargo build --release

Copy-Item "target/release/vectorlite.dll" `
    "../bindings/python/vectorlite_py/vectorlite.dll" -Force

Write-Host "Deployed Rust vectorlite.dll to bindings/python/vectorlite_py/"
Write-Host "Run tests with:"
Write-Host "  `$env:PYTHONPATH='bindings/python'; python -m pytest bindings/python/vectorlite_py/test"
