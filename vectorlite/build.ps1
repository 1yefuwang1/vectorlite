# Compatibility build-only shortcut for Windows/MSVC. CMake builds and deploys
# the primary Rust extension; use the root release script for all test suites.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    cmake --preset release
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE."
    }
    cmake --build build/release --target vectorlite --parallel 8
    if ($LASTEXITCODE -ne 0) {
        throw "Vectorlite build failed with exit code $LASTEXITCODE."
    }
    Write-Host "Built and deployed vectorlite.dll from Rust via CMake."
} finally {
    Pop-Location
}
