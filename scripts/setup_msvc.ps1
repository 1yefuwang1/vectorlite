#requires -Version 7.0
# Run in a Windows GitHub Actions step with shell: pwsh.
# Microsoft documents vswhere discovery and the maintained VS shell initializer:
# https://github.com/microsoft/vswhere/wiki/Find-VC
# https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell
$ErrorActionPreference = 'Stop'

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'MSVC environment setup requires Windows.'
}
if (-not [Environment]::Is64BitProcess) {
    throw 'MSVC environment setup requires 64-bit PowerShell.'
}
$githubEnv = $env:GITHUB_ENV
if ([string]::IsNullOrWhiteSpace($githubEnv) -or
    -not (Test-Path -LiteralPath $githubEnv -PathType Leaf)) {
    throw 'GITHUB_ENV must name the environment file for this GitHub Actions step.'
}

$programFilesX86 = ${env:ProgramFiles(x86)}
if ([string]::IsNullOrWhiteSpace($programFilesX86)) {
    throw 'ProgramFiles(x86) is not set; cannot locate the Visual Studio installer.'
}
$vswhere = Join-Path $programFilesX86 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw "Visual Studio locator not found: $vswhere"
}
$installationPaths = & $vswhere -latest -products '*' -requires `
    Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -utf8
if ($LASTEXITCODE -ne 0) {
    throw "vswhere failed with exit code $LASTEXITCODE."
}
$installationPath = ($installationPaths -join [Environment]::NewLine).Trim()
if ([string]::IsNullOrWhiteSpace($installationPath)) {
    throw 'No installed Visual Studio instance provides the x86/x64 MSVC tools.'
}
$devShell = Join-Path $installationPath 'Common7\Tools\Launch-VsDevShell.ps1'
if (-not (Test-Path -LiteralPath $devShell -PathType Leaf)) {
    throw "Visual Studio developer shell initializer not found: $devShell"
}

$before = @{}
Get-ChildItem Env: | ForEach-Object { $before[$_.Name] = $_.Value }
Write-Host "Initializing x64 MSVC from $installationPath"
try {
    # These architecture arguments are supported by VS 2022 17.1 and newer.
    # Keep the checkout as the working directory, rather than VS's project folder.
    & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
    if (-not $?) {
        throw 'The developer shell initializer reported failure.'
    }
} catch {
    throw "Visual Studio developer environment initialization failed: $($_.Exception.Message)"
}
if ($env:VSCMD_ARG_TGT_ARCH -notin @('x64', 'amd64') -or
    $env:VSCMD_ARG_HOST_ARCH -notin @('x64', 'amd64')) {
    throw 'Visual Studio did not initialize an x64 host/target environment.'
}
if ([string]::IsNullOrWhiteSpace($env:VSINSTALLDIR) -or
    $env:VSINSTALLDIR.TrimEnd([IO.Path]::DirectorySeparatorChar) -ne
    $installationPath.TrimEnd([IO.Path]::DirectorySeparatorChar)) {
    throw 'The developer environment does not match the discovered Visual Studio installation.'
}
if ([string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) {
    throw 'Visual Studio did not set VCToolsInstallDir.'
}
foreach ($variable in @('PATH', 'INCLUDE', 'LIB', 'LIBPATH')) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($variable))) {
        throw "Visual Studio did not initialize $variable."
    }
}
foreach ($tool in @('cl.exe', 'link.exe')) {
    $toolPath = Join-Path $env:VCToolsInstallDir "bin\Hostx64\x64\$tool"
    if (-not (Test-Path -LiteralPath $toolPath -PathType Leaf)) {
        throw "The initialized x64 MSVC tool is missing: $toolPath"
    }
}

# VS can restore its own pre-initialization PATH when re-entering a developer
# shell. Retain any current user/Python entries it dropped, after the VS tools.
$pathEntries = @($env:PATH -split ';')
foreach ($entry in ($before['PATH'] -split ';')) {
    if ($entry.Length -gt 0 -and $pathEntries -notcontains $entry) {
        $pathEntries += $entry
    }
}
$env:PATH = $pathEntries -join ';'

# Persist only changed variables, without printing potentially sensitive values.
# Multiline records and BOM-free UTF-8 also handle unusual environment values.
$lines = [System.Collections.Generic.List[string]]::new()
$exported = 0
foreach ($variable in (Get-ChildItem Env: | Sort-Object Name)) {
    $name = $variable.Name
    if ($name -match '^(GITHUB_|RUNNER_|NODE_OPTIONS$)') {
        continue
    }
    if ($before.ContainsKey($name) -and $before[$name] -ceq $variable.Value) {
        continue
    }
    if ($name -match '[\r\n=]') {
        throw 'Cannot export an environment variable with an invalid name.'
    }
    $delimiter = "MSVC_ENV_$([Guid]::NewGuid().ToString('N'))"
    $lines.Add("$name<<$delimiter")
    $lines.Add($variable.Value)
    $lines.Add($delimiter)
    $exported += 1
}
[IO.File]::AppendAllLines($githubEnv, $lines.ToArray(), [Text.UTF8Encoding]::new($false))
Write-Host "Exported $exported changed MSVC environment variables for later steps."
Write-Host "MSVC tools directory: $env:VCToolsInstallDir"
