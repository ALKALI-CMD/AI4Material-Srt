$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$checks = New-Object System.Collections.Generic.List[object]

function Add-Check($Name, $Passed, $Detail) {
  $checks.Add([ordered]@{
    name = $Name
    passed = [bool]$Passed
    detail = $Detail
  })
}

function Test-RequiredFile($Path) {
  $full = Join-Path $root $Path
  Add-Check "file:$Path" (Test-Path -LiteralPath $full) $Path
}

$requiredFiles = @(
  "README.md",
  "pyproject.toml",
  "docs/project_detailed_notes.md",
  "docs/literature_support.md",
  "docs/phase1_protocol.md",
  "docs/artifact_index.md",
  "docs/remote_runbook.md",
  "configs/phase1_smoke.yaml",
  "configs/phase1_remote_matrix.yaml",
  "scripts/run_phase1_smoke.ps1",
  "scripts/upload_phase1_package.ps1",
  "scripts/archive_remote_results.sh",
  "scripts/check_remote_gpu.sh",
  "scripts/bootstrap_remote.sh",
  "scripts/run_remote_model_smoke.sh",
  "scripts/run_remote_parallel.sh",
  "src/atomlens/attribution.py",
  "src/atomlens/compare.py",
  "src/atomlens/export.py",
  "src/atomlens/models.py",
  "src/atomlens/plotting.py",
  "src/atomlens/reporting.py",
  "src/atomlens/validation.py",
  "tests/test_attribution.py",
  "results/phase1_attribute_harmonic.json",
  "results/phase1_attribute_harmonic.png",
  "results/phase1_attribute_harmonic.extxyz",
  "results/phase1_convergence_harmonic.json",
  "results/phase1_convergence_harmonic.png",
  "results/phase1_validation_harmonic.json",
  "results/phase1_compare_self.json",
  "results/phase1_compare_self.png",
  "results/phase1_report.md"
)

foreach ($file in $requiredFiles) {
  Test-RequiredFile $file
}

$validationPath = Join-Path $root "results/phase1_validation_harmonic.json"
if (Test-Path -LiteralPath $validationPath) {
  $validation = Get-Content -LiteralPath $validationPath -Raw | ConvertFrom-Json
  Add-Check "validation:phase1_harmonic" ($validation.passed -eq $true) "results/phase1_validation_harmonic.json"
}

$latestZip = Join-Path $root "dist/atomlens_phase1_latest.zip"
Add-Check "package:latest_zip" (Test-Path -LiteralPath $latestZip) $latestZip

$pytestOutput = & python -m pytest -q 2>&1
$pytestPassed = $LASTEXITCODE -eq 0
Add-Check "tests:pytest" $pytestPassed ($pytestOutput -join "`n")

$payload = [ordered]@{
  passed = -not ($checks | Where-Object { -not $_.passed })
  generated_at = (Get-Date).ToString("s")
  checks = $checks
}

$out = Join-Path $root "results/phase1_readiness.json"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
$payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $out -Encoding UTF8

Get-Content -LiteralPath $out -Raw
