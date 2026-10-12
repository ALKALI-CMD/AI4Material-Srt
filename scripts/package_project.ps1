$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$outDir = Join-Path $root "dist"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $outDir "atomlens_phase1_$stamp.zip"
$latest = Join-Path $outDir "atomlens_phase1_latest.zip"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$items = @(
  ".gitignore",
  "README.md",
  "pyproject.toml",
  "configs",
  "docs",
  "scripts",
  "src",
  "tests",
  "results/manifest.json",
  "results/phase1_convergence_harmonic.json",
  "results/phase1_convergence_harmonic.png",
  "results/phase1_validation_harmonic.json",
  "results/phase1_smoke_harmonic.json",
  "results/phase1_attribute_harmonic.json",
  "results/phase1_attribute_harmonic.png",
  "results/phase1_attribute_harmonic.extxyz",
  "results/phase1_report.md",
  "results/phase1_compare_self.json",
  "results/phase1_compare_self.png",
  "results/phase1_readiness.json"
)

$staging = Join-Path $outDir "package_staging"
if (Test-Path -LiteralPath $staging) {
  Remove-Item -LiteralPath $staging -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $staging | Out-Null

foreach ($item in $items) {
  $source = Join-Path $root $item
  if (-not (Test-Path -LiteralPath $source)) {
    continue
  }
  $target = Join-Path $staging $item
  $targetParent = Split-Path -Parent $target
  New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
  Copy-Item -LiteralPath $source -Destination $target -Recurse -Force
}

Get-ChildItem -LiteralPath $staging -Recurse -Force -Directory |
  Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") -or $_.Name.EndsWith(".egg-info") } |
  Remove-Item -Recurse -Force

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $out -Force
Copy-Item -LiteralPath $out -Destination $latest -Force
Remove-Item -LiteralPath $staging -Recurse -Force

Write-Output $out
Write-Output $latest
