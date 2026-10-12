param(
  [Parameter(Mandatory = $true)]
  [string]$HostName,

  [string]$User = "root",

  [string]$RemoteDir = "~/atomlens_phase1",

  [string]$Package = "dist/atomlens_phase1_latest.zip",

  [string]$IdentityFile = ""
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$packagePath = Resolve-Path -LiteralPath (Join-Path $root $Package)
$target = "${User}@${HostName}"
$sshArgs = @()
if ($IdentityFile -ne "") {
  $sshArgs += @("-i", $IdentityFile)
}

ssh @sshArgs $target "mkdir -p $RemoteDir"
scp @sshArgs $packagePath.Path "${target}:${RemoteDir}/atomlens_phase1_latest.zip"
ssh @sshArgs $target "cd $RemoteDir && rm -rf atomlens_phase1 && unzip -q atomlens_phase1_latest.zip -d atomlens_phase1 && cd atomlens_phase1 && bash scripts/bootstrap_remote.sh"
