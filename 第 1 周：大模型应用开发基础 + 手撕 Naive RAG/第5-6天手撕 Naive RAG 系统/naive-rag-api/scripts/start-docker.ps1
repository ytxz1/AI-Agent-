param(
    [switch] $Dev,
    [switch] $Detached
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.docker.example" ".env"
    Write-Host "Created .env from .env.docker.example."
    Write-Host "Default Docker mode uses EMBEDDING_PROVIDER=hash and CHAT_PROVIDER=mock."
}

New-Item -ItemType Directory -Force "data", "data\uploads", "data\chroma", "data\metadata" | Out-Null

$composeFiles = @("docker-compose.yml")
if ($Dev) {
    $composeFiles = @("docker-compose.dev.yml")
}

$argsList = @()
foreach ($file in $composeFiles) {
    $argsList += @("-f", $file)
}

$argsList += @("up", "--build")
if ($Detached) {
    $argsList += "-d"
}

docker compose @argsList

