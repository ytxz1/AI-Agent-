param(
    [string]$InputPath = "data/raw_pdfs",
    [string]$OutputPath = "outputs/mineru",
    [string]$Backend = "pipeline"
)

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

Write-Host "[mineru] input:  $InputPath"
Write-Host "[mineru] output: $OutputPath"
Write-Host "[mineru] backend: $Backend"

mineru -p $InputPath -o $OutputPath -b $Backend

