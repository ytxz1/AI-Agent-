param(
    [string] $BaseUrl = "http://127.0.0.1:8000",
    [string] $SampleFile = "samples\rag_notes.md"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path $SampleFile)) {
    throw "Sample file not found: $SampleFile"
}

Write-Host "1. Checking health..."
$health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
$health | ConvertTo-Json -Depth 10

Write-Host "2. Uploading sample document..."
$uploadOutput = curl.exe -sS -X POST "$BaseUrl/api/v1/documents/upload" -F "file=@$SampleFile"
Write-Host $uploadOutput

Write-Host "3. Listing documents..."
$documents = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/v1/documents"
$documents | ConvertTo-Json -Depth 10

Write-Host "4. Asking a RAG question..."
$body = @{
    question = "RAG 的索引阶段包括哪些步骤？"
    top_k = 4
} | ConvertTo-Json -Depth 10

$answer = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/v1/chat/query" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

$answer | ConvertTo-Json -Depth 10

Write-Host "Smoke test finished."

