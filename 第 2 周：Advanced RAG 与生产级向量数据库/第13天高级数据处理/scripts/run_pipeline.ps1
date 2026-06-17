param(
    [string]$InputPath = "data/raw_pdfs",
    [switch]$RunUnstructured,
    [switch]$RunMinerU,
    [switch]$RunDocling,
    [string]$MinerUBackend = "pipeline"
)

New-Item -ItemType Directory -Force -Path "outputs/unstructured","outputs/mineru","outputs/docling","outputs/normalized","outputs/reports" | Out-Null

if ($RunUnstructured) {
    python scripts/parse_unstructured.py --input $InputPath --output outputs/unstructured --strategy hi_res --extract-image-blocks
    python scripts/normalize_unstructured.py --input outputs/unstructured --output outputs/normalized
}

if ($RunMinerU) {
    powershell -ExecutionPolicy Bypass -File scripts/parse_mineru_cli.ps1 -InputPath $InputPath -OutputPath outputs/mineru -Backend $MinerUBackend
    python scripts/normalize_mineru.py --input outputs/mineru --output outputs/normalized
}

if ($RunDocling) {
    python scripts/parse_docling.py --input $InputPath --output outputs/docling
    python scripts/normalize_docling.py --input outputs/docling --output outputs/normalized
}

python scripts/normalize_blocks.py --input outputs/normalized --output outputs/normalized/all_blocks.jsonl --manifest outputs/normalized/manifest.json
python scripts/build_chunks.py --input outputs/normalized/all_blocks.jsonl --output outputs/normalized/rag_chunks.jsonl
python scripts/evaluate_parse_quality.py --input outputs/normalized/all_blocks.jsonl --output outputs/reports/parse_quality_report.json

Write-Host "Pipeline finished."
Write-Host "Blocks: outputs/normalized/all_blocks.jsonl"
Write-Host "Chunks: outputs/normalized/rag_chunks.jsonl"
Write-Host "Report: outputs/reports/parse_quality_report.json"

