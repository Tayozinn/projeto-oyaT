# ==============================================================================
# ServerMind Neural Engine — Maratona Autônoma Opção B (~113M Parâmetros, 50/50)
# Hardware Alvo: Intel Core i5-9600K + 16 GB RAM + AMD Radeon RX 7600 (8 GB VRAM)
# ==============================================================================
# Uso:
#   1. Primeira noite (cria do zero, treina por 8h, gera sequência evolutiva + INT8):
#      .\scripts\run_marathon_110m.ps1
#   2. Noite customizada (ex.: 6h):
#      .\scripts\run_marathon_110m.ps1 -Duration "6h"
#   3. Continuar nas noites seguintes (Noite 2, Noite 3... acumulando inteligência):
#      .\scripts\run_marathon_110m.ps1 -Resume -Duration "8h"
# ==============================================================================

param(
    [string]$Duration = "8h",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $ProjectRoot "target\release\servermind-ai.exe"

if (-not (Test-Path $Exe)) {
    Write-Host "[ERRO] Executável '$Exe' não encontrado. Rode 'cargo build --release --locked' primeiro." -ForegroundColor Red
    exit 1
}

$OutDir = "checkpoints/servermind-neural-110m"
$QuantDir = "checkpoints/servermind-neural-110m-int8"

$ArgsList = @(
    "train-neural",
    "--marathon",
    "--max-duration", $Duration,
    "--corpus-dir", "datasets/pretrain_corpus_v2/servidores",
    "--corpus-dir", "datasets/pretrain_corpus_v2/programacao",
    "--dataset", "datasets/servermind_corpus_v3_balanced.jsonl",
    "--out", $OutDir,
    "--export-quant", $QuantDir,
    "--bits", "int8",
    "--device", "gpu",
    "--vocab", "4096",
    "--seq-len", "512",
    "--d-model", "768",
    "--layers", "15",
    "--heads", "12",
    "--ff", "3072",
    "--batch-size", "4",
    "--lr", "0.0006"
)

if ($Resume) {
    $ArgsList += "--resume"
}

Write-Host "Iniciando ServerMind Neural 113M (50% Servidores / 50% Programação) na AMD RX 7600 por $Duration..." -ForegroundColor Cyan
Push-Location $ProjectRoot
try {
    & $Exe @ArgsList
} finally {
    Pop-Location
}
