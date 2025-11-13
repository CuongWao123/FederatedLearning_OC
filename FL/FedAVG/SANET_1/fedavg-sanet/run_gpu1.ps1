# PowerShell script to run Flower federated learning on GPU 1
# Usage: .\run_gpu1.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "SANet Federated Learning - GPU 1" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variable to force GPU 1
$env:CUDA_VISIBLE_DEVICES = "1"

Write-Host "Environment Configuration:" -ForegroundColor Yellow
Write-Host "  CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES" -ForegroundColor Green
Write-Host ""

# Test GPU configuration (optional)
Write-Host "Testing GPU configuration..." -ForegroundColor Yellow
python test_gpu.py
Write-Host ""

# Run Flower server
Write-Host "Starting Flower Federated Learning..." -ForegroundColor Yellow
Write-Host "  Strategy: FedAvg" -ForegroundColor Green
Write-Host "  Model: SANet (ECCV 2018)" -ForegroundColor Green
Write-Host "  Device: GPU 1 (cuda:1)" -ForegroundColor Green
Write-Host ""

# Start Flower with configuration
flwr run

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
