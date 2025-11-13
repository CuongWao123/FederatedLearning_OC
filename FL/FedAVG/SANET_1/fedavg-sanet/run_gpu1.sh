#!/bin/bash
# Bash script to run Flower federated learning on GPU 1
# Usage: bash run_gpu1.sh or ./run_gpu1.sh

echo "======================================"
echo "SANet Federated Learning - GPU 1"
echo "======================================"
echo ""

# Set environment variable to force GPU 1
export CUDA_VISIBLE_DEVICES=1

echo "Environment Configuration:"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo ""

# Test GPU configuration (optional)
echo "Testing GPU configuration..."
python test_gpu.py
echo ""

# Run Flower server
echo "Starting Flower Federated Learning..."
echo "  Strategy: FedAvg"
echo "  Model: SANet (ECCV 2018)"
echo "  Device: GPU 1 (cuda:1)"
echo ""

# Start Flower with configuration
flwr run

echo ""
echo "======================================"
echo "Training Complete!"
echo "======================================"
