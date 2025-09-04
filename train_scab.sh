#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Starting SCAB Training with train_val.py"
echo "=========================================="

echo "Training with SCAB configuration..."
python -u tools/train_val.py --config configs/monodgp_scab.yaml > logs/monodgp_scab.log 2>&1

echo "Training completed!"
echo "Check logs/monodgp_scab.log for details" 