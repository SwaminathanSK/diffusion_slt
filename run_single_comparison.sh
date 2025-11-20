#!/bin/bash
#
# Quick script to run a single comparison between noise and direct prediction.
# Useful for fast validation on cluster.

set -e

DATASET=${1:-mnist}
EPOCHS=${2:-20}
SUBSET_SIZE=${3:-5000}
SEED=${4:-42}

echo "=========================================="
echo "Quick Comparison: Noise vs Direct Prediction"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Subset Size: $SUBSET_SIZE"
echo "Seed: $SEED"
echo "=========================================="
echo ""

# Train noise prediction model
echo "Training NOISE PREDICTION model..."
python train.py \
    --model_type noise \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --subset_size $SUBSET_SIZE \
    --seed $SEED \
    --device cuda

echo ""
echo "=========================================="
echo ""

# Train direct prediction model
echo "Training DIRECT PREDICTION model..."
python train.py \
    --model_type direct \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --subset_size $SUBSET_SIZE \
    --seed $SEED \
    --device cuda

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Check checkpoints/ directory for saved models and results."
