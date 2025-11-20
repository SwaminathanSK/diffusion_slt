#!/bin/bash
#
# Script to run all experiments from the paper on cluster.
# Optimized for L40 GPU cluster.

set -e

echo "=========================================="
echo "Diffusion SLT Experiments"
echo "=========================================="

# Create necessary directories
mkdir -p checkpoints
mkdir -p results/exp1
mkdir -p results/exp2
mkdir -p results/exp3
mkdir -p logs

# Set dataset (mnist is faster for validation)
DATASET="mnist"
DEVICE="cuda"

echo ""
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo ""

# Experiment 1: Sample Complexity Curves
# Tests Theorem 3 from the paper
echo "=========================================="
echo "EXPERIMENT 1: Sample Complexity Curves"
echo "=========================================="
echo "This validates Theorem 3: noise prediction should achieve"
echo "lower error for all m, with gap widening for smaller datasets."
echo ""

python experiments.py \
    --experiment 1 \
    --dataset $DATASET \
    --device $DEVICE \
    2>&1 | tee logs/exp1_${DATASET}.log

echo ""
echo "Experiment 1 complete. Results saved to results/exp1/"
echo ""

# Experiment 2: Variance Estimation
# Tests Proposition 1 from the paper
echo "=========================================="
echo "EXPERIMENT 2: Variance Estimation"
echo "=========================================="
echo "This validates Proposition 1: Var_epsilon < Var_x"
echo ""

python experiments.py \
    --experiment 2 \
    --dataset $DATASET \
    --device $DEVICE \
    2>&1 | tee logs/exp2_${DATASET}.log

echo ""
echo "Experiment 2 complete. Results saved to results/exp2/"
echo ""

# Experiment 3: Bias Estimation
# Tests Proposition 2 from the paper
echo "=========================================="
echo "EXPERIMENT 3: Bias Estimation"
echo "=========================================="
echo "This validates Proposition 2: Bias_epsilon ≈ Bias_x"
echo "for sufficiently expressive models."
echo ""

python experiments.py \
    --experiment 3 \
    --dataset $DATASET \
    --device $DEVICE \
    2>&1 | tee logs/exp3_${DATASET}.log

echo ""
echo "Experiment 3 complete. Results saved to results/exp3/"
echo ""

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "To visualize results, run:"
echo "  python visualize_results.py"
echo ""
