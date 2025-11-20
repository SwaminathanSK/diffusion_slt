#!/bin/bash
#SBATCH --job-name=diffusion_quick
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/quick_%j.out
#SBATCH --error=logs/quick_%j.err

# Quick Comparison: Noise vs Direct Prediction
# Fast validation script for testing setup

echo "=========================================="
echo "Quick Comparison: Noise vs Direct Prediction"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Set parameters
DATASET=${DATASET:-mnist}
EPOCHS=${EPOCHS:-20}
SUBSET_SIZE=${SUBSET_SIZE:-5000}
SEED=${SEED:-42}

echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Subset Size: $SUBSET_SIZE"
echo "Seed: $SEED"
echo "Start time: $(date)"
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
echo "End time: $(date)"
echo "Check checkpoints/ directory for saved models and results."
