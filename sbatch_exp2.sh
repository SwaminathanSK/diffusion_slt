#!/bin/bash
#SBATCH --job-name=diffusion_exp2
#SBATCH --partition=gpu_24hour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err

# Experiment 2: Variance Estimation via Bootstrapping
# Tests Proposition 1: Var_epsilon < Var_x
# Trains 20 models of each type on different data subsets

echo "=========================================="
echo "EXPERIMENT 2: Variance Estimation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p checkpoints
mkdir -p results/exp2
mkdir -p logs

# Set dataset and device
DATASET=${DATASET:-mnist}
DEVICE="cuda"

echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Start time: $(date)"
echo ""

# Run experiment
python experiments.py \
    --experiment 2 \
    --dataset $DATASET \
    --device $DEVICE

echo ""
echo "Experiment 2 complete!"
echo "End time: $(date)"
echo "Results saved to results/exp2/"
