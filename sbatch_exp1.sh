#!/bin/bash
#SBATCH --job-name=diffusion_exp1
#SBATCH --partition=gpu_24hour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err

# Experiment 1: Sample Complexity Curves
# Tests Theorem 3 from the paper - compares noise vs direct prediction across different dataset sizes

echo "=========================================="
echo "EXPERIMENT 1: Sample Complexity Curves"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p checkpoints
mkdir -p results/exp1
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
    --experiment 1 \
    --dataset $DATASET \
    --device $DEVICE

echo ""
echo "Experiment 1 complete!"
echo "End time: $(date)"
echo "Results saved to results/exp1/"
