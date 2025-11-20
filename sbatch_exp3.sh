#!/bin/bash
#SBATCH --job-name=diffusion_exp3
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/exp3_%j.out
#SBATCH --error=logs/exp3_%j.err

# Experiment 3: Bias Estimation
# Tests Proposition 2: Bias_epsilon ≈ Bias_x
# Trains models to convergence on full dataset (50 epochs, 5 runs each)

echo "=========================================="
echo "EXPERIMENT 3: Bias Estimation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p checkpoints
mkdir -p results/exp3
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
    --experiment 3 \
    --dataset $DATASET \
    --device $DEVICE

echo ""
echo "Experiment 3 complete!"
echo "End time: $(date)"
echo "Results saved to results/exp3/"
