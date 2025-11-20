#!/bin/bash
#
# Master script to submit all experiments to the cluster
# Each experiment runs as a separate job

echo "=========================================="
echo "Diffusion SLT - Submitting All Experiments"
echo "=========================================="
echo ""

# Ensure logs directory exists
mkdir -p logs

# Option to set dataset (default: mnist)
DATASET=${1:-mnist}
echo "Dataset: $DATASET"
echo ""

# Submit Experiment 1: Sample Complexity Curves
echo "Submitting Experiment 1 (Sample Complexity)..."
JOB1=$(sbatch --export=DATASET=$DATASET sbatch_exp1.sh | awk '{print $4}')
echo "  Job ID: $JOB1"
echo "  Partition: gpu_l40"
echo "  Expected runtime: ~6-10 hours"
echo ""

# Submit Experiment 2: Variance Estimation
echo "Submitting Experiment 2 (Variance Estimation)..."
JOB2=$(sbatch --export=DATASET=$DATASET sbatch_exp2.sh | awk '{print $4}')
echo "  Job ID: $JOB2"
echo "  Partition: gpu_l40"
echo "  Expected runtime: ~12-16 hours (trains 20 models)"
echo ""

# Submit Experiment 3: Bias Estimation
echo "Submitting Experiment 3 (Bias Estimation)..."
JOB3=$(sbatch --export=DATASET=$DATASET sbatch_exp3.sh | awk '{print $4}')
echo "  Job ID: $JOB3"
echo "  Partition: gpu_l40"
echo "  Expected runtime: ~20-24 hours (100 epochs, 10 runs)"
echo ""

echo "=========================================="
echo "All experiments submitted!"
echo "=========================================="
echo ""
echo "To monitor your jobs:"
echo "  squeue -u \$USER"
echo ""
echo "To check a specific job:"
echo "  squeue -j <job_id>"
echo ""
echo "To view output logs:"
echo "  tail -f logs/exp1_<job_id>.out"
echo "  tail -f logs/exp2_<job_id>.out"
echo "  tail -f logs/exp3_<job_id>.out"
echo ""
echo "To cancel a job:"
echo "  scancel <job_id>"
echo ""
echo "To cancel all your jobs:"
echo "  scancel -u \$USER"
echo ""

# Save job IDs for reference
cat > logs/submitted_jobs.txt <<EOF
Experiment 1: $JOB1
Experiment 2: $JOB2
Experiment 3: $JOB3
Dataset: $DATASET
Submitted: $(date)
EOF

echo "Job IDs saved to logs/submitted_jobs.txt"
