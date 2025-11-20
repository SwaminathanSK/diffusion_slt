# Statistical Learning Theory Perspective on Noise Prediction in Diffusion Models

Implementation of experiments from the paper "A Statistical Learning Theory Perspective on Noise Prediction in Diffusion Models" by Swaminathan S K.

## Overview

This repository contains code to empirically validate the theoretical findings about why noise prediction (ε-prediction) outperforms direct x₀ prediction in Denoising Diffusion Probabilistic Models (DDPMs).

### Main Theoretical Contributions

1. **Theorem 3**: Noise prediction achieves superior generalization through favorable bias-variance tradeoff
2. **Proposition 1**: Variance of noise prediction is lower than direct prediction (Var_ε < Var_x)
3. **Proposition 2**: For sufficiently expressive models, bias is comparable (Bias_ε ≈ Bias_x)

## Project Structure

```
diffusion_slt/
├── models.py              # DDPM implementations (noise & direct prediction)
├── train.py              # Training loops and loss functions
├── experiments.py        # Three main experiments from the paper
├── visualize_results.py  # Result visualization and analysis
├── run_experiments.sh    # Run all experiments
├── run_single_comparison.sh  # Quick comparison script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

## Running Experiments

### Quick Start: Single Comparison

For a fast validation on your L40 cluster:

```bash
chmod +x run_single_comparison.sh
./run_single_comparison.sh mnist 20 5000 42
```

This trains both models for 20 epochs on 5000 MNIST samples. Results are saved to `checkpoints/`.

### Full Experimental Suite

To run all three experiments from the paper:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This will:
1. **Experiment 1**: Sample complexity curves (tests Theorem 3)
2. **Experiment 2**: Variance estimation via bootstrapping (tests Proposition 1)
3. **Experiment 3**: Bias estimation (tests Proposition 2)

Results are saved to `results/exp{1,2,3}/`.

### Individual Experiments

Run specific experiments:

```bash
# Experiment 1: Sample Complexity
python experiments.py --experiment 1 --dataset mnist --device cuda

# Experiment 2: Variance Estimation
python experiments.py --experiment 2 --dataset mnist --device cuda

# Experiment 3: Bias Estimation
python experiments.py --experiment 3 --dataset mnist --device cuda
```

### Custom Training

Train individual models with custom parameters:

```bash
# Noise prediction model
python train.py --model_type noise --dataset mnist --epochs 50 --batch_size 128 --subset_size 10000

# Direct x0 prediction model
python train.py --model_type direct --dataset mnist --epochs 50 --batch_size 128 --subset_size 10000
```

## Visualization

After running experiments, generate plots:

```bash
python visualize_results.py
```

This creates:
- `results/exp1/sample_complexity_plot.png` - Learning curves
- `results/exp2/variance_plot.png` - Variance comparison
- `results/exp3/bias_plot.png` - Bias comparison
- `results/summary_plot.png` - Combined overview

## Implementation Details

### Model Architecture

- **Base**: Simplified U-Net with residual blocks
- **Time Embedding**: Sinusoidal position embeddings
- **Dimensions**: Base 64, multipliers (1, 2, 4)
- **Parameters**: ~1-2M (comparable capacity for fair comparison)

### Loss Functions

**Noise Prediction (Equation 8)**:
```
L_ε(h) = E_{x₀,ε,t}[‖ε - h(x_t, t)‖²]
```

**Direct Prediction (Equation 10)**:
```
L_x(h) = E_{x₀,ε,t}[‖x₀ - h(x_t, t)‖²]
```

### Diffusion Process

- **Schedule**: Linear beta schedule (β_start=1e-4, β_end=0.02)
- **Timesteps**: T=1000
- **Forward Process**: x_t = √(ᾱ_t)x₀ + √(1-ᾱ_t)ε (Equation 2)

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train.py --batch_size 64 ...
```

### Slow Training

- Ensure CUDA is being used: check logs for "device: cuda"
- Increase num_workers in data loaders
- Use mixed precision (can add to train.py)

### Missing Data

The first run will download MNIST/CIFAR-10 automatically to `./data/`

## Extending the Code

### Add New Datasets

Modify `train.py`:
```python
def get_custom_dataloader(batch_size, subset_size=None):
    # Add your dataset loading code
    pass
```

### Try Different Architectures

Modify `models.py` to experiment with:
- Different base dimensions
- Attention mechanisms
- Different time embeddings

### Additional Experiments

Add to `experiments.py`:
```python
def experiment4_custom(...):
    # Your custom experiment
    pass
```

## Citation

If you use this code, please cite the original paper:

```
@article{swaminathan2024diffusion,
  title={A Statistical Learning Theory Perspective on Noise Prediction in Diffusion Models},
  author={Swaminathan S K},
  year={2024},
  institution={Indian Institute of Technology Kharagpur}
}
```

## References

Key papers implemented:
- [Ho et al. 2020] - DDPM original paper
- [Vincent 2011] - Denoising score matching
- [Song & Ermon 2019] - Score-based generative modeling

## License

This code is for research purposes only.

## Contact

For questions about the implementation or experiments, please open an issue in the repository.
