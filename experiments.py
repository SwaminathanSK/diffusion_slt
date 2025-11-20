"""
Implementation of experiments from Section 6 of the paper.
"""
import torch
import numpy as np
from train import train_model, get_mnist_dataloader, get_cifar10_dataloader
from train import evaluate_noise_prediction, evaluate_direct_prediction
from models import DDPMNoisePrediction, DDPMDirectPrediction, DiffusionProcess
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def experiment1_sample_complexity(
    dataset='mnist',
    sample_sizes=[100, 500, 1000, 5000, 10000, 50000],
    epochs=50,
    num_seeds=3,
    save_dir='./results/exp1',
    device='cuda'
):
    """
    Experiment 1: Sample Complexity Curves

    Train noise prediction and direct prediction models with varying training set sizes.
    Validates Theorem 3: noise prediction should achieve lower error for all m,
    with the gap widening for smaller datasets.

    Args:
        dataset: 'mnist' or 'cifar10'
        sample_sizes: List of training set sizes to test
        epochs: Number of epochs per training run
        num_seeds: Number of random seeds to average over
        save_dir: Directory to save results
        device: 'cuda' or 'cpu'
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {
        'sample_sizes': sample_sizes,
        'noise_pred': {'mean': [], 'std': [], 'all_runs': []},
        'direct_pred': {'mean': [], 'std': [], 'all_runs': []}
    }

    for subset_size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Training with {subset_size} samples")
        print(f"{'='*60}")

        noise_test_losses = []
        direct_test_losses = []

        for seed in range(num_seeds):
            print(f"\nSeed {seed + 1}/{num_seeds}")

            # Train noise prediction model
            print("Training noise prediction model...")
            _, _, noise_losses = train_model(
                model_type='noise',
                dataset=dataset,
                epochs=epochs,
                subset_size=subset_size,
                save_dir=os.path.join(save_dir, 'checkpoints'),
                device=device,
                seed=seed
            )
            noise_test_losses.append(noise_losses[-1])

            # Train direct prediction model
            print("Training direct prediction model...")
            _, _, direct_losses = train_model(
                model_type='direct',
                dataset=dataset,
                epochs=epochs,
                subset_size=subset_size,
                save_dir=os.path.join(save_dir, 'checkpoints'),
                device=device,
                seed=seed
            )
            direct_test_losses.append(direct_losses[-1])

        # Store results
        results['noise_pred']['all_runs'].append(noise_test_losses)
        results['noise_pred']['mean'].append(np.mean(noise_test_losses))
        results['noise_pred']['std'].append(np.std(noise_test_losses))

        results['direct_pred']['all_runs'].append(direct_test_losses)
        results['direct_pred']['mean'].append(np.mean(direct_test_losses))
        results['direct_pred']['std'].append(np.std(direct_test_losses))

        print(f"\nNoise Prediction - Mean Loss: {results['noise_pred']['mean'][-1]:.6f} +/- {results['noise_pred']['std'][-1]:.6f}")
        print(f"Direct Prediction - Mean Loss: {results['direct_pred']['mean'][-1]:.6f} +/- {results['direct_pred']['std'][-1]:.6f}")

    # Save results
    results_path = os.path.join(save_dir, 'sample_complexity_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


def experiment2_variance_estimation(
    dataset='mnist',
    subset_size=10000,
    epochs=50,
    num_models=20,
    save_dir='./results/exp2',
    device='cuda'
):
    """
    Experiment 2: Variance Estimation via Bootstrapping

    Train multiple models on different random subsets of the training data.
    Measure the variance across these models in their predictions on a fixed test set.
    Validates Proposition 1: Var_epsilon < Var_x

    Args:
        dataset: 'mnist' or 'cifar10'
        subset_size: Size of training subset
        epochs: Number of epochs per training run
        num_models: Number of models to train (k=20 in paper)
        save_dir: Directory to save results
        device: 'cuda' or 'cpu'
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load test data once
    if dataset == 'mnist':
        _, test_loader = get_mnist_dataloader(batch_size=128)
        img_channels = 1
    else:
        _, test_loader = get_cifar10_dataloader(batch_size=128)
        img_channels = 3

    # Get fixed test batch for evaluation
    test_batch = next(iter(test_loader))
    x_0_test = test_batch[0].to(device)

    # Initialize diffusion
    diffusion = DiffusionProcess(timesteps=1000, beta_schedule='linear').to(device)

    # Storage for predictions
    noise_predictions = []
    direct_predictions = []

    print(f"Training {num_models} models of each type...")

    for i in range(num_models):
        print(f"\nModel {i + 1}/{num_models}")

        # Train noise prediction model with different seed
        print("Training noise prediction model...")
        noise_model, _, _ = train_model(
            model_type='noise',
            dataset=dataset,
            epochs=epochs,
            subset_size=subset_size,
            save_dir=os.path.join(save_dir, 'checkpoints'),
            device=device,
            seed=i * 100  # Different seed for each run
        )

        # Train direct prediction model with different seed
        print("Training direct prediction model...")
        direct_model, _, _ = train_model(
            model_type='direct',
            dataset=dataset,
            epochs=epochs,
            subset_size=subset_size,
            save_dir=os.path.join(save_dir, 'checkpoints'),
            device=device,
            seed=i * 100
        )

        # Evaluate on fixed test batch
        noise_model.eval()
        direct_model.eval()

        with torch.no_grad():
            # Sample same timesteps for fair comparison
            batch_size = x_0_test.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

            # Create noisy images with fixed noise for consistency
            torch.manual_seed(42)
            noise = torch.randn_like(x_0_test)
            x_t, _ = diffusion.q_sample(x_0_test, t, noise)

            # Get predictions
            noise_pred = noise_model(x_t, t)
            direct_pred = direct_model(x_t, t)

            noise_predictions.append(noise_pred.cpu().numpy())
            direct_predictions.append(direct_pred.cpu().numpy())

    # Calculate variance across models
    noise_predictions = np.array(noise_predictions)  # Shape: (num_models, batch_size, C, H, W)
    direct_predictions = np.array(direct_predictions)

    # Variance across model ensemble (axis=0)
    var_noise = np.var(noise_predictions, axis=0).mean()
    var_direct = np.var(direct_predictions, axis=0).mean()

    results = {
        'num_models': num_models,
        'subset_size': subset_size,
        'variance_noise': float(var_noise),
        'variance_direct': float(var_direct),
        'variance_ratio': float(var_direct / var_noise),
    }

    print(f"\n{'='*60}")
    print(f"Variance Results:")
    print(f"Noise Prediction Variance: {var_noise:.6f}")
    print(f"Direct Prediction Variance: {var_direct:.6f}")
    print(f"Ratio (Var_x / Var_epsilon): {var_direct / var_noise:.4f}")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(save_dir, 'variance_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


def experiment3_bias_estimation(
    dataset='mnist',
    subset_size=50000,
    epochs=50,
    num_runs=5,
    save_dir='./results/exp3',
    device='cuda'
):
    """
    Experiment 3: Bias Estimation

    Train models to convergence on the full dataset.
    Estimate bias by comparing the average prediction (over multiple training runs)
    to the empirical optimal predictor.
    Validates Proposition 2: Bias_epsilon ≈ Bias_x for sufficiently expressive models.

    Args:
        dataset: 'mnist' or 'cifar10'
        subset_size: Size of training subset (or full dataset)
        epochs: Number of epochs per training run (default: 50 for faster convergence)
        num_runs: Number of training runs to average over (default: 5, sufficient for bias estimation)
        save_dir: Directory to save results
        device: 'cuda' or 'cpu'
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load test data
    if dataset == 'mnist':
        _, test_loader = get_mnist_dataloader(batch_size=128)
        img_channels = 1
    else:
        _, test_loader = get_cifar10_dataloader(batch_size=128)
        img_channels = 3

    # Get fixed test batch
    test_batch = next(iter(test_loader))
    x_0_test = test_batch[0].to(device)

    # Initialize diffusion
    diffusion = DiffusionProcess(timesteps=1000, beta_schedule='linear').to(device)

    # Storage for averaged predictions
    noise_predictions = []
    direct_predictions = []

    print(f"Training {num_runs} models of each type to convergence...")

    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}")

        # Train noise prediction model
        print("Training noise prediction model...")
        noise_model, _, _ = train_model(
            model_type='noise',
            dataset=dataset,
            epochs=epochs,
            subset_size=subset_size,
            save_dir=os.path.join(save_dir, 'checkpoints'),
            device=device,
            seed=i * 100
        )

        # Train direct prediction model
        print("Training direct prediction model...")
        direct_model, _, _ = train_model(
            model_type='direct',
            dataset=dataset,
            epochs=epochs,
            subset_size=subset_size,
            save_dir=os.path.join(save_dir, 'checkpoints'),
            device=device,
            seed=i * 100
        )

        # Evaluate
        noise_model.eval()
        direct_model.eval()

        with torch.no_grad():
            batch_size = x_0_test.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

            torch.manual_seed(42)
            noise = torch.randn_like(x_0_test)
            x_t, _ = diffusion.q_sample(x_0_test, t, noise)

            noise_pred = noise_model(x_t, t)
            direct_pred = direct_model(x_t, t)

            noise_predictions.append(noise_pred.cpu().numpy())
            direct_predictions.append(direct_pred.cpu().numpy())

    # Calculate average predictions (expected hypothesis)
    noise_predictions = np.array(noise_predictions)
    direct_predictions = np.array(direct_predictions)

    avg_noise_pred = noise_predictions.mean(axis=0)
    avg_direct_pred = direct_predictions.mean(axis=0)

    # Calculate bias (MSE to ground truth)
    # For noise prediction: bias = ||E[epsilon_pred] - epsilon_true||^2
    # For direct prediction: bias = ||E[x0_pred] - x0_true||^2

    with torch.no_grad():
        batch_size = x_0_test.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        torch.manual_seed(42)
        noise_true = torch.randn_like(x_0_test)
        x_t, _ = diffusion.q_sample(x_0_test, t, noise_true)

        noise_true_np = noise_true.cpu().numpy()
        x0_true_np = x_0_test.cpu().numpy()

        bias_noise = np.mean((avg_noise_pred - noise_true_np) ** 2)
        bias_direct = np.mean((avg_direct_pred - x0_true_np) ** 2)

    results = {
        'num_runs': num_runs,
        'subset_size': subset_size,
        'epochs': epochs,
        'bias_squared_noise': float(bias_noise),
        'bias_squared_direct': float(bias_direct),
        'bias_ratio': float(bias_direct / bias_noise) if bias_noise > 0 else None,
    }

    print(f"\n{'='*60}")
    print(f"Bias Results:")
    print(f"Noise Prediction Bias^2: {bias_noise:.6f}")
    print(f"Direct Prediction Bias^2: {bias_direct:.6f}")
    print(f"Ratio (Bias^2_x / Bias^2_epsilon): {bias_direct / bias_noise:.4f}" if bias_noise > 0 else "N/A")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(save_dir, 'bias_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiments from the paper')
    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3],
                        help='Which experiment to run (1, 2, or 3)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.experiment == 1:
        print("Running Experiment 1: Sample Complexity Curves")
        experiment1_sample_complexity(dataset=args.dataset, device=args.device)
    elif args.experiment == 2:
        print("Running Experiment 2: Variance Estimation")
        experiment2_variance_estimation(dataset=args.dataset, device=args.device)
    elif args.experiment == 3:
        print("Running Experiment 3: Bias Estimation")
        experiment3_bias_estimation(dataset=args.dataset, device=args.device)
