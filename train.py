"""
Training script for diffusion models.
Implements loss functions L_epsilon (Eq 8) and L_x (Eq 10) from the paper.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import json
from models import DDPMNoisePrediction, DDPMDirectPrediction, DiffusionProcess


def get_mnist_dataloader(batch_size=128, subset_size=None, data_dir='./data'):
    """
    Load MNIST dataset with optional subset sampling.

    Args:
        batch_size: Batch size for training
        subset_size: If specified, use only this many samples (for sample complexity experiments)
        data_dir: Directory to store/load data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    if subset_size is not None and subset_size < len(train_dataset):
        # Create a subset for sample complexity experiments
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_cifar10_dataloader(batch_size=128, subset_size=None, data_dir='./data'):
    """Load CIFAR-10 dataset with optional subset sampling."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    if subset_size is not None and subset_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def train_noise_prediction(model, diffusion, train_loader, optimizer, device, epoch):
    """
    Train noise prediction model.
    Minimizes L_epsilon from Equation (8):
    L_epsilon(h) = E[||epsilon - h(x_t, t)||^2]
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Noise Pred]')
    for batch_idx, (x_0, _) in enumerate(pbar):
        x_0 = x_0.to(device)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        # Sample noise and create noisy images - Equation (2)
        noise = torch.randn_like(x_0)
        x_t, _ = diffusion.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = model(x_t, t)

        # Compute loss - Equation (8)
        loss = F.mse_loss(predicted_noise, noise)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def train_direct_prediction(model, diffusion, train_loader, optimizer, device, epoch):
    """
    Train direct x0 prediction model.
    Minimizes L_x from Equation (10):
    L_x(h) = E[||x_0 - h(x_t, t)||^2]
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Direct Pred]')
    for batch_idx, (x_0, _) in enumerate(pbar):
        x_0 = x_0.to(device)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        # Sample noise and create noisy images - Equation (2)
        noise = torch.randn_like(x_0)
        x_t, _ = diffusion.q_sample(x_0, t, noise)

        # Directly predict x_0
        predicted_x0 = model(x_t, t)

        # Compute loss - Equation (10)
        loss = F.mse_loss(predicted_x0, x_0)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


@torch.no_grad()
def evaluate_noise_prediction(model, diffusion, test_loader, device):
    """Evaluate noise prediction model on test set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for x_0, _ in test_loader:
        x_0 = x_0.to(device)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        # Sample noise and create noisy images
        noise = torch.randn_like(x_0)
        x_t, _ = diffusion.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = model(x_t, t)

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate_direct_prediction(model, diffusion, test_loader, device):
    """Evaluate direct x0 prediction model on test set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for x_0, _ in test_loader:
        x_0 = x_0.to(device)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        # Sample noise and create noisy images
        noise = torch.randn_like(x_0)
        x_t, _ = diffusion.q_sample(x_0, t, noise)

        # Predict x_0
        predicted_x0 = model(x_t, t)

        # Compute loss
        loss = F.mse_loss(predicted_x0, x_0)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train_model(
    model_type='noise',
    dataset='mnist',
    epochs=50,
    batch_size=128,
    learning_rate=1e-4,
    subset_size=None,
    save_dir='./checkpoints',
    device='cuda',
    seed=42
):
    """
    Main training function.

    Args:
        model_type: 'noise' or 'direct' for noise prediction or direct x0 prediction
        dataset: 'mnist' or 'cifar10'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        subset_size: Optional subset size for sample complexity experiments
        save_dir: Directory to save checkpoints
        device: 'cuda' or 'cpu'
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloader(batch_size, subset_size)
        img_channels = 1
    elif dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloader(batch_size, subset_size)
        img_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Initialize diffusion process
    diffusion = DiffusionProcess(timesteps=1000, beta_schedule='linear')
    diffusion = diffusion.to(device)

    # Initialize model
    if model_type == 'noise':
        model = DDPMNoisePrediction(img_channels=img_channels, base_dim=64, dim_mults=(1, 2, 4))
        train_fn = train_noise_prediction
        eval_fn = evaluate_noise_prediction
    elif model_type == 'direct':
        model = DDPMDirectPrediction(img_channels=img_channels, base_dim=64, dim_mults=(1, 2, 4))
        train_fn = train_direct_prediction
        eval_fn = evaluate_direct_prediction
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train_fn(model, diffusion, train_loader, optimizer, device, epoch)
        test_loss = eval_fn(model, diffusion, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

        # Save checkpoint
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = os.path.join(
                save_dir,
                f'{model_type}_{dataset}_subset{subset_size}_epoch{epoch}_seed{seed}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, checkpoint_path)

            # Also save final losses
            results_path = os.path.join(
                save_dir,
                f'{model_type}_{dataset}_subset{subset_size}_seed{seed}_results.json'
            )
            with open(results_path, 'w') as f:
                json.dump({
                    'model_type': model_type,
                    'dataset': dataset,
                    'subset_size': subset_size,
                    'epochs': epoch,
                    'seed': seed,
                    'final_train_loss': train_loss,
                    'final_test_loss': test_loss,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                }, f, indent=2)

    return model, train_losses, test_losses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--model_type', type=str, default='noise', choices=['noise', 'direct'],
                        help='Model type: noise prediction or direct x0 prediction')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Subset size for sample complexity experiments')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_size=args.subset_size,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed
    )
