"""
Diffusion model implementations for noise prediction vs direct x0 prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUNet(nn.Module):
    """
    Simplified U-Net architecture for diffusion models.
    Can be used for both noise prediction and direct x0 prediction.
    """

    def __init__(self, img_channels=1, base_dim=64, dim_mults=(1, 2, 4), time_dim=256):
        super().__init__()

        self.img_channels = img_channels
        self.time_dim = time_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Initial convolution
        self.conv0 = nn.Conv2d(img_channels, base_dim, 3, padding=1)

        # Downsample blocks
        self.downs = nn.ModuleList([])
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_out, time_dim),
                ResBlock(dim_out, dim_out, time_dim),
                nn.Conv2d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity()
            ]))

        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_dim)

        # Upsample blocks
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResBlock(dim_out * 2, dim_in, time_dim),
                ResBlock(dim_in, dim_in, time_dim),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, img_channels, 1)
        )

    def forward(self, x, time):
        # Time embedding
        t = self.time_mlp(time)

        # Initial conv
        x = self.conv0(x)
        r = x.clone()

        # Downsample
        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # Upsample
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)

        # Final
        x = x + r
        x = self.final_conv(x)

        return x


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, dim_in, dim_out, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim_out)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim_in),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)

        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]

        h = self.block2(h)

        return h + self.res_conv(x)


class DDPMNoisePrediction(nn.Module):
    """
    DDPM model that predicts the noise epsilon.
    This corresponds to the H_epsilon hypothesis class in the paper.
    """

    def __init__(self, img_channels=1, base_dim=64, dim_mults=(1, 2, 4)):
        super().__init__()
        self.model = SimpleUNet(img_channels, base_dim, dim_mults)

    def forward(self, x_t, t):
        """
        Predicts noise epsilon given noisy image x_t and timestep t.

        Args:
            x_t: Noisy image at timestep t
            t: Timestep (scalar tensor)

        Returns:
            predicted noise epsilon
        """
        return self.model(x_t, t)


class DDPMDirectPrediction(nn.Module):
    """
    DDPM model that directly predicts x0.
    This corresponds to the H_x hypothesis class in the paper.
    """

    def __init__(self, img_channels=1, base_dim=64, dim_mults=(1, 2, 4)):
        super().__init__()
        self.model = SimpleUNet(img_channels, base_dim, dim_mults)

    def forward(self, x_t, t):
        """
        Directly predicts clean image x0 given noisy image x_t and timestep t.

        Args:
            x_t: Noisy image at timestep t
            t: Timestep (scalar tensor)

        Returns:
            predicted clean image x0
        """
        return self.model(x_t, t)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear beta schedule as used in DDPM paper.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule as proposed in Improved DDPM.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionProcess:
    """
    Handles the forward and reverse diffusion process.
    Implements equations (1) and (2) from the paper.
    """

    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps

        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-calculate useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0) - Equation (2)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        Implements equation (2) from the paper:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0: Clean image
            t: Timestep
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy image at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def predict_x0_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from noise prediction.
        Uses equation (11) from the paper (rearranged):
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def to(self, device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
