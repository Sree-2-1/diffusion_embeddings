from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class VAEConfig:
    """Configuration for the production 128x128 TLPP Matryoshka VAE."""

    probability_mode: str = "log01"
    latent_dim: int = 128
    reconstruction_loss: str = "hellinger_squared"
    beta: float = 1e-3
    matryoshka_dims: tuple[int, ...] = (4, 8, 16, 32, 64, 128)

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    logvar_min: float = -10.0
    logvar_max: float = 8.0
    kl_warmup_epochs: int = 10
    seed: int = 1729

    @classmethod
    def from_dict(cls, values: dict) -> "VAEConfig":
        """Load current or older project checkpoints without carrying old features forward."""
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def counts_to_probability(counts: np.ndarray) -> np.ndarray:
    """Normalize raw TLPP occupancy counts to a probability map per sample."""
    values = np.asarray(counts, dtype=np.float32)
    total = values.sum(axis=(-2, -1), keepdims=True)
    return np.divide(
        values,
        total,
        out=np.zeros_like(values, dtype=np.float32),
        where=total > 0,
    ).astype(np.float32, copy=False)


# Compatibility alias used by checkpoints/scripts from the development repo.
prepare_cached_occupancy = lambda raw, _cfg=None: counts_to_probability(raw)


class MapVAE(nn.Module):
    """Convolutional VAE used for 128x128 TLPP probability maps.

    ``encoder_features`` exposes the three spatial encoder stages without changing
    the checkpoint/state-dict layout.  Those tensors are the natural attachment
    points for future residual blocks or ControlNet-style gated conditioning.
    """

    def __init__(self, channels: int, grid: int, latent_dim: int):
        super().__init__()
        self.grid = int(grid)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        with torch.no_grad():
            encoded = self.encoder(torch.zeros(1, channels, grid, grid))
        self.hidden_shape = tuple(encoded.shape[1:])
        hidden = encoded.numel()

        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.from_z = nn.Linear(latent_dim, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, 2, 1),
        )

    def encoder_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features: list[torch.Tensor] = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                features.append(h)
        return tuple(features)

    def stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.from_z(z).reshape(-1, *self.hidden_shape)
        y = self.decoder(hidden)
        return y[..., : self.grid, : self.grid]


class VAE(nn.Module):
    """Thin wrapper retained so current checkpoint state-dict keys remain unchanged."""

    def __init__(self, sample_shape: tuple[int, ...], latent_dim: int):
        super().__init__()
        if tuple(sample_shape) != (1, 128, 128):
            raise ValueError(f"Production VAE expects sample shape (1,128,128), got {sample_shape}")
        self.net = MapVAE(sample_shape[0], sample_shape[1], latent_dim)

    def stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net.stats(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.stats(x)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.net.decode(z)

    def encoder_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.net.encoder_features(x)


def reconstruction_loss(
    logits: torch.Tensor,
    raw_target: torch.Tensor,
    name: str = "hellinger_squared",
) -> torch.Tensor:
    """Squared Hellinger reconstruction loss used by the production model.

    The decoder returns logits.  We normalize them with softmax and compute
    H^2(P,Q) = 1 - sum_i sqrt(P_i Q_i).
    """
    if name not in {"hellinger_squared", "hellinger"}:
        raise ValueError("Production workflow supports reconstruction_loss='hellinger_squared' only")

    log_q = torch.log_softmax(logits.flatten(2), dim=-1)
    sqrt_q = torch.exp(0.5 * log_q)

    p = raw_target.flatten(2)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sqrt_p = torch.sqrt(p.clamp_min(0.0))

    coefficient = (sqrt_p * sqrt_q).sum(dim=-1)
    return (1.0 - coefficient).clamp_min(0.0).mean()


def prefix_dims(cfg: VAEConfig) -> tuple[int, ...]:
    dims = sorted({int(d) for d in cfg.matryoshka_dims if 0 < int(d) <= cfg.latent_dim})
    if cfg.latent_dim not in dims:
        dims.append(cfg.latent_dim)
    return tuple(dims)


def load_vae_checkpoint(checkpoint: Path | str, device: str = "auto"):
    """Load a current HDF5-native checkpoint, including older compatible config dicts."""
    dev = resolve_device(device)
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = VAEConfig.from_dict(state["vae"])
    sample_shape = tuple(state["sample_shape"])
    model = VAE(sample_shape, cfg.latent_dim).to(dev)
    model.load_state_dict(state["model_state"])
    model.eval()
    return state, cfg, model, dev
