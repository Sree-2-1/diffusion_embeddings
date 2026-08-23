from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .core import (
    TLPPConfig,
    make_multilag_TLPP,
    tlpp_probability_from_signal,
    transform_occupancy,
)
from .data import Item, TraceConfig, load_trace


@dataclass(frozen=True)
class VAEConfig:
    representation: str = "adaptive"  # adaptive, fixed, multilag_coords
    probability_mode: str = "log01"
    latent_dim: int = 32
    reconstruction_loss: str = "mse"  # mse, l1, cosine, hellinger
    beta: float = 1e-3
    matryoshka_dims: tuple[int, ...] = ()

    multilag_points: int = 256
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5

    # Numerical-stability controls for VAE training.
    grad_clip_norm: float = 5.0
    logvar_min: float = -10.0
    logvar_max: float = 8.0
    kl_warmup_epochs: int = 10

    seed: int = 1729
    device: str = "auto"
    train_split: str = "train"
    val_split: str = "val_small"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resample_rows(x: np.ndarray, count: int) -> np.ndarray:
    if len(x) == count:
        return x.astype(np.float32)
    old = np.linspace(0.0, 1.0, len(x))
    new = np.linspace(0.0, 1.0, count)
    return np.column_stack([
        np.interp(new, old, x[:, j]) for j in range(x.shape[1])
    ]).astype(np.float32)


def _multilag_scaled(
    signal,
    sampling_hz: float,
    tlpp_cfg: TLPPConfig,
    points: int,
) -> np.ndarray:
    x = make_multilag_TLPP(signal, sampling_hz, tlpp_cfg.multi_lags_us)
    x = _resample_rows(x, points)
    span = tlpp_cfg.current_max - tlpp_cfg.current_min
    x = 2.0 * (x - tlpp_cfg.current_min) / span - 1.0
    return np.clip(x.T, -1.0, 1.0).astype(np.float32)


def raw_representation(
    signal,
    sampling_hz: float,
    tlpp_cfg: TLPPConfig,
    vae_cfg: VAEConfig,
) -> np.ndarray:
    """Raw cache representation: probability maps or scaled N-D coordinates."""
    if vae_cfg.representation == "multilag_coords":
        return _multilag_scaled(
            signal,
            sampling_hz,
            tlpp_cfg,
            vae_cfg.multilag_points,
        )
    return tlpp_probability_from_signal(
        signal,
        sampling_hz,
        vae_cfg.representation,
        tlpp_cfg,
    ).astype(np.float32)


def model_input(raw: np.ndarray, tlpp_cfg: TLPPConfig, vae_cfg: VAEConfig) -> np.ndarray:
    if vae_cfg.representation == "multilag_coords":
        return np.asarray(raw, dtype=np.float32)
    return transform_occupancy(
        raw,
        vae_cfg.probability_mode,
        tlpp_cfg.log_epsilon,
    )


def build_cache(
    items: list[Item],
    output: Path,
    trace_cfg: TraceConfig,
    tlpp_cfg: TLPPConfig,
    vae_cfg: VAEConfig,
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if not items:
        raise ValueError("No traces were selected for the cache")

    started = time.perf_counter()

    # Read the first trace once to determine the cache shape, then keep that
    # result instead of loading the same network file a second time.
    first_trace = load_trace(items[0], trace_cfg)
    first_sample = raw_representation(
        first_trace.current_a,
        first_trace.sampling_hz,
        tlpp_cfg,
        vae_cfg,
    )
    if not np.all(np.isfinite(first_sample)):
        raise ValueError(
            f"First cached representation is non-finite: {items[0].uuid}"
        )

    values = np.lib.format.open_memmap(
        output / "representations.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(items), *first_sample.shape),
    )

    rows = []
    values[0] = first_sample
    rows.append((0, items[0].uuid, items[0].split, "ok", ""))
    print(f"Cached 1/{len(items)} traces", flush=True)

    for i, item in enumerate(items[1:], start=1):
        try:
            trace = load_trace(item, trace_cfg)
            sample = raw_representation(
                trace.current_a,
                trace.sampling_hz,
                tlpp_cfg,
                vae_cfg,
            )
            if not np.all(np.isfinite(sample)):
                raise ValueError("representation contains NaN or Inf")
            values[i] = sample
            rows.append((i, item.uuid, item.split, "ok", ""))
        except Exception as exc:
            values[i] = np.nan
            rows.append((i, item.uuid, item.split, "failed", str(exc)))

        if (i + 1) % 25 == 0 or i + 1 == len(items):
            elapsed = time.perf_counter() - started
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"Cached {i + 1}/{len(items)} traces "
                f"({rate:.1f} traces/s)",
                flush=True,
            )

    values.flush()

    with (output / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "uuid", "split", "status", "error"])
        w.writerows(rows)

    (output / "config.json").write_text(
        json.dumps(
            {
                "trace": trace_cfg.to_dict(),
                "tlpp": tlpp_cfg.to_dict(),
                "vae": asdict(vae_cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = sum(row[3] == "ok" for row in rows)
    failed = len(rows) - ok
    elapsed = time.perf_counter() - started
    print(
        f"Cache complete: {ok} succeeded, {failed} failed, "
        f"{elapsed:.1f} s total.",
        flush=True,
    )
    return output / "representations.npy"


class CacheDataset(Dataset):
    def __init__(
        self,
        cache: Path,
        split: str,
        tlpp_cfg: TLPPConfig,
        vae_cfg: VAEConfig,
    ):
        cache = Path(cache)
        self.values = np.load(cache / "representations.npy", mmap_mode="r")
        self.tlpp_cfg = tlpp_cfg
        self.vae_cfg = vae_cfg
        with (cache / "index.csv").open("r", encoding="utf-8-sig") as f:
            self.ids = [
                int(row["index"])
                for row in csv.DictReader(f)
                if row["status"] == "ok" and row["split"] == split
            ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        raw = np.asarray(self.values[self.ids[i]], dtype=np.float32).copy()
        x = model_input(raw, self.tlpp_cfg, self.vae_cfg)
        return torch.from_numpy(x), torch.from_numpy(raw)


class _MapVAE(nn.Module):
    def __init__(self, channels: int, grid: int, latent_dim: int):
        super().__init__()
        self.grid = grid
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

    def stats(self, x):
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        y = self.decoder(self.from_z(z).reshape(-1, *self.hidden_shape))
        return y[..., : self.grid, : self.grid]


class _CoordinateVAE(nn.Module):
    def __init__(self, channels: int, length: int, latent_dim: int):
        super().__init__()
        self.length = length
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.ReLU(),
        )
        with torch.no_grad():
            encoded = self.encoder(torch.zeros(1, channels, length))
        self.hidden_shape = tuple(encoded.shape[1:])
        hidden = encoded.numel()
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.from_z = nn.Linear(latent_dim, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(32, channels, 4, 2, 1), nn.Tanh(),
        )

    def stats(self, x):
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        y = self.decoder(self.from_z(z).reshape(-1, *self.hidden_shape))
        return y[..., : self.length]


class _VAE(nn.Module):
    def __init__(self, sample_shape: tuple[int, ...], latent_dim: int):
        super().__init__()
        if len(sample_shape) == 3:
            self.net = _MapVAE(sample_shape[0], sample_shape[1], latent_dim)
        else:
            self.net = _CoordinateVAE(sample_shape[0], sample_shape[1], latent_dim)

    def stats(self, x):
        return self.net.stats(x)

    def encode(self, x):
        return self.stats(x)[0]

    def decode(self, z):
        return self.net.decode(z)


def _reconstruction_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    raw_target: torch.Tensor,
    name: str,
    probability_maps: bool,
) -> torch.Tensor:
    if name == "mse":
        return nn.functional.mse_loss(output, target)
    if name == "l1":
        return nn.functional.l1_loss(output, target)
    if name == "cosine":
        return 1.0 - nn.functional.cosine_similarity(
            output.flatten(1), target.flatten(1), dim=1
        ).mean()
    if name == "hellinger":
        if not probability_maps:
            raise ValueError(
                "Hellinger loss applies to occupancy-map VAEs, not coordinates"
            )

        # Compute sqrt(q) directly from log-softmax.
        # This avoids taking sqrt(0) after softmax underflow,
        # which can produce infinite gradients.
        log_q = torch.log_softmax(
            output.flatten(2),
            dim=-1,
        )
        sqrt_q = torch.exp(0.5 * log_q)

        p = raw_target.flatten(2)
        p = p / p.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        sqrt_p = torch.sqrt(
            p.clamp_min(0.0)
        )

        # Squared Hellinger distance:
        # H^2(p,q) = 1 - sum_i sqrt(p_i * q_i)
        coefficient = (
            sqrt_p * sqrt_q
        ).sum(dim=-1)

        return (
            1.0 - coefficient
        ).clamp_min(0.0).mean()
    raise ValueError("loss must be mse, l1, cosine, or hellinger")


def _prefix_dims(cfg: VAEConfig) -> tuple[int, ...]:
    if not cfg.matryoshka_dims:
        return (cfg.latent_dim,)
    dims = sorted({d for d in cfg.matryoshka_dims if 0 < d <= cfg.latent_dim})
    if cfg.latent_dim not in dims:
        dims.append(cfg.latent_dim)
    return tuple(dims)


def train_vae(cache: Path, output: Path, cfg: VAEConfig) -> Path:
    """
    Train a numerically-stable TLPP VAE.

    Stability changes compared with the original version:
    - verifies cached/model inputs are finite;
    - clamps log-variance before exp();
    - warms the KL weight in gradually;
    - clips gradient norm;
    - uses the latent mean during validation so validation is deterministic;
    - aborts immediately with a useful error if a non-finite value appears.
    """
    cache = Path(cache)
    saved = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    trace_cfg = TraceConfig(**saved["trace"])
    tlpp_cfg = TLPPConfig(**saved["tlpp"])

    train_set = CacheDataset(cache, cfg.train_split, tlpp_cfg, cfg)
    val_set = CacheDataset(cache, cfg.val_split, tlpp_cfg, cfg)

    if len(train_set) == 0:
        raise ValueError(f"No usable samples in training split {cfg.train_split!r}")
    if len(val_set) == 0:
        raise ValueError(f"No usable samples in validation split {cfg.val_split!r}")

    sample_shape = tuple(train_set[0][0].shape)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    dev = _device(cfg.device)
    model = _VAE(sample_shape, cfg.latent_dim).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, cfg.batch_size)
    prefixes = _prefix_dims(cfg)
    probability_maps = cfg.representation != "multilag_coords"

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []

    def finite_or_raise(name: str, value: torch.Tensor, epoch_number: int, batch_number: int):
        if not torch.isfinite(value).all():
            finite = value[torch.isfinite(value)]
            if finite.numel():
                detail = (
                    f"finite range=[{finite.min().item():.4g}, "
                    f"{finite.max().item():.4g}]"
                )
            else:
                detail = "no finite values"
            raise FloatingPointError(
                f"{name} became NaN/Inf at epoch {epoch_number}, "
                f"batch {batch_number}; {detail}"
            )

    def run_epoch(loader, training: bool, epoch_number: int):
        model.train(training)

        total_values = []
        reconstruction_values = []
        kl_values = []
        grad_values = []

        warmup = max(int(cfg.kl_warmup_epochs), 0)
        if warmup:
            beta_scale = min(1.0, epoch_number / float(warmup))
        else:
            beta_scale = 1.0
        effective_beta = cfg.beta * beta_scale

        for batch_number, (x, raw) in enumerate(loader, start=1):
            x, raw = x.to(dev), raw.to(dev)
            finite_or_raise("model input", x, epoch_number, batch_number)
            finite_or_raise("raw cache input", raw, epoch_number, batch_number)

            if training:
                optimizer.zero_grad(set_to_none=True)

            mu, raw_logvar = model.stats(x)
            finite_or_raise("latent mean", mu, epoch_number, batch_number)
            finite_or_raise("raw latent log-variance", raw_logvar, epoch_number, batch_number)

            logvar = torch.clamp(
                raw_logvar,
                min=float(cfg.logvar_min),
                max=float(cfg.logvar_max),
            )

            # Sampling is used for VAE training.  Validation uses mu directly,
            # which makes the validation curve deterministic and easier to interpret.
            if training:
                std = torch.exp(0.5 * logvar)
                z = mu + torch.randn_like(mu) * std
            else:
                z = mu

            reconstruction_terms = []
            for width in prefixes:
                zp = z.clone()
                zp[:, width:] = 0.0
                decoded = model.decode(zp)
                finite_or_raise("decoder output", decoded, epoch_number, batch_number)

                reconstruction_terms.append(
                    _reconstruction_loss(
                        decoded,
                        x,
                        raw,
                        cfg.reconstruction_loss,
                        probability_maps,
                    )
                )

            reconstruction = torch.stack(reconstruction_terms).mean()

            # logvar is already bounded, so exp(logvar) cannot overflow.
            kl = -0.5 * torch.mean(
                1.0 + logvar - mu.pow(2) - torch.exp(logvar)
            )
            loss = reconstruction + effective_beta * kl

            finite_or_raise("reconstruction loss", reconstruction, epoch_number, batch_number)
            finite_or_raise("KL loss", kl, epoch_number, batch_number)
            finite_or_raise("total loss", loss, epoch_number, batch_number)

            if training:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg.grad_clip_norm),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                grad_values.append(float(grad_norm.detach().cpu()))

            total_values.append(float(loss.detach().cpu()))
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            kl_values.append(float(kl.detach().cpu()))

        return {
            "total": float(np.mean(total_values)),
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
            "grad_norm": float(np.mean(grad_values)) if grad_values else 0.0,
            "effective_beta": float(effective_beta),
        }

    print(
        f"Training on {len(train_set)} traces, validating on {len(val_set)} traces, "
        f"device={dev}, lr={cfg.learning_rate:g}, grad_clip={cfg.grad_clip_norm:g}",
        flush=True,
    )

    for e in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(train_loader, True, e)
        with torch.no_grad():
            val_metrics = run_epoch(val_loader, False, e)

        history.append(
            (
                e,
                train_metrics["total"],
                train_metrics["reconstruction"],
                train_metrics["kl"],
                val_metrics["total"],
                val_metrics["reconstruction"],
                val_metrics["kl"],
                train_metrics["effective_beta"],
                train_metrics["grad_norm"],
            )
        )

        print(
            f"epoch {e:03d} "
            f"train={train_metrics['total']:.6g} "
            f"(recon={train_metrics['reconstruction']:.6g}, kl={train_metrics['kl']:.6g}) "
            f"val={val_metrics['total']:.6g} "
            f"(recon={val_metrics['reconstruction']:.6g}, kl={val_metrics['kl']:.6g}) "
            f"beta={train_metrics['effective_beta']:.3g}",
            flush=True,
        )

        state = {
            "model_state": model.state_dict(),
            "vae": asdict(cfg),
            "tlpp": tlpp_cfg.to_dict(),
            "trace": trace_cfg.to_dict(),
            "sample_shape": sample_shape,
        }
        torch.save(state, output / "last.pt")

        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(state, output / "best.pt")

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "train_loss",
                "train_reconstruction",
                "train_kl",
                "val_loss",
                "val_reconstruction",
                "val_kl",
                "effective_beta",
                "mean_unclipped_grad_norm",
            ]
        )
        w.writerows(history)

    return output / "best.pt"


class TLPPVAE:
    """Loaded VAE with signal-to-embedding convenience methods."""

    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.device = _device(device)
        state = torch.load(checkpoint, map_location=self.device)
        self.cfg = VAEConfig(**state["vae"])
        self.tlpp_cfg = TLPPConfig(**state["tlpp"])
        self.trace_cfg = TraceConfig(**state["trace"])
        self.model = _VAE(tuple(state["sample_shape"]), self.cfg.latent_dim).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

    def raw_input(self, signal, sampling_hz: float) -> np.ndarray:
        return raw_representation(signal, sampling_hz, self.tlpp_cfg, self.cfg)

    def model_input(self, signal, sampling_hz: float) -> np.ndarray:
        return model_input(self.raw_input(signal, sampling_hz), self.tlpp_cfg, self.cfg)

    def embed_signal(
        self,
        signal,
        sampling_hz: float,
        dimensions: int | None = None,
    ) -> np.ndarray:
        x = torch.from_numpy(self.model_input(signal, sampling_hz)[None]).to(self.device)
        with torch.no_grad():
            z = self.model.encode(x).cpu().numpy()[0]
        return z if dimensions is None else z[: int(dimensions)]

    def reconstruct_signal(
        self,
        signal,
        sampling_hz: float,
        dimensions: int | None = None,
    ) -> np.ndarray:
        x = torch.from_numpy(self.model_input(signal, sampling_hz)[None]).to(self.device)
        with torch.no_grad():
            z = self.model.encode(x)
            if dimensions is not None:
                z[:, int(dimensions) :] = 0.0
            y = self.model.decode(z)
            if self.cfg.reconstruction_loss == "hellinger" and self.cfg.representation != "multilag_coords":
                shape = y.shape
                y = torch.softmax(y.flatten(2), dim=-1).reshape(shape)
            return y.cpu().numpy()[0]


def load_vae(checkpoint: Path, device: str = "auto") -> TLPPVAE:
    return TLPPVAE(checkpoint, device)


def embed_signal(
    signal,
    sampling_hz: float,
    model: TLPPVAE,
    dimensions: int | None = None,
) -> np.ndarray:
    return model.embed_signal(signal, sampling_hz, dimensions)
