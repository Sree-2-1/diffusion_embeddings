from __future__ import annotations

import csv
import json
import os
import random
import time
from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .core import TLPPConfig, transform_occupancy
from .hdf5 import validate_universal_tlpp
from .models import VAE, VAEConfig, counts_to_probability, prefix_dims, reconstruction_loss, resolve_device

HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "train_reconstruction",
    "train_kl",
    "val_loss",
    "val_reconstruction",
    "val_kl",
    "effective_beta",
    "mean_unclipped_grad_norm",
    "epoch_seconds",
]

PathInput = Path | str
PathInputs = PathInput | Sequence[PathInput]


class HDF5TLPPDataset(Dataset):
    """Lazy dataset over one finalized universal TLPP HDF5 file."""

    def __init__(self, path: PathInput, cfg: VAEConfig, log_epsilon: float = 1e-6):
        self.path = str(Path(path))
        self.cfg = cfg
        self.log_epsilon = float(log_epsilon)
        self.metadata = validate_universal_tlpp(self.path)
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return self.metadata.count

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, index: int):
        counts = np.asarray(self._file()["tlpp_counts"][int(index)], dtype=np.float32)[None]
        probability = counts_to_probability(counts)
        model_input = transform_occupancy(
            probability,
            self.cfg.probability_mode,
            self.log_epsilon,
        )
        return torch.from_numpy(model_input), torch.from_numpy(probability)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def normalize_h5_paths(paths: PathInputs) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        result = (Path(paths),)
    else:
        result = tuple(Path(path) for path in paths)
    if not result:
        raise ValueError("At least one HDF5 path is required")
    return result


class MultiHDF5TLPPDataset(Dataset):
    """Treat several universal HDF5 files as one logical dataset without copying them."""

    def __init__(self, paths: PathInputs, cfg: VAEConfig, log_epsilon: float = 1e-6):
        self.paths = normalize_h5_paths(paths)
        self.datasets = tuple(HDF5TLPPDataset(path, cfg, log_epsilon) for path in self.paths)
        self.component_counts = tuple(len(dataset) for dataset in self.datasets)
        ends: list[int] = []
        running = 0
        for count in self.component_counts:
            running += int(count)
            ends.append(running)
        self._ends = tuple(ends)
        self.total_count = running

    def __len__(self) -> int:
        return self.total_count

    def locate(self, index: int) -> tuple[int, int]:
        index = int(index)
        if index < 0:
            index += self.total_count
        if not 0 <= index < self.total_count:
            raise IndexError(index)
        component = bisect_right(self._ends, index)
        start = 0 if component == 0 else self._ends[component - 1]
        return component, index - start

    def __getitem__(self, index: int):
        component, local_index = self.locate(index)
        return self.datasets[component][local_index]

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def default_hdf5_vae_config(epochs: int = 50) -> VAEConfig:
    return VAEConfig(epochs=int(epochs))


def _validate_paths(paths: PathInputs):
    normalized = normalize_h5_paths(paths)
    metadata = tuple(validate_universal_tlpp(path) for path in normalized)
    return normalized, metadata


def _path_strings(paths: tuple[Path, ...]) -> list[str]:
    return [str(path) for path in paths]


def _atomic_torch_save(obj, path: Path) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def _atomic_json(path: Path, obj) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _write_history(path: Path, history: list[dict]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in HISTORY_COLUMNS})
    os.replace(temp, path)


def _rng_state() -> dict:
    state = {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(checkpoint: dict) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].detach().cpu().to(torch.uint8))
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all([
            state.detach().cpu().to(torch.uint8)
            for state in checkpoint["cuda_rng_state_all"]
        ])


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _make_loader(dataset, cfg: VAEConfig, *, shuffle: bool, workers: int, pin: bool):
    kwargs = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _finite_or_raise(name: str, value: torch.Tensor, epoch: int, batch: int) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} became NaN/Inf at epoch {epoch}, batch {batch}")


def _run_epoch(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    cfg: VAEConfig,
    prefixes: tuple[int, ...],
    device: torch.device,
    *,
    training: bool,
    epoch: int,
    beta: float,
) -> dict:
    model.train(training)
    n_seen = 0
    total_sum = recon_sum = kl_sum = grad_sum = 0.0
    grad_batches = 0

    masks = torch.zeros((len(prefixes), cfg.latent_dim), device=device)
    for j, width in enumerate(prefixes):
        masks[j, :width] = 1.0

    for batch_no, (x, probability) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        probability = probability.to(device, non_blocking=True)
        _finite_or_raise("model input", x, epoch, batch_no)
        _finite_or_raise("probability map", probability, epoch, batch_no)

        if training:
            optimizer.zero_grad(set_to_none=True)

        mu, raw_logvar = model.stats(x)
        logvar = torch.clamp(raw_logvar, cfg.logvar_min, cfg.logvar_max)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if training else mu

        nested = (z.unsqueeze(0) * masks[:, None, :]).reshape(
            len(prefixes) * z.shape[0], cfg.latent_dim
        )
        decoded = model.decode(nested).reshape(len(prefixes), z.shape[0], *x.shape[1:])
        reconstruction = torch.stack([
            reconstruction_loss(decoded[j], probability, cfg.reconstruction_loss)
            for j in range(len(prefixes))
        ]).mean()
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar))
        loss = reconstruction + float(beta) * kl

        _finite_or_raise("reconstruction loss", reconstruction, epoch, batch_no)
        _finite_or_raise("KL loss", kl, epoch, batch_no)
        _finite_or_raise("total loss", loss, epoch, batch_no)

        if training:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            grad_sum += float(grad_norm.detach().cpu())
            grad_batches += 1

        batch_size = int(x.shape[0])
        n_seen += batch_size
        total_sum += float(loss.detach().cpu()) * batch_size
        recon_sum += float(reconstruction.detach().cpu()) * batch_size
        kl_sum += float(kl.detach().cpu()) * batch_size

    return {
        "total": total_sum / n_seen,
        "reconstruction": recon_sum / n_seen,
        "kl": kl_sum / n_seen,
        "grad_norm": grad_sum / grad_batches if grad_batches else 0.0,
        "n": n_seen,
    }


def preflight_hdf5_vae(train_h5: PathInputs, val_h5: PathInputs, cfg: VAEConfig) -> dict:
    train_paths, train_meta = _validate_paths(train_h5)
    val_paths, val_meta = _validate_paths(val_h5)
    log_epsilon = TLPPConfig().log_epsilon

    train_dataset = MultiHDF5TLPPDataset(train_paths, cfg, log_epsilon)
    x, probability = train_dataset[0]
    train_dataset.close()

    sample_shape = tuple(x.shape)
    model = VAE(sample_shape, cfg.latent_dim)
    with torch.no_grad():
        mu, _ = model.stats(x.unsqueeze(0))
        decoded = model.decode(mu)

    return {
        "train_count": sum(meta.count for meta in train_meta),
        "val_count": sum(meta.count for meta in val_meta),
        "train_component_counts": [meta.count for meta in train_meta],
        "val_component_counts": [meta.count for meta in val_meta],
        "train_paths": _path_strings(train_paths),
        "val_paths": _path_strings(val_paths),
        "sample_shape": sample_shape,
        "latent_shape": tuple(mu.shape),
        "decoder_shape": tuple(decoded.shape),
        "probability_sum": float(probability.sum()),
        "train_compression": [meta.compression for meta in train_meta],
        "val_compression": [meta.compression for meta in val_meta],
    }


def train_hdf5_vae(
    train_h5: PathInputs,
    val_h5: PathInputs,
    output: Path | str,
    *,
    cfg: VAEConfig | None = None,
    target_epochs: int | None = None,
    num_workers: int = 6,
    max_runtime_minutes: float | None = None,
    device: str = "auto",
) -> Path:
    """Train or resume the production VAE.

    ``target_epochs`` is the total desired epoch.  A pre-existing ``last.pt``
    restores model/optimizer/RNG/history state and resumes at the next epoch.
    The checkpoint also locks the training/validation HDF5 path lists so an
    interrupted run cannot silently resume on different data.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    requested_cfg = cfg or default_hdf5_vae_config(target_epochs or 50)
    if target_epochs is not None:
        requested_cfg = VAEConfig.from_dict({**requested_cfg.to_dict(), "epochs": int(target_epochs)})

    train_paths, train_meta = _validate_paths(train_h5)
    val_paths, val_meta = _validate_paths(val_h5)
    requested_train = _path_strings(train_paths)
    requested_val = _path_strings(val_paths)

    last_path = output / "last.pt"
    best_path = output / "best.pt"
    existing = None

    if last_path.exists():
        existing = torch.load(last_path, map_location="cpu", weights_only=False)
        cfg = VAEConfig.from_dict(existing["vae"])
        cfg = VAEConfig.from_dict({**cfg.to_dict(), "epochs": requested_cfg.epochs})
        saved_paths = existing.get("data_paths", {})
        saved_train = saved_paths.get("train")
        saved_val = saved_paths.get("val_small")
        if isinstance(saved_train, str):
            saved_train = [saved_train]
        if isinstance(saved_val, str):
            saved_val = [saved_val]
        if saved_train not in (None, requested_train) or saved_val not in (None, requested_val):
            raise RuntimeError(
                "Refusing to resume with different data files. "
                f"checkpoint train={saved_train}, val={saved_val}; "
                f"requested train={requested_train}, val={requested_val}"
            )
    else:
        cfg = requested_cfg

    log_epsilon = TLPPConfig().log_epsilon
    train_dataset = MultiHDF5TLPPDataset(train_paths, cfg, log_epsilon)
    val_dataset = MultiHDF5TLPPDataset(val_paths, cfg, log_epsilon)
    sample_x, _ = train_dataset[0]
    train_dataset.close()
    sample_shape = tuple(sample_x.shape)
    if sample_shape != (1, 128, 128):
        raise RuntimeError(f"Production run requires 128x128 TLPP input, got {sample_shape}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    dev = resolve_device(device)
    model = VAE(sample_shape, cfg.latent_dim).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    prefixes = prefix_dims(cfg)

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    start_epoch = 1

    if existing is not None:
        model.load_state_dict(existing["model_state"])
        optimizer.load_state_dict(existing["optimizer_state"])
        _optimizer_to(optimizer, dev)
        history = list(existing.get("history", []))
        best_val = float(existing.get("best_validation_loss", float("inf")))
        best_epoch = int(existing.get("best_epoch", 0))
        start_epoch = int(existing["epoch"]) + 1
        _restore_rng_state(existing)
        print(
            f"RESUME: starting epoch {start_epoch}; best_epoch={best_epoch}; "
            f"best_val={best_val:.8g}",
            flush=True,
        )

    _atomic_json(output / "config.json", {
        "created_or_updated_utc": datetime.now(timezone.utc).isoformat(),
        "train_h5": requested_train,
        "val_h5": requested_val,
        "train_component_counts": [meta.count for meta in train_meta],
        "val_component_counts": [meta.count for meta in val_meta],
        "train_count": sum(meta.count for meta in train_meta),
        "val_count": sum(meta.count for meta in val_meta),
        "input_grid": [128, 128],
        "validation_policy": "val_small only; val_large is held out",
        "target_epochs": cfg.epochs,
        "vae": cfg.to_dict(),
    })

    pin = dev.type == "cuda"
    train_loader = _make_loader(train_dataset, cfg, shuffle=True, workers=num_workers, pin=pin)
    val_loader = _make_loader(val_dataset, cfg, shuffle=False, workers=num_workers, pin=pin)

    print(
        f"train={len(train_dataset):,}, val={len(val_dataset):,}, input={sample_shape}, "
        f"latent={cfg.latent_dim}, prefixes={prefixes}, target_epochs={cfg.epochs}",
        flush=True,
    )

    segment_start = time.monotonic()
    previous_epoch_seconds = None
    if history:
        try:
            previous_epoch_seconds = float(history[-1]["epoch_seconds"])
        except Exception:
            pass

    for epoch in range(start_epoch, cfg.epochs + 1):
        if max_runtime_minutes is not None and previous_epoch_seconds is not None:
            elapsed = time.monotonic() - segment_start
            remaining = max_runtime_minutes * 60.0 - elapsed
            needed = max(previous_epoch_seconds * 1.35 + 300.0, 900.0)
            if remaining < needed:
                print(f"Clean segment stop before epoch {epoch}; resubmit to resume from last.pt", flush=True)
                break

        started = time.monotonic()
        beta_scale = min(1.0, epoch / cfg.kl_warmup_epochs) if cfg.kl_warmup_epochs else 1.0
        effective_beta = cfg.beta * beta_scale

        train_metrics = _run_epoch(
            model, optimizer, train_loader, cfg, prefixes, dev,
            training=True, epoch=epoch, beta=effective_beta,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, optimizer, val_loader, cfg, prefixes, dev,
                training=False, epoch=epoch, beta=effective_beta,
            )

        epoch_seconds = time.monotonic() - started
        previous_epoch_seconds = epoch_seconds
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["total"],
            "train_reconstruction": train_metrics["reconstruction"],
            "train_kl": train_metrics["kl"],
            "val_loss": val_metrics["total"],
            "val_reconstruction": val_metrics["reconstruction"],
            "val_kl": val_metrics["kl"],
            "effective_beta": effective_beta,
            "mean_unclipped_grad_norm": train_metrics["grad_norm"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        eligible = cfg.kl_warmup_epochs == 0 or epoch >= cfg.kl_warmup_epochs
        is_best = eligible and val_metrics["total"] < best_val
        if is_best:
            best_val = float(val_metrics["total"])
            best_epoch = epoch

        checkpoint = {
            "checkpoint_version": "tlpp-hdf5-vae-v3",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "vae": cfg.to_dict(),
            "sample_shape": sample_shape,
            "data_paths": {"train": requested_train, "val_small": requested_val},
            "input_grid": [128, 128],
            "input_source": "precomputed_fixed_10us_universal_hdf5",
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        }
        checkpoint.update(_rng_state())
        _atomic_torch_save(checkpoint, last_path)
        if is_best:
            _atomic_torch_save(checkpoint, best_path)
        _write_history(output / "history.csv", history)

        print(
            f"epoch {epoch:03d} train={train_metrics['total']:.6g} "
            f"(recon={train_metrics['reconstruction']:.6g}, kl={train_metrics['kl']:.6g}) "
            f"val={val_metrics['total']:.6g} "
            f"(recon={val_metrics['reconstruction']:.6g}, kl={val_metrics['kl']:.6g}) "
            f"beta={effective_beta:.3g} time={epoch_seconds/60:.2f}m "
            f"{'BEST' if is_best else ''}",
            flush=True,
        )

    completed_epoch = int(history[-1]["epoch"]) if history else 0
    print(
        f"Segment complete: completed_epoch={completed_epoch}, target={cfg.epochs}, "
        f"best_epoch={best_epoch}",
        flush=True,
    )
    return best_path if best_path.exists() else last_path
