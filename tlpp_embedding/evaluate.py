from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .core import TLPPConfig, transform_occupancy
from .hdf5 import validate_universal_tlpp
from .models import counts_to_probability, load_vae_checkpoint
from .training import PathInputs, normalize_h5_paths


def _effective_rank_from_moments(
    count: int,
    sum_x: np.ndarray,
    sum_x2: np.ndarray,
    sum_outer: np.ndarray,
) -> tuple[float, float]:
    if count < 2:
        return 0.0, 0.0
    mean = sum_x / count
    variance = np.clip(sum_x2 / count - mean * mean, 0.0, None)
    std = np.sqrt(variance)
    active = std > 1e-12
    if not np.any(active):
        return 0.0, 0.0
    covariance = sum_outer / count - np.outer(mean, mean)
    denom = np.outer(std, std)
    correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 1e-24)
    correlation = correlation[np.ix_(active, active)]
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-20)
    probabilities = probabilities[probabilities > 1e-15]
    rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    maximum = min(count - 1, int(np.count_nonzero(active)))
    return rank, rank / maximum if maximum else 0.0


def evaluate_vae_hdf5(
    checkpoint: Path | str,
    h5_paths: PathInputs,
    output: Path | str,
    embedding_dims: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
    batch_size: int = 64,
    device: str = "auto",
    make_plots: bool = True,
) -> dict:
    """Evaluate one checkpoint over one or several universal TLPP HDF5 files.

    Multiple files are treated as one logical split, matching the combined
    subscale+fullscale training/validation setup.
    """
    checkpoint = Path(checkpoint)
    paths = normalize_h5_paths(h5_paths)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    metadata = tuple(validate_universal_tlpp(path) for path in paths)
    component_counts = [item.count for item in metadata]
    total_count = sum(component_counts)

    state, cfg, model, dev = load_vae_checkpoint(checkpoint, device)
    sample_shape = tuple(state["sample_shape"])
    if sample_shape != (1, 128, 128):
        raise ValueError(f"Expected checkpoint input shape (1,128,128), got {sample_shape}")

    dims = tuple(sorted({int(width) for width in embedding_dims if 0 < int(width) <= cfg.latent_dim}))
    if cfg.latent_dim not in dims:
        dims = (*dims, cfg.latent_dim)
    if not dims:
        raise ValueError("No valid embedding dimensions requested")

    prefix_errors = {width: np.empty(total_count, dtype=np.float32) for width in dims}
    latent_sum = np.zeros(cfg.latent_dim, dtype=np.float64)
    latent_sq_sum = np.zeros(cfg.latent_dim, dtype=np.float64)
    latent_outer_sum = np.zeros((cfg.latent_dim, cfg.latent_dim), dtype=np.float64)
    kl_sum = np.zeros(cfg.latent_dim, dtype=np.float64)
    log_epsilon = TLPPConfig().log_epsilon

    trace_rows: list[tuple[int, int, str, int, str]] = []
    aggregate_offset = 0

    with torch.no_grad():
        for component_index, path in enumerate(paths):
            with h5py.File(path, "r") as h5:
                counts_ds = h5["tlpp_counts"]
                n = len(counts_ds)
                for start in range(0, n, batch_size):
                    stop = min(start + batch_size, n)
                    counts = np.asarray(counts_ds[start:stop], dtype=np.float32)[:, None]
                    probability = counts_to_probability(counts)
                    model_input = transform_occupancy(probability, cfg.probability_mode, log_epsilon)
                    x = torch.from_numpy(model_input).to(dev)
                    probability_t = torch.from_numpy(probability).to(dev)

                    mu, raw_logvar = model.stats(x)
                    logvar = torch.clamp(raw_logvar, cfg.logvar_min, cfg.logvar_max)
                    mu_np = mu.detach().cpu().numpy().astype(np.float64, copy=False)
                    latent_sum += mu_np.sum(axis=0)
                    latent_sq_sum += np.square(mu_np).sum(axis=0)
                    latent_outer_sum += mu_np.T @ mu_np
                    kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar))
                    kl_sum += kl_dim.sum(dim=0).detach().cpu().numpy()

                    p = probability_t.flatten(2)
                    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    sqrt_p = torch.sqrt(p.clamp_min(0.0))

                    aggregate_slice = slice(aggregate_offset + start, aggregate_offset + stop)
                    for width in dims:
                        z = torch.zeros_like(mu)
                        z[:, :width] = mu[:, :width]
                        logits = model.decode(z)
                        sqrt_q = torch.exp(0.5 * torch.log_softmax(logits.flatten(2), dim=-1))
                        h2 = (1.0 - (sqrt_p * sqrt_q).sum(dim=-1)).clamp_min(0.0).mean(dim=1)
                        prefix_errors[width][aggregate_slice] = h2.detach().cpu().numpy()

                global_index = (
                    np.asarray(h5["global_index"], dtype=np.uint64)
                    if "global_index" in h5
                    else np.arange(n, dtype=np.uint64)
                )
                trace_keys = list(h5["trace_key"].asstr()[:]) if "trace_key" in h5 else [str(i) for i in range(n)]
                trace_rows.extend(
                    (component_index, local_row, path.name, int(global_index[local_row]), trace_keys[local_row])
                    for local_row in range(n)
                )
            aggregate_offset += n

    latent_mean = latent_sum / total_count
    latent_variance = np.clip(latent_sq_sum / total_count - latent_mean * latent_mean, 0.0, None)
    latent_std = np.sqrt(latent_variance)
    mean_kl_dim = kl_sum / total_count
    effective_rank, rank_fraction = _effective_rank_from_moments(
        total_count, latent_sum, latent_sq_sum, latent_outer_sum
    )

    prefix_metrics: dict[str, dict[str, float]] = {}
    for width in dims:
        values = prefix_errors[width]
        prefix_metrics[str(width)] = {
            "mean_hellinger_squared": float(np.mean(values)),
            "median_hellinger_squared": float(np.median(values)),
            "p90_hellinger_squared": float(np.percentile(values, 90)),
            "p95_hellinger_squared": float(np.percentile(values, 95)),
            "p99_hellinger_squared": float(np.percentile(values, 99)),
            "mean_hellinger": float(np.mean(np.sqrt(values))),
        }

    metrics = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "best_validation_loss": float(state.get("best_validation_loss", float("nan"))),
        "datasets": [str(path) for path in paths],
        "component_counts": component_counts,
        "count": total_count,
        "input_shape": list(sample_shape),
        "latent_dim": int(cfg.latent_dim),
        "embedding_dims": list(dims),
        "probability_mode": cfg.probability_mode,
        "reconstruction_loss": cfg.reconstruction_loss,
        "prefix_metrics": prefix_metrics,
        "latent_statistics": {
            "effective_rank": effective_rank,
            "effective_rank_fraction": rank_fraction,
            "mean_abs_latent_mean": float(np.mean(np.abs(latent_mean))),
            "mean_latent_std": float(np.mean(latent_std)),
            "min_latent_std": float(np.min(latent_std)),
            "max_latent_std": float(np.max(latent_std)),
            "mean_kl_per_dimension": float(np.mean(mean_kl_dim)),
            "active_dims_kl_gt_0.001": int(np.count_nonzero(mean_kl_dim > 0.001)),
            "active_dims_kl_gt_0.01": int(np.count_nonzero(mean_kl_dim > 0.01)),
            "active_dims_kl_gt_0.1": int(np.count_nonzero(mean_kl_dim > 0.1)),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (output / "prefix_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "embedding_dim", "mean_hellinger_squared", "median_hellinger_squared",
            "p90_hellinger_squared", "p95_hellinger_squared", "p99_hellinger_squared",
            "mean_hellinger",
        ])
        for width in dims:
            row = prefix_metrics[str(width)]
            writer.writerow([
                width,
                row["mean_hellinger_squared"],
                row["median_hellinger_squared"],
                row["p90_hellinger_squared"],
                row["p95_hellinger_squared"],
                row["p99_hellinger_squared"],
                row["mean_hellinger"],
            ])

    with (output / "latent_dimensions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "mu_mean", "mu_std", "mean_kl"])
        for index in range(cfg.latent_dim):
            writer.writerow([index + 1, latent_mean[index], latent_std[index], mean_kl_dim[index]])

    with (output / "per_trace_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "row", "component", "local_row", "dataset", "global_index", "trace_key",
            *[f"h2_{width}d" for width in dims],
        ])
        for aggregate_row, (component, local_row, dataset, global_index, trace_key) in enumerate(trace_rows):
            writer.writerow([
                aggregate_row, component, local_row, dataset, global_index, trace_key,
                *[float(prefix_errors[width][aggregate_row]) for width in dims],
            ])

    if make_plots:
        from .plotting import plot_evaluation_summary
        plot_evaluation_summary(output, checkpoint.parent / "history.csv")

    print(f"HDF5 VAE evaluation complete: {output}")
    print(f"datasets: {len(paths)}; traces: {total_count:,}")
    print(f"effective rank: {effective_rank:.3f} ({rank_fraction:.3f})")
    for width in dims:
        print(f"  {width:3d}D: H^2 mean={prefix_metrics[str(width)]['mean_hellinger_squared']:.8f}")
    return metrics
