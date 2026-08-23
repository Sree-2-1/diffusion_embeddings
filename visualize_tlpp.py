#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tlpp_embedding.core import (
    TLPPConfig,
    adaptive_lag_samples,
    make_TLPP,
    make_multilag_TLPP,
    tlpp_probability_from_signal,
    transform_occupancy,
)
from tlpp_embedding.data import TraceConfig, load_trace, select_items
from tlpp_embedding.models import TLPPVAE


DATA_ROOT = Path(r"T:\h9_diffusion_model")


def save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_map(array, cfg: TLPPConfig, title: str, path: Path):
    plt.figure(figsize=(6.2, 5.4))
    image = plt.imshow(
        array.T,
        origin="lower",
        extent=[cfg.current_min, cfg.current_max, cfg.current_min, cfg.current_max],
        aspect="equal",
    )
    plt.xlabel("I(t) / A")
    plt.ylabel("Delayed current / A")
    plt.title(title)
    plt.colorbar(image)
    save(path)


def plot_2d_variant(trace, cfg: TLPPConfig, variant: str, folder: Path, number: int):
    if variant == "adaptive":
        lag = adaptive_lag_samples(trace.current_a, trace.sampling_hz, cfg)
        points = make_TLPP(trace.current_a, trace.sampling_hz, lag_samples=lag)
        lag_text = f"{lag / trace.sampling_hz * 1e6:.3g} us"
    else:
        points = make_TLPP(trace.current_a, trace.sampling_hz, lag_us=cfg.fixed_lag_us)
        lag_text = f"{cfg.fixed_lag_us:g} us"

    plt.figure(figsize=(6.2, 5.4))
    plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.45)
    plt.xlim(cfg.current_min, cfg.current_max)
    plt.ylim(cfg.current_min, cfg.current_max)
    plt.xlabel("I(t) / A")
    plt.ylabel("I(t-delay) / A")
    plt.title(f"{variant.capitalize()} TLPP points, lag={lag_text}")
    save(folder / f"{number:02d}_{variant}_points.png")

    raw = tlpp_probability_from_signal(
        trace.current_a, trace.sampling_hz, variant, cfg
    )[0]
    transformed = transform_occupancy(raw, cfg.probability_mode, cfg.log_epsilon)
    plot_map(
        raw,
        cfg,
        f"{variant.capitalize()} occupancy probability",
        folder / f"{number + 1:02d}_{variant}_probability.png",
    )
    plot_map(
        transformed,
        cfg,
        f"{variant.capitalize()} TLPP embedding ({cfg.probability_mode})",
        folder / f"{number + 2:02d}_{variant}_embedding.png",
    )


def plot_multilag(trace, cfg: TLPPConfig, folder: Path):
    coords = make_multilag_TLPP(
        trace.current_a, trace.sampling_hz, cfg.multi_lags_us
    )
    fig = plt.figure(figsize=(6.5, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=5, alpha=0.4)
    ax.set_xlabel("I(t)")
    ax.set_ylabel(f"I(t-{cfg.multi_lags_us[0]:g} us)")
    ax.set_zlabel(f"I(t-{cfg.multi_lags_us[1]:g} us)")
    ax.set_title("First three dimensions of the multi-lag TLPP")
    save(folder / "08_multilag_3d.png")

    raw_maps = tlpp_probability_from_signal(
        trace.current_a, trace.sampling_hz, "multilag", cfg
    )
    maps = transform_occupancy(raw_maps, cfg.probability_mode, cfg.log_epsilon)
    pairs = list(combinations(range(coords.shape[1]), 2))
    cols = min(4, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for k, (i, j) in enumerate(pairs):
        ax = axes[k // cols][k % cols]
        image = ax.imshow(
            maps[k].T,
            origin="lower",
            extent=[cfg.current_min, cfg.current_max, cfg.current_min, cfg.current_max],
            aspect="equal",
        )
        ax.set_title(f"dimensions {i} vs {j}")
        fig.colorbar(image, ax=ax, fraction=0.046)
    for k in range(len(pairs), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"Multi-lag pairwise TLPP maps ({cfg.probability_mode})")
    save(folder / "09_multilag_pairwise.png")


def plot_vae(trace, checkpoint: Path, folder: Path, dimensions: int | None):
    model = TLPPVAE(checkpoint)
    target = model.model_input(trace.current_a, trace.sampling_hz)
    reconstruction = model.reconstruct_signal(
        trace.current_a, trace.sampling_hz, dimensions
    )

    if model.cfg.reconstruction_loss == "hellinger" and model.cfg.representation != "multilag_coords":
        target = model.raw_input(trace.current_a, trace.sampling_hz)

    if target.ndim == 3:
        plt.figure(figsize=(10, 4.2))
        plt.subplot(1, 2, 1)
        plt.imshow(target[0].T, origin="lower", aspect="auto")
        plt.title("VAE target")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(reconstruction[0].T, origin="lower", aspect="auto")
        plt.title("VAE reconstruction")
        plt.colorbar()
    else:
        plt.figure(figsize=(9, 4.5))
        for i in range(min(3, target.shape[0])):
            plt.plot(target[i], label=f"target {i}")
            plt.plot(reconstruction[i], "--", label=f"recon {i}")
        plt.legend()
        plt.title("Multi-lag coordinate VAE reconstruction")
    save(folder / "10_vae_reconstruction.png")


def main():
    p = argparse.ArgumentParser(description="Visualize TLPP variants and optional VAE reconstruction")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--split", default="val_small")
    p.add_argument("--pool-size", type=int, default=500)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--window-us", type=float, default=700.0)
    p.add_argument("--window-start", type=float, default=0.5)
    p.add_argument("--tlpp-bins", type=int, default=64)
    p.add_argument("--tlpp-min", type=float, default=-5.0)
    p.add_argument("--tlpp-max", type=float, default=105.0)
    p.add_argument("--smoothing", type=float, default=0.75)
    p.add_argument("--probability-mode", choices=["raw", "sqrt", "log", "log01"], default="log01")
    p.add_argument("--log-epsilon", type=float, default=1e-6)
    p.add_argument("--adaptive-period-fraction", type=float, default=0.25)
    p.add_argument("--fixed-lag-us", type=float, default=1.0)
    p.add_argument("--multi-lags-us", default="1,2,3,5")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--embedding-dim", type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    items = select_items(
        a.data_root, a.split, a.pool_size, a.seed, a.input_dir, a.manifest
    )
    rng = np.random.default_rng(a.seed)
    ids = np.sort(rng.choice(len(items), min(a.count, len(items)), replace=False))
    trace_cfg = TraceConfig(window_us=a.window_us, window_start=a.window_start)
    cfg = TLPPConfig(
        bins=a.tlpp_bins,
        current_min=a.tlpp_min,
        current_max=a.tlpp_max,
        smoothing_sigma=a.smoothing,
        probability_mode=a.probability_mode,
        log_epsilon=a.log_epsilon,
        adaptive_period_fraction=a.adaptive_period_fraction,
        fixed_lag_us=a.fixed_lag_us,
        multi_lags_us=tuple(float(x) for x in a.multi_lags_us.split(",") if x.strip()),
    )

    for number, index in enumerate(ids, 1):
        trace = load_trace(items[int(index)], trace_cfg)
        folder = a.output / f"{number:02d}_{trace.item.uuid}"
        folder.mkdir(parents=True, exist_ok=True)

        time_us = (trace.time_s - trace.time_s[0]) * 1e6
        plt.figure(figsize=(9, 4.2))
        plt.plot(time_us, trace.current_a)
        plt.xlabel("Time / us")
        plt.ylabel("Discharge current / A")
        plt.title("Selected current window")
        save(folder / "01_window.png")

        plot_2d_variant(trace, cfg, "adaptive", folder, 2)
        plot_2d_variant(trace, cfg, "fixed", folder, 5)
        plot_multilag(trace, cfg, folder)
        if a.checkpoint:
            plot_vae(trace, a.checkpoint, folder, a.embedding_dim)

        (folder / "summary.txt").write_text(
            "\n".join([
                f"uuid: {trace.item.uuid}",
                f"sampling_hz: {trace.sampling_hz}",
                f"window_us: {a.window_us}",
                f"mean_current_a: {trace.current_mean}",
                f"rms_current_a: {trace.current_rms}",
                f"dominant_frequency_hz: {trace.dominant_frequency_hz}",
                f"tlpp_bins: {cfg.bins}",
                f"probability_mode: {cfg.probability_mode}",
                f"fixed_lag_us: {cfg.fixed_lag_us}",
                f"multi_lags_us: {cfg.multi_lags_us}",
            ]) + "\n",
            encoding="utf-8",
        )

    print(f"Plots: {a.output}")


if __name__ == "__main__":
    main()
