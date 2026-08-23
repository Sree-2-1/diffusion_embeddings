#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tlpp_embedding.core import TLPPConfig
from tlpp_embedding.data import TraceConfig, select_items, write_manifest
from tlpp_embedding.evaluate import evaluate_tlpp, evaluate_vae
from tlpp_embedding.models import VAEConfig, build_cache, train_vae


DATA_ROOT = Path(r"T:\h9_diffusion_model")
OUTPUT_ROOT = Path(r"D:\PEPL\diffusion_embeddings")


def csv_text(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def csv_float(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in csv_text(text))


def csv_int(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in csv_text(text)) if text.strip() else ()


def trace_cfg(a) -> TraceConfig:
    return TraceConfig(window_us=a.window_us, window_start=a.window_start)


def tlpp_cfg(a) -> TLPPConfig:
    return TLPPConfig(
        bins=a.tlpp_bins,
        current_min=a.tlpp_min,
        current_max=a.tlpp_max,
        smoothing_sigma=a.smoothing,
        probability_mode=getattr(a, "probability_mode", "log01"),
        log_epsilon=a.log_epsilon,
        adaptive_period_fraction=a.adaptive_period_fraction,
        fixed_lag_us=a.fixed_lag_us,
        multi_lags_us=csv_float(a.multi_lags_us),
    )


def selected(a, split=None):
    return select_items(
        a.data_root,
        split or a.split,
        a.max_files,
        a.seed,
        a.input_dir,
        a.manifest,
    )


def cmd_manifest(a):
    write_manifest(a.data_root, a.output, a.max_files, a.seed)
    print(f"Manifest: {a.output}")


def cmd_evaluate(a):
    evaluate_tlpp(
        selected(a),
        a.output,
        trace_cfg(a),
        tlpp_cfg(a),
        variants=csv_text(a.variants),
        probability_modes=csv_text(a.probability_modes),
        window_starts=csv_float(a.window_starts),
        noise_snr_db=a.noise_snr_db,
        seed=a.seed,
    )


def cmd_cache(a):
    items = []
    for i, split in enumerate(csv_text(a.splits)):
        print(f"Selecting {split} files...", flush=True)
        chosen = select_items(
            a.data_root,
            split,
            a.max_files,
            a.seed + i,
            a.input_dir,
            a.manifest,
        )
        print(f"Selected {len(chosen)} {split} files.", flush=True)
        items += chosen
    print(f"Building cache from {len(items)} traces...", flush=True)
    cfg = VAEConfig(
        representation=a.representation,
        multilag_points=a.multilag_points,
        seed=a.seed,
    )
    path = build_cache(items, a.output, trace_cfg(a), tlpp_cfg(a), cfg)
    print(f"Cache: {path}")


def cmd_train(a):
    saved = json.loads((a.cache / "config.json").read_text(encoding="utf-8"))
    cached = VAEConfig(**saved["vae"])
    cfg = VAEConfig(
        representation=cached.representation,
        probability_mode=a.probability_mode,
        latent_dim=a.latent_dim,
        reconstruction_loss=a.loss,
        beta=a.beta,
        matryoshka_dims=csv_int(a.matryoshka_dims),
        multilag_points=cached.multilag_points,
        batch_size=a.batch_size,
        epochs=a.epochs,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        grad_clip_norm=a.grad_clip_norm,
        logvar_min=a.logvar_min,
        logvar_max=a.logvar_max,
        kl_warmup_epochs=a.kl_warmup_epochs,
        seed=a.seed,
        device=a.device,
        train_split=a.train_split,
        val_split=a.val_split,
    )
    checkpoint = train_vae(a.cache, a.output, cfg)
    print(f"Best checkpoint: {checkpoint}")


def cmd_evaluate_vae(a):
    evaluate_vae(
        selected(a),
        a.checkpoint,
        a.output,
        trace_cfg(a),
        embedding_dims=csv_int(a.embedding_dims),
        window_starts=csv_float(a.window_starts),
        noise_snr_db=a.noise_snr_db,
        seed=a.seed,
        device=a.device,
    )


def add_data_args(p):
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--split", default="val_small")
    p.add_argument("--max-files", type=int, default=500)
    p.add_argument("--seed", type=int, default=1729)


def add_trace_args(p):
    p.add_argument("--window-us", type=float, default=700.0)
    p.add_argument("--window-start", type=float, default=0.5)


def add_tlpp_args(p):
    p.add_argument("--tlpp-bins", type=int, default=64)
    p.add_argument("--tlpp-min", type=float, default=-5.0)
    p.add_argument("--tlpp-max", type=float, default=105.0)
    p.add_argument("--smoothing", type=float, default=0.75)
    p.add_argument("--log-epsilon", type=float, default=1e-6)
    p.add_argument("--adaptive-period-fraction", type=float, default=0.25)
    p.add_argument("--fixed-lag-us", type=float, default=1.0)
    p.add_argument("--multi-lags-us", default="1,2,3,5")


def parser():
    p = argparse.ArgumentParser(description="TLPP library experiments and VAE training")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest")
    m.add_argument("--data-root", type=Path, default=DATA_ROOT)
    m.add_argument("--max-files", type=int)
    m.add_argument("--seed", type=int, default=1729)
    m.add_argument("--output", type=Path, required=True)
    m.set_defaults(func=cmd_manifest)

    e = sub.add_parser("evaluate", help="Compare TLPP lag choices and occupancy transforms")
    add_data_args(e); add_trace_args(e); add_tlpp_args(e)
    e.add_argument("--variants", default="adaptive,fixed,multilag")
    e.add_argument("--probability-modes", default="raw,sqrt,log,log01")
    e.add_argument("--window-starts", default="0,0.5,1")
    e.add_argument("--noise-snr-db", type=float, default=20.0)
    e.add_argument("--output", type=Path, default=OUTPUT_ROOT / "runs" / "tlpp")
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("cache", help="Cache VAE inputs from the PEPL data")
    c.add_argument("--data-root", type=Path, default=DATA_ROOT)
    c.add_argument("--input-dir", type=Path)
    c.add_argument("--manifest", type=Path)
    c.add_argument("--splits", default="train,val_small")
    c.add_argument("--max-files", type=int, default=500)
    c.add_argument("--seed", type=int, default=1729)
    add_trace_args(c); add_tlpp_args(c)
    c.add_argument("--representation", choices=["adaptive", "fixed", "multilag_coords"], required=True)
    c.add_argument("--multilag-points", type=int, default=256)
    c.add_argument("--output", type=Path, required=True)
    c.set_defaults(func=cmd_cache)

    t = sub.add_parser("train", help="Train a TLPP variational autoencoder")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--probability-mode", choices=["raw", "sqrt", "log", "log01"], default="log01")
    t.add_argument("--loss", choices=["mse", "l1", "cosine", "hellinger"], default="mse")
    t.add_argument("--latent-dim", type=int, default=32)
    t.add_argument("--matryoshka-dims", default="", help="Example: 4,8,16,32. Empty disables nested-prefix training.")
    t.add_argument("--beta", type=float, default=1e-3, help="KL-divergence weight")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-5)
    t.add_argument("--grad-clip-norm", type=float, default=5.0)
    t.add_argument("--logvar-min", type=float, default=-10.0)
    t.add_argument("--logvar-max", type=float, default=8.0)
    t.add_argument("--kl-warmup-epochs", type=int, default=10)
    t.add_argument("--seed", type=int, default=1729)
    t.add_argument("--device", default="auto")
    t.add_argument("--train-split", default="train")
    t.add_argument("--val-split", default="val_small")
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("evaluate-vae", help="Evaluate a trained VAE and Matryoshka prefixes")
    add_data_args(v); add_trace_args(v)
    v.add_argument("--checkpoint", type=Path, required=True)
    v.add_argument("--embedding-dims", default="", help="Example: 4,8,16,32. Empty uses the full latent vector.")
    v.add_argument("--window-starts", default="0,0.5,1")
    v.add_argument("--noise-snr-db", type=float, default=20.0)
    v.add_argument("--device", default="auto")
    v.add_argument("--output", type=Path, required=True)
    v.set_defaults(func=cmd_evaluate_vae)

    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
