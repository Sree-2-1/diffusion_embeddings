from __future__ import annotations

import argparse
from pathlib import Path

from .training import default_hdf5_vae_config, preflight_hdf5_vae, train_hdf5_vae

SUBSCALE_ROOT = Path("/nfs/turbo/coe-marksta/h9_subscale/2026_06_subscale_data_TLPPs")
FULLSCALE_ROOT = Path("/nfs/turbo/coe-marksta/h9_fullscale")
DEFAULT_RUN_DIR = FULLSCALE_ROOT / "vae_runs" / "combined_subscale_fullscale_128x128_latent128_run1"


def combined_paths() -> dict[str, list[Path]]:
    return {
        "train": [
            SUBSCALE_ROOT / "h9_batch3_train_TLPPs_compressed.h5",
            FULLSCALE_ROOT / "h9_train_TLPPs_compressed.h5",
        ],
        "val_small": [
            SUBSCALE_ROOT / "h9_batch3_val_small_TLPPs_compressed.h5",
            FULLSCALE_ROOT / "h9_val_small_TLPPs_compressed.h5",
        ],
        "val_large": [
            SUBSCALE_ROOT / "h9_batch3_val_large_TLPPs_compressed.h5",
            FULLSCALE_ROOT / "h9_val_large_TLPPs_compressed.h5",
        ],
    }


def _print_preflight(result: dict, cfg) -> None:
    print("input_grid: 128x128")
    print("latent_dim:", cfg.latent_dim)
    print("matryoshka_dims:", cfg.matryoshka_dims)
    print("epochs:", cfg.epochs)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("PREFLIGHT OK")


def command_preflight(args) -> None:
    cfg = default_hdf5_vae_config(args.epochs)
    _print_preflight(preflight_hdf5_vae(args.train, args.val, cfg), cfg)


def command_train(args) -> None:
    cfg = default_hdf5_vae_config(args.epochs)
    train_hdf5_vae(
        args.train,
        args.val,
        args.run_dir,
        cfg=cfg,
        target_epochs=args.epochs,
        num_workers=args.num_workers,
        max_runtime_minutes=args.max_runtime_minutes,
        device=args.device,
    )


def command_preflight_combined(args) -> None:
    paths = combined_paths()
    cfg = default_hdf5_vae_config(args.epochs)
    print("=== COMBINED SUBSCALE + FULLSCALE PREFLIGHT ===")
    _print_preflight(preflight_hdf5_vae(paths["train"], paths["val_small"], cfg), cfg)
    print("held_out_paths:", [str(path) for path in paths["val_large"]])


def command_train_combined(args) -> None:
    paths = combined_paths()
    cfg = default_hdf5_vae_config(args.epochs)
    train_hdf5_vae(
        paths["train"],
        paths["val_small"],
        args.run_dir,
        cfg=cfg,
        target_epochs=args.epochs,
        num_workers=args.num_workers,
        max_runtime_minutes=args.max_runtime_minutes,
        device=args.device,
    )


def command_evaluate(args) -> None:
    from .evaluate import evaluate_vae_hdf5
    evaluate_vae_hdf5(
        args.checkpoint,
        args.data,
        args.output,
        embedding_dims=tuple(args.dims),
        batch_size=args.batch_size,
        device=args.device,
        make_plots=True,
    )


def _evaluate_run(checkpoint: Path, run_dir: Path, val_small, val_large, args) -> None:
    from .evaluate import evaluate_vae_hdf5
    from .plotting import build_dashboard, plot_trace_gallery

    evaluation_root = run_dir / "evaluation"
    dims = tuple(args.dims)
    evaluate_vae_hdf5(
        checkpoint, val_small, evaluation_root / "val_small",
        embedding_dims=dims, batch_size=args.batch_size, device=args.device, make_plots=True,
    )
    evaluate_vae_hdf5(
        checkpoint, val_large, evaluation_root / "val_large",
        embedding_dims=dims, batch_size=args.batch_size, device=args.device, make_plots=True,
    )
    if args.gallery_count > 0:
        plot_trace_gallery(
            checkpoint,
            val_large,
            evaluation_root / "val_large",
            evaluation_root / "dashboard" / "trace TLPP plots",
            trace_count=args.gallery_count,
            embedding_dims=dims,
            device=args.device,
        )
    build_dashboard(run_dir)


def command_evaluate_combined(args) -> None:
    paths = combined_paths()
    checkpoint = args.checkpoint or (args.run_dir / "best.pt")
    _evaluate_run(checkpoint, args.run_dir, paths["val_small"], paths["val_large"], args)


def command_dashboard(args) -> None:
    from .plotting import build_dashboard
    build_dashboard(args.run_dir)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tlpp_embedding.cli",
        description="TLPP universal-HDF5 VAE workflow",
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("preflight", help="Check arbitrary train/validation HDF5 lists")
    q.add_argument("--train", nargs="+", type=Path, required=True)
    q.add_argument("--val", nargs="+", type=Path, required=True)
    q.add_argument("--epochs", type=int, default=50)
    q.set_defaults(func=command_preflight)

    q = sub.add_parser("train", help="Train/resume on arbitrary HDF5 lists")
    q.add_argument("--train", nargs="+", type=Path, required=True)
    q.add_argument("--val", nargs="+", type=Path, required=True)
    q.add_argument("--run-dir", type=Path, required=True)
    q.add_argument("--epochs", type=int, default=50)
    q.add_argument("--num-workers", type=int, default=6)
    q.add_argument("--max-runtime-minutes", type=float, default=100.0)
    q.add_argument("--device", default="auto")
    q.set_defaults(func=command_train)

    q = sub.add_parser("preflight-combined", help="Preflight the project subscale+fullscale split")
    q.add_argument("--epochs", type=int, default=50)
    q.set_defaults(func=command_preflight_combined)

    q = sub.add_parser("train-combined", help="Train/resume the project subscale+fullscale model")
    q.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    q.add_argument("--epochs", type=int, default=50)
    q.add_argument("--num-workers", type=int, default=6)
    q.add_argument("--max-runtime-minutes", type=float, default=100.0)
    q.add_argument("--device", default="auto")
    q.set_defaults(func=command_train_combined)

    q = sub.add_parser("evaluate", help="Evaluate one checkpoint over one or more HDF5 files")
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--data", nargs="+", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    q.add_argument("--batch-size", type=int, default=64)
    q.add_argument("--device", default="auto")
    q.set_defaults(func=command_evaluate)

    q = sub.add_parser("evaluate-combined", help="Evaluate val_small + held-out val_large and build plots/dashboard")
    q.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    q.add_argument("--checkpoint", type=Path)
    q.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    q.add_argument("--batch-size", type=int, default=64)
    q.add_argument("--gallery-count", type=int, default=50)
    q.add_argument("--device", default="auto")
    q.set_defaults(func=command_evaluate_combined)

    q = sub.add_parser("dashboard", help="Rebuild dashboard from existing evaluation outputs")
    q.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    q.set_defaults(func=command_dashboard)

    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
