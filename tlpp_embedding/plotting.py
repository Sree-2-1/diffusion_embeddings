from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

import h5py
import numpy as np

from .training import PathInputs, normalize_h5_paths


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_evaluation_summary(evaluation_dir: Path | str, history_csv: Path | str | None = None) -> None:
    evaluation_dir = Path(evaluation_dir)
    plot_dir = evaluation_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()

    prefix_rows = _read_csv(evaluation_dir / "prefix_metrics.csv")
    dims = [int(row["embedding_dim"]) for row in prefix_rows]
    means = [float(row["mean_hellinger_squared"]) for row in prefix_rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(dims, means, marker="o")
    ax.set_xlabel("Latent prefix dimension")
    ax.set_ylabel("Mean squared Hellinger reconstruction error")
    ax.set_title("Matryoshka reconstruction")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(plot_dir / "prefix_reconstruction.png", dpi=200); plt.close(fig)

    latent_rows = _read_csv(evaluation_dir / "latent_dimensions.csv")
    dimensions = np.array([int(row["dimension"]) for row in latent_rows])
    latent_std = np.array([float(row["mu_std"]) for row in latent_rows])
    latent_kl = np.array([float(row["mean_kl"]) for row in latent_rows])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(dimensions, latent_std)
    axes[0].set_ylabel("Std of latent mean"); axes[0].set_title("Latent utilization"); axes[0].grid(True, alpha=0.2)
    axes[1].plot(dimensions, latent_kl)
    axes[1].set_xlabel("Latent dimension"); axes[1].set_ylabel("Mean KL contribution"); axes[1].grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(plot_dir / "latent_activity.png", dpi=200); plt.close(fig)

    per_trace = _read_csv(evaluation_dir / "per_trace_errors.csv")
    full_key = f"h2_{max(dims)}d"
    full_errors = np.array([float(row[full_key]) for row in per_trace])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.hist(full_errors, bins=60)
    ax.set_xlabel(f"{max(dims)}-D squared Hellinger reconstruction error")
    ax.set_ylabel("Trace count"); ax.set_title("Reconstruction-error distribution")
    fig.tight_layout(); fig.savefig(plot_dir / "reconstruction_error_distribution.png", dpi=200); plt.close(fig)

    if history_csv is not None and Path(history_csv).exists():
        history = _read_csv(Path(history_csv))
        epochs = [int(row["epoch"]) for row in history]
        train_total = [float(row["train_loss"]) for row in history]
        val_total = [float(row["val_loss"]) for row in history]
        train_recon = [float(row["train_reconstruction"]) for row in history]
        val_recon = [float(row["val_reconstruction"]) for row in history]
        train_kl = [float(row["train_kl"]) for row in history]
        val_kl = [float(row["val_kl"]) for row in history]
        beta = [float(row["effective_beta"]) for row in history]

        fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
        axes[0].plot(epochs, train_total, label="train")
        axes[0].plot(epochs, val_total, label="val_small")
        axes[0].set_ylabel("Total VAE objective"); axes[0].legend(); axes[0].grid(True, alpha=0.25)
        axes[1].plot(epochs, train_recon, label="train")
        axes[1].plot(epochs, val_recon, label="val_small")
        axes[1].set_ylabel("Reconstruction H²"); axes[1].legend(); axes[1].grid(True, alpha=0.25)
        axes[2].plot(epochs, train_kl, label="train")
        axes[2].plot(epochs, val_kl, label="val_small")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("KL divergence"); axes[2].legend(); axes[2].grid(True, alpha=0.25)
        fig.suptitle("Training and validation history")
        fig.tight_layout(); fig.savefig(plot_dir / "training_history.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, beta)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Effective beta"); ax.set_title("KL warmup"); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(plot_dir / "kl_warmup.png", dpi=200); plt.close(fig)


def _decode_scalar(dataset, index: int):
    value = dataset[index]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_name(value) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text[:80] or "trace"


def _save_map(array, path: Path, title: str, colorbar_label: str) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    image = ax.imshow(array, origin="lower", aspect="equal")
    ax.set_xlabel("I(t - 10 µs) bin"); ax.set_ylabel("I(t) bin"); ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_trace_gallery(
    checkpoint: Path | str,
    h5_paths: PathInputs,
    evaluation_dir: Path | str,
    output: Path | str,
    *,
    trace_count: int = 50,
    embedding_dims: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
    device: str = "auto",
) -> list[dict]:
    """Create detailed plots for traces spanning the combined error distribution."""
    import torch

    from .core import TLPPConfig, transform_occupancy
    from .models import counts_to_probability, load_vae_checkpoint

    checkpoint = Path(checkpoint)
    paths = normalize_h5_paths(h5_paths)
    evaluation_dir = Path(evaluation_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    state, cfg, model, dev = load_vae_checkpoint(checkpoint, device)
    dims = tuple(sorted({int(d) for d in embedding_dims if 0 < int(d) <= cfg.latent_dim}))
    if cfg.latent_dim not in dims:
        dims = (*dims, cfg.latent_dim)

    error_rows = _read_csv(evaluation_dir / "per_trace_errors.csv")
    full_key = f"h2_{cfg.latent_dim}d"
    error_rows.sort(key=lambda row: float(row[full_key]))
    count = min(int(trace_count), len(error_rows))
    positions = np.linspace(0, len(error_rows) - 1, count).round().astype(int)
    selected = [error_rows[int(position)] for position in positions]
    log_epsilon = TLPPConfig().log_epsilon
    plt = _matplotlib()
    index_rows: list[dict] = []
    html_sections: list[str] = []

    handles = [h5py.File(path, "r") for path in paths]
    try:
        with torch.no_grad():
            for gallery_number, (position, selected_row) in enumerate(zip(positions, selected), start=1):
                component = int(selected_row["component"])
                local_row = int(selected_row["local_row"])
                h5 = handles[component]
                path = paths[component]

                counts = np.asarray(h5["tlpp_counts"][local_row], dtype=np.float32)
                probability = counts_to_probability(counts[None, None])
                model_input = transform_occupancy(probability, cfg.probability_mode, log_epsilon)
                x = torch.from_numpy(model_input).to(dev)
                target = torch.from_numpy(probability).to(dev)
                mu, raw_logvar = model.stats(x)
                logvar = torch.clamp(raw_logvar, cfg.logvar_min, cfg.logvar_max)
                latent = mu[0].detach().cpu().numpy()

                p = target.flatten(2)
                sqrt_p = torch.sqrt((p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_min(0.0))
                recon_probability: dict[int, np.ndarray] = {}
                recon_display: dict[int, np.ndarray] = {}
                h2_errors: dict[int, float] = {}

                for width in dims:
                    z = torch.zeros_like(mu); z[:, :width] = mu[:, :width]
                    logits = model.decode(z)
                    q = torch.softmax(logits.flatten(2), dim=-1).reshape(logits.shape)
                    h2 = (1.0 - (sqrt_p * torch.sqrt(q.flatten(2).clamp_min(0.0))).sum(dim=-1)).clamp_min(0.0).mean()
                    h2_errors[width] = float(h2.detach().cpu())
                    q_np = q.detach().cpu().numpy()
                    recon_probability[width] = q_np[0, 0]
                    recon_display[width] = transform_occupancy(q_np, cfg.probability_mode, log_epsilon)[0, 0]

                trace_key = _decode_scalar(h5["trace_key"], local_row)
                trace_uuid = _decode_scalar(h5["trace_uuid"], local_row)
                global_index = int(_decode_scalar(h5["global_index"], local_row))
                percentile = 100.0 * float(position) / max(len(error_rows) - 1, 1)
                folder_name = f"trace_{gallery_number:02d}_{_safe_name(path.stem)}_row_{local_row:07d}_{_safe_name(trace_uuid)}"
                folder = output / folder_name
                folder.mkdir(parents=True, exist_ok=True)

                metadata = {
                    "gallery_number": gallery_number,
                    "dataset": str(path),
                    "local_row": local_row,
                    "global_index": global_index,
                    "trace_key": trace_key,
                    "trace_uuid": trace_uuid,
                    "source_filename": _decode_scalar(h5["source_filename"], local_row),
                    "sampling_hz": float(_decode_scalar(h5["sampling_hz"], local_row)),
                    "selection_percentile": percentile,
                    "checkpoint_epoch": int(state.get("epoch", -1)),
                    "prefix_h2": {str(width): h2_errors[width] for width in dims},
                    "latent_mean": latent.tolist(),
                    "mean_posterior_logvar": float(logvar.mean().detach().cpu()),
                }
                (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                _save_map(counts, folder / "01_raw_counts.png", "Raw TLPP occupancy counts", "Occupancy count")
                _save_map(probability[0, 0], folder / "02_normalized_probability.png", "Normalized TLPP probability", "Probability")
                _save_map(model_input[0, 0], folder / "03_normalized_log01.png", "Exact VAE input (log01)", "log01 intensity")

                titles = ["Target", *[f"{width}D" for width in dims]]
                prob_arrays = [probability[0, 0], *[recon_probability[width] for width in dims]]
                display_arrays = [model_input[0, 0], *[recon_display[width] for width in dims]]
                vmax = max(float(array.max()) for array in prob_arrays)
                display_min = min(float(array.min()) for array in display_arrays)
                display_max = max(float(array.max()) for array in display_arrays)

                fig, axes = plt.subplots(1, len(titles), figsize=(3.0 * len(titles), 3.5), squeeze=False)
                for ax, array, title in zip(axes[0], prob_arrays, titles):
                    image = ax.imshow(array, origin="lower", aspect="equal", vmin=0.0, vmax=vmax)
                    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
                fig.suptitle(f"Probability-space Matryoshka reconstructions — trace {gallery_number}")
                fig.colorbar(image, ax=list(axes[0]), label="Probability", shrink=0.8)
                fig.subplots_adjust(left=0.02, right=0.96, bottom=0.04, top=0.84, wspace=0.08)
                fig.savefig(folder / "04_reconstructions_probability.png", dpi=200); plt.close(fig)

                fig, axes = plt.subplots(1, len(titles), figsize=(3.0 * len(titles), 3.5), squeeze=False)
                for ax, array, title in zip(axes[0], display_arrays, titles):
                    image = ax.imshow(array, origin="lower", aspect="equal", vmin=display_min, vmax=display_max)
                    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
                fig.suptitle(f"log01 Matryoshka reconstructions — trace {gallery_number}")
                fig.colorbar(image, ax=list(axes[0]), label="log01 intensity", shrink=0.8)
                fig.subplots_adjust(left=0.02, right=0.96, bottom=0.04, top=0.84, wspace=0.08)
                fig.savefig(folder / "05_reconstructions_log01.png", dpi=200); plt.close(fig)

                fig, axes = plt.subplots(2, len(titles), figsize=(3.0 * len(titles), 6.4), squeeze=False)
                for column, title in enumerate(titles):
                    axes[0, column].imshow(prob_arrays[column], origin="lower", aspect="equal", vmin=0.0, vmax=vmax)
                    axes[1, column].imshow(display_arrays[column], origin="lower", aspect="equal", vmin=display_min, vmax=display_max)
                    axes[0, column].set_title(title)
                    for r in (0, 1):
                        axes[r, column].set_xticks([]); axes[r, column].set_yticks([])
                axes[0, 0].set_ylabel("Probability"); axes[1, 0].set_ylabel("log01")
                fig.suptitle(f"TLPP target and Matryoshka reconstructions — trace {gallery_number} ({percentile:.1f}th error percentile)")
                fig.subplots_adjust(left=0.04, right=0.99, bottom=0.04, top=0.88, wspace=0.07, hspace=0.12)
                fig.savefig(folder / "06_reconstruction_comparison.png", dpi=200); plt.close(fig)

                fig, ax = plt.subplots(figsize=(13, 5))
                coordinates = np.arange(1, cfg.latent_dim + 1)
                ax.plot(coordinates, latent)
                for boundary in dims[:-1]:
                    ax.axvline(boundary + 0.5, linestyle="--", linewidth=0.8)
                ax.set_xlim(1, cfg.latent_dim); ax.set_xlabel("Latent coordinate"); ax.set_ylabel("Latent mean (µ)")
                ax.set_title(f"{cfg.latent_dim}-D latent vector — trace {gallery_number}"); ax.grid(True, alpha=0.2)
                fig.tight_layout(); fig.savefig(folder / "07_latent_vector.png", dpi=200); plt.close(fig)

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(list(dims), [h2_errors[width] for width in dims], marker="o")
                ax.set_xlabel("Latent prefix dimension"); ax.set_ylabel("Squared Hellinger error")
                ax.set_title(f"Reconstruction error by prefix — trace {gallery_number}"); ax.grid(True, alpha=0.25)
                fig.tight_layout(); fig.savefig(folder / "08_prefix_error_curve.png", dpi=200); plt.close(fig)

                index_record = {
                    "gallery_number": gallery_number,
                    "folder": folder_name,
                    "dataset": path.name,
                    "local_row": local_row,
                    "global_index": global_index,
                    "trace_key": trace_key,
                    "selection_percentile": percentile,
                    **{f"h2_{width}d": h2_errors[width] for width in dims},
                }
                index_rows.append(index_record)
                escaped_folder = html.escape(folder_name)
                html_sections.append(f"""
<section><h2>Trace {gallery_number:02d} — {html.escape(path.name)} row {local_row}</h2>
<p>Error percentile: {percentile:.1f}% &nbsp; | &nbsp; {cfg.latent_dim}-D H²: {h2_errors[cfg.latent_dim]:.6f}<br>
Trace key: <code>{html.escape(str(trace_key))}</code></p>
<a href="{escaped_folder}/06_reconstruction_comparison.png"><img src="{escaped_folder}/06_reconstruction_comparison.png"></a>
<p><a href="{escaped_folder}/01_raw_counts.png">raw counts</a> | <a href="{escaped_folder}/02_normalized_probability.png">probability</a> | <a href="{escaped_folder}/03_normalized_log01.png">log01</a> | <a href="{escaped_folder}/04_reconstructions_probability.png">probability reconstructions</a> | <a href="{escaped_folder}/05_reconstructions_log01.png">log01 reconstructions</a> | <a href="{escaped_folder}/07_latent_vector.png">latent</a> | <a href="{escaped_folder}/08_prefix_error_curve.png">prefix errors</a> | <a href="{escaped_folder}/metadata.json">metadata</a></p></section><hr>
""")
                print(f"[{gallery_number:02d}/{count:02d}] {path.name} row={local_row} H2={h2_errors[cfg.latent_dim]:.6f}", flush=True)
    finally:
        for handle in handles:
            handle.close()

    fields = [
        "gallery_number", "folder", "dataset", "local_row", "global_index", "trace_key",
        "selection_percentile", *[f"h2_{width}d" for width in dims],
    ]
    with (output / "trace_gallery_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(index_rows)

    (output / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Trace TLPP plots</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;line-height:1.35}}section{{margin-bottom:32px}}img{{max-width:100%;border:1px solid #bbb}}code{{background:#f2f2f2;padding:2px 4px}}</style></head>
<body><h1>Trace TLPP plots</h1><p>{count} traces sampled across the combined held-out reconstruction-error distribution.</p>{''.join(html_sections)}</body></html>""", encoding="utf-8")
    print(f"Trace TLPP gallery complete: {output}")
    return index_rows


def build_dashboard(run_dir: Path | str, split_names: tuple[str, ...] = ("val_small", "val_large")) -> Path:
    run_dir = Path(run_dir)
    evaluation_root = run_dir / "evaluation"
    dashboard = evaluation_root / "dashboard"
    dashboard.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()

    available = [name for name in split_names if (evaluation_root / name / "metrics.json").exists()]
    if not available:
        raise FileNotFoundError("No evaluation split metrics found")
    metrics = {name: json.loads((evaluation_root / name / "metrics.json").read_text(encoding="utf-8")) for name in available}
    prefix = {name: _read_csv(evaluation_root / name / "prefix_metrics.csv") for name in available}
    latent = {name: _read_csv(evaluation_root / name / "latent_dimensions.csv") for name in available}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in available:
        ax.plot([int(row["embedding_dim"]) for row in prefix[name]], [float(row["mean_hellinger_squared"]) for row in prefix[name]], marker="o", label=name)
    ax.set_xlabel("Latent prefix dimension"); ax.set_ylabel("Mean squared Hellinger reconstruction error")
    ax.set_title("Matryoshka reconstruction"); ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(dashboard / "combined_prefix_reconstruction.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name in available:
        dimensions = [int(row["dimension"]) for row in latent[name]]
        axes[0].plot(dimensions, [float(row["mu_std"]) for row in latent[name]], label=name)
        axes[1].plot(dimensions, [float(row["mean_kl"]) for row in latent[name]], label=name)
    axes[0].set_ylabel("Std of latent mean"); axes[0].set_title("Latent activity"); axes[0].legend(); axes[0].grid(True, alpha=0.2)
    axes[1].set_xlabel("Latent dimension"); axes[1].set_ylabel("Mean KL contribution"); axes[1].legend(); axes[1].grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(dashboard / "combined_latent_activity.png", dpi=200); plt.close(fig)

    def summary_table(name: str) -> str:
        m = metrics[name]; s = m["latent_statistics"]
        rows = [
            ("checkpoint epoch", m["checkpoint_epoch"]), ("count", m["count"]),
            ("component counts", m.get("component_counts", [])), ("latent dim", m["latent_dim"]),
            ("effective rank", f"{s['effective_rank']:.3f}"),
            ("effective-rank fraction", f"{s['effective_rank_fraction']:.3f}"),
            ("mean latent std", f"{s['mean_latent_std']:.6f}"),
            ("active dims KL > 0.001", s["active_dims_kl_gt_0.001"]),
            ("active dims KL > 0.01", s["active_dims_kl_gt_0.01"]),
            ("active dims KL > 0.1", s["active_dims_kl_gt_0.1"]),
        ]
        return "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)

    split_sections = []
    for name in available:
        split_sections.append(f"""
<h2>{html.escape(name)}</h2><table>{summary_table(name)}</table>
<div class="grid"><img src="../{name}/plots/prefix_reconstruction.png"><img src="../{name}/plots/latent_activity.png"><img src="../{name}/plots/reconstruction_error_distribution.png"><img src="../{name}/plots/training_history.png"></div>
""")
    gallery_path = dashboard / "trace TLPP plots" / "index.html"
    gallery_link = '<h2>Trace TLPP gallery</h2><p><a href="trace%20TLPP%20plots/index.html">Open raw / normalized / log01 / Matryoshka reconstruction gallery</a></p>' if gallery_path.exists() else ""

    index = dashboard / "index.html"
    index.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>TLPP VAE evaluation dashboard</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;line-height:1.35}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:6px 8px;text-align:left}}th{{background:#eee}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}img{{max-width:100%;border:1px solid #ccc}}</style></head>
<body><h1>TLPP VAE evaluation dashboard</h1><h2>Combined Matryoshka performance</h2><img src="combined_prefix_reconstruction.png"><h2>Combined latent activity</h2><img src="combined_latent_activity.png">{''.join(split_sections)}{gallery_link}</body></html>""", encoding="utf-8")
    print(f"Dashboard complete: {index}")
    return index
