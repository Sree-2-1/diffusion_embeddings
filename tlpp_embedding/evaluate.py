from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .core import TLPPConfig, tlpp_from_signal
from .data import TraceConfig, add_noise, alternate_windows, load_trace, phase_scramble
from .models import TLPPVAE


def _vector(trace, cfg: TLPPConfig, variant: str) -> np.ndarray:
    return tlpp_from_signal(
        trace.current_a,
        trace.sampling_hz,
        variant,
        cfg,
    ).ravel().astype(np.float64)


def _symmetric_l2(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na + nb <= 1e-12 else float(np.linalg.norm(a - b) / (na + nb))


def _pairwise_distance(matrix: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if mode in {"log", "log01"}:
        y = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return np.clip(1.0 - y @ y.T, 0.0, 2.0)

    norms = np.sum(x * x, axis=1)
    d2 = norms[:, None] + norms[None, :] - 2.0 * (x @ x.T)
    d = np.sqrt(np.clip(d2, 0.0, None))
    return d / np.sqrt(2.0) if mode == "sqrt" else d


def _waveform_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 and nb <= 1e-12:
        return 1.0
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _neighbor_score(
    vectors: np.ndarray,
    waveforms: list[np.ndarray],
    mode: str,
    seed: int,
) -> tuple[float, float]:
    distance = _pairwise_distance(vectors, mode)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    rng = np.random.default_rng(seed)
    near, random = [], []
    for i, j in enumerate(nearest):
        j = int(j)
        near.append(_waveform_corr(waveforms[i], waveforms[j]))
        candidates = [k for k in range(len(waveforms)) if k not in {i, j}]
        if candidates:
            random.append(_waveform_corr(waveforms[i], waveforms[int(rng.choice(candidates))]))
    return float(np.mean(near)), float(np.mean(random)) if random else 0.0


def _effective_rank(matrix: np.ndarray) -> tuple[float, float]:
    x = np.asarray(matrix, dtype=np.float64)
    active = np.std(x, axis=0) > 1e-12
    if not np.any(active) or len(x) < 2:
        return 0.0, 0.0
    x = x[:, active]
    x = (x - x.mean(0)) / x.std(0)
    eig = np.clip(np.linalg.eigvalsh(x @ x.T), 0.0, None)
    p = eig / max(eig.sum(), 1e-20)
    p = p[p > 1e-15]
    rank = float(np.exp(-np.sum(p * np.log(p))))
    maximum = min(len(x) - 1, x.shape[1])
    return rank, rank / maximum if maximum else 0.0


def evaluate_tlpp(
    items,
    output: Path,
    trace_cfg: TraceConfig,
    base_cfg: TLPPConfig,
    variants: tuple[str, ...] = ("adaptive", "fixed", "multilag"),
    probability_modes: tuple[str, ...] = ("raw", "sqrt", "log", "log01"),
    window_starts: tuple[float, ...] = (0.0, 0.5, 1.0),
    noise_snr_db: float = 20.0,
    seed: int = 1729,
) -> list[dict]:
    traces, failures = [], []
    for item in items:
        try:
            traces.append(load_trace(item, trace_cfg))
        except Exception as exc:
            failures.append((item.uuid, str(exc)))

    rows = []
    for variant in variants:
        for probability_mode in probability_modes:
            cfg = TLPPConfig(**{**base_cfg.to_dict(), "probability_mode": probability_mode})
            rng = np.random.default_rng(seed)
            vectors, waveforms, runtimes = [], [], []
            window_error, scramble_error, noise_error = [], [], []

            for trace in traces:
                start = time.perf_counter()
                reference = _vector(trace, cfg, variant)
                runtimes.append((time.perf_counter() - start) * 1000.0)
                vectors.append(reference)
                waveforms.append(trace.current_a)

                alternate = alternate_windows(trace, trace_cfg, window_starts)
                if alternate:
                    window_error.append(np.mean([
                        _symmetric_l2(reference, _vector(other, cfg, variant))
                        for other in alternate
                    ]))

                scrambled = phase_scramble(trace, trace_cfg, rng)
                noisy = add_noise(trace, trace_cfg, noise_snr_db, rng)
                scramble_error.append(_symmetric_l2(reference, _vector(scrambled, cfg, variant)))
                noise_error.append(_symmetric_l2(reference, _vector(noisy, cfg, variant)))

            matrix = np.asarray(vectors)
            near, random = _neighbor_score(matrix, waveforms, probability_mode, seed)
            rank, rank_fraction = _effective_rank(matrix)
            rows.append({
                "variant": variant,
                "probability_mode": probability_mode,
                "embedding_dim": int(matrix.shape[1]),
                "processed": len(traces),
                "failed": len(failures),
                "window_consistency_error": float(np.mean(window_error)) if window_error else 0.0,
                "scramble_error": float(np.mean(scramble_error)),
                "noise_error": float(np.mean(noise_error)),
                "nearest_waveform_corr": near,
                "random_waveform_corr": random,
                "effective_rank": rank,
                "effective_rank_fraction": rank_fraction,
                "milliseconds_per_trace": float(np.mean(runtimes)),
            })

    _write_results(
        output,
        rows,
        failures,
        {"trace": trace_cfg.to_dict(), "tlpp": base_cfg.to_dict()},
    )
    return rows


def evaluate_vae(
    items,
    checkpoint: Path,
    output: Path,
    trace_cfg: TraceConfig,
    embedding_dims: tuple[int, ...] = (),
    window_starts: tuple[float, ...] = (0.0, 0.5, 1.0),
    noise_snr_db: float = 20.0,
    seed: int = 1729,
    device: str = "auto",
) -> list[dict]:
    model = TLPPVAE(checkpoint, device)
    traces = [load_trace(item, trace_cfg) for item in items]
    dims = embedding_dims or (model.cfg.latent_dim,)
    rows = []

    for width in dims:
        rng = np.random.default_rng(seed)
        vectors, waveforms, runtimes = [], [], []
        window_error, scramble_error, noise_error = [], [], []

        for trace in traces:
            start = time.perf_counter()
            reference = model.embed_signal(trace.current_a, trace.sampling_hz, width)
            runtimes.append((time.perf_counter() - start) * 1000.0)
            vectors.append(reference)
            waveforms.append(trace.current_a)

            alternate = alternate_windows(trace, trace_cfg, window_starts)
            if alternate:
                window_error.append(np.mean([
                    _symmetric_l2(
                        reference,
                        model.embed_signal(other.current_a, other.sampling_hz, width),
                    )
                    for other in alternate
                ]))

            scrambled = phase_scramble(trace, trace_cfg, rng)
            noisy = add_noise(trace, trace_cfg, noise_snr_db, rng)
            scramble_error.append(_symmetric_l2(
                reference,
                model.embed_signal(scrambled.current_a, scrambled.sampling_hz, width),
            ))
            noise_error.append(_symmetric_l2(
                reference,
                model.embed_signal(noisy.current_a, noisy.sampling_hz, width),
            ))

        matrix = np.asarray(vectors)
        std = matrix.std(0)
        standardized = (matrix - matrix.mean(0)) / np.where(std > 1e-12, std, 1.0)
        near, random = _neighbor_score(standardized, waveforms, "raw", seed)
        rank, rank_fraction = _effective_rank(matrix)
        rows.append({
            "variant": f"vae_{model.cfg.representation}",
            "probability_mode": model.cfg.probability_mode,
            "loss": model.cfg.reconstruction_loss,
            "embedding_dim": int(width),
            "processed": len(traces),
            "failed": 0,
            "window_consistency_error": float(np.mean(window_error)) if window_error else 0.0,
            "scramble_error": float(np.mean(scramble_error)),
            "noise_error": float(np.mean(noise_error)),
            "nearest_waveform_corr": near,
            "random_waveform_corr": random,
            "effective_rank": rank,
            "effective_rank_fraction": rank_fraction,
            "milliseconds_per_trace": float(np.mean(runtimes)),
        })

    _write_results(
        output,
        rows,
        [],
        {"checkpoint": str(checkpoint), "trace": trace_cfg.to_dict()},
    )
    return rows


def _write_results(output: Path, rows: list[dict], failures: list, config: dict) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if failures:
        with (output / "failures.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uuid", "error"])
            w.writerows(failures)

    for row in rows:
        print("=" * 72)
        print(f"{row['variant']} / {row['probability_mode']}")
        print("=" * 72)
        for key, value in row.items():
            print(f"{key}: {value}")
