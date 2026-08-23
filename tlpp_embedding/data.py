from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .core import dominant_frequency


SPLITS = {
    "train": "h9_batch3",
    "val_small": "h9_batch3_val_small",
}


@dataclass(frozen=True)
class TraceConfig:
    key: str = "time"
    time_row: int = 0
    current_row: int = 2
    steady_ms: float = 1.0
    window_us: float = 700.0
    window_start: float = 0.5
    fmin_hz: float = 5_000.0
    fmax_hz: float = 450_000.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Item:
    path: Path
    split: str
    uuid: str


@dataclass
class Trace:
    item: Item
    time_s: np.ndarray
    current_a: np.ndarray
    sampling_hz: float
    current_mean: float
    current_rms: float
    dominant_frequency_hz: float
    steady_time_s: np.ndarray
    steady_current_a: np.ndarray
    full_time_s: np.ndarray
    full_current_a: np.ndarray


def _data_dir(path: Path) -> Path:
    path = Path(path)
    return path / "data" if (path / "data").is_dir() else path


def split_dirs(root: Path) -> dict[str, Path]:
    root = Path(root)
    return {name: _data_dir(root / folder) for name, folder in SPLITS.items()}


def _sample(folder: Path, max_files: int | None, seed: int) -> list[Path]:
    paths = [
        Path(e.path)
        for e in os.scandir(_data_dir(folder))
        if e.is_file() and e.name.lower().endswith(".npz")
    ]
    paths.sort(key=lambda p: p.name)
    if max_files is not None and len(paths) > max_files:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(len(paths), max_files, replace=False))
        paths = [paths[int(i)] for i in ids]
    return paths


def select_items(
    data_root: Path,
    split: str = "val_small",
    max_files: int | None = 500,
    seed: int = 1729,
    input_dir: Path | None = None,
    manifest: Path | None = None,
) -> list[Item]:
    dirs = split_dirs(data_root)

    if manifest:
        rows = []
        with Path(manifest).open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row_split = (row.get("split") or "val_small").strip()
                if split != "all" and row_split != split:
                    continue
                uuid = (row.get("uuid") or "").strip()
                if uuid:
                    rows.append((uuid, row_split))
        if max_files is not None and len(rows) > max_files:
            rng = np.random.default_rng(seed)
            ids = np.sort(rng.choice(len(rows), max_files, replace=False))
            rows = [rows[int(i)] for i in ids]
        return [
            Item(
                (_data_dir(input_dir) if input_dir else dirs[row_split]) / f"{uuid}.npz",
                row_split,
                uuid,
            )
            for uuid, row_split in rows
        ]

    if input_dir:
        return [Item(p, split, p.stem) for p in _sample(input_dir, max_files, seed)]

    if split == "all":
        items = []
        for offset, (name, folder) in enumerate(dirs.items()):
            items += [
                Item(p, name, p.stem)
                for p in _sample(folder, max_files, seed + offset)
            ]
        return items

    return [
        Item(p, split, p.stem)
        for p in _sample(dirs[split], max_files, seed)
    ]


def write_manifest(
    data_root: Path,
    output: Path,
    max_files: int | None = None,
    seed: int = 1729,
) -> None:
    rows = []
    for offset, (name, folder) in enumerate(split_dirs(data_root).items()):
        rows += [
            (p.stem, name)
            for p in _sample(folder, max_files, seed + offset)
        ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "split"])
        w.writerows(rows)


def _load_arrays(path: Path, cfg: TraceConfig) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        a = np.asarray(z[cfg.key])
    if a.shape[0] > a.shape[1] and a.shape[1] >= 3:
        a = a.T
    t = np.asarray(a[cfg.time_row], dtype=float).ravel()
    x = np.asarray(a[cfg.current_row], dtype=float).ravel()
    keep = np.isfinite(t) & np.isfinite(x)
    t, x = t[keep], x[keep]
    order = np.argsort(t)
    t, x = t[order], x[order]
    t, unique = np.unique(t, return_index=True)
    return t, x[unique]


def _window(
    item: Item,
    full_t: np.ndarray,
    full_x: np.ndarray,
    steady_t: np.ndarray,
    steady_x: np.ndarray,
    fs: float,
    cfg: TraceConfig,
    start_fraction: float,
) -> Trace:
    n = int(round(cfg.window_us * 1e-6 * fs))
    start = int(round(start_fraction * max(len(steady_x) - n, 0)))
    x = steady_x[start : start + n].copy()
    t = steady_t[start : start + n].copy()
    mean = float(np.mean(x))
    centered = x - mean
    rms = float(np.sqrt(np.mean(centered**2)))
    return Trace(
        item=item,
        time_s=t,
        current_a=x,
        sampling_hz=fs,
        current_mean=mean,
        current_rms=rms,
        dominant_frequency_hz=dominant_frequency(
            centered, fs, cfg.fmin_hz, cfg.fmax_hz
        ),
        steady_time_s=steady_t,
        steady_current_a=steady_x,
        full_time_s=full_t,
        full_current_a=full_x,
    )


def load_trace(item: Item, cfg: TraceConfig) -> Trace:
    t, x = _load_arrays(item.path, cfg)
    fs = 1.0 / float(np.median(np.diff(t)))
    first = np.searchsorted(t, t[-1] - cfg.steady_ms * 1e-3, side="left")
    return _window(
        item, t, x, t[first:], x[first:], fs, cfg, cfg.window_start
    )


def alternate_windows(
    trace: Trace,
    cfg: TraceConfig,
    starts: tuple[float, ...],
) -> list[Trace]:
    return [
        _window(
            trace.item,
            trace.full_time_s,
            trace.full_current_a,
            trace.steady_time_s,
            trace.steady_current_a,
            trace.sampling_hz,
            cfg,
            start,
        )
        for start in starts
        if abs(start - cfg.window_start) > 1e-12
    ]


def _with_current(trace: Trace, current: np.ndarray, cfg: TraceConfig) -> Trace:
    x = np.asarray(current, dtype=float)
    mean = float(np.mean(x))
    centered = x - mean
    rms = float(np.sqrt(np.mean(centered**2)))
    return Trace(
        item=trace.item,
        time_s=trace.time_s,
        current_a=x,
        sampling_hz=trace.sampling_hz,
        current_mean=mean,
        current_rms=rms,
        dominant_frequency_hz=dominant_frequency(
            centered, trace.sampling_hz, cfg.fmin_hz, cfg.fmax_hz
        ),
        steady_time_s=trace.steady_time_s,
        steady_current_a=trace.steady_current_a,
        full_time_s=trace.full_time_s,
        full_current_a=trace.full_current_a,
    )


def phase_scramble(
    trace: Trace,
    cfg: TraceConfig,
    rng: np.random.Generator,
) -> Trace:
    centered = trace.current_a - trace.current_mean
    spectrum = np.fft.rfft(centered)
    phase = rng.uniform(-np.pi, np.pi, len(spectrum))
    phase[0] = 0.0
    if len(centered) % 2 == 0:
        phase[-1] = 0.0
    x = np.fft.irfft(
        np.abs(spectrum) * np.exp(1j * phase),
        n=len(centered),
    ) + trace.current_mean
    return _with_current(trace, x, cfg)


def add_noise(
    trace: Trace,
    cfg: TraceConfig,
    snr_db: float,
    rng: np.random.Generator,
) -> Trace:
    if trace.current_rms <= 1e-12:
        return _with_current(trace, trace.current_a.copy(), cfg)
    sigma = trace.current_rms / 10 ** (snr_db / 20.0)
    return _with_current(
        trace,
        trace.current_a + rng.normal(0.0, sigma, len(trace.current_a)),
        cfg,
    )
