from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TLPPConfig:
    """Canonical TLPP definition used by the stored universal HDF5 files."""

    bins: int = 128
    current_min: float = -5.0
    current_max: float = 105.0
    fixed_lag_us: float = 10.0
    interpolation_factor: int = 16
    interpolation_method: str = "linear"
    steady_region_us: float = 1000.0
    window_us: float = 700.0
    window_start_fraction: float = 0.5
    probability_mode: str = "log01"
    log_epsilon: float = 1e-6

    def to_dict(self) -> dict:
        return asdict(self)


def _signal(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).ravel()


def interpolate_signal(
    signal,
    sampling_hz: float,
    factor: int = 1,
    method: str = "linear",
) -> tuple[np.ndarray, float]:
    """Densify a uniformly sampled signal without changing physical duration."""
    x = _signal(signal)
    factor = int(factor)
    if factor < 1:
        raise ValueError("interpolation factor must be >= 1")
    if method != "linear":
        raise ValueError("production TLPP pipeline supports linear interpolation only")
    if len(x) < 2 or factor == 1:
        return x.copy(), float(sampling_hz)

    fractions = np.arange(factor, dtype=np.float64) / float(factor)
    delta = x[1:] - x[:-1]
    y = (x[:-1, None] + delta[:, None] * fractions[None, :]).reshape(-1)
    y = np.concatenate((y, x[-1:]))
    return y, float(sampling_hz) * factor


def lag_us_to_samples(lag_us: float, sampling_hz: float) -> int:
    return max(1, int(round(float(lag_us) * 1e-6 * float(sampling_hz))))


def make_TLPP(
    signal,
    sampling_hz: float,
    *,
    lag_us: float | None = None,
    lag_samples: int | None = None,
    interpolation_factor: int = 1,
    interpolation_method: str = "linear",
) -> np.ndarray:
    """Return non-circular delay coordinates ``[I(t), I(t-tau)]``."""
    original_fs = float(sampling_hz)
    x, effective_fs = interpolate_signal(
        signal,
        original_fs,
        interpolation_factor,
        interpolation_method,
    )
    if lag_samples is not None:
        lag_seconds = int(lag_samples) / original_fs
        lag = max(1, int(round(lag_seconds * effective_fs)))
    else:
        lag = lag_us_to_samples(10.0 if lag_us is None else lag_us, effective_fs)
    if lag >= len(x):
        raise ValueError(f"lag ({lag}) must be smaller than signal length ({len(x)})")
    return np.column_stack((x[lag:], x[:-lag]))


def _uniform_histogram2d_counts(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    low: float,
    high: float,
    dtype,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have equal length")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("histogram inputs must be finite")

    bins = int(bins)
    scale = bins / (float(high) - float(low))
    ix = np.floor((x - low) * scale).astype(np.int64)
    iy = np.floor((y - low) * scale).astype(np.int64)
    np.clip(ix, 0, bins - 1, out=ix)
    np.clip(iy, 0, bins - 1, out=iy)
    image = np.bincount(ix * bins + iy, minlength=bins * bins).reshape(bins, bins)

    out_dtype = np.dtype(dtype)
    if out_dtype.kind != "u":
        raise ValueError("occupancy count dtype must be unsigned integer")
    if int(image.max(initial=0)) > np.iinfo(out_dtype).max:
        raise OverflowError(f"occupancy count exceeds {out_dtype} range")
    return image.astype(out_dtype, copy=False)


def make_occupancy_counts(
    points: np.ndarray,
    bins: int = 128,
    value_range: tuple[float, float] = (-5.0, 105.0),
    *,
    dtype=np.uint16,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    low, high = map(float, value_range)
    return _uniform_histogram2d_counts(points[:, 0], points[:, 1], bins, low, high, dtype)


def make_tlpp_counts(
    signal,
    sampling_hz: float,
    *,
    lag_us: float | None = None,
    lag_samples: int | None = None,
    bins: int = 128,
    value_range: tuple[float, float] = (-5.0, 105.0),
    interpolation_factor: int = 16,
    interpolation_method: str = "linear",
    dtype=np.uint16,
) -> np.ndarray:
    """Build the canonical unsmoothed integer TLPP map without an intermediate point array."""
    original_fs = float(sampling_hz)
    x, effective_fs = interpolate_signal(signal, original_fs, interpolation_factor, interpolation_method)
    if lag_samples is not None:
        lag_seconds = int(lag_samples) / original_fs
        lag = max(1, int(round(lag_seconds * effective_fs)))
    else:
        lag = lag_us_to_samples(10.0 if lag_us is None else lag_us, effective_fs)
    if lag >= len(x):
        raise ValueError(f"lag ({lag}) must be smaller than signal length ({len(x)})")
    low, high = map(float, value_range)
    return _uniform_histogram2d_counts(x[lag:], x[:-lag], bins, low, high, dtype)


def make_occupancy(
    points: np.ndarray,
    bins: int = 128,
    value_range: tuple[float, float] = (-5.0, 105.0),
    smoothing_sigma: float = 0.0,
) -> np.ndarray:
    if float(smoothing_sigma) != 0.0:
        raise ValueError("production TLPP pipeline does not smooth stored occupancy maps")
    counts = make_occupancy_counts(points, bins, value_range, dtype=np.uint32).astype(np.float64)
    total = counts.sum()
    return (counts / total if total > 0 else counts).astype(np.float32)


def transform_occupancy(
    probability: np.ndarray,
    mode: str = "log01",
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Transform a normalized TLPP probability map for the VAE input."""
    p = np.asarray(probability, dtype=np.float64)
    if mode == "raw":
        y = p
    elif mode == "sqrt":
        y = np.sqrt(np.clip(p, 0.0, None))
    elif mode in {"log", "log01"}:
        y = np.log1p(np.clip(p, 0.0, None) / epsilon)
        if mode == "log01":
            y /= np.log1p(1.0 / epsilon)
    else:
        raise ValueError("probability mode must be raw, sqrt, log, or log01")
    return y.astype(np.float32)
