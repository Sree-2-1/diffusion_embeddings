from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TLPPConfig:
    bins: int = 64
    current_min: float = -5.0
    current_max: float = 105.0
    smoothing_sigma: float = 0.75
    probability_mode: str = "log01"   # raw, sqrt, log, log01
    log_epsilon: float = 1e-6

    adaptive_period_fraction: float = 0.25
    fixed_lag_us: float = 1.0
    multi_lags_us: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

    fmin_hz: float = 5_000.0
    fmax_hz: float = 450_000.0

    def to_dict(self) -> dict:
        return asdict(self)


def _signal(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def dominant_frequency(
    signal,
    sampling_hz: float,
    fmin_hz: float = 5_000.0,
    fmax_hz: float = 450_000.0,
) -> float:
    x = _signal(signal)
    x = x - np.mean(x)
    if np.linalg.norm(x) <= 1e-12:
        return 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), 1.0 / sampling_hz)
    valid = (freq >= fmin_hz) & (
        freq <= min(fmax_hz, 0.95 * sampling_hz / 2.0)
    )
    ids = np.flatnonzero(valid)
    return float(freq[ids[np.argmax(power[ids])]]) if len(ids) else 0.0


def lag_us_to_samples(lag_us: float, sampling_hz: float) -> int:
    return max(1, int(round(float(lag_us) * 1e-6 * float(sampling_hz))))


def adaptive_lag_samples(signal, sampling_hz: float, cfg: TLPPConfig) -> int:
    fdom = dominant_frequency(signal, sampling_hz, cfg.fmin_hz, cfg.fmax_hz)
    if fdom <= 0:
        return 1
    return max(
        1,
        int(round(cfg.adaptive_period_fraction * sampling_hz / fdom)),
    )


def make_TLPP(
    signal,
    sampling_hz: float,
    *,
    lag_us: float | None = None,
    lag_samples: int | None = None,
) -> np.ndarray:
    """Return non-circular 2-D points [x(t), x(t-tau)]."""
    x = _signal(signal)
    lag = int(lag_samples) if lag_samples is not None else lag_us_to_samples(lag_us or 1.0, sampling_hz)
    return np.column_stack((x[lag:], x[:-lag]))


def make_multilag_TLPP(
    signal,
    sampling_hz: float,
    lags_us: tuple[float, ...] | list[float],
) -> np.ndarray:
    """Return N-D delay coordinates [x(t), x(t-tau1), x(t-tau2), ...]."""
    x = _signal(signal)
    lags = tuple(dict.fromkeys(lag_us_to_samples(v, sampling_hz) for v in lags_us))
    maximum = max(lags)
    columns = [x[maximum:]]
    columns += [x[maximum - lag : len(x) - lag] for lag in lags]
    return np.column_stack(columns).astype(np.float32)


def make_occupancy(
    points: np.ndarray,
    bins: int = 64,
    value_range: tuple[float, float] = (-5.0, 105.0),
    smoothing_sigma: float = 0.75,
) -> np.ndarray:
    """Convert 2-D TLPP points into a probability map whose cells sum to 1."""
    points = np.asarray(points, dtype=np.float64)
    low, high = map(float, value_range)
    edges = np.linspace(low, high, int(bins) + 1)
    image, _, _ = np.histogram2d(
        np.clip(points[:, 0], low, high),
        np.clip(points[:, 1], low, high),
        bins=(edges, edges),
    )
    if smoothing_sigma > 0:
        image = ndimage.gaussian_filter(image, smoothing_sigma, mode="constant")
    total = image.sum()
    return (image / total if total > 0 else image).astype(np.float32)


def transform_occupancy(
    probability: np.ndarray,
    mode: str = "log01",
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    raw   = p
    sqrt  = sqrt(p)
    log   = log(1 + p/epsilon)
    log01 = log(1 + p/epsilon) / log(1 + 1/epsilon)

    Because occupancy p is in [0,1], log01 uses one fixed mapping to [0,1]
    rather than per-trace min-max scaling.
    """
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


def make_tlpp_occupancy(
    signal,
    sampling_hz: float,
    *,
    lag_us: float | None = None,
    lag_samples: int | None = None,
    bins: int = 64,
    value_range: tuple[float, float] = (-5.0, 105.0),
    smoothing_sigma: float = 0.75,
    probability_mode: str = "raw",
    log_epsilon: float = 1e-6,
) -> np.ndarray:
    points = make_TLPP(
        signal,
        sampling_hz,
        lag_us=lag_us,
        lag_samples=lag_samples,
    )
    p = make_occupancy(points, bins, value_range, smoothing_sigma)
    return transform_occupancy(p, probability_mode, log_epsilon)


def pairwise_multilag_occupancies(
    coordinates: np.ndarray,
    cfg: TLPPConfig,
    *,
    transform: bool = True,
) -> np.ndarray:
    maps = []
    for i, j in combinations(range(coordinates.shape[1]), 2):
        p = make_occupancy(
            coordinates[:, (i, j)],
            cfg.bins,
            (cfg.current_min, cfg.current_max),
            cfg.smoothing_sigma,
        )
        maps.append(
            transform_occupancy(p, cfg.probability_mode, cfg.log_epsilon)
            if transform else p
        )
    return np.stack(maps)


def tlpp_probability_from_signal(
    signal,
    sampling_hz: float,
    mode: str,
    cfg: TLPPConfig,
) -> np.ndarray:
    """Return raw occupancy probability map(s), channel-first."""
    if mode == "adaptive":
        lag = adaptive_lag_samples(signal, sampling_hz, cfg)
        p = make_tlpp_occupancy(
            signal,
            sampling_hz,
            lag_samples=lag,
            bins=cfg.bins,
            value_range=(cfg.current_min, cfg.current_max),
            smoothing_sigma=cfg.smoothing_sigma,
            probability_mode="raw",
        )
        return p[None]

    if mode == "fixed":
        p = make_tlpp_occupancy(
            signal,
            sampling_hz,
            lag_us=cfg.fixed_lag_us,
            bins=cfg.bins,
            value_range=(cfg.current_min, cfg.current_max),
            smoothing_sigma=cfg.smoothing_sigma,
            probability_mode="raw",
        )
        return p[None]

    if mode == "multilag":
        coordinates = make_multilag_TLPP(signal, sampling_hz, cfg.multi_lags_us)
        return pairwise_multilag_occupancies(coordinates, cfg, transform=False)

    raise ValueError("mode must be adaptive, fixed, or multilag")


def tlpp_from_signal(
    signal,
    sampling_hz: float,
    mode: str,
    cfg: TLPPConfig,
) -> np.ndarray:
    p = tlpp_probability_from_signal(signal, sampling_hz, mode, cfg)
    y = transform_occupancy(p, cfg.probability_mode, cfg.log_epsilon)
    return y[0] if mode in {"adaptive", "fixed"} else y
