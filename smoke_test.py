import numpy as np

from tlpp_embedding import (
    TLPPConfig,
    make_TLPP,
    make_multilag_TLPP,
    make_tlpp_occupancy,
    tlpp_from_signal,
    transform_occupancy,
)

fs = 1_000_000.0
t = np.arange(700) / fs
x = 40 + 8 * np.sin(2 * np.pi * 30_000 * t) + 2 * np.sin(2 * np.pi * 70_000 * t)

cfg = TLPPConfig(bins=32, probability_mode="log01")
assert make_TLPP(x, fs, lag_us=1).shape[1] == 2
assert make_multilag_TLPP(x, fs, (1, 2, 3, 5)).shape[1] == 5
assert make_tlpp_occupancy(x, fs, lag_us=1, bins=32).shape == (32, 32)
assert tlpp_from_signal(x, fs, "adaptive", cfg).shape == (32, 32)
assert tlpp_from_signal(x, fs, "multilag", cfg).shape[1:] == (32, 32)

p = np.array([0.0, 0.1, 1.0])
y = transform_occupancy(p, "log01", 1e-6)
assert np.all(y >= 0) and np.all(y <= 1.0 + 1e-7)
assert abs(float(y[-1]) - 1.0) < 1e-6
print("TLPP smoke test passed")
