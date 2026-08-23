# TLPP_embedding

A compact, library-first codebase for time-lagged phase portraits (TLPPs), TLPP occupancy representations, and TLPP variational autoencoders.

## Functionality of the TLPP embedding suite

This suit has the following TLPP functionality:

- Adaptive 2-D TLPP: lag = a configurable fraction of the dominant Fourier period.
- Fixed 2-D TLPP: one physical lag in microseconds for every signal.
- Multidimensional TLPP: `[I(t), I(t-tau1), I(t-tau2), ...]` with non-circular alignment.
- Pairwise occupancy maps for the multidimensional TLPP.
- Configurable grid size, current bounds, smoothing, and final-window selection.
- Occupancy transforms: `raw`, `sqrt`, `log`, and `log01`.
- Window-consistency, phase-scramble, noise, nearest-neighbor, effective-rank, and runtime evaluation.

## Log occupancy and log01

For one occupancy cell,

`p = number of TLPP points in that cell / total number of TLPP points`.

The log representation is:

`log(1 + p / epsilon)`

with default `epsilon = 1e-6`.

The new `log01` representation is:

`log(1 + p / epsilon) / log(1 + 1 / epsilon)`.

Because `0 <= p <= 1`, this maps the possible log values to `[0,1]` with one fixed global scaling. It does not independently min-max normalize every trace, so values remain comparable between traces.


## Installation as a library

From another project, install this folder in editable mode once:

```powershell
uv pip install -e D:\PEPL\diffusion_embeddings\TLPP_embedding
```

After that, `import tlpp_embedding` works from other scripts and projects while edits to this code remain immediately available.

## Library API

```python
from tlpp_embedding import (
    TLPPConfig,
    make_TLPP,
    make_multilag_TLPP,
    make_occupancy,
    make_tlpp_occupancy,
    tlpp_from_signal,
    load_vae,
    embed_signal,
)

# 2-D phase portrait points
points = make_TLPP(signal, sampling_hz=1_000_000, lag_us=2.0)

# Raw occupancy probability
p = make_occupancy(points, bins=64, value_range=(-5, 105))

# Direct TLPP occupancy representation
log01_map = make_tlpp_occupancy(
    signal,
    sampling_hz=1_000_000,
    lag_us=2.0,
    bins=64,
    value_range=(-5, 105),
    probability_mode="log01",
)

# Adaptive, fixed, or multidimensional pairwise TLPP maps
cfg = TLPPConfig(
    bins=64,
    fixed_lag_us=1.0,
    multi_lags_us=(1, 2, 3, 5),
    probability_mode="log01",
)
adaptive = tlpp_from_signal(signal, 1_000_000, "adaptive", cfg)
fixed = tlpp_from_signal(signal, 1_000_000, "fixed", cfg)
multilag_maps = tlpp_from_signal(signal, 1_000_000, "multilag", cfg)

# True N-D delay coordinates
multilag_coordinates = make_multilag_TLPP(
    signal, 1_000_000, (1, 2, 3, 5)
)

# Learned VAE embedding
vae = load_vae("best.pt")
z = embed_signal(signal, 1_000_000, vae)
z8 = embed_signal(signal, 1_000_000, vae, dimensions=8)
```

## Variational autoencoder

The VAE supports three input representations:

- `adaptive`: adaptive-lag 2-D occupancy map.
- `fixed`: fixed-lag 2-D occupancy map.
- `multilag_coords`: the true N-D delay-coordinate sequence directly, rather than a huge N-D histogram.

The occupancy-map cache stores raw probabilities. This lets the same cache be reused while training with `raw`, `sqrt`, `log`, or `log01` encoder inputs.

Available reconstruction losses:

- `mse`
- `l1`
- `cosine`
- `hellinger` for occupancy-map VAEs

For Hellinger training, the decoder output is normalized into a probability distribution and compared with the raw occupancy probabilities.

## Matryoshka option

Matryoshka mode is optional. Example:

`--latent-dim 32 --matryoshka-dims 4,8,16,32`

During training, the same sampled latent vector is decoded repeatedly using only its first 4, 8, 16, and 32 coordinates; later coordinates are zeroed. The reconstruction loss is averaged over those prefixes. This is an experimental VAE adaptation of the nested-prefix idea: it basically pressures early latent coordinates to remain useful on their own.

At inference, evaluating a prefix is simply `embedding[:k]`, and `evaluate-vae` can test several prefix sizes from one checkpoint.

## Scripts

- `run_tlpp.py`: manifests, classical TLPP evaluation, VAE cache creation, VAE training, VAE evaluation.
- `visualize_tlpp.py`: current window, adaptive/fixed TLPPs, raw occupancy, transformed occupancy, multidimensional TLPP plots, and optional VAE reconstruction.
- `smoke_test.py`: quick synthetic sanity check.

The PEPL defaults currently use `T:\h9_diffusion_model\h9_batch3\data` for training and `T:\h9_diffusion_model\h9_batch3_val_small\data` for validation. Cache creation prints selection and caching progress so network-drive reads do not look frozen.
