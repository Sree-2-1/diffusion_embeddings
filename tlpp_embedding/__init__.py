"""Library-first tools for time-lagged phase portrait (TLPP) embeddings."""

from .core import (
    TLPPConfig,
    adaptive_lag_samples,
    dominant_frequency,
    lag_us_to_samples,
    make_TLPP,
    make_multilag_TLPP,
    make_occupancy,
    make_tlpp_occupancy,
    pairwise_multilag_occupancies,
    tlpp_from_signal,
    tlpp_probability_from_signal,
    transform_occupancy,
)
from .models import TLPPVAE, VAEConfig, embed_signal, load_vae

__all__ = [
    "TLPPConfig",
    "VAEConfig",
    "TLPPVAE",
    "load_vae",
    "embed_signal",
    "dominant_frequency",
    "lag_us_to_samples",
    "adaptive_lag_samples",
    "make_TLPP",
    "make_multilag_TLPP",
    "make_occupancy",
    "make_tlpp_occupancy",
    "pairwise_multilag_occupancies",
    "transform_occupancy",
    "tlpp_probability_from_signal",
    "tlpp_from_signal",
]
