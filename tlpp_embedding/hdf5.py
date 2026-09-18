from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

UNIVERSAL_SCHEMA_VERSION = "tlpp-universal-v1"
CANONICAL_BINS = 128

REQUIRED_DATASETS = (
    "tlpp_counts",
    "trace_key",
    "source_filename",
    "trace_id",
    "trace_uuid",
    "global_index",
    "sampling_hz",
    "lag_samples",
    "window_samples",
    "point_count",
)


@dataclass(frozen=True)
class UniversalTLPPMetadata:
    path: Path
    count: int
    bins: int
    schema_version: str
    storage_variant: str
    compression: str | None
    chunks: tuple[int, ...] | None
    sampling_hz_nominal: float | None
    lookup_capacity: int


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_universal_tlpp(path: Path | str, *, require_finalized: bool = True) -> UniversalTLPPMetadata:
    """Validate the production universal-HDF5 contract without scanning TLPP payloads."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5:
        missing = [name for name in REQUIRED_DATASETS if name not in h5]
        if missing:
            raise ValueError(f"{path}: missing required datasets: {missing}")

        counts = h5["tlpp_counts"]
        if counts.ndim != 3 or tuple(counts.shape[1:]) != (CANONICAL_BINS, CANONICAL_BINS):
            raise ValueError(f"{path}: expected tlpp_counts shape (N,128,128), got {counts.shape}")
        if counts.dtype != np.dtype("uint16"):
            raise ValueError(f"{path}: tlpp_counts must be uint16, got {counts.dtype}")

        n = int(counts.shape[0])
        for name in REQUIRED_DATASETS[1:]:
            if len(h5[name]) != n:
                raise ValueError(f"{path}: {name} length {len(h5[name])} != {n}")

        if "source_index" not in h5 or h5["source_index"].id != h5["global_index"].id:
            raise ValueError(f"{path}: source_index must be a hard link to global_index")

        if "lookup" not in h5 or not isinstance(h5["lookup"], h5py.Group):
            raise ValueError(f"{path}: missing canonical /lookup group")
        lookup = h5["lookup"]
        if set(lookup.keys()) != {"hash", "row"}:
            raise ValueError(f"{path}: /lookup must contain exactly hash and row, got {list(lookup.keys())}")
        if lookup["hash"].dtype != np.dtype("uint64") or lookup["row"].dtype != np.dtype("uint64"):
            raise ValueError(f"{path}: /lookup hash/row must be uint64")
        if lookup["hash"].shape != lookup["row"].shape:
            raise ValueError(f"{path}: /lookup hash and row shapes differ")

        schema = str(_decode(h5.attrs.get("schema_version", "")))
        if schema and schema != UNIVERSAL_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported schema_version={schema!r}")
        if require_finalized and int(h5.attrs.get("finalized", 0)) != 1:
            raise ValueError(f"{path}: file is not finalized")
        bins = int(h5.attrs.get("bins", CANONICAL_BINS))
        if bins != CANONICAL_BINS:
            raise ValueError(f"{path}: expected bins=128, got {bins}")

        return UniversalTLPPMetadata(
            path=path,
            count=n,
            bins=bins,
            schema_version=schema or UNIVERSAL_SCHEMA_VERSION,
            storage_variant=str(_decode(h5.attrs.get("storage_variant", ""))),
            compression=counts.compression,
            chunks=counts.chunks,
            sampling_hz_nominal=float(h5.attrs["sampling_hz_nominal"]) if "sampling_hz_nominal" in h5.attrs else None,
            lookup_capacity=int(lookup["hash"].shape[0]),
        )


class UniversalTLPPReader:
    """Lazy row reader; safe to pickle because open HDF5 handles are dropped."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.metadata = validate_universal_tlpp(self.path)
        self._file: h5py.File | None = None

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return self.metadata.count

    def counts(self, index: int) -> np.ndarray:
        return np.asarray(self._ensure_open()["tlpp_counts"][int(index)])

    def scalar(self, name: str, index: int) -> Any:
        return _decode(self._ensure_open()[name][int(index)])

    def trace_metadata(self, index: int) -> dict[str, Any]:
        return {name: self.scalar(name, index) for name in REQUIRED_DATASETS if name != "tlpp_counts"}

    def close(self) -> None:
        if self._file is not None:
            self._file.close(); self._file = None

    def __enter__(self):
        self._ensure_open(); return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy(); state["_file"] = None; return state
