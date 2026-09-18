from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = "tlpp-universal-v1"
SOFTWARE_VERSION = "2026-09-18-universal-hdf5-v2-advisor"

BINS = 128
CURRENT_MIN_A = -5.0
CURRENT_MAX_A = 105.0
LAG_US = 10.0
INTERPOLATION_FACTOR = 16
INTERPOLATION_METHOD = "linear"
STEADY_REGION_US = 1000.0
WINDOW_US = 700.0
WINDOW_START_FRACTION = 0.5

STR_DTYPE = h5py.string_dtype(encoding="utf-8")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
UUID_HASH_TABLE_GROUP = "lookup"
UUID_HASH_TABLE_VERSION = "uuid-open-addressing-linear-v1"
UUID_HASH_TABLE_EMPTY_ROW = np.iinfo(np.uint64).max
UUID_HASH_TABLE_DEFAULT_MAX_LOAD = 0.70

def decode_text(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)

def hash64(text: str) -> np.uint64:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))

def text_array(values):
    return np.asarray([decode_text(v) for v in values], dtype=object)

def split_source_paths(source_path: Path, dataset_name: str | None = None,
                       work_dir: Path | None = None, output_dir: Path | None = None):
    source_path = Path(source_path)
    dataset_name = dataset_name or source_path.stem
    work_dir = Path(work_dir) if work_dir is not None else source_path.parent / f".{dataset_name}_TLPP_work"
    output_dir = Path(output_dir) if output_dir is not None else source_path.parent
    return source_path, dataset_name, work_dir, output_dir

def time_source_layout(src, source_path: Path):
    """Validate a waveform source and locate required channels by name.

    Source files may differ in sample count, sampling rate, and auxiliary channels.
    The TLPP builder only requires a rank-3 ``/time`` dataset plus named
    ``time_s`` and ``discharge_current_A`` channels.
    """
    if "time" not in src:
        raise RuntimeError(f"{source_path} has no /time dataset")
    time_ds = src["time"]
    if time_ds.ndim != 3:
        raise RuntimeError(f"unexpected /time rank in {source_path}: {time_ds.shape}")
    if int(time_ds.shape[1]) < 2:
        raise RuntimeError(f"/time has fewer than 2 samples in {source_path}: {time_ds.shape}")
    if "time_names" not in src:
        raise RuntimeError(f"{source_path} has no /time_names dataset")

    names = [decode_text(v).strip() for v in src["time_names"][:]]
    if len(names) != int(time_ds.shape[2]):
        raise RuntimeError(
            f"/time_names length {len(names)} does not match /time channels "
            f"{time_ds.shape[2]} in {source_path}"
        )
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate /time_names in {source_path}: {names}")

    required = ("time_s", "discharge_current_A")
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"missing required time channels in {source_path}: {missing}; names={names}")

    n = int(time_ds.shape[0])
    if n <= 0:
        raise RuntimeError(f"empty /time dataset in {source_path}")

    if "UUID" in src:
        uuid_ds = src["UUID"]
        if uuid_ds.ndim != 1 or len(uuid_ds) != n:
            raise RuntimeError(f"/UUID is not row-aligned in {source_path}: {uuid_ds.shape} vs {n}")
        for i in sorted(set((0, n // 2, n - 1))):
            value = decode_text(uuid_ds[i]).strip()
            if not UUID_RE.fullmatch(value):
                raise RuntimeError(f"invalid UUID at row {i} in {source_path}: {value!r}")

    return {
        "count": n,
        "sample_count": int(time_ds.shape[1]),
        "channel_count": int(time_ds.shape[2]),
        "time_names": names,
        "time_index": names.index("time_s"),
        "current_index": names.index("discharge_current_A"),
    }

def validate_time_source(src, source_path: Path):
    return int(time_source_layout(src, source_path)["count"])

def set_common_attrs(f, *, dataset_name: str, source_path: str, storage_variant: str):
    f.attrs["schema_version"] = SCHEMA_VERSION
    f.attrs["software_version"] = SOFTWARE_VERSION
    f.attrs["dataset_name"] = dataset_name
    f.attrs["source_path"] = source_path
    f.attrs["storage_variant"] = storage_variant
    f.attrs["record_axis"] = np.int32(0)

    f.attrs["tlpp_definition"] = "non-circular [I(t), I(t-tau)] raw occupancy counts"
    f.attrs["bins"] = np.int32(BINS)
    f.attrs["current_min_A"] = np.float64(CURRENT_MIN_A)
    f.attrs["current_max_A"] = np.float64(CURRENT_MAX_A)
    f.attrs["out_of_range_behavior"] = "clip_to_edge_bins"
    f.attrs["lag_us"] = np.float64(LAG_US)
    f.attrs["interpolation_factor"] = np.int32(INTERPOLATION_FACTOR)
    f.attrs["interpolation_method"] = INTERPOLATION_METHOD
    f.attrs["steady_region_us"] = np.float64(STEADY_REGION_US)
    f.attrs["window_us"] = np.float64(WINDOW_US)
    f.attrs["window_start_fraction"] = np.float64(WINDOW_START_FRACTION)
    f.attrs["smoothing"] = "none"
    f.attrs["normalization"] = "none"
    f.attrs["counts_dtype"] = "uint16"
    f.attrs["lookup_hash"] = "blake2b-64-little-endian"

def create_universal_file(path: Path, n: int, *, dataset_name: str, source_path: str,
                          compressed: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    storage_variant = "compressed_lzf" if compressed else "uncompressed_contiguous"
    set_common_attrs(
        f,
        dataset_name=dataset_name,
        source_path=source_path,
        storage_variant=storage_variant,
    )
    f.attrs["record_count"] = np.uint64(n)

    if compressed:
        f.create_dataset(
            "tlpp_counts",
            shape=(n, BINS, BINS),
            dtype=np.uint16,
            chunks=(1, BINS, BINS),
            compression="lzf",
            shuffle=True,
            fletcher32=True,
        )
    else:
        f.create_dataset(
            "tlpp_counts",
            shape=(n, BINS, BINS),
            dtype=np.uint16,
        )

    for name in ("trace_key", "source_filename", "trace_id", "trace_uuid"):
        f.create_dataset(name, shape=(n,), dtype=STR_DTYPE)

    f.create_dataset("global_index", shape=(n,), dtype=np.uint64)
    f["source_index"] = f["global_index"]  # hard link; zero extra storage
    f.create_dataset("sampling_hz", shape=(n,), dtype=np.float64)
    f.create_dataset("lag_samples", shape=(n,), dtype=np.uint32)
    f.create_dataset("window_samples", shape=(n,), dtype=np.uint32)
    f.create_dataset("point_count", shape=(n,), dtype=np.uint32)

    return f

def _make_uuid_lookup_arrays(f, uuid_hashes: np.ndarray | None = None,
                             max_load: float = UUID_HASH_TABLE_DEFAULT_MAX_LOAD):
    n = int(len(f["trace_uuid"]))
    if n <= 0:
        raise RuntimeError("cannot build UUID lookup for an empty file")

    if uuid_hashes is None:
        uuid_hashes = np.empty(n, dtype=np.uint64)
        ds = f["trace_uuid"]
        block = 50_000
        for start in range(0, n, block):
            stop = min(start + block, n)
            values = [decode_text(v).strip() for v in ds[start:stop]]
            if any(not v for v in values):
                raise RuntimeError(
                    "canonical universal UUID lookup requires a non-empty trace_uuid for every row"
                )
            uuid_hashes[start:stop] = np.fromiter(
                (hash64(v) for v in values),
                dtype=np.uint64,
                count=stop - start,
            )
    else:
        uuid_hashes = np.asarray(uuid_hashes, dtype=np.uint64)
        if uuid_hashes.shape != (n,):
            raise RuntimeError(f"UUID hash array shape mismatch: {uuid_hashes.shape} vs {(n,)}")

    capacity = uuid_hash_table_capacity(n, max_load)
    mask = capacity - 1
    empty = int(UUID_HASH_TABLE_EMPTY_ROW)
    table_hash = np.zeros(capacity, dtype=np.uint64)
    table_row = np.full(capacity, np.uint64(UUID_HASH_TABLE_EMPTY_ROW), dtype=np.uint64)
    uuid_ds = f["trace_uuid"]

    total_displacements = 0
    max_displacements = 0
    for row in range(n):
        h = int(uuid_hashes[row])
        slot = h & mask
        displaced = 0
        while int(table_row[slot]) != empty:
            existing_row = int(table_row[slot])
            if int(table_hash[slot]) == h:
                existing_uuid = decode_text(uuid_ds[existing_row]).strip()
                incoming_uuid = decode_text(uuid_ds[row]).strip()
                if existing_uuid == incoming_uuid:
                    raise RuntimeError(
                        f"duplicate UUID {incoming_uuid!r} at rows {existing_row} and {row}"
                    )
            displaced += 1
            slot = (slot + 1) & mask
            if displaced >= capacity:
                raise RuntimeError("UUID lookup table unexpectedly full")
        table_hash[slot] = np.uint64(h)
        table_row[slot] = np.uint64(row)
        total_displacements += displaced
        max_displacements = max(max_displacements, displaced)

    stats = {
        "count": n,
        "capacity": capacity,
        "load_factor": float(n / capacity),
        "mean_displacements_on_build": float(total_displacements / n),
        "max_displacements_on_build": int(max_displacements),
    }
    return table_hash, table_row, stats

def _write_uuid_lookup_group(f, group_name: str, table_hash, table_row, stats,
                             max_load: float = UUID_HASH_TABLE_DEFAULT_MAX_LOAD):
    if group_name in f:
        raise RuntimeError(f"/{group_name} already exists")
    g = f.create_group(group_name)
    g.create_dataset("hash", data=np.asarray(table_hash, dtype=np.uint64), dtype=np.uint64)
    g.create_dataset("row", data=np.asarray(table_row, dtype=np.uint64), dtype=np.uint64)
    g.attrs["version"] = UUID_HASH_TABLE_VERSION
    g.attrs["field"] = "trace_uuid"
    g.attrs["hash_algorithm"] = "blake2b-64-little-endian"
    g.attrs["probe_strategy"] = "linear"
    g.attrs["count"] = np.uint64(stats["count"])
    g.attrs["capacity"] = np.uint64(stats["capacity"])
    g.attrs["load_factor"] = float(stats["load_factor"])
    g.attrs["configured_max_load"] = float(max_load)
    g.attrs["empty_row_sentinel"] = np.uint64(UUID_HASH_TABLE_EMPTY_ROW)
    g.attrs["mean_displacements_on_build"] = float(stats["mean_displacements_on_build"])
    g.attrs["max_displacements_on_build"] = np.uint64(stats["max_displacements_on_build"])

def create_lookup_group(f, uuid_hashes: np.ndarray | None = None,
                        max_load: float = UUID_HASH_TABLE_DEFAULT_MAX_LOAD,
                        group_name: str = UUID_HASH_TABLE_GROUP):
    """Create the canonical persistent UUID -> row lookup table.

    The universal format indexes only ``trace_uuid``. Provenance fields remain
    stored datasets but are not duplicated into secondary indexes.
    """
    table_hash, table_row, stats = _make_uuid_lookup_arrays(f, uuid_hashes, max_load)
    _write_uuid_lookup_group(f, group_name, table_hash, table_row, stats, max_load)
    f.attrs["lookup_field"] = "trace_uuid"
    f.attrs["lookup_method"] = UUID_HASH_TABLE_VERSION
    f.attrs["lookup_hash"] = "blake2b-64-little-endian"

def copy_root_attrs_preserving(src, dst):
    for k, v in src.attrs.items():
        if k in dst.attrs:
            dst.attrs[f"source_attr__{k}"] = v
        else:
            dst.attrs[k] = v

def write_metadata_block(outputs, start: int, stop: int, meta):
    for f in outputs:
        for name, values in meta.items():
            f[name][start:stop] = values

def update_uuid_hash_array(uuid_hashes, start, stop, meta):
    values = [str(v).strip() for v in meta["trace_uuid"]]
    if any(not v for v in values):
        raise RuntimeError(
            "canonical universal UUID lookup requires a non-empty trace_uuid for every row"
        )
    uuid_hashes[start:stop] = np.fromiter(
        (hash64(v) for v in values),
        dtype=np.uint64,
        count=stop - start,
    )

def finalize_common_attrs(outputs):
    for f in outputs:
        sampling = f["sampling_hz"][:]
        finite = sampling[np.isfinite(sampling)]
        if len(finite):
            first = finite[0]
            uniform = bool(np.allclose(finite, first, rtol=1e-10, atol=1e-8))
            f.attrs["sampling_hz_uniform"] = np.uint8(uniform)
            f.attrs["sampling_hz_nominal"] = np.float64(first if uniform else np.median(finite))
        else:
            f.attrs["sampling_hz_uniform"] = np.uint8(0)
            f.attrs["sampling_hz_nominal"] = np.float64(np.nan)

def select_window(time_s: np.ndarray, current_a: np.ndarray):
    t = np.asarray(time_s, dtype=np.float64)
    x = np.asarray(current_a, dtype=np.float64)

    if t.ndim != 1 or x.ndim != 1 or len(t) != len(x):
        raise ValueError("time/current must be 1-D and equal length")
    if len(t) < 2:
        raise ValueError("trace has fewer than 2 samples")

    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("time grid is not strictly increasing")

    # Use the full trace span to estimate the measured sampling rate.
    # This is much less sensitive than median(diff(t)) to float32
    # quantization of the stored timestamps.
    dt_med = float(np.median(dt))
    sampling_hz = float((len(t) - 1) / (t[-1] - t[0]))

    steady_s = STEADY_REGION_US * 1e-6
    window_s = WINDOW_US * 1e-6
    if window_s > steady_s:
        raise ValueError("window exceeds steady region")

    steady_start = float(t[-1]) - steady_s
    window_start = steady_start + WINDOW_START_FRACTION * (steady_s - window_s)
    window_stop = window_start + window_s

    tol = 0.25 * dt_med
    a = int(np.searchsorted(t, window_start - tol, side="left"))
    b = int(np.searchsorted(t, window_stop + tol, side="right"))

    signal = x[a:b]
    if len(signal) < 2:
        raise ValueError("selected TLPP window has fewer than 2 samples")

    return signal, sampling_hz

def tlpp_counts_from_signal(signal: np.ndarray, sampling_hz: float):
    # Equivalent to the validated production definition:
    # 16x linear densification, physical 10 us lag, clipped 128x128 raw counts.
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("signal too short")

    new_n = (len(x) - 1) * INTERPOLATION_FACTOR + 1
    old_idx = np.arange(len(x), dtype=np.float64)
    new_idx = np.linspace(0.0, float(len(x) - 1), new_n)
    dense = np.interp(new_idx, old_idx, x)

    effective_fs = float(sampling_hz) * INTERPOLATION_FACTOR
    lag_dense = max(1, int(round(LAG_US * 1e-6 * effective_fs)))
    if lag_dense >= len(dense):
        raise ValueError("lag is longer than selected signal")

    now = dense[lag_dense:]
    delayed = dense[:-lag_dense]

    hi_clip = np.nextafter(np.float64(CURRENT_MAX_A), np.float64(CURRENT_MIN_A))
    now = np.clip(now, CURRENT_MIN_A, hi_clip)
    delayed = np.clip(delayed, CURRENT_MIN_A, hi_clip)

    scale = BINS / (CURRENT_MAX_A - CURRENT_MIN_A)
    ix = np.floor((now - CURRENT_MIN_A) * scale).astype(np.int64)
    iy = np.floor((delayed - CURRENT_MIN_A) * scale).astype(np.int64)

    flat = ix * BINS + iy
    counts = np.bincount(flat, minlength=BINS * BINS).reshape(BINS, BINS)

    if counts.max(initial=0) > np.iinfo(np.uint16).max:
        raise OverflowError("TLPP bin count exceeds uint16")

    return counts.astype(np.uint16), int(len(now))

def create_worker_shard(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.create_dataset(
        "tlpp_counts", shape=(n, BINS, BINS), dtype=np.uint16,
        chunks=(1, BINS, BINS), compression="lzf",
        shuffle=True, fletcher32=True,
    )
    for name in ("trace_key", "source_filename", "trace_id", "trace_uuid"):
        f.create_dataset(name, shape=(n,), dtype=STR_DTYPE)
    f.create_dataset("global_index", shape=(n,), dtype=np.uint64)
    f.create_dataset("sampling_hz", shape=(n,), dtype=np.float64)
    f.create_dataset("lag_samples", shape=(n,), dtype=np.uint32)
    f.create_dataset("window_samples", shape=(n,), dtype=np.uint32)
    f.create_dataset("point_count", shape=(n,), dtype=np.uint32)
    return f

def build_source_shard(source_path: Path, dataset_name: str | None, work_dir: Path | None,
                       shard_index: int, shard_size: int = 5000, read_block: int = 16):
    source_path, dataset_name, work, _ = split_source_paths(
        source_path, dataset_name, work_dir, None
    )
    shard_dir = work / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as src:
        layout = time_source_layout(src, source_path)
        n_total = int(layout["count"])
        time_index = int(layout["time_index"])
        current_index = int(layout["current_index"])
        start = shard_index * shard_size
        stop = min(start + shard_size, n_total)
        if start >= n_total:
            raise ValueError(f"shard {shard_index} starts beyond dataset")

        final = shard_dir / f"shard_{shard_index:06d}.h5"
        if final.exists():
            with h5py.File(final, "r") as existing:
                if (
                    int(existing.attrs.get("start", -1)) == start
                    and int(existing.attrs.get("stop", -1)) == stop
                    and existing.attrs.get("software_version", "") == SOFTWARE_VERSION
                    and decode_text(existing.attrs.get("source_path", "")) == str(source_path)
                    and decode_text(existing.attrs.get("dataset_name", "")) == dataset_name
                ):
                    print(f"SKIP existing valid shard {final}")
                    return
            raise RuntimeError(f"Incompatible existing shard: {final}")

        tmp = final.with_name(f".{final.name}.partial.{os.getpid()}")
        if tmp.exists():
            tmp.unlink()

        out = create_worker_shard(tmp, stop - start)
        out.attrs["software_version"] = SOFTWARE_VERSION
        out.attrs["schema_version"] = SCHEMA_VERSION
        out.attrs["dataset_name"] = dataset_name
        out.attrs["start"] = np.uint64(start)
        out.attrs["stop"] = np.uint64(stop)
        out.attrs["source_path"] = str(source_path)

        if "UUID" not in src:
            raise RuntimeError(f"{source_path}: /UUID is required by the production source contract")

        try:
            local = 0
            for b0 in range(start, stop, read_block):
                b1 = min(b0 + read_block, stop)
                time_block = np.asarray(src["time"][b0:b1], dtype=np.float32)
                identity_block = src["UUID"][b0:b1]

                for j in range(b1 - b0):
                    global_i = b0 + j
                    trace = time_block[j]
                    signal, sampling_hz = select_window(trace[:, time_index], trace[:, current_index])
                    counts, point_count = tlpp_counts_from_signal(signal, sampling_hz)

                    trace_uuid = decode_text(identity_block[j]).strip()
                    if not UUID_RE.fullmatch(trace_uuid):
                        raise RuntimeError(f"invalid UUID at row {global_i}: {trace_uuid!r}")
                    source_filename = source_path.name
                    trace_id = trace_uuid
                    key = f"{source_filename}/{trace_uuid}"

                    out["tlpp_counts"][local] = counts
                    out["trace_key"][local] = key
                    out["source_filename"][local] = source_filename
                    out["trace_id"][local] = trace_id
                    out["trace_uuid"][local] = trace_uuid
                    out["global_index"][local] = global_i
                    out["sampling_hz"][local] = sampling_hz
                    out["lag_samples"][local] = max(
                        1, int(round(LAG_US * 1e-6 * sampling_hz))
                    )
                    out["window_samples"][local] = len(signal)
                    out["point_count"][local] = point_count
                    local += 1

                if local == (b1 - start) or local % 500 < read_block:
                    print(
                        f"{dataset_name} shard {shard_index:06d}: "
                        f"{local:,}/{stop-start:,}",
                        flush=True,
                    )

            out.flush()
        except Exception:
            out.close()
            if tmp.exists():
                tmp.unlink()
            raise
        else:
            out.close()
            os.replace(tmp, final)
            print(f"COMPLETE {final}")

def source_shard_count(source_path: Path, shard_size: int = 5000):
    source_path = Path(source_path)
    with h5py.File(source_path, "r") as src:
        n = validate_time_source(src, source_path)
    return math.ceil(n / shard_size)

def finalize_source(source_path: Path, dataset_name: str | None = None,
                    work_dir: Path | None = None, output_dir: Path | None = None,
                    shard_size: int = 5000):
    source_path, dataset_name, work, output_dir = split_source_paths(
        source_path, dataset_name, work_dir, output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as src:
        source_layout = time_source_layout(src, source_path)
        n = int(source_layout["count"])
        if "UUID" not in src:
            raise RuntimeError(f"{source_path}: /UUID is required by the production source contract")

    out_plain = output_dir / f"{dataset_name}_TLPPs.h5"
    out_comp = output_dir / f"{dataset_name}_TLPPs_compressed.h5"
    for p in (out_plain, out_comp):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {p}")

    tmp_plain = out_plain.with_name(f".{out_plain.name}.partial.{os.getpid()}")
    tmp_comp = out_comp.with_name(f".{out_comp.name}.partial.{os.getpid()}")

    shard_count = math.ceil(n / shard_size)
    shard_paths = [
        work / "shards" / f"shard_{i:06d}.h5"
        for i in range(shard_count)
    ]
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} shards; first missing: {missing[0]}")

    plain = create_universal_file(
        tmp_plain, n, dataset_name=dataset_name, source_path=str(source_path), compressed=False
    )
    comp = create_universal_file(
        tmp_comp, n, dataset_name=dataset_name, source_path=str(source_path), compressed=True
    )
    outputs = (plain, comp)

    with h5py.File(source_path, "r") as src:
        for f in outputs:
            copy_root_attrs_preserving(src, f)
            f.attrs["source_kind"] = "hdf5_time_series_container"
            f.attrs["source_identity_dataset"] = "/UUID"
            f.attrs["source_time_sample_count"] = np.uint32(source_layout["sample_count"])
            f.attrs["source_time_channel_count"] = np.uint32(source_layout["channel_count"])
            f.attrs["source_time_names"] = ",".join(source_layout["time_names"])
            f.attrs["source_time_index"] = np.uint32(source_layout["time_index"])
            f.attrs["source_current_index"] = np.uint32(source_layout["current_index"])

    uuid_hashes = np.empty(n, dtype=np.uint64)

    try:
        written = 0
        for shard_i, shard_path in enumerate(shard_paths):
            with h5py.File(shard_path, "r") as sh:
                start = int(sh.attrs["start"])
                stop = int(sh.attrs["stop"])
                expected_start = shard_i * shard_size
                expected_stop = min(expected_start + shard_size, n)
                if (start, stop) != (expected_start, expected_stop):
                    raise RuntimeError(
                        f"Shard range mismatch for {shard_path}: "
                        f"{start}:{stop} vs {expected_start}:{expected_stop}"
                    )
                if decode_text(sh.attrs.get("source_path", "")) != str(source_path):
                    raise RuntimeError(f"Shard source mismatch: {shard_path}")

                counts = np.asarray(sh["tlpp_counts"][:], dtype=np.uint16)
                plain["tlpp_counts"][start:stop] = counts
                comp["tlpp_counts"][start:stop] = counts

                meta = {
                    name: (
                        text_array(sh[name][:])
                        if name in ("trace_key", "source_filename", "trace_id", "trace_uuid")
                        else np.asarray(sh[name][:])
                    )
                    for name in (
                        "trace_key", "source_filename", "trace_id", "trace_uuid",
                        "global_index", "sampling_hz", "lag_samples",
                        "window_samples", "point_count"
                    )
                }
                write_metadata_block(outputs, start, stop, meta)
                update_uuid_hash_array(uuid_hashes, start, stop, meta)
                written += stop - start

            print(
                f"merge {dataset_name}: shard {shard_i+1:,}/{shard_count:,}; "
                f"{written:,}/{n:,} rows",
                flush=True,
            )

        for f in outputs:
            create_lookup_group(f, uuid_hashes)
        finalize_common_attrs(outputs)

        for f in outputs:
            # ``finalize_common_attrs`` derives the nominal source rate from the
            # measured per-trace sampling frequencies. This keeps 1 MHz subscale
            # and 500 kHz fullscale sources in one pipeline without hard-coding.
            f.attrs["sampling_hz_nominal_basis"] = "measured_from_source_time_span_per_trace"
            f.attrs["trace_count"] = np.uint64(n)
            f.attrs["finalized"] = np.uint8(1)
            f.attrs["finalized_utc"] = datetime.now(timezone.utc).isoformat()
            f.flush()
    except Exception:
        for f in outputs:
            f.close()
        for p in (tmp_plain, tmp_comp):
            if p.exists():
                p.unlink()
        raise
    else:
        for f in outputs:
            f.close()
        os.replace(tmp_plain, out_plain)
        os.replace(tmp_comp, out_comp)
        print(f"SUCCESS: {out_plain}")
        print(f"SUCCESS: {out_comp}")

def preflight_source(source_path: Path, dataset_name: str | None = None, shard_size: int = 5000):
    source_path, dataset_name, work, output_dir = split_source_paths(
        source_path, dataset_name, None, None
    )
    with h5py.File(source_path, "r") as f:
        layout = time_source_layout(f, source_path)
        n = int(layout["count"])
        one = np.asarray(f["time"][0], dtype=np.float64)
        signal, fs = select_window(
            one[:, int(layout["time_index"])],
            one[:, int(layout["current_index"])],
        )
        counts, pc = tlpp_counts_from_signal(signal, fs)
        if "UUID" not in f:
            raise RuntimeError(f"{source_path}: /UUID is required by the production source contract")
        tu0 = decode_text(f["UUID"][0]).strip()
        sf0 = source_path.name
        tid0 = tu0
        key0 = f"{sf0}/{tu0}"

        print(f"source={source_path}")
        print(f"dataset_name={dataset_name}")
        print(f"records={n:,}")
        print(f"time_shape={f['time'].shape}")
        print(f"time_names={layout['time_names']}")
        print(f"time_channel_index={layout['time_index']}")
        print(f"current_channel_index={layout['current_index']}")
        print("UUID=yes")
        print(f"example_trace_key={key0}")
        print(f"example_source_filename={sf0}")
        print(f"example_trace_id={tid0}")
        print(f"example_trace_uuid={tu0}")
        print(f"example_sampling_hz={fs:.9g}")
        print(f"example_window_samples={len(signal)}")
        print(f"example_lag_samples={max(1, int(round(LAG_US*1e-6*fs)))}")
        print(f"example_point_count={pc}")
        print(f"example_count_sum={int(counts.sum())}")
        print(f"work_dir={work}")
        print(f"output_dir={output_dir}")
        print(f"shard_size={shard_size:,}")
        print(f"shards={math.ceil(n/shard_size):,}")
        print(
            "logical_TLPP_GiB="
            f"{n * BINS * BINS * np.dtype(np.uint16).itemsize / 1024**3:.3f}"
        )

def verify_pair(plain_path: Path, comp_path: Path, sample_count: int = 11):
    with h5py.File(plain_path, "r") as a, h5py.File(comp_path, "r") as b:
        required = (
            "tlpp_counts", "trace_key", "source_filename", "trace_id", "trace_uuid",
            "global_index", "source_index", "sampling_hz", "lag_samples",
            "window_samples", "point_count", "lookup",
        )
        for name in required:
            if name not in a or name not in b:
                raise RuntimeError(f"Missing {name}")

        if a["tlpp_counts"].compression is not None:
            raise RuntimeError("Uncompressed file is unexpectedly compressed")
        if b["tlpp_counts"].compression != "lzf":
            raise RuntimeError("Compressed file is not LZF")

        if a["tlpp_counts"].is_virtual or b["tlpp_counts"].is_virtual:
            raise RuntimeError("Final file still contains virtual tlpp_counts")

        if a["tlpp_counts"].shape != b["tlpp_counts"].shape:
            raise RuntimeError("Count shapes differ")

        n = a["tlpp_counts"].shape[0]
        indices = np.linspace(0, n - 1, min(sample_count, n), dtype=np.int64)
        for i in indices:
            if not np.array_equal(a["tlpp_counts"][i], b["tlpp_counts"][i]):
                raise RuntimeError(f"TLPP mismatch at {i}")
            for name in (
                "trace_key", "source_filename", "trace_id", "trace_uuid",
                "global_index", "sampling_hz", "lag_samples",
                "window_samples", "point_count"
            ):
                av = a[name][i]
                bv = b[name][i]
                if isinstance(av, bytes) or isinstance(bv, bytes):
                    av, bv = decode_text(av), decode_text(bv)
                if av != bv:
                    raise RuntimeError(f"Metadata mismatch {name}[{i}]")

        print(f"VERIFIED {plain_path.name} <-> {comp_path.name}")
        print(f"records={n:,}")
        print(f"uncompressed={plain_path.stat().st_size / 1024**3:.3f} GiB")
        print(f"compressed={comp_path.stat().st_size / 1024**3:.3f} GiB")

def deep_verify(path: Path, block: int = 512):
    with h5py.File(path, "r") as f:
        n = f["tlpp_counts"].shape[0]
        for start in range(0, n, block):
            stop = min(start + block, n)
            counts = np.asarray(f["tlpp_counts"][start:stop], dtype=np.uint64)
            sums = counts.sum(axis=(1, 2))
            expected = np.asarray(f["point_count"][start:stop], dtype=np.uint64)
            if not np.array_equal(sums, expected):
                bad = np.flatnonzero(sums != expected)[0]
                raise RuntimeError(f"point_count mismatch at row {start + int(bad)}")
            if start == 0 or stop == n or stop % 50000 < block:
                print(f"deep verify: {stop:,}/{n:,}", flush=True)
    print("DEEP VERIFY OK")

def _lookup_group_is_open_addressing(g) -> bool:
    return (
        isinstance(g, h5py.Group)
        and "hash" in g
        and "row" in g
        and decode_text(g.attrs.get("field", "")) == "trace_uuid"
        and decode_text(g.attrs.get("probe_strategy", "")) == "linear"
    )

def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()

def uuid_hash_table_capacity(count: int, max_load: float = UUID_HASH_TABLE_DEFAULT_MAX_LOAD) -> int:
    if count < 0:
        raise ValueError("count must be non-negative")
    if not (0.0 < float(max_load) < 1.0):
        raise ValueError("max_load must be between 0 and 1")
    minimum = max(2, int(math.ceil(max(1, count) / float(max_load))))
    return _next_power_of_two(minimum)

def _lookup_uuid_open_addressing(f: h5py.File, value: str, group_name: str = UUID_HASH_TABLE_GROUP):
    if group_name not in f:
        raise KeyError(f"/{group_name} is not present")
    g = f[group_name]
    hashes = g["hash"]
    rows = g["row"]
    capacity = int(g.attrs["capacity"])
    if capacity <= 0 or capacity & (capacity - 1):
        raise RuntimeError(f"/{group_name}: capacity is not a power of two")
    if len(hashes) != capacity or len(rows) != capacity:
        raise RuntimeError(f"/{group_name}: dataset length/capacity mismatch")

    target = int(hash64(value))
    mask = capacity - 1
    slot = target & mask
    empty = int(UUID_HASH_TABLE_EMPTY_ROW)

    for probes in range(1, capacity + 1):
        row = int(rows[slot])
        if row == empty:
            return [], probes
        if int(hashes[slot]) == target:
            # The 64-bit hash is only a filter.  The full UUID is authoritative,
            # so a genuine BLAKE2b-64 collision cannot return the wrong trace.
            if decode_text(f["trace_uuid"][row]).strip() == value:
                return [row], probes
        slot = (slot + 1) & mask

    raise RuntimeError(f"/{group_name}: corrupt/full hash table (no empty slot encountered)")

def find_uuid_rows(path: Path, value: str):
    if not value:
        raise ValueError("UUID cannot be empty")
    with h5py.File(path, "r") as f:
        if UUID_HASH_TABLE_GROUP not in f or not _lookup_group_is_open_addressing(f[UUID_HASH_TABLE_GROUP]):
            raise RuntimeError(
                f"{path}: canonical /{UUID_HASH_TABLE_GROUP} UUID hash table is missing; "
                "rebuild/finalize the file with the canonical writer"
            )
        matches, probes = _lookup_uuid_open_addressing(f, value)
        return matches, probes

def _verify_uuid_lookup_open(f, group_name: str = UUID_HASH_TABLE_GROUP,
                             sample_count: int = 1000):
    if group_name not in f or not _lookup_group_is_open_addressing(f[group_name]):
        raise RuntimeError(f"/{group_name} is not a canonical UUID hash table")
    g = f[group_name]
    n = int(len(f["trace_uuid"]))
    capacity = int(g.attrs["capacity"])
    count = int(g.attrs["count"])
    if count != n or len(g["hash"]) != capacity or len(g["row"]) != capacity:
        raise RuntimeError(f"/{group_name}: schema/count mismatch")
    if capacity <= 0 or capacity & (capacity - 1):
        raise RuntimeError(f"/{group_name}: capacity is not a power of two")

    rows = np.asarray(g["row"][:], dtype=np.uint64)
    occupied = rows != np.uint64(UUID_HASH_TABLE_EMPTY_ROW)
    if int(np.count_nonzero(occupied)) != n:
        raise RuntimeError(f"/{group_name}: occupied count != record count")
    occupied_rows = rows[occupied]
    if np.any(occupied_rows >= np.uint64(n)):
        raise RuntimeError(f"/{group_name}: invalid row")
    seen = np.zeros(n, dtype=np.uint8)
    np.add.at(seen, occupied_rows.astype(np.int64, copy=False), 1)
    if not np.all(seen == 1):
        raise RuntimeError(f"/{group_name}: rows are not a one-to-one permutation")

    sample_rows = np.linspace(0, n - 1, min(sample_count, n), dtype=np.int64)
    probes = []
    for expected in sample_rows:
        value = decode_text(f["trace_uuid"][int(expected)]).strip()
        if not value:
            raise RuntimeError(f"empty UUID at row {int(expected)}")
        found, count_probes = _lookup_uuid_open_addressing(f, value, group_name=group_name)
        if found != [int(expected)]:
            raise RuntimeError(
                f"/{group_name}: UUID lookup mismatch at row {int(expected)}: {found}"
            )
        probes.append(count_probes)
    return probes

def verify_uuid_hash_table(path: Path, sample_count: int = 1000):
    path = Path(path)
    with h5py.File(path, "r") as f:
        probes = _verify_uuid_lookup_open(f, sample_count=sample_count)
        g = f[UUID_HASH_TABLE_GROUP]
        n = int(g.attrs["count"])
        capacity = int(g.attrs["capacity"])
        print(
            f"{path.name}: UUID lookup VERIFIED; count={n:,}; capacity={capacity:,}; "
            f"load={n/capacity:.3f}; sampled_mean_probes={float(np.mean(probes)):.3f}; "
            f"sampled_max_probes={max(probes) if probes else 0}"
        )

def selftest():
    from tlpp_embedding.core import make_TLPP

    rng = np.random.default_rng(1729)
    for fs in (500_000.0, 1_000_000.0, 2_000_000.0):
        for n in (351, 701, 997):
            signal = (
                20.0
                + 8.0 * np.sin(2.0 * np.pi * 23_000.0 * np.arange(n) / fs)
                + rng.normal(0.0, 0.4, n)
            )
            signal[0] = -20.0
            signal[-1] = 120.0

            fast, pc = tlpp_counts_from_signal(signal, fs)

            points = make_TLPP(
                signal,
                fs,
                lag_us=LAG_US,
                interpolation_factor=INTERPOLATION_FACTOR,
                interpolation_method=INTERPOLATION_METHOD,
            )
            hi_clip = np.nextafter(
                np.float64(CURRENT_MAX_A), np.float64(CURRENT_MIN_A)
            )
            clipped = np.clip(
                points,
                np.float64(CURRENT_MIN_A),
                hi_clip,
            )
            reference, _, _ = np.histogram2d(
                clipped[:, 0],
                clipped[:, 1],
                bins=BINS,
                range=[
                    [CURRENT_MIN_A, CURRENT_MAX_A],
                    [CURRENT_MIN_A, CURRENT_MAX_A],
                ],
            )
            reference = reference.astype(np.uint16)

            if not np.array_equal(fast, reference):
                raise RuntimeError(f"TLPP regression mismatch for fs={fs}, n={n}")
            if pc != int(reference.sum()):
                raise RuntimeError("point_count regression mismatch")

    print("SELFTEST OK: universal TLPP counts match validated make_TLPP + numpy histogram2d")

def main():
    parser = argparse.ArgumentParser(description="Build and inspect universal TLPP HDF5 files")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("selftest")

    p = sub.add_parser("preflight-source")
    p.add_argument("source", type=Path)
    p.add_argument("--dataset-name")
    p.add_argument("--shard-size", type=int, default=5000)

    p = sub.add_parser("build-source-shard")
    p.add_argument("source", type=Path)
    p.add_argument("--dataset-name")
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--read-block", type=int, default=16)

    p = sub.add_parser("finalize-source")
    p.add_argument("source", type=Path)
    p.add_argument("--dataset-name")
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--shard-size", type=int, default=5000)

    p = sub.add_parser("verify-pair")
    p.add_argument("plain", type=Path)
    p.add_argument("compressed", type=Path)

    p = sub.add_parser("deep-verify")
    p.add_argument("path", type=Path)
    p.add_argument("--block", type=int, default=512)

    p = sub.add_parser("verify-lookup")
    p.add_argument("path", type=Path)
    p.add_argument("--sample-count", type=int, default=1000)

    p = sub.add_parser("find")
    p.add_argument("path", type=Path)
    p.add_argument("--uuid", required=True)

    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "preflight-source":
        preflight_source(args.source, args.dataset_name, args.shard_size)
    elif args.command == "build-source-shard":
        build_source_shard(args.source, args.dataset_name, args.work_dir, args.shard_index, args.shard_size, args.read_block)
    elif args.command == "finalize-source":
        finalize_source(args.source, args.dataset_name, args.work_dir, args.output_dir, args.shard_size)
    elif args.command == "verify-pair":
        verify_pair(args.plain, args.compressed)
    elif args.command == "deep-verify":
        deep_verify(args.path, args.block)
    elif args.command == "verify-lookup":
        verify_uuid_hash_table(args.path, args.sample_count)
    elif args.command == "find":
        matches, probes = find_uuid_rows(args.path, args.uuid)
        if not matches:
            print("NO MATCH")
        else:
            with h5py.File(args.path, "r") as h5:
                row = matches[0]
                print(f"row={row} probes={probes} trace_key={decode_text(h5['trace_key'][row])} trace_uuid={decode_text(h5['trace_uuid'][row])}")


if __name__ == "__main__":
    main()
