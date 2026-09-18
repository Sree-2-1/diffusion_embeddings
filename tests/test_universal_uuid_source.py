from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

import h5py
import numpy as np

import universal_tlpp as u


class UniversalUUIDSourceTests(unittest.TestCase):
    def _run_source_case(self, *, sample_count, names, expected_fs, dataset_name):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / f"{dataset_name}.h5"
            work = root / "work"
            out = root / "out"
            out.mkdir()

            n = 7
            t = np.linspace(0.0, 0.002, sample_count, dtype=np.float32)
            data = np.empty((n, sample_count, len(names)), dtype=np.float32)
            ids = [str(uuid.uuid4()) for _ in range(n)]
            time_index = names.index("time_s")
            current_index = names.index("discharge_current_A")

            for i in range(n):
                data[i, :, :] = 0.0
                data[i, :, time_index] = t
                data[i, :, current_index] = 20 + i + 5 * np.sin(2 * np.pi * 23000 * t)
                if "discharge_voltage_V" in names:
                    data[i, :, names.index("discharge_voltage_V")] = 300.0
                if "thrust_N" in names:
                    data[i, :, names.index("thrust_N")] = 0.1

            text = h5py.string_dtype("utf-8")
            with h5py.File(source, "w") as f:
                f.create_dataset("time", data=data, chunks=(2, sample_count, len(names)))
                f.create_dataset("time_names", data=np.asarray(names, dtype=object), dtype=text)
                f.create_dataset("UUID", data=np.asarray(ids, dtype=object), dtype=text)
                f.attrs["split"] = dataset_name

            self.assertEqual(u.source_shard_count(source, 3), 3)
            for shard in range(3):
                u.build_source_shard(source, dataset_name, work, shard, 3, 2)

            u.finalize_source(source, dataset_name, work, out, 3)
            plain = out / f"{dataset_name}_TLPPs.h5"
            comp = out / f"{dataset_name}_TLPPs_compressed.h5"
            u.verify_pair(plain, comp)
            u.deep_verify(comp, block=4)

            with h5py.File(comp, "r") as f:
                self.assertEqual(f["trace_uuid"][0].decode(), ids[0])
                self.assertEqual(f["trace_id"][0].decode(), ids[0])
                self.assertEqual(f["source_filename"][0].decode(), source.name)
                self.assertEqual(f["trace_key"][0].decode(), f"{source.name}/{ids[0]}")
                self.assertEqual(f.attrs["source_identity_dataset"], "/UUID")
                self.assertEqual(f.attrs["split"], dataset_name)
                self.assertEqual(int(f.attrs["source_time_sample_count"]), sample_count)
                self.assertEqual(int(f.attrs["source_time_channel_count"]), len(names))
                self.assertEqual(int(f.attrs["source_time_index"]), time_index)
                self.assertEqual(int(f.attrs["source_current_index"]), current_index)
                self.assertEqual(f.attrs["source_time_names"], ",".join(names))
                self.assertAlmostEqual(float(f.attrs["sampling_hz_nominal"]), expected_fs, delta=1.0)
                self.assertEqual(
                    f.attrs["sampling_hz_nominal_basis"],
                    "measured_from_source_time_span_per_trace",
                )
                self.assertEqual(int(f.attrs["finalized"]), 1)
                expected_lag = round(10e-6 * expected_fs)
                self.assertEqual(int(f["lag_samples"][0]), expected_lag)

    def test_fullscale_1001x4_source_end_to_end(self):
        self._run_source_case(
            sample_count=1001,
            names=["time_s", "discharge_voltage_V", "discharge_current_A", "thrust_N"],
            expected_fs=500000.0,
            dataset_name="h9_train",
        )

    def test_subscale_2001x3_source_end_to_end(self):
        self._run_source_case(
            sample_count=2001,
            names=["time_s", "thrust_N", "discharge_current_A"],
            expected_fs=1000000.0,
            dataset_name="h9_batch3_train",
        )

    def test_rejects_missing_current_channel(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "bad.h5"
            text = h5py.string_dtype("utf-8")
            with h5py.File(source, "w") as f:
                f.create_dataset("time", data=np.zeros((2, 11, 2), dtype=np.float32))
                f.create_dataset(
                    "time_names",
                    data=np.asarray(["time_s", "thrust_N"], dtype=object),
                    dtype=text,
                )
                f.create_dataset(
                    "UUID",
                    data=np.asarray([str(uuid.uuid4()), str(uuid.uuid4())], dtype=object),
                    dtype=text,
                )
            with h5py.File(source, "r") as f:
                with self.assertRaisesRegex(RuntimeError, "discharge_current_A"):
                    u.time_source_layout(f, source)


if __name__ == "__main__":
    unittest.main()
