from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

import universal_tlpp as ut
from tlpp_embedding.evaluate import evaluate_vae_hdf5
from tlpp_embedding.hdf5 import UniversalTLPPReader, validate_universal_tlpp
from tlpp_embedding.models import VAE, VAEConfig, load_vae_checkpoint
from tlpp_embedding.plotting import build_dashboard, plot_trace_gallery
from tlpp_embedding.training import MultiHDF5TLPPDataset, preflight_hdf5_vae, train_hdf5_vae


def make_universal(path: Path, count: int = 8) -> None:
    rng = np.random.default_rng(4)
    counts = rng.poisson(0.2, size=(count, 128, 128)).astype(np.uint16)
    counts[:, 64, 64] += 10
    string_dtype = h5py.string_dtype("utf-8")
    ids = [str(uuid.uuid4()) for _ in range(count)]
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "tlpp-universal-v1"
        h5.attrs["finalized"] = 1
        h5.attrs["bins"] = 128
        h5.attrs["storage_variant"] = "compressed_lzf"
        h5.create_dataset("tlpp_counts", data=counts, compression="lzf", chunks=(1, 128, 128))
        h5.create_dataset("global_index", data=np.arange(count, dtype=np.uint64))
        h5["source_index"] = h5["global_index"]
        h5.create_dataset("lag_samples", data=np.full(count, 10, dtype=np.uint32))
        h5.create_dataset("point_count", data=counts.sum(axis=(1, 2)).astype(np.uint32))
        h5.create_dataset("sampling_hz", data=np.full(count, 1_000_000.0))
        h5.create_dataset("window_samples", data=np.full(count, 701, dtype=np.uint32))
        h5.create_dataset("source_filename", data=np.asarray([path.name] * count, dtype=object), dtype=string_dtype)
        h5.create_dataset("trace_id", data=np.asarray(ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("trace_uuid", data=np.asarray(ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("trace_key", data=np.asarray([f"{path.name}/{x}" for x in ids], dtype=object), dtype=string_dtype)
        ut.create_lookup_group(h5)


def make_checkpoint(path: Path, latent_dim: int = 8) -> None:
    cfg = VAEConfig(latent_dim=latent_dim, matryoshka_dims=(2, 4, 8), batch_size=4, epochs=2)
    model = VAE((1, 128, 128), latent_dim)
    old_style_cfg = {**cfg.to_dict(), "representation": "fixed", "input_bins": None, "occupancy_smoothing_sigma": 0.0}
    torch.save({
        "checkpoint_version": "test",
        "epoch": 2,
        "best_epoch": 2,
        "best_validation_loss": 0.1,
        "model_state": model.state_dict(),
        "vae": old_style_cfg,
        "sample_shape": (1, 128, 128),
    }, path)


class CurrentWorkflowTests(unittest.TestCase):
    def test_reader_checkpoint_and_encoder_feature_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); h5 = root / "data.h5"; ckpt = root / "best.pt"
            make_universal(h5); make_checkpoint(ckpt)
            meta = validate_universal_tlpp(h5)
            self.assertEqual(meta.count, 8)
            with UniversalTLPPReader(h5) as reader:
                self.assertEqual(reader.counts(0).shape, (128, 128))
            state, cfg, model, device = load_vae_checkpoint(ckpt, "cpu")
            self.assertEqual(cfg.latent_dim, 8)
            features = model.encoder_features(torch.zeros(1, 1, 128, 128))
            self.assertEqual([tuple(x.shape) for x in features], [(1,16,64,64),(1,32,32,32),(1,64,16,16)])
            self.assertEqual(device.type, "cpu")
            self.assertEqual(tuple(state["sample_shape"]), (1,128,128))

    def test_multi_hdf5_training_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_a, train_b, val_a, val_b = [root / name for name in ("train_a.h5","train_b.h5","val_a.h5","val_b.h5")]
            for path, count in ((train_a,3),(train_b,5),(val_a,2),(val_b,3)):
                make_universal(path, count)
            run = root / "run"
            cfg = VAEConfig(latent_dim=4, matryoshka_dims=(2,4), batch_size=2, epochs=1, kl_warmup_epochs=0)
            ds = MultiHDF5TLPPDataset([train_a, train_b], cfg)
            self.assertEqual(len(ds), 8); self.assertEqual(ds.component_counts, (3,5)); ds.close()
            result = preflight_hdf5_vae([train_a, train_b], [val_a, val_b], cfg)
            self.assertEqual(result["train_count"], 8); self.assertEqual(result["val_count"], 5)
            train_hdf5_vae([train_a,train_b],[val_a,val_b],run,cfg=cfg,target_epochs=1,num_workers=0,device="cpu")
            first = torch.load(run/"last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(first["epoch"],1)
            train_hdf5_vae([train_a,train_b],[val_a,val_b],run,cfg=cfg,target_epochs=2,num_workers=0,device="cpu")
            second = torch.load(run/"last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(second["epoch"],2); self.assertEqual(len(second["history"]),2)
            self.assertEqual(second["data_paths"]["train"],[str(train_a),str(train_b)])

    def test_combined_evaluation_gallery_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/"run"; run.mkdir(); ckpt=run/"best.pt"; make_checkpoint(ckpt)
            (run/"history.csv").write_text(
                "epoch,train_loss,train_reconstruction,train_kl,val_loss,val_reconstruction,val_kl,effective_beta,mean_unclipped_grad_norm,epoch_seconds\n"
                "1,0.2,0.19,1,0.21,0.20,1,0.0005,0.5,1\n2,0.1,0.09,0.8,0.11,0.10,0.9,0.001,0.4,1\n",
                encoding="utf-8",
            )
            vs1,vs2,vl1,vl2=[root/name for name in ("vs1.h5","vs2.h5","vl1.h5","vl2.h5")]
            for path,count in ((vs1,3),(vs2,4),(vl1,5),(vl2,6)): make_universal(path,count)
            eval_root=run/"evaluation"
            m1=evaluate_vae_hdf5(ckpt,[vs1,vs2],eval_root/"val_small",embedding_dims=(2,4,8),batch_size=4,device="cpu")
            m2=evaluate_vae_hdf5(ckpt,[vl1,vl2],eval_root/"val_large",embedding_dims=(2,4,8),batch_size=4,device="cpu")
            self.assertEqual(m1["count"],7); self.assertEqual(m2["count"],11)
            rows=plot_trace_gallery(ckpt,[vl1,vl2],eval_root/"val_large",eval_root/"dashboard"/"trace TLPP plots",trace_count=3,embedding_dims=(2,4,8),device="cpu")
            self.assertEqual(len(rows),3)
            dashboard=build_dashboard(run); self.assertTrue(dashboard.exists())
            self.assertTrue((eval_root/"val_large"/"plots"/"training_history.png").exists())

    def test_uuid_lookup_collision_handling(self):
        ids=["00000000-0000-4000-8000-000000000001","00000000-0000-4000-8000-000000000002","00000000-0000-4000-8000-000000000003"]
        text=h5py.string_dtype("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"collision.h5"
            with h5py.File(path,"w") as h5:
                h5.create_dataset("trace_uuid",data=np.asarray(ids,dtype=object),dtype=text)
                original=ut.hash64
                try:
                    ut.hash64=lambda _value: np.uint64(7)
                    ut.create_lookup_group(h5)
                    for row,value in enumerate(ids):
                        found,probes=ut._lookup_uuid_open_addressing(h5,value)
                        self.assertEqual(found,[row]); self.assertGreaterEqual(probes,1)
                finally:
                    ut.hash64=original


if __name__ == "__main__":
    unittest.main()
