# TLPP embedding pipeline

This repo is the cleaned up version of the TLPP work we have been using for the H9 diffusion-control project. I stripped out the old cache experiments that I was using, a bunch of one-off migration scripts, an older, obsolete lookup formats (now that we have our sick O(1) hash table lol), adaptive/multilag experiments, and old versions of the VAE code so that what is left is basically the path we are actually using now:

```text
raw waveform HDF5
        ↓
fixed-physics TLPP generation
        ↓
universal TLPP HDF5
        ↓
128-D Matryoshka VAE
        ↓
validation / held-out evaluation / plots
```

The code is written around the current Great Lakes/Turbo layout, but the actual Python interfaces accept arbitrary compatible HDF5 paths, so it should be fairly easy to fold into another workflow. Hopefully this is useful for the ControlNet type idea
we talked about in the lab.

## What the TLPP actually is

For every discharge-current trace we take a centered **700 µs window inside the final 1 ms** of the trace, linearly densify it by **16×**, and construct the non-circular delay portrait

```text
[I(t), I(t - 10 µs)]
```

The points are histogrammed into a **128 × 128** grid spanning **-5 A to 105 A**. Values outside that range are clipped into the edge bins. What gets stored is the raw `uint16` occupancy count image—no smoothing and no normalization.

The physical definition is the same for the subscale and fullscale datasets even though their original sampling rates differ. The builder measures the sampling rate from each trace's timestamps instead of hard-coding it:

- subscale: 2001 samples over 2 ms, roughly 1 MHz, so 10 µs is about 10 raw samples;
- fullscale: 1001 samples over 2 ms, roughly 500 kHz, so 10 µs is about 5 raw samples.

After 16× interpolation the physical lag is still 10 µs in both cases.

## Raw source HDF5 contract

The builder does not assume a fixed number or order of time-series channels. It requires:

```text
/time       rank-3 array: (records, time samples, channels)
/time_names one name per channel
/UUID       one UUID string per record
```

`time_names` must contain at least:

```text
time_s
discharge_current_A
```

The current files are:

```text
SUBSCALE
/nfs/turbo/coe-marksta/h9_subscale/2026_06_subscale_data/
    h9_batch3_train.h5       1,044,539 traces
    h9_batch3_val_small.h5       2,644 traces
    h9_batch3_val_large.h5      10,578 traces

FULLSCALE
/nfs/turbo/coe-marksta/h9_fullscale/
    h9_train.h5             2,372,757 traces
    h9_val_small.h5            40,971 traces
    h9_val_large.h5           131,056 traces
```

The new subscale splits are mutually disjoint by UUID and add back up to the historical 1,057,761-trace subscale dataset size. The fullscale splits are also mutually disjoint. I had to make sure of this because of what happened earlier with the h9_batch3_norm issue I mentioned on slack.

## Universal TLPP HDF5 format

Both the plain and LZF-compressed files represent exactly the same data. We normally train from the compressed files because the much smaller Turbo I/O footprint is worth far more than the LZF decompression cost.

Each finalized file has this logical layout:

```text
/
├── tlpp_counts          uint16, (N, 128, 128)
├── trace_key            UTF-8, (N,)
├── source_filename      UTF-8, (N,)
├── trace_id             UTF-8, (N,)
├── trace_uuid           UTF-8, (N,)
├── global_index         uint64, (N,)
├── source_index         hard link to global_index
├── sampling_hz          float64, (N,)
├── lag_samples          uint32, (N,)
├── window_samples       uint32, (N,)
├── point_count          uint32, (N,)
└── lookup/
    ├── hash             uint64, (capacity,)
    └── row              uint64, (capacity,)
```

`point_count` is useful for integrity checks: for every record,

```text
sum(tlpp_counts) == point_count
```

## Using this as a Python library

The Slurm scripts and command-line tools are mostly just wrappers around regular Python functions/classes, so none of this is tied to the Great Lakes workflow. If you're integrating this into another codebase, you can import the pieces you need directly.

The main public interfaces are exported from `tlpp_embedding`:

```python
from tlpp_embedding import (
    TLPPConfig,
    make_TLPP,
    make_tlpp_counts,
    make_occupancy_counts,
    transform_occupancy,
    UniversalTLPPReader,
    VAE,
    VAEConfig,
    load_vae_checkpoint,
)
```

For example, to construct a TLPP directly from a discharge-current trace:

```python
from tlpp_embedding import make_tlpp_counts

counts = make_tlpp_counts(
    current_signal,
    sampling_hz,
    lag_us=10.0,
    bins=128,
    value_range=(-5.0, 105.0),
    interpolation_factor=16,
)
```

If you want the delay-coordinate points themselves instead of the binned image:

```python
from tlpp_embedding import make_TLPP

points = make_TLPP(
    current_signal,
    sampling_hz,
    lag_us=10.0,
    interpolation_factor=16,
)
```

The universal HDF5 files can also be accessed lazily without loading the whole dataset:

```python
from tlpp_embedding import UniversalTLPPReader

reader = UniversalTLPPReader(
    "/path/to/h9_train_TLPPs_compressed.h5"
)

sample = reader[0]

counts = sample["tlpp_counts"]
uuid = sample["trace_uuid"]

row = reader.find_uuid(uuid)
```

And a trained VAE can be loaded and used directly:

```python
from tlpp_embedding import load_vae_checkpoint

vae, state = load_vae_checkpoint(
    "/path/to/best.pt",
    device="cuda",
)

vae.eval()

mu, logvar = vae.stats(x)
embedding = mu

features = vae.encoder_features(x)
```

`mu` is what we use as the deterministic embedding during validation/evaluation, while `encoder_features()` exposes the intermediate convolutional feature maps that could be useful for the ControlNet-style work discussed later.


### UUID lookup

`/lookup` is a persistent open-addressed hash table for `trace_uuid → HDF5 row`.

The UUID is hashed with BLAKE2b truncated to 64 bits. The initial slot is selected from the hash and collisions are handled by linear probing. A matching 64-bit hash is **not** treated as proof of identity: the full stored UUID string is always compared before a row is returned. This means both ordinary bucket collisions and a true 64-bit hash collision are handled correctly. This was an unexpectedly annoying problem to diagnose.

The expected lookup complexity is O(1). The table uses a `uint64` max-value sentinel to represent an empty row, so every possible 64-bit hash value remains legal.

Example:

```bash
python universal_tlpp.py find \
  /path/to/h9_train_TLPPs_compressed.h5 \
  --uuid 00002150-b7ef-45bf-9853-0f9f30cdc332
```

## Building TLPP HDF5 files

First check a source without writing anything:

```bash
python universal_tlpp.py preflight-source /path/to/h9_train.h5 \
  --dataset-name h9_train \
  --shard-size 5000
```

The Great Lakes build path is shard-based so a multi-million-trace dataset can be generated in parallel without making every worker write into the same HDF5 file. It saves an absolute crap-ton of time.

A reusable launcher is included:

```bash
SOURCE_ROOT=/nfs/turbo/coe-marksta/h9_fullscale \
OUTPUT_ROOT=/nfs/turbo/coe-marksta/h9_fullscale \
DATASETS="h9_train h9_val_small h9_val_large" \
bash slurm/submit_tlpp_family.sh
```

For the current subscale files:

```bash
SOURCE_ROOT=/nfs/turbo/coe-marksta/h9_subscale/2026_06_subscale_data \
OUTPUT_ROOT=/nfs/turbo/coe-marksta/h9_subscale/2026_06_subscale_data_TLPPs \
DATASETS="h9_batch3_train h9_batch3_val_small h9_batch3_val_large" \
bash slurm/submit_tlpp_family.sh
```

Each finalizer merges the shards on scratch, produces both plain and compressed files, compares them, performs the full `point_count` check, SHA256-checks the copy to Turbo, and only then publishes the final filename.

The outputs are named:

```text
<dataset>_TLPPs.h5
<dataset>_TLPPs_compressed.h5
```

## Model input

The VAE never sees the raw integer counts directly. For every TLPP:

1. convert counts to `float32`;
2. normalize the whole 128×128 map so it sums to 1;
3. apply the fixed `log01` transform

```text
log(1 + p / ε) / log(1 + 1 / ε),   ε = 1e-6
```

There is no per-trace min/max scaling and no spatial smoothing. This way, if we wanted to use a different scaling function or whatever, we don't need to regenerate the TLPPs.

## Current VAE

The model is intentionally oretty simple right now because the main question has been whether the TLPP representation itself gives us a useful latent space. I haven't gotten to implement the Res-net idea or the ControlNet architecture because I haven't had much time, but that'll be my next task. I've mainly sort of focused on just getting the TLPP h5 files working and the VAE trained on all data preliminarily. Eventually though, we'll probably change the encoder training using the new diffusion-loss function and stuff.

Encoder:

```text
1 × 128 × 128
    ↓ Conv 3×3, stride 2 + ReLU
16 × 64 × 64
    ↓ Conv 3×3, stride 2 + ReLU
32 × 32 × 32
    ↓ Conv 3×3, stride 2 + ReLU
64 × 16 × 16
    ↓ flatten
mu:     128
logvar: 128
```

Decoder:

```text
latent 128
    ↓ linear
64 × 16 × 16
    ↓ transpose conv + ReLU
32 × 32 × 32
    ↓ transpose conv + ReLU
16 × 64 × 64
    ↓ transpose conv
1 × 128 × 128 logits
```

The external `VAE` wrapper preserves the checkpoint tensor names used by the current training runs.

### Matryoshka latent objective

We train the same latent vector at prefixes:

```text
4, 8, 16, 32, 64, 128 dimensions
```

For a prefix of width `d`, dimensions after `d` are zeroed before decoding. All six reconstruction losses are averaged into the reconstruction objective. The reconstruction loss is squared Hellinger distance between the target TLPP probability map and the decoder softmax probability map:

```text
H²(P,Q) = 1 - Σ sqrt(P_i Q_i)
```

The total VAE objective is

```text
reconstruction + beta * KL
```

with:

```text
beta = 1e-3
10-epoch linear KL warmup
AdamW
learning rate = 3e-4
weight decay = 1e-5
gradient clipping = 5
batch size = 64
seed = 1729
```

I should probably explain the KL warmup since it might not seem very useful at first glance. The Kullback Leibler loss term is a metric that penalizes the model when the latent variable distribution is differs significantly from another distribution, which for the purposes of VAEs, is just the normal distribution. It basically forces the distribution to have a better spread of possible embedding vectors rather than just collapsing the distribution and making the selection small. But it needs to "warm-up" at the start to avoid Posterior Collapse, where the model learns to minimize loss by just minimizing KL loss and generating normal distributions despite not learning much at all from the training data. Warm-up allows the model to first learn useful data on the training set itself before then making good latent vector distributions for sampling later. You probably already know about all of this though.

Validation uses the latent mean (`mu`) rather than sampling, so validation is deterministic. `best.pt` is selected by the `val_small` total VAE objective only after the KL warmup has reached its final beta.

## Combined model split

The current combined experiment treats the two data families as one logical PyTorch dataset without physically merging the HDF5 files.

```text
TRAIN
subscale train     1,044,539
fullscale train    2,372,757
                  ---------
                   3,417,296

VAL_SMALL / checkpoint selection
subscale               2,644
fullscale              40,971
                      -------
                       43,615

VAL_LARGE / held out
subscale              10,578
fullscale            131,056
                    -------
                     141,634
```

`MultiHDF5TLPPDataset` maps one global dataset index into the correct component file/local row and lazily opens HDF5 handles per worker. We deliberately do not create another giant merged HDF5 just for training.

## Training and resume

On Great Lakes:

```bash
cd ~/TLPP_embedding
module load python
source ~/tlpp_venv/bin/activate
```

Preflight:

```bash
python -m tlpp_embedding.cli preflight-combined --epochs 50
```

Interactive/direct training:

```bash
python -m tlpp_embedding.cli train-combined --epochs 50 --device cuda
```

Or use the Slurm chain:

```bash
bash slurm/submit_combined_train_chain.sh
```

I made it so that the training job can be split into multiple jobs. Not ONLY that, but now we can also take an older model and train it for more epochs if we choose to, basically allowing us to resume training whenever we want. I think this is hella neat. Each Slurm segment stops cleanly before its wall-time budget and writes `last.pt`. The next segment restores:

- model parameters;
- AdamW state;
- completed epoch;
- best-validation tracking;
- full history;
- Python RNG;
- NumPy RNG;
- CPU torch RNG;
- CUDA RNG state(s).

It also records the exact train/validation HDF5 path lists in the checkpoint and refuses to resume if those files change. In other words, an epoch-17 segment ending and the next job starting at epoch 18 is a normal path, not a special recovery path or anything.

Run directory currently defaults to:

```text
/nfs/turbo/coe-marksta/h9_fullscale/vae_runs/
combined_subscale_fullscale_128x128_latent128_run1/
```

Important outputs:

```text
last.pt       latest completed epoch; resume checkpoint
best.pt       best val_small checkpoint after KL warmup
history.csv   per-epoch train/validation metrics
config.json   data paths + run/model configuration
```

## Evaluation

The evaluator handles **multiple HDF5 files as one logical split**, just like training. This matters because both `val_small` and `val_large` contain a subscale component and a fullscale component.

Run everything:

```bash
sbatch slurm/evaluate_combined_vae.sbatch
```

or:

```bash
python -m tlpp_embedding.cli evaluate-combined --device cuda --gallery-count 50
```

Evaluation produces, for both combined `val_small` and combined `val_large`:

```text
metrics.json
prefix_metrics.csv
latent_dimensions.csv
per_trace_errors.csv
plots/
    prefix_reconstruction.png
    latent_activity.png
    reconstruction_error_distribution.png
    training_history.png
    kl_warmup.png
```

`training_history.png` contains train vs validation curves for:

- total VAE objective;
- reconstruction H²;
- KL divergence.

The dashboard is written to:

```text
evaluation/dashboard/index.html
```

The optional trace gallery selects examples across the full held-out 128-D reconstruction-error distribution. Each trace folder contains the raw count TLPP, probability map, exact `log01` input, all Matryoshka reconstructions, latent vector, prefix-error curve, and trace metadata.

## Where the next model idea could plug into what I have

The next thing we've been discussing is replacing the plain convolutional encoder/decoder blocks with residual blocks and, more importantly, using the encoder as a conditioning network for the diffusion model in the ControlNet-like way from the paper. I still don't fully understand the implementation yet (partially because I haven't been focusing on it) but I guess now's the time.

I haven't implemented any of that in this repo yet; this repo is just meant to be a clean baseline. I did make one useful hook without changing the trained model or its checkpoint keys though:

```python
features = vae.encoder_features(x)
```

For a 128×128 input, that returns:

```text
features[0]  (B, 16, 64, 64)
features[1]  (B, 32, 32, 32)
features[2]  (B, 64, 16, 16)
```

Those are the obvious places to attach gated/zero-initialized projection layers into diffusion-model blocks with matching spatial scales. If the diffusion channels do not match, a 1×1 convolution (or another small learned adapter or something) can probably do the channel projection. Keeping the external `stats()`, `decode()`, and `encoder_features()` contracts stable should make it possible to experiment with residual encoders or ControlNet-style conditioning without rewriting the whole HDF5/training pipeline. I've honestly revised it so many times that I'm a bit sick of it, and would like it to stay as is without having to tweak or overhaul it anymore.

## UUID lookup from Python / CLI

The simplest command-line lookup is:

```bash
python universal_tlpp.py find FILE.h5 --uuid UUID
```

The lookup table lives entirely inside the HDF5, so there is no `.pkl`, `.npy`, SQLite database, or sidecar index that has to stay synchronized with the file.

## Installing elsewhere

The repo can still be run the way we currently run it with `PYTHONPATH` from the repo root, but it is also installable:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . --no-build-isolation
```

(`--no-build-isolation` is useful on Great Lakes if the compute/login environment can't reach PyPI; on a normal internet-connected machine `pip install -e .` should be fine.)

Then either use:

```bash
python -m tlpp_embedding.cli ...
```

or the installed shortcut:

```bash
tlpp-vae ...
```

## Tests

```bash
python -m unittest discover -s tests -v
python universal_tlpp.py selftest
```

The focused tests cover:

- exact TLPP interpolation/histogram behavior;
- 1 MHz `(2001,3)` subscale sources;
- 500 kHz `(1001,4)` fullscale sources;
- shard → finalize → plain/compressed verification;
- canonical UUID hash lookup and forced hash collisions;
- loading current checkpoints while ignoring old unused config keys;
- multi-HDF5 training;
- cross-job checkpoint/resume;
- combined subscale+fullscale evaluation;
- training curves, dashboard, and trace-gallery generation;
- encoder feature-map outputs for the future conditioning work.

## What is intentionally not in this repo

I removed the historical 64×64 models, old NPZ/cache workflow, adaptive/multilag TLPP experiments from way back (although if you think it might be useful I can probably just bring it back), raw-waveform evaluator, old sorted lookup format and migration scripts (I made an older lookup that was O(log(n)) but it was pointless now), old `h9_batch3_norm`/`h9_normalized` training paths, a bunch of dated backups, one-off consolidation scripts I was using to adapt my old code to the changing Turbo files, and Slurm logs. They were useful during development, but they make the production path way too complicated and weren't needed to reproduce the current model.

## That's all, folks!

That's everything I have for you right now. I sort of ulted so I'll be taking a bit of time to breathe and relax lmao