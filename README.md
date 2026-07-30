# Self-Supervised Vision Transformers for Radar Grid Learning Radar-DINO

PyTorch implementation for Radar-DINO.

This project is based in the [original DINO](https://github.com/facebookresearch/dino) repository by [Facebook AI research group](https://ai.facebook.com/).

## Python package

The `wip` field-token architecture is now exposed through an installable
`radar_dino` package. Each radar field receives its own patch-token sequence and
learned field embedding. A released model artifact contains:

- `manifest.json`: the exact field order, normalization, altitude, input size,
  patch size, and architecture used during training.
- `model.safetensors`: inference-only teacher-backbone weights.
- Optional reference-catalog files for PCA/HDBSCAN cluster assignment, UMAP
  projection, reference t-SNE visualization, and cosine-nearest scans.

Install the local checkout during development:

```bash
python -m pip install -e '.[netcdf,hub,analysis,plot,dev]'
```

Install the public package directly from GitHub:

```bash
python -m pip install \
  "radar-dino[netcdf,hub,analysis,plot] @ git+https://github.com/DanielWefer/radar_dino.git@main"
```

Analyze one NetCDF scan from a notebook:

```python
from radar_dino import RadarDINO

dino = RadarDINO.from_pretrained(
    "dwefer/radar-dino-fieldtoken-v1",
    device="auto",
)
result = dino.analyze("/path/to/scan.nc")
pngs = dino.save_plots(result, "/path/to/output")

result.feature                 # (384,) for vit_small
result.attention               # heads x fields x height x width
result.attention_for("reflectivity", head="mean")
result.umap                    # present when a reference catalog is shipped
result.tsne                    # display-only nearest-neighbor interpolation
result.cluster                 # HDBSCAN label, or -1 for noise
result.cluster_probability
result.neighbors               # five cosine-nearest reference scans
pngs                           # multi-head field-attention PNGs plus UMAP and t-SNE
```

UMAP supports an out-of-sample transform. Scikit-learn t-SNE does not, so
`result.tsne` is explicitly an approximate display position interpolated from
nearby reference scans. Cluster labels are predicted by the fitted
PCA/HDBSCAN pipeline, never from either two-dimensional visualization.

The requested NetCDF fields may be supplied in any order, but they must match
the model manifest. Radar-DINO always reorders them to the training order before
inference. Missing fields, unknown fields, incompatible spatial dimensions,
undersized grids, and incompatible grid spacing raise errors instead of
silently changing the model input.

The same operation is available from the command line:

```bash
radar-dino analyze /path/to/scan.nc \
  --model dwefer/radar-dino-fieldtoken-v1 \
  --output /path/to/result \
  --plots
```

The output directory contains `feature.npy`, `attention.npy`, `umap.npy`, and
`tsne.npy` when available. `result.json` records the cluster label,
membership strength, five most similar reference scans, and generated PNG
filenames. Each field-attention PNG shows the normalized radar field followed
by every attention head on a shared scale. Reflectivity uses the bundled
ChaseSpectral colormap. UMAP and t-SNE plots highlight the input scan.

Export an existing trusted training checkpoint after creating a manifest that
matches its training arguments:

```bash
radar-dino export /path/to/checkpoint.pth \
  --manifest /path/to/manifest.json \
  --output /path/to/radar-dino-model-artifact \
  --allow-unsafe-pickle
```

Only use `--allow-unsafe-pickle` for checkpoints you trust. The exported public
artifact contains tensor-only `safetensors` weights. Reference PCA, HDBSCAN,
and UMAP estimators use joblib serialization, so only load a reference catalog
from a model repository you trust.

## Testing and test-driven development

The test suite is CPU-only by default and uses small synthetic radar NetCDF
files. It does not require production radar data, distributed initialization, or
a CUDA device.

Install the test dependencies into the same environment used for Radar-DINO:

```bash
python3 -m pip install -r requirements-dev.txt
```

Run the complete suite:

```bash
make test
```

Run only fast unit tests or the end-to-end synthetic pipeline test:

```bash
make test-unit
make test-integration
```

Generate a coverage report:

```bash
make test-cov
```

For test-driven changes:

1. Add or update a focused test describing the desired behavior.
2. Run that test and confirm it fails for the expected reason.
3. Make the smallest production change that satisfies the test.
4. Run `make test` before committing.

Tests that eventually require CUDA should use the `gpu` pytest marker and remain
separate from the default CPU suite.

## Training the WIP field-token model

The current training path is `radar_dino_training.py`. Unlike the original
mixed-channel image tokenizer, it projects every radar field independently
with a shared single-channel patch projection and adds a learned field
embedding. The token sequence therefore preserves both field identity and
spatial patch location.

### Training data contract

Training recursively reads regridded NetCDF files. The reproduced WIP run uses this exact field order:

| Field | Clipping range |
| --- | ---: |
| `reflectivity` | 10 to 75 dBZ |
| `specific_differential_phase` | -5 to 10 deg/km |
| `differential_reflectivity` | -8 to 12 dB |
| `cross_correlation_ratio` | 0.2 to 1.05 |
| `spectrum_width` | 0 to 20 m/s |

Each field is clipped and normalized to `[0, 1]`, and its original NaNs are
set to `--radar_nan_fill`. At locations where reflectivity is missing or outside
10-75 dBZ, every field is set to that sentinel (`-1.0` in the WIP run). The job
selects the grid level nearest 2000 m. Use `--z_level none` only when column
maxima are intended.

### Crops, patches, and asymmetric masking

The WIP configuration uses 1 km grid spacing, 300 km global crops, 100 km local
crops, and 10x10 grid-cell patches. Crops are random physical subwindows; they
are not resized to 224x224. Crop dimensions are rounded down to a patch-size
multiple.

For every scan, the augmentation produces two global views and four local
views. It applies independent horizontal and vertical flips, and may add
Gaussian noise to the second global and local views while preserving missing
pixels. The teacher receives the two global views without artificial
whole-field masking. The student receives all six views; with
`--channel_nan_prob 1.0`, exactly one randomly chosen field in each student
sample is replaced by `-1.0`.

With five fields and patch size 10, a 300x300 global crop has 4,500 field-patch
tokens and a 100x100 local crop has 500, plus the CLS token in each view.

### Reproduce the WIP run

Run this from the repository root. Change the process count to match the number
of visible GPUs:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  radar_dino_training.py \
  --arch vit_small \
  --patch_size 10 \
  --epochs 100 \
  --batch_size_per_gpu 1 \
  --num_workers 2 \
  --data_path /path/to/gridnc \
  --output_dir /path/to/radar_train_wip_tokenized/model \
  --saveckp_freq 1 \
  --radar_fields reflectivity specific_differential_phase differential_reflectivity cross_correlation_ratio spectrum_width \
  --in_chans 5 \
  --z_level 2000.0 \
  --radar_nan_fill -1.0 \
  --global_crop_size_km 300 \
  --local_crop_size_km 100 \
  --grid_spacing_km 1.0 \
  --local_crops_number 4 \
  --channel_nan_prob 1.0 \
  --use_fp16 true
```

The output directory contains the rolling `checkpoint.pth`, numbered
`checkpointNNNN.pth` snapshots according to `--saveckp_freq`, and JSON-lines
training metrics in `log.txt`. Relaunching with the same output directory
automatically resumes from `checkpoint.pth`, including student, teacher,
optimizer, DINO loss, epoch, and mixed-precision scaler state.

On Polaris, copy and adapt
[`examples/pbs/wip_train.pbs`](examples/pbs/wip_train.pbs), then submit it with:

```bash
qsub examples/pbs/wip_train.pbs
```

The published PBS file retains the original project allocation, filesystem
paths, queue, and four-GPU configuration as a reference.

### Matching checkpoint inference

Raw-checkpoint inference must use the same architecture, field order, patch
size, altitude, sentinel, and 300x300 input contract:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  radar_dino_inference.py \
  --arch vit_small \
  --patch_size 10 \
  --data_path /path/to/gridnc \
  --pretrained_weights /path/to/model/checkpoint.pth \
  --checkpoint_key teacher \
  --dump_features /path/to/features \
  --batch_size_per_gpu 1 \
  --image_size 300 \
  --radar_fields reflectivity specific_differential_phase differential_reflectivity cross_correlation_ratio spectrum_width \
  --in_chans 5 \
  --z_level 2000.0 \
  --radar_nan_fill -1.0 \
  --plot_attention_overlays true \
  --max_attention_plots 20
```

Use `radar_dino_associative_inference.py` with the same model/data arguments
when feature rows must be saved alongside their source paths. Add
`--inference_up_to 100` for a 100-scan smoke test. For new user-facing
inference, prefer the versioned package and Hugging Face artifact described at
the top of this README.

## For visualizing attention maps

The public package API shown above is the recommended path. The legacy training
checkpoint utility can also produce the six-head ChaseSpectral reflectivity
multiplot directly:

```bash
python3 post_processing_utilities/visualize_attention.py \
  --pretrained_weights /path/to/checkpoint.pth \
  --radar_path /path/to/scan.nc \
  --radar_fields reflectivity specific_differential_phase differential_reflectivity cross_correlation_ratio spectrum_width \
  --in_chans 5 \
  --image_size 300 \
  --patch_size 10 \
  --output_dir /path/to/attention_output
```

## Sample PBS jobs

The original Polaris PBS jobs are published unchanged as reference templates:

- [`examples/pbs/wip_train.pbs`](examples/pbs/wip_train.pbs)
- [`examples/pbs/infer_radar_dino_wip_tokenized.pbs`](examples/pbs/infer_radar_dino_wip_tokenized.pbs)
- [`examples/pbs/assoc_infer_radar_dino_wip_tokenized.pbs`](examples/pbs/assoc_infer_radar_dino_wip_tokenized.pbs)

They retain the original allocation, queue, module, and filesystem paths so
readers can see an authentic example. Users should copy a script and adapt
those site-specific values and entry-point names for their own HPC environment.
