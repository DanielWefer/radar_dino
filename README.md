# Self-Supervised Vision Transformers for Radar Grid Learning Radar-DINO

PyTorch implementation for Radar-DINO.

This project is based in the [original DINO](https://github.com/facebookresearch/dino) repository by [Facebook AI research group](https://ai.facebook.com/).

## Python package

The custom field-token architecture is now the `main` training and inference
path and is exposed through the installable `radar_dino` package. Each radar
field receives its own patch-token sequence and learned field embedding. A released model artifact contains:

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

## Training the field-token model on `main`

The current training path is `radar_dino_training.py`. Unlike the original
mixed-channel image tokenizer, it projects every radar field independently
with a shared single-channel patch projection and adds a learned field
embedding. The token sequence therefore preserves both field identity and
spatial patch location.

### Training data contract

Training recursively reads regridded NetCDF files. The reference main-branch
run uses this exact field order:

| Field | Clipping range |
| --- | ---: |
| `reflectivity` | 10 to 75 dBZ |
| `specific_differential_phase` | -5 to 10 deg/km |
| `differential_reflectivity` | -8 to 12 dB |
| `cross_correlation_ratio` | 0.2 to 1.05 |
| `spectrum_width` | 0 to 20 m/s |

Each field is clipped and normalized to `[0, 1]`, and its original NaNs are
set to `--radar_nan_fill`. At locations where reflectivity is missing or outside
10-75 dBZ, every field is set to that sentinel (`-1.0` in the reference
run). The job selects the grid level nearest 2000 m. Use `--z_level none` only when column
maxima are intended.

### Crops, patches, and asymmetric masking

The main configuration uses 1 km grid spacing, 300 km global crops, 100 km local
crops, and 10x10 grid-cell patches.

For every scan, the augmentation produces two global views and four local
views. It applies independent horizontal and vertical flips, and may add
Gaussian noise to the second global and local views while preserving missing
pixels. The teacher receives the two global views without artificial
whole-field masking. The student receives all six views; with
`--channel_nan_prob 1.0`, exactly one randomly chosen field in each student
sample is replaced by `-1.0`.

With five fields and patch size 10, a 300x300 global crop has 4,500 field-patch
tokens and a 100x100 local crop has 500, plus the CLS token in each view.

### Reproduce the run

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
  --output_dir /path/to/radar_train_fieldtoken/model \
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

On a PBS-based cluster, copy and adapt
[`examples/pbs/train_radino.pbs`](examples/pbs/train_radino.pbs), then submit it
with:

```bash
qsub examples/pbs/train_radino.pbs
```

The supplied example targets Polaris and requests one complete node with four
GPUs. Account, queue, filesystem, environment-module, and storage settings must
be adapted for another PBS installation. The complete setup and submission
workflow is described below.

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

## Running on a PBS-based HPC system

The repository includes three example jobs for the field-token implementation
on `main`:

- [`examples/pbs/train_radino.pbs`](examples/pbs/train_radino.pbs): train or
  resume a model.
- [`examples/pbs/infer_radino.pbs`](examples/pbs/infer_radino.pbs): extract
  features and save a capped set of attention plots.
- [`examples/pbs/assoc_infer_radino.pbs`](examples/pbs/assoc_infer_radino.pbs):
  extract features together with source filenames and paths for a reference
  catalog.

These are working Polaris examples, not scheduler-independent scripts. PBS
resource syntax and site software differ between clusters, so a new user should
copy a job and review its header before submitting it. In particular:

| Setting | Polaris example | What another site must supply |
| --- | --- | --- |
| Allocation | `#PBS -A SSL-SULI2026` | Project or allocation name |
| Nodes | `#PBS -l select=1:system=polaris` | Syntax for one GPU node |
| Queue | `capacity` or `preemptable` | An accessible GPU queue |
| Filesystems | `home:eagle` | Site storage resources, or remove this directive |
| Environment | ALCF `conda` module | Site CUDA/PyTorch module or environment |
| Storage | `/eagle/SSL-SULI2026/$USER` | High-throughput project or scratch storage |

Each job uses one node and starts four processes with
`torchrun --nnodes=1 --nproc_per_node=4`, one process per GPU. If a node at
another site exposes a different number of GPUs, change both the PBS GPU
request and `NPROC_PER_NODE` in the copied script. The examples are not
configured for multi-node rendezvous.

### 1. Clone the code

Run setup commands on the cluster login node:

```bash
cd "$HOME"
git clone https://github.com/DanielWefer/radar_dino.git
git -C "$HOME/radar_dino" switch main
git -C "$HOME/radar_dino" pull --ff-only
```

### 2. Create and install the Python environment

The PBS jobs do not install packages. They expect a prepared Python environment
containing a CUDA-compatible PyTorch build and the Radar-DINO dependencies. On
Polaris, the intended starting point is the ALCF-provided Conda environment:

```bash
module use /soft/modulefiles
module load conda
conda activate base

python -m venv --system-site-packages /eagle/SSL-SULI2026/$USER/venvs/radar-dino
source /eagle/SSL-SULI2026/$USER/venvs/radar-dino/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "$HOME/radar_dino[netcdf,hub,analysis,plot,dev]"
python -m pip check
```

Using `--system-site-packages` allows the virtual environment to reuse the
site-supported GPU PyTorch build. On another cluster, use that site's
recommended CUDA/PyTorch module or container instead.

If dependencies were installed into a virtual environment, add its `source`
command after `conda activate` in each copied PBS file. Activating an
environment in the login shell does not automatically activate it in a later
batch job. Before a full run, verify imports and GPU visibility in an
interactive GPU allocation:

```bash
python -c "import torch, torchvision, xarray, radar_dino; print(torch.__version__)"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())"
```

### 3. Place data and outputs on cluster storage

Training recursively searches the data directory for `.nc` files. Keep the
repository in home storage if convenient, but place large NetCDF collections,
checkpoints, features, and plots on the site's project or scratch filesystem.
The Polaris defaults are:

```text
repository:  $HOME/radar_dino
data:        /eagle/SSL-SULI2026/$USER/gridnc_gt10pct
training:    /eagle/SSL-SULI2026/$USER/radar_train_fieldtoken
checkpoint:  /eagle/SSL-SULI2026/$USER/radar_train_fieldtoken/model/checkpoint.pth
features:    /eagle/SSL-SULI2026/$USER/radar/features_fieldtoken
catalog:     /eagle/SSL-SULI2026/$USER/radar/assoc_features_fieldtoken
```

Confirm that at least one input file is visible before submission:

```bash
find -L /path/to/gridnc -type f -name '*.nc' | wc -l
```

### 4. Adapt and submit the jobs

Copy the examples before changing site-specific PBS directives or environment
activation:

```bash
cp examples/pbs/train_radino.pbs train_radino.pbs
cp examples/pbs/infer_radino.pbs infer_radino.pbs
cp examples/pbs/assoc_infer_radino.pbs assoc_infer_radino.pbs
```

The data and output locations can be edited in those copies or supplied through
environment variables. For example:

```bash
qsub -v RADAR_DINO_REPO_DIR="$HOME/radar_dino",RADAR_DINO_DATA_DIR="/path/to/gridnc",RADAR_DINO_RUN_DIR="/path/to/training-run" \
  train_radino.pbs
```

For a first test, use the site's debug queue, a short walltime, one epoch, and a
small NetCDF subset. After it passes, submit the full training job. Monitor it
with the PBS tools and the log written under the run directory:

```bash
qstat -u "$USER"
qstat -f JOB_ID
tail -f /path/to/training-run/logs/JOB_ID.log
tail -f /path/to/training-run/model/log.txt
```

Training writes a rolling `checkpoint.pth`, numbered checkpoint snapshots, and
`log.txt`. Submitting the same training command with the same output directory
resumes from the rolling checkpoint, including the model, optimizer, epoch,
DINO loss, and mixed-precision scaler state.

After training, provide `RADAR_DINO_MODEL_PATH` when the checkpoint is not at
the Polaris default location:

```bash
qsub -v RADAR_DINO_REPO_DIR="$HOME/radar_dino",RADAR_DINO_DATA_DIR="/path/to/gridnc",RADAR_DINO_MODEL_PATH="/path/to/checkpoint.pth",RADAR_DINO_FEATURE_DIR="/path/to/features" \
  infer_radino.pbs

qsub -v RADAR_DINO_REPO_DIR="$HOME/radar_dino",RADAR_DINO_DATA_DIR="/path/to/gridnc",RADAR_DINO_MODEL_PATH="/path/to/checkpoint.pth",RADAR_DINO_ASSOC_FEATURE_DIR="/path/to/associated-features" \
  assoc_infer_radino.pbs
```

Regular inference saves `feat.pth` and limits attention output to 20 scans by
default. Change `RADAR_DINO_MAX_ATTENTION_PLOTS` to adjust the cap. Associative
inference saves `feat.pth`, `file_name.pth`, and `file_path.pth`; use those
matched outputs when constructing a searchable reference catalog.
