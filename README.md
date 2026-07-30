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

## For training Radar-DINO

`python3 -m torch.distributed.launch --nproc_per_node=1 radar_dino_training.py --data_path ../radar_dino_test/KHTX/gridnc --output_dir /path/to/your/model/ --use_fp16 false`

By default, Radar-DINO selects reflectivity at 2 km altitude, clips reflectivity to 10-75 dBZ, maps original NaNs and reflectivity outside 10-75 dBZ to -1.0 after normalization across all channels, and tokenizes with 5x5 grid-cell patches. On the KHTX 1 km horizontal grid, this gives 5x5 km ViT patches. On the field-token `wip` architecture, `--channel_nan_prob` controls asymmetric student-field masking while teacher crops retain all fields.

The released `radar-dino-fieldtoken-v1` artifact has its own versioned contract:
a 300x300 input, 10x10 grid-cell patches, five fields, and a positional table
initialized at the DINO default size and interpolated during inference. The
artifact manifest, rather than the training CLI defaults, is authoritative.

## Running inference to obtain Radar-DINO's features

`python3 -m torch.distributed.launch --nproc_per_node=1 radar_dino_inference.py --data_path ../radar_dino_test/KHTX/gridnc --pretrained_weights /path/to/your/model/checkpoint0000.pth --dump_features /path/to/your/features/`

Inference saves reflectivity attention overlays for every head plus a mean-attention two-panel plot to `/path/to/your/features/attention_overlays` by default. Use `--plot_attention_overlays false` to disable this for large runs.

## Running associative inference to obtain Radar-DINO's features and their respective file input names

Here we are truncating the inference ptocess to only 100 samples. That means that after the first 100 samples the inference process will be stoped.

`python3 -m torch.distributed.launch --nproc_per_node=1 radar_dino_associative_inference.py --data_path ../radar_dino_test/KHTX/gridnc --pretrained_weights /path/to/your/model/checkpoint0000.pth --dump_features /path/to/your/features/ --inference_up_to 100`

## Training Radar-DINO in a node on 8 GPUs

To run it during 10 min do

`qsub -n 1 -q full-node -t 10 -A your_project ./train_Radar_DINO.sh`

This is the `train_Radar_DINO.sh` script for training Radar-DINO in a node with 8 GPUs

```
#!/bin/sh

# Common paths
radar_gridnc_path='/path/to/KHTX/gridnc'
singularity_image_path='/path/to/the/singularity/container/your_singularity_image_file.sif'
radar_dino_path='/path/to/radar_dino'
train_radar_dino_path='/path/to/radar_dino/radar_dino_training.py'
model_path='/path/to/the/model'

cd $radar_dino_path
singularity exec --nv -B $radar_gridnc_path:/RadarGridNC $singularity_image_path python -m torch.distributed.launch --nproc_per_node=8 $train_radar_dino_path --arch vit_small --data_path /RadarGridNC --output_dir $model_path
```


## Inferencing using a Radar-DINO trained model on 8 GPUs

To run it during 10 min do

`qsub -n 1 -q full-node -t 10 -A your_project ./inference_Radar_DINO.sh`

This is the `inference_Radar_DINO.sh` script for making inference on Radar-DINO in a node with 8 GPUs

```
#!/bin/sh

# Common paths
radar_gridnc_path='/path/to/KHTX/gridnc'
singularity_image_path='/path/to/the/singularity/container/your_singularity_image_file.sif'
radar_dino_path='/path/to/radar_dino'
inference_radar_dino_path='/path/to/radar_dino/radar_dino_inference.py'
model_path='/path/to/the/model/checkpoint0000.pth'
output_path='/path/to/output/features'

cd $radar_dino_path
singularity exec --nv -B $radar_gridnc_path:/RadarGridNC $singularity_image_path python -m torch.distributed.launch --nproc_per_node=8 $inference_radar_dino_path --data_path /RadarGridNC --pretrained_weights $model_path --dump_features $output_path
```


## For visualizing attention maps

The public package API shown above is the recommended path. The legacy training
checkpoint utility can also produce the six-head ChaseSpectral reflectivity
multiplot directly:

```bash
python3 post_processing_utilities/visualize_attention.py \
  --pretrained_weights /path/to/checkpoint.pth \
  --radar_path /path/to/scan.nc \
  --radar_fields reflectivity \
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
