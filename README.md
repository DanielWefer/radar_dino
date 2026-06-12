# Self-Supervised Vision Transformers for Radar Grid Learning Radar-DINO

PyTorch implementation for Radar-DINO.

This project is based in the [original DINO](https://github.com/facebookresearch/dino) repository by [Facebook AI research group](https://ai.facebook.com/).

## For training Radar-DINO

`python3 -m torch.distributed.launch --nproc_per_node=1 radar_dino_training.py --data_path ../radar_dino_test/KHTX/gridnc --output_dir /path/to/your/model/ --use_fp16 false`

By default, Radar-DINO uses 1-km reflectivity, clips reflectivity to 10-75 dBZ, maps original NaNs and reflectivity outside 10-75 dBZ to -1.0 after normalization across all channels, and tokenizes with 5x5 grid-cell patches. On the KHTX 1 km grid, this gives 5x5 km ViT patches. Training also uses --channel_nan_prob 0.1 to randomly set one crop channel to the NaN sentinel.

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

For attention maps visualiztions, run the following command

`python3 visualize_attention.py --pretrained_weights /path/to/your/pretrained/weights/checkpoint.pth --image_path /path/to/the/image.jpg --image_size size_of_the_image --output_dir /where/to/save/the/maps/ --threshold (a number between 0 and 1) --patch_size 8-16`
