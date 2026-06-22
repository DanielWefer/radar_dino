from __future__ import annotations

import pytest
import torch

from radar_dino_training import DataAugmentationRadarDINO
from utils import UnlabeledRadarNetCDFDataset
from vision_transformer import VisionTransformer


pytestmark = pytest.mark.integration


def test_synthetic_radar_file_flows_through_augmentation_and_vit(radar_netcdf):
    dataset = UnlabeledRadarNetCDFDataset(
        str(radar_netcdf),
        fields=("reflectivity", "differential_reflectivity"),
        z_level=2000.0,
    )
    augmentation = DataAugmentationRadarDINO(
        global_crop_size_km=2.0,
        local_crop_size_km=2.0,
        grid_spacing_km=1.0,
        patch_size=1,
        local_crops_number=0,
        nan_fill=-1.0,
        channel_nan_prob=0.0,
    )
    model = VisionTransformer(
        img_size=[2],
        patch_size=1,
        in_chans=2,
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
    )

    sample, _ = dataset[0]
    crops = augmentation(sample)
    output = model(torch.stack(crops))

    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
