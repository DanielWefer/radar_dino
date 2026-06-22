from __future__ import annotations

import pytest
import torch

from radar_dino_training import DataAugmentationRadarDINO


pytestmark = pytest.mark.unit


def make_augmentation(**overrides):
    defaults = {
        "global_crop_size_km": 20.0,
        "local_crop_size_km": 10.0,
        "grid_spacing_km": 1.0,
        "patch_size": 5,
        "local_crops_number": 2,
        "nan_fill": -1.0,
        "channel_nan_prob": 0.0,
    }
    defaults.update(overrides)
    return DataAugmentationRadarDINO(**defaults)


def test_crop_sizes_align_to_patch_size():
    augmentation = make_augmentation(
        global_crop_size_km=23.0,
        local_crop_size_km=12.0,
    )

    assert augmentation.global_crop_size == 20
    assert augmentation.local_crop_size == 10


def test_augmentation_returns_two_global_and_requested_local_crops(monkeypatch):
    augmentation = make_augmentation()
    image = torch.full((3, 30, 30), 0.5)
    monkeypatch.setattr(augmentation, "_random_flip", lambda crop: crop)

    crops = augmentation(image)

    assert len(crops) == 4
    assert [tuple(crop.shape) for crop in crops] == [
        (3, 20, 20),
        (3, 20, 20),
        (3, 10, 10),
        (3, 10, 10),
    ]
    assert all(torch.all((crop == -1.0) | ((crop >= 0.0) & (crop <= 1.0))) for crop in crops)


def test_random_channel_mask_replaces_exactly_one_channel(monkeypatch):
    augmentation = make_augmentation(channel_nan_prob=1.0)
    crop = torch.zeros((3, 4, 4))
    monkeypatch.setattr("radar_dino_training.random.random", lambda: 0.0)
    monkeypatch.setattr("radar_dino_training.random.randrange", lambda _: 1)

    masked = augmentation._random_channel_nan(crop)

    assert torch.all(masked[1] == -1.0)
    assert torch.all(masked[0] == 0.0)
    assert torch.all(masked[2] == 0.0)
    assert torch.all(crop == 0.0)


def test_random_crop_rejects_non_channel_first_input():
    augmentation = make_augmentation()

    with pytest.raises(ValueError, match="C x H x W"):
        augmentation._random_crop(torch.zeros((10, 10)), crop_size=5)

