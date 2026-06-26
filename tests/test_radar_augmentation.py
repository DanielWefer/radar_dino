from __future__ import annotations

import pytest
import torch

from radar_dino_training import DataAugmentationRadarDINO, mask_random_student_channels


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



def test_augmentation_keeps_teacher_candidate_crops_full_field(monkeypatch):
    augmentation = make_augmentation(channel_nan_prob=1.0)
    image = torch.full((3, 30, 30), 0.5)
    monkeypatch.setattr(augmentation, "_random_flip", lambda crop: crop)

    crops = augmentation(image)

    assert all(torch.all(crop != -1.0) for crop in crops)


def test_student_channel_mask_clones_and_masks_each_batch_sample(monkeypatch):
    crop = torch.zeros((2, 3, 4, 4))
    monkeypatch.setattr("radar_dino_training.random.randrange", lambda _: 1)

    masked = mask_random_student_channels([crop], nan_fill=-1.0, channel_nan_prob=1.0)

    assert len(masked) == 1
    assert torch.all(masked[0][:, 1] == -1.0)
    assert torch.all(masked[0][:, 0] == 0.0)
    assert torch.all(masked[0][:, 2] == 0.0)
    assert torch.all(crop == 0.0)


def test_student_channel_mask_can_leave_images_unchanged():
    crop = torch.zeros((2, 3, 4, 4))

    masked = mask_random_student_channels([crop], nan_fill=-1.0, channel_nan_prob=0.0)

    assert masked[0] is crop


def test_student_channel_mask_respects_per_sample_probability(monkeypatch):
    crop = torch.zeros((3, 2, 2, 2))
    random_values = iter([0.9, 0.1, 0.8])
    monkeypatch.setattr("radar_dino_training.random.random", lambda: next(random_values))
    monkeypatch.setattr("radar_dino_training.random.randrange", lambda _: 0)

    masked = mask_random_student_channels([crop], nan_fill=-1.0, channel_nan_prob=0.5)[0]

    assert torch.all(masked[0] == 0.0)
    assert torch.all(masked[1, 0] == -1.0)
    assert torch.all(masked[1, 1] == 0.0)
    assert torch.all(masked[2] == 0.0)


def test_student_channel_mask_rejects_unbatched_crops():
    with pytest.raises(ValueError, match="B x C x H x W"):
        mask_random_student_channels([torch.zeros((2, 4, 4))], nan_fill=-1.0, channel_nan_prob=1.0)
