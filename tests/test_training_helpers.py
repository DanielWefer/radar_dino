from __future__ import annotations

import numpy as np
import pytest
import torch

import radar_dino_training
from radar_dino_training import RadarDINOLoss
from utils import bool_flag, cosine_scheduler


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("off", False), ("0", False)],
)
def test_bool_flag(value, expected):
    assert bool_flag(value) is expected


def test_cosine_scheduler_has_expected_length_and_endpoints():
    schedule = cosine_scheduler(
        base_value=1.0,
        final_value=0.0,
        epochs=2,
        niter_per_ep=5,
        warmup_epochs=1,
        start_warmup_value=0.2,
    )

    assert len(schedule) == 10
    assert schedule[0] == pytest.approx(0.2)
    assert schedule[4] == pytest.approx(1.0)
    assert np.all(np.diff(schedule[4:]) <= 0)


def test_radar_dino_loss_is_finite_and_updates_center(monkeypatch):
    monkeypatch.setattr(radar_dino_training.dist, "all_reduce", lambda tensor: None)
    monkeypatch.setattr(radar_dino_training.dist, "get_world_size", lambda: 1)
    loss_fn = RadarDINOLoss(
        out_dim=4,
        ncrops=4,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        nepochs=2,
    )
    student_output = torch.randn(8, 4, requires_grad=True)
    teacher_output = torch.randn(4, 4)

    loss = loss_fn(student_output, teacher_output, epoch=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert student_output.grad is not None
    assert not torch.equal(loss_fn.center, torch.zeros_like(loss_fn.center))

