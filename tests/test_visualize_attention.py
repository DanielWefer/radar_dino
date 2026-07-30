from __future__ import annotations

import numpy as np
import pytest
import torch

from post_processing_utilities.visualize_attention import field_cls_attention_maps


pytestmark = pytest.mark.unit


def test_field_cls_attention_maps_preserves_head_and_field_axes():
    attentions = torch.zeros((1, 1, 9, 9))
    attentions[0, 0, 0, 1:] = torch.arange(8, dtype=torch.float32)

    maps = field_cls_attention_maps(
        attentions,
        num_fields=2,
        feat_height=2,
        feat_width=2,
        height=4,
        width=4,
    )

    assert maps.shape == (1, 2, 4, 4)
    np.testing.assert_array_equal(
        maps[0, 0],
        np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        maps[0, 1],
        np.array([
            [4, 4, 5, 5],
            [4, 4, 5, 5],
            [6, 6, 7, 7],
            [6, 6, 7, 7],
        ], dtype=np.float32),
    )


def test_field_cls_attention_maps_rejects_unexpected_token_count():
    attentions = torch.zeros((1, 1, 8, 8))

    with pytest.raises(ValueError, match="Expected 8 field-patch tokens, got 7"):
        field_cls_attention_maps(
            attentions,
            num_fields=2,
            feat_height=2,
            feat_width=2,
            height=4,
            width=4,
        )
