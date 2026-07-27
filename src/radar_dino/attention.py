"""Field-aware transformations for last-layer transformer attention."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def field_attention_maps(
    attention: torch.Tensor,
    *,
    num_fields: int,
    patch_grid: tuple[int, int],
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Return CLS attention as ``batch x heads x fields x height x width``."""

    if attention.ndim != 4:
        raise ValueError(
            "Expected attention with shape batch x heads x tokens x tokens, "
            f"got {tuple(attention.shape)}"
        )
    patch_height, patch_width = patch_grid
    expected_patch_tokens = num_fields * patch_height * patch_width
    cls_attention = attention[:, :, 0, 1:]
    if cls_attention.shape[-1] != expected_patch_tokens:
        raise ValueError(
            f"Expected {expected_patch_tokens} field-patch tokens, "
            f"got {cls_attention.shape[-1]}"
        )

    batch_size, num_heads = cls_attention.shape[:2]
    maps = cls_attention.reshape(
        batch_size,
        num_heads,
        num_fields,
        patch_height,
        patch_width,
    )
    if output_size is None or output_size == patch_grid:
        return maps

    maps = maps.reshape(
        batch_size * num_heads * num_fields,
        1,
        patch_height,
        patch_width,
    )
    maps = F.interpolate(maps, size=output_size, mode="nearest")
    return maps.reshape(batch_size, num_heads, num_fields, *output_size)
