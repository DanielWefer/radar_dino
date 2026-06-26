from __future__ import annotations

import pytest
import torch

from vision_transformer import PatchEmbed, VisionTransformer


pytestmark = pytest.mark.unit


def test_patch_embedding_keeps_input_channels_as_separate_token_groups():
    patch_embed = PatchEmbed(
        img_size=4,
        patch_size=2,
        in_chans=2,
        embed_dim=1,
    )
    with torch.no_grad():
        patch_embed.proj.weight.fill_(1.0)
        patch_embed.proj.bias.zero_()
        patch_embed.field_embed.zero_()

    channel_zero = torch.ones((1, 1, 4, 4))
    channel_one = torch.full((1, 1, 4, 4), 2.0)
    tokens = patch_embed(torch.cat([channel_zero, channel_one], dim=1))

    assert tokens.shape == (1, 8, 1)
    torch.testing.assert_close(tokens[:, :4], torch.full((1, 4, 1), 4.0))
    torch.testing.assert_close(tokens[:, 4:], torch.full((1, 4, 1), 8.0))


def test_field_embeddings_make_same_patch_values_field_specific():
    patch_embed = PatchEmbed(
        img_size=2,
        patch_size=2,
        in_chans=2,
        embed_dim=1,
    )
    with torch.no_grad():
        patch_embed.proj.weight.zero_()
        patch_embed.proj.bias.zero_()
        patch_embed.field_embed[0, 0, 0, 0] = 3.0
        patch_embed.field_embed[0, 1, 0, 0] = 7.0

    tokens = patch_embed(torch.zeros((1, 2, 2, 2)))

    torch.testing.assert_close(tokens, torch.tensor([[[3.0], [7.0]]]))


def test_patch_embedding_rejects_more_fields_than_configured():
    patch_embed = PatchEmbed(
        img_size=4,
        patch_size=2,
        in_chans=2,
        embed_dim=1,
    )

    with pytest.raises(ValueError, match="Expected at most 2 radar fields"):
        patch_embed(torch.zeros((1, 3, 4, 4)))


def test_vit_attention_uses_actual_field_count_when_fewer_than_configured():
    model = VisionTransformer(
        img_size=[20],
        patch_size=5,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
    )

    attention = model.get_last_selfattention(torch.randn(1, 2, 10, 15))

    token_count = 1 + 2 * (10 // 5) * (15 // 5)
    assert attention.shape == (1, 4, token_count, token_count)


def test_vit_supports_rectangular_radar_inputs_and_returns_cls_embedding():
    model = VisionTransformer(
        img_size=[20],
        patch_size=5,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
    )

    output = model(torch.randn(2, 3, 10, 15))

    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()


def test_last_self_attention_has_one_cls_plus_all_field_patch_tokens():
    model = VisionTransformer(
        img_size=[20],
        patch_size=5,
        in_chans=2,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
    )

    attention = model.get_last_selfattention(torch.randn(1, 2, 10, 15))

    token_count = 1 + 2 * (10 // 5) * (15 // 5)
    assert attention.shape == (1, 4, token_count, token_count)
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        atol=1e-5,
        rtol=1e-5,
    )

