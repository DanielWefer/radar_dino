# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import utils
import vision_transformer as vits


class RadarCenterCropTransform(object):
    def __init__(self, crop_size, patch_size):
        self.crop_size = int(crop_size)
        self.patch_size = int(patch_size)

    def __call__(self, sample):
        if sample.ndim != 3:
            raise ValueError(f"Expected radar tensor with shape C x H x W, got {tuple(sample.shape)}")
        _, height, width = sample.shape
        crop_size = min(self.crop_size, height, width)
        crop_size = max(self.patch_size, (crop_size // self.patch_size) * self.patch_size)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        sample = sample[:, top:top + crop_size, left:left + crop_size]
        missing = sample < 0.0
        return torch.where(missing, sample, sample.clamp(0.0, 1.0))


def load_model(args, device):
    in_chans = args.in_chans if args.in_chans is not None else len(args.radar_fields)
    model = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0,
        in_chans=in_chans,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)

    if not os.path.isfile(args.pretrained_weights):
        raise FileNotFoundError(
            "Please use --pretrained_weights to provide a Radar-DINO checkpoint. "
            "The public RGB DINO weights are not compatible with radar channels."
        )

    state_dict = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
    if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
        print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[args.checkpoint_key]
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    state_dict = {key.replace("backbone.", ""): value for key, value in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Pretrained weights found at {args.pretrained_weights} and loaded with msg: {msg}")
    return model


def load_radar_sample(args):
    radar_path = args.radar_path or args.image_path
    if radar_path is None:
        raise ValueError("Please provide --radar_path pointing to a NetCDF file or directory.")

    crop_size = args.image_size[0] if isinstance(args.image_size, list) else args.image_size
    z_level = None if args.z_level.lower() == "none" else float(args.z_level)
    dataset = utils.UnlabeledRadarNetCDFDataset(
        radar_path,
        fields=args.radar_fields,
        z_level=z_level,
        transform=RadarCenterCropTransform(crop_size, args.patch_size),
        nan_fill=args.radar_nan_fill,
        return_paths=True,
    )
    sample, _, path = dataset[args.sample_index]
    return sample, path


def resize_attention(attentions, height, width):
    return nn.functional.interpolate(
        attentions.unsqueeze(0),
        size=(height, width),
        mode="nearest",
    )[0].cpu().numpy()


def threshold_attention(attentions, threshold, feat_height, feat_width, height, width):
    val, idx = torch.sort(attentions)
    val = val / torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(attentions.shape[0]):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(attentions.shape[0], feat_height, feat_width).float()
    return resize_attention(th_attn, height, width)


def field_cls_attention_maps(attentions, num_fields, feat_height, feat_width, height, width):
    num_heads = attentions.shape[1]
    expected_tokens = num_fields * feat_height * feat_width
    cls_attention = attentions[0, :, 0, 1:]
    if cls_attention.shape[-1] != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} field-patch tokens, got {cls_attention.shape[-1]}"
        )
    field_attentions = cls_attention.reshape(num_heads, num_fields, feat_height, feat_width)
    return resize_attention(
        field_attentions.reshape(num_heads * num_fields, feat_height, feat_width),
        height,
        width,
    ).reshape(num_heads, num_fields, height, width)


def channel_display_values(channel):
    channel = channel.copy()
    missing = channel < 0.0
    channel[missing] = np.nan
    return channel, missing


def bundled_chasespectral_colormap():
    """Load the ChaseSpectral RGB table bundled from Py-ART/cmweather."""
    from matplotlib.colors import LinearSegmentedColormap

    rgb_path = os.path.join(REPO_ROOT, "src", "radar_dino", "data", "chase-spectral-rgb.txt")
    rgb_values = np.loadtxt(rgb_path)
    return LinearSegmentedColormap.from_list("ChaseSpectral", rgb_values)


def pyart_chasespectral_colormap():
    """Return Py-ART's ChaseSpectral colormap across Py-ART versions."""
    try:
        import pyart
    except ImportError:
        # Current Py-ART versions source their colormaps from cmweather. This
        # fallback also supports environments that install that dependency
        # separately from the full radar I/O stack.
        try:
            from cmweather import cm_colorblind
        except ImportError:
            return bundled_chasespectral_colormap()
        return cm_colorblind.ChaseSpectral

    colorblind_module = getattr(pyart.graph, "cm_colorblind", None)
    if colorblind_module is not None and hasattr(colorblind_module, "ChaseSpectral"):
        return colorblind_module.ChaseSpectral

    cmweather_module = getattr(pyart.graph, "cmweather", None)
    if cmweather_module is not None:
        return cmweather_module.cm_colorblind.ChaseSpectral

    for cmap_name in ("ChaseSpectral", "pyart_ChaseSpectral"):
        try:
            return plt.get_cmap(cmap_name)
        except ValueError:
            pass
    return bundled_chasespectral_colormap()


def save_reflectivity_attention_multiplot(reflectivity, attentions, output_path):
    attentions = np.asarray(attentions)
    if attentions.ndim != 3 or attentions.shape[0] != 6:
        raise ValueError(
            "The 4x2 reflectivity multiplot requires exactly six attention "
            f"heads with shape (6, H, W), got {attentions.shape}."
        )

    values, missing = channel_display_values(reflectivity)
    reflectivity_cmap = pyart_chasespectral_colormap().copy()
    reflectivity_cmap.set_bad(color="black")
    attention_cmap = plt.get_cmap("magma").copy()
    attention_cmap.set_bad(color="black")

    finite_attention = attentions[np.isfinite(attentions)]
    if finite_attention.size == 0:
        raise ValueError("Attention maps contain no finite values to plot.")
    attention_max = float(finite_attention.max())
    if attention_max <= 0.0:
        attention_max = 1.0

    fig, axes = plt.subplots(4, 2, figsize=(12, 18), constrained_layout=True)
    axes = axes.ravel()

    reflectivity_image = axes[0].imshow(
        values,
        cmap=reflectivity_cmap,
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    if missing.any():
        axes[0].contour(
            missing.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.4,
        )
    axes[0].set_title("Reflectivity")
    axes[0].set_axis_off()
    fig.colorbar(
        reflectivity_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="normalized reflectivity",
    )

    for head in range(6):
        ax = axes[head + 1]
        attention_image = ax.imshow(
            attentions[head],
            cmap=attention_cmap,
            vmin=0.0,
            vmax=attention_max,
            origin="upper",
        )
        ax.set_title(f"Attention head {head}")
        ax.set_axis_off()
        fig.colorbar(
            attention_image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="attention",
        )

    axes[7].set_axis_off()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"{output_path} saved.")


def visualize_attention(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(args, device)
    sample, sample_path = load_radar_sample(args)
    print(f"Loaded radar sample {sample_path} with shape {tuple(sample.shape)}")

    height, width = sample.shape[-2:]
    img = sample.unsqueeze(0)
    feat_height = height // args.patch_size
    feat_width = width // args.patch_size

    with torch.no_grad():
        attentions = model.get_last_selfattention(img.to(device))

    num_heads = attentions.shape[1]
    num_fields = sample.shape[0]
    upsampled_attentions = field_cls_attention_maps(
        attentions,
        num_fields,
        feat_height,
        feat_width,
        height,
        width,
    )

    threshold_masks = None
    if args.threshold is not None:
        attentions = attentions[0, :, 0, 1:].reshape(num_heads, num_fields, feat_height, feat_width)
        flat_attentions = attentions.reshape(num_heads * num_fields, -1)
        threshold_masks = threshold_attention(
            flat_attentions,
            args.threshold,
            feat_height,
            feat_width,
            height,
            width,
        ).reshape(num_heads, num_fields, height, width)

    os.makedirs(args.output_dir, exist_ok=True)
    sample_np = sample.cpu().numpy()
    torch.save(sample.cpu(), os.path.join(args.output_dir, "radar_sample.pt"))
    torch.save(torch.from_numpy(upsampled_attentions), os.path.join(args.output_dir, "attention_maps.pt"))
    if threshold_masks is not None:
        torch.save(torch.from_numpy(threshold_masks), os.path.join(args.output_dir, "masks_th.pt"))

    if "reflectivity" not in args.radar_fields:
        raise ValueError("The attention multiplot requires reflectivity in --radar_fields.")
    reflectivity_index = args.radar_fields.index("reflectivity")
    multiplot_path = os.path.join(args.output_dir, "reflectivity_attention_heads.png")
    save_reflectivity_attention_multiplot(
        sample_np[reflectivity_index],
        upsampled_attentions[:, reflectivity_index],
        multiplot_path,
    )


def get_args_parser():
    parser = argparse.ArgumentParser('Visualize Radar-DINO self-attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=5, type=int, help='Patch size in radar grid cells.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help='Path to pretrained Radar-DINO weights to load.')
    parser.add_argument('--checkpoint_key', default='teacher', type=str,
        help='Key to use in the checkpoint (example: teacher).')
    parser.add_argument('--radar_path', default=None, type=str,
        help='Path to a radar NetCDF file or directory of NetCDF files.')
    parser.add_argument('--image_path', default=None, type=str,
        help='Deprecated alias for --radar_path.')
    parser.add_argument('--image_size', default=[300], type=int, nargs='+',
        help='Center-crop radar grid before attention extraction, in grid cells.')
    parser.add_argument('--radar_fields', default=['reflectivity'], nargs='+', type=str,
        help='Radar variable names to stack as input channels.')
    parser.add_argument('--z_level', default='2000.0', type=str,
        help='Altitude in meters to select from 3D radar grids. Use --z_level none for column max.')
    parser.add_argument('--radar_nan_fill', default=-1.0, type=float,
        help='Normalized sentinel value assigned where a selected radar field contains NaNs.')
    parser.add_argument('--in_chans', default=None, type=int,
        help='Number of input channels for ViT patch embedding. Defaults to len(--radar_fields).')
    parser.add_argument('--sample_index', default=0, type=int,
        help='Sample index to visualize when --radar_path is a directory.')
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument('--threshold', type=float, default=None, help='Keep this fraction of attention mass as binary masks.')
    return parser


if __name__ == '__main__':
    visualize_attention(get_args_parser().parse_args())
