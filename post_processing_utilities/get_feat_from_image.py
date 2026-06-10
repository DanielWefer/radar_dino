# Copyright (c) Northwestern Argonne Institute of Science and Engineering (NAISE)
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
from torch import nn

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
        raise FileNotFoundError("Please provide --pretrained_weights pointing to a Radar-DINO checkpoint.")

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


def save_channel_plot(channel, field_name, output_path, cmap):
    values = channel.copy()
    missing = values < 0.0
    values[missing] = np.nan
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="black")
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(values, cmap=cmap_obj, vmin=0.0, vmax=1.0, origin="upper")
    if missing.any():
        ax.contour(missing.astype(float), levels=[0.5], colors="white", linewidths=0.4)
    ax.set_title(field_name)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized value")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"{output_path} saved.")


def get_feat_from_radar(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(args, device)
    sample, sample_path = load_radar_sample(args)
    print(f"Loaded radar sample {sample_path} with shape {tuple(sample.shape)}")

    with torch.no_grad():
        features = model(sample.unsqueeze(0).to(device))
    features = nn.functional.normalize(features, dim=1, p=2)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(features.cpu(), os.path.join(args.output_dir, "feats.pt"))
    torch.save(sample.cpu(), os.path.join(args.output_dir, "radar_sample.pt"))
    sample_np = sample.cpu().numpy()
    for channel_index, field_name in enumerate(args.radar_fields):
        save_channel_plot(
            sample_np[channel_index],
            field_name,
            os.path.join(args.output_dir, f"channel-{channel_index}_{field_name}.png"),
            args.cmap,
        )
    print("Radar sample and its features saved.")


# Backwards-compatible function name for notebooks that imported the old helper.
def get_feat_from_image(args):
    return get_feat_from_radar(args)


def get_args_parser():
    parser = argparse.ArgumentParser('Get Radar-DINO features for one radar NetCDF sample')
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
        help='Center-crop radar grid before feature extraction, in grid cells.')
    parser.add_argument('--radar_fields', default=['reflectivity'], nargs='+', type=str,
        help='Radar variable names to stack as input channels.')
    parser.add_argument('--z_level', default='1000.0', type=str,
        help='Altitude in meters to select from 3D radar grids. Use --z_level none for column max.')
    parser.add_argument('--radar_nan_fill', default=-1.0, type=float,
        help='Normalized sentinel value assigned where a selected radar field contains NaNs.')
    parser.add_argument('--in_chans', default=None, type=int,
        help='Number of input channels for ViT patch embedding. Defaults to len(--radar_fields).')
    parser.add_argument('--sample_index', default=0, type=int,
        help='Sample index to process when --radar_path is a directory.')
    parser.add_argument('--output_dir', default='.', help='Path where to save the radar sample and its features.')
    parser.add_argument('--cmap', default='turbo', help='Matplotlib colormap for radar channels.')
    return parser


if __name__ == '__main__':
    get_feat_from_radar(get_args_parser().parse_args())
