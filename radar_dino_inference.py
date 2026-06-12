import os
import sys
import argparse
import datetime
import time
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    crop_size = args.image_size[0] if isinstance(args.image_size, list) else args.image_size
    transform = RadarCenterCropTransform(crop_size, args.patch_size)
    z_level = None if args.z_level.lower() == "none" else float(args.z_level)
    dataset = ReturnIndexDataset(
        args.data_path,
        fields=args.radar_fields,
        z_level=z_level,
        transform=transform,
        nan_fill=args.radar_nan_fill,
    )

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} radar NetCDF files.")

    # ============ building network ... ============
    if "vit" in args.arch:
        if args.in_chans is None:
            args.in_chans = len(args.radar_fields)
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=args.in_chans)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    start_time = time.time()
    # ============ extract features ... ============
    print("Extracting features ...")
    features = extract_features(model, data_loader, args.use_cuda)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Inference time {}'.format(total_time_str))

    if utils.get_rank() == 0:
        features = nn.functional.normalize(features, dim=1, p=2)

    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        os.makedirs(args.dump_features, exist_ok=True)
        torch.save(features.cpu(), os.path.join(args.dump_features, "feat.pth"))
        print(f"Features are saved in {args.dump_features}!")
    return features


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    if args.inference_up_to:
        cumulative_indexes = torch.zeros(len(data_loader.dataset), dtype=torch.bool)
        counter = 0

    for images, index, path in metric_logger.log_every(data_loader, 10):
        if args.inference_up_to and counter >= args.inference_up_to:
            break

        # move images to gpu
        images = images.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward pass
        feats = model(images).clone()
        save_reflectivity_attention_overlays(model, images, index, path)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)
        if args.inference_up_to:
            counter += (dist.get_world_size() * args.batch_size_per_gpu)
            cumulative_indexes[index_all] = True

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())

    if args.inference_up_to:
        features = features[cumulative_indexes]

    return features


def attention_output_dir():
    if args.attention_output_dir:
        return args.attention_output_dir
    if args.dump_features:
        return os.path.join(args.dump_features, "attention_overlays")
    return None


@torch.no_grad()
def save_reflectivity_attention_overlays(model, images, indices, paths):
    if not args.plot_attention_overlays:
        return
    output_dir = attention_output_dir()
    if output_dir is None:
        return
    if "reflectivity" not in args.radar_fields:
        if dist.get_rank() == 0:
            print("Skipping attention overlays because 'reflectivity' is not in --radar_fields.")
        return

    reflectivity_channel = args.radar_fields.index("reflectivity")
    os.makedirs(output_dir, exist_ok=True)
    attentions = model.get_last_selfattention(images)
    batch_size, num_heads = attentions.shape[:2]
    height, width = images.shape[-2:]
    feat_height = height // args.patch_size
    feat_width = width // args.patch_size
    attentions = attentions[:, :, 0, 1:].reshape(batch_size, num_heads, feat_height, feat_width)
    mean_attentions = attentions.mean(dim=1)
    all_attentions = torch.cat([attentions, mean_attentions.unsqueeze(1)], dim=1)
    all_attentions = nn.functional.interpolate(
        all_attentions,
        size=(height, width),
        mode="nearest",
    ).detach().cpu().numpy()
    reflectivity = images[:, reflectivity_channel].detach().cpu().numpy()

    labels = [f"head{head}" for head in range(num_heads)] + ["mean"]
    titles = [f"reflectivity + attention head {head}" for head in range(num_heads)] + ["reflectivity and mean attention"]
    for batch_pos, sample_index in enumerate(indices.detach().cpu().tolist()):
        if args.max_attention_plots is not None and sample_index >= args.max_attention_plots:
            continue
        basename = os.path.splitext(os.path.basename(paths[batch_pos]))[0]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename)
        for attention_index, label in enumerate(labels):
            output_path = os.path.join(
                output_dir,
                f"idx{sample_index:06d}_{safe_name}_reflectivity_attn-{label}.png",
            )
            if label == "mean":
                save_reflectivity_mean_attention_panel(
                    reflectivity[batch_pos],
                    all_attentions[batch_pos, attention_index],
                    output_path,
                    titles[attention_index],
                )
            else:
                save_reflectivity_attention_overlay(
                    reflectivity[batch_pos],
                    all_attentions[batch_pos, attention_index],
                    output_path,
                    titles[attention_index],
                    label,
                )


def reflectivity_dbz_for_display(reflectivity):
    values = reflectivity.copy()
    missing = values < 0.0
    below_floor = values <= 0.0
    low, high = utils.RADAR_FIELD_CLIPS["reflectivity"]
    values = low + values * (high - low)
    mask = missing | below_floor
    values[mask] = np.nan
    return values, mask, low, high


def save_reflectivity_attention_overlay(reflectivity, attention, output_path, title, attention_label):
    reflectivity_dbz, missing, low, high = reflectivity_dbz_for_display(reflectivity)
    attention = normalize_for_display(attention)

    reflectivity_cmap = plt.get_cmap(args.reflectivity_cmap).copy()
    reflectivity_cmap.set_bad(color="black")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    radar_im = axes[0].imshow(
        reflectivity_dbz,
        cmap=reflectivity_cmap,
        vmin=low,
        vmax=high,
        origin="upper",
    )
    if missing.any():
        axes[0].contour(missing.astype(float), levels=[0.5], colors="white", linewidths=0.4)
    axes[0].set_title("reflectivity")
    axes[0].set_axis_off()
    fig.colorbar(radar_im, ax=axes[0], fraction=0.046, pad=0.04, label="reflectivity (dBZ)")

    attention_im = axes[1].imshow(
        attention,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    axes[1].set_title(attention_label)
    axes[1].set_axis_off()
    fig.colorbar(attention_im, ax=axes[1], fraction=0.046, pad=0.04, label=attention_label)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"{output_path} saved.")


def save_reflectivity_mean_attention_panel(reflectivity, attention, output_path, title):
    reflectivity_dbz, missing, low, high = reflectivity_dbz_for_display(reflectivity)
    attention = normalize_for_display(attention)

    reflectivity_cmap = plt.get_cmap(args.reflectivity_cmap).copy()
    reflectivity_cmap.set_bad(color="black")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    radar_im = axes[0].imshow(
        reflectivity_dbz,
        cmap=reflectivity_cmap,
        vmin=low,
        vmax=high,
        origin="upper",
    )
    if missing.any():
        axes[0].contour(missing.astype(float), levels=[0.5], colors="white", linewidths=0.4)
    axes[0].set_title("reflectivity")
    axes[0].set_axis_off()
    fig.colorbar(radar_im, ax=axes[0], fraction=0.046, pad=0.04, label="reflectivity (dBZ)")

    attention_im = axes[1].imshow(
        attention,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    axes[1].set_title("mean attention")
    axes[1].set_axis_off()
    fig.colorbar(attention_im, ax=axes[1], fraction=0.046, pad=0.04, label="mean attention")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"{output_path} saved.")


def normalize_for_display(array):
    array = np.asarray(array, dtype=np.float32)
    low = np.nanmin(array)
    high = np.nanmax(array)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


class RadarCenterCropTransform(object):
    def __init__(self, crop_size, patch_size):
        self.crop_size = int(crop_size)
        self.patch_size = int(patch_size)

    def __call__(self, image):
        if image.ndim != 3:
            raise ValueError(f"Expected radar tensor with shape C x H x W, got {tuple(image.shape)}")
        _, height, width = image.shape
        crop_size = min(self.crop_size, height, width)
        crop_size = max(self.patch_size, (crop_size // self.patch_size) * self.patch_size)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        image = image[:, top:top + crop_size, left:left + crop_size]
        missing = image < 0.0
        return torch.where(missing, image, image.clamp(0.0, 1.0))


class ReturnIndexDataset(utils.UnlabeledRadarNetCDFDataset):
    def __init__(self, root, fields=("reflectivity",), z_level=1000.0, transform=None, nan_fill=-1.0):
        super().__init__(root, fields=fields, z_level=z_level, transform=transform, nan_fill=nan_fill, return_paths=True)

    def __getitem__(self, idx):
        img, _, path = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx, path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference using pretrained weights')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int, help='Per-GPU batch-size')
    parser.add_argument("--image_size", default=[300], type=int, nargs="+", help="Center-crop radar grid before feature extraction, in 1 km grid cells by default.")
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=5, type=int, help='Patch size in grid cells; default 5 is 5x5 km on the KHTX 1 km grid.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='../radar_dino_test/KHTX/gridnc', type=str)
    parser.add_argument('--radar_fields', default=['reflectivity'], nargs='+', type=str,
        help='Radar variable names to stack as input channels.')
    parser.add_argument('--z_level', default='1000.0', type=str,
        help='Altitude in meters to select from 3D radar grids. Use --z_level none for column max.')
    parser.add_argument('--radar_nan_fill', default=-1.0, type=float,
        help='Normalized sentinel value assigned where a selected radar field contains NaNs.')
    parser.add_argument('--in_chans', default=None, type=int,
        help='Number of input channels for ViT patch embedding. Defaults to len(--radar_fields).')
    parser.add_argument('--plot_attention_overlays', default=True, type=utils.bool_flag,
        help='Save reflectivity plots with each attention head and mean last-layer attention overlaid during inference.')
    parser.add_argument('--attention_output_dir', default=None, type=str,
        help='Directory for attention overlay plots. Defaults to dump_features/attention_overlays.')
    parser.add_argument('--max_attention_plots', default=None, type=int,
        help='Only save attention overlays for dataset indices below this value. Default saves all processed samples.')
    parser.add_argument('--attention_alpha', default=0.45, type=float,
        help='Alpha blending for the mean-attention overlay.')
    parser.add_argument('--reflectivity_cmap', default='turbo', type=str,
        help='Matplotlib colormap for the reflectivity background.')
    parser.add_argument('--inference_up_to', default=None, type=int, help='Inference up to n samples from the complete dataset')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        features = torch.load(os.path.join(args.load_features, "feat.pth"), map_location="cpu", weights_only=False)
    else:
        # need to extract features !
        features = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            features = features.cuda()

    dist.barrier()
