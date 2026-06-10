import os
import sys
import argparse
import datetime
import time

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
    print("Extracting features and attentional maps ...")
    features, attentional_maps, selected_indices = extract_features(model, data_loader, args.use_cuda)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Inference time {}'.format(total_time_str))

    if utils.get_rank() == 0:
        features = nn.functional.normalize(features, dim=1, p=2)

    if args.dump_features and dist.get_rank() == 0:
        os.makedirs(args.dump_features, exist_ok=True)
        file_paths = [dataset.samples[index] for index in selected_indices]
        file_names = [os.path.basename(path) for path in file_paths]
        torch.save(features.cpu(), os.path.join(args.dump_features, "feat.pth"))
        torch.save(file_names, os.path.join(args.dump_features, "file_name.pth"))
        torch.save(file_paths, os.path.join(args.dump_features, "file_path.pth"))
        torch.save(attentional_maps.cpu(), os.path.join(args.dump_features, "att_map.pth"))
        print(f"Features and attentional maps with their corresponding radar file names are saved in {args.dump_features}!")
    return features


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    attentional_maps = None
    selected_mask = None
    selected_indices = None
    if args.inference_up_to:
        selected_mask = torch.zeros(len(data_loader.dataset), dtype=torch.bool)
        counter = 0

    printing = True
    for images, index in metric_logger.log_every(data_loader, 10):
        if args.inference_up_to and counter >= args.inference_up_to:
            break

        # move images to gpu
        images = images.cuda(non_blocking=True)
        if printing and dist.get_rank() == 0:
            printing = False
            print('radar tensor shape is ', images.shape)
        index = index.cuda(non_blocking=True)

        # forward pass
        feats = model(images).clone()

        # getting attentional maps
        local_attentional_maps = model.get_last_selfattention(images).clone()
        batch_size = local_attentional_maps.shape[0]
        local_attentional_maps = local_attentional_maps[:, :, 0, 1:].reshape(batch_size, -1)

        # init storage tensors
        if dist.get_rank() == 0 and features is None and attentional_maps is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            attentional_maps = torch.zeros(len(data_loader.dataset), local_attentional_maps.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
                attentional_maps = attentional_maps.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            print(f"Storing attentional maps into tensor of shape {attentional_maps.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)
        if args.inference_up_to:
            counter += (dist.get_world_size() * args.batch_size_per_gpu)
            selected_mask[index_all.cpu()] = True

        # share attentional maps between processes
        attentional_maps_all = torch.empty(
            dist.get_world_size(),
            local_attentional_maps.size(0),
            local_attentional_maps.size(1),
            dtype=local_attentional_maps.dtype,
            device=local_attentional_maps.device,
        )
        attentional_maps_output_l = list(attentional_maps_all.unbind(0))
        attentional_maps_output_all_reduce = torch.distributed.all_gather(attentional_maps_output_l, local_attentional_maps, async_op=True)
        attentional_maps_output_all_reduce.wait()

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

        # update storage tensors
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                attentional_maps.index_copy_(0, index_all, torch.cat(attentional_maps_output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                attentional_maps.index_copy_(0, index_all.cpu(), torch.cat(attentional_maps_output_l).cpu())

    if args.inference_up_to:
        selected_indices = selected_mask.nonzero(as_tuple=False).flatten().tolist()
        if dist.get_rank() == 0:
            features = features[selected_mask.to(features.device)]
            attentional_maps = attentional_maps[selected_mask.to(attentional_maps.device)]
    else:
        selected_indices = list(range(len(data_loader.dataset)))

    return features, attentional_maps, selected_indices


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
    def __getitem__(self, idx):
        img, _ = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def get_args_parser():
    parser = argparse.ArgumentParser('Associative attentional Radar-DINO inference using pretrained weights')
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
    parser.add_argument('--inference_up_to', default=None, type=int, help='Inference up to n samples from the complete dataset')
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        features = torch.load(os.path.join(args.load_features, "feat.pth"), map_location="cpu", weights_only=False)
    else:
        features = extract_feature_pipeline(args)

    if utils.get_rank() == 0 and args.use_cuda:
        features = features.cuda()

    dist.barrier()
