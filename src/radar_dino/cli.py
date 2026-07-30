"""Command-line entry point for packaged Radar-DINO inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifact import export_checkpoint
from .config import RadarDINOConfig
from .model import RadarDINO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="radar-dino")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze one radar NetCDF file")
    analyze.add_argument("netcdf", type=Path)
    analyze.add_argument("--model", required=True, help="Artifact directory or Hub repository")
    analyze.add_argument("--revision", default=None)
    analyze.add_argument("--device", default="auto")
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--no-attention", action="store_true")
    analyze.add_argument(
        "--plots",
        action="store_true",
        help="Save attention, UMAP, and t-SNE PNG files",
    )
    analyze.add_argument("--plot-dpi", type=int, default=200)

    export = subparsers.add_parser(
        "export",
        help="Export a trusted training checkpoint as an inference artifact",
    )
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-unsafe-pickle", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        config = RadarDINOConfig.from_json(args.manifest)
        export_checkpoint(
            args.checkpoint,
            args.output,
            config,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        )
        return 0
    if args.command != "analyze":
        raise AssertionError(f"Unhandled command: {args.command}")

    dino = RadarDINO.from_pretrained(
        args.model,
        device=args.device,
        revision=args.revision,
    )
    result = dino.analyze(
        args.netcdf,
        include_attention=not args.no_attention,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "feature.npy", result.feature)
    if result.attention is not None:
        np.save(args.output / "attention.npy", result.attention)
    if result.umap is not None:
        np.save(args.output / "umap.npy", result.umap)
    if result.tsne is not None:
        np.save(args.output / "tsne.npy", result.tsne)
    plot_files = (
        dino.save_plots(result, args.output, dpi=args.plot_dpi)
        if args.plots
        else {}
    )
    summary = {
        "model_id": dino.config.model_id,
        "source": str(result.path),
        "fields": list(result.fields),
        "feature_shape": list(result.feature.shape),
        "attention_shape": (
            None if result.attention is None else list(result.attention.shape)
        ),
        "umap": None if result.umap is None else result.umap.tolist(),
        "tsne": None if result.tsne is None else result.tsne.tolist(),
        "tsne_note": (
            None
            if result.tsne is None
            else "Display-only interpolation from nearest reference scans; "
            "cluster assignment is not performed in t-SNE space."
        ),
        "cluster": result.cluster,
        "cluster_probability": result.cluster_probability,
        "neighbors": list(result.neighbors),
        "plots": {name: path.name for name, path in plot_files.items()},
    }
    with (args.output / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
