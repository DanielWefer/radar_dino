"""Export trusted training checkpoints into inference-only model artifacts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from . import vision_transformer
from .config import RadarDINOConfig
from .model import _load_checkpoint


def export_checkpoint(
    checkpoint_path: str | Path,
    output_directory: str | Path,
    config: RadarDINOConfig,
    *,
    allow_unsafe_pickle: bool = False,
) -> Path:
    """Validate and export only backbone tensors plus a plain JSON manifest."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "Exporting model artifacts requires the 'hub' extra"
        ) from exc

    factory = getattr(vision_transformer, config.architecture, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Unsupported architecture '{config.architecture}'")
    network = factory(
        patch_size=config.patch_size,
        num_classes=0,
        in_chans=len(config.fields),
        img_size=[config.input_size[0]],
    )
    checkpoint = _load_checkpoint(
        Path(checkpoint_path),
        checkpoint_key=config.checkpoint_key,
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    expected = set(network.state_dict())
    backbone = {
        key: value.detach().cpu().contiguous()
        for key, value in checkpoint.items()
        if key in expected
    }
    network.load_state_dict(backbone, strict=True)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    weights_name = "model.safetensors"
    save_file(backbone, str(output / weights_name))
    exported_config = replace(
        config,
        weights_file=weights_name,
        checkpoint_key=None,
    )
    exported_config.to_json(output / "manifest.json")
    return output
