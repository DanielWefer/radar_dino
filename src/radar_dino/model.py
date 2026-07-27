"""Public model-loading and inference API for Radar-DINO."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from . import vision_transformer
from .attention import field_attention_maps
from .catalog import ReferenceCatalog
from .config import RadarDINOConfig
from .netcdf import RadarSample, load_radar_netcdf
from .result import RadarDINOResult


def _artifact_directory(
    model: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.exists():
        raise ValueError(f"Expected a model artifact directory, got {candidate}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Loading a remote model requires the 'hub' extra: "
            "python -m pip install 'radar-dino[hub]'"
        ) from exc

    downloaded = snapshot_download(
        repo_id=str(model),
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    return Path(downloaded)


def _strip_training_prefixes(key: str) -> str:
    prefixes = ("module.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _load_checkpoint(
    path: Path,
    *,
    checkpoint_key: str | None,
    allow_unsafe_pickle: bool,
) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading safetensors weights requires the 'hub' extra: "
                "python -m pip install 'radar-dino[hub]'"
            ) from exc
        values = load_file(str(path), device="cpu")
    else:
        try:
            values = torch.load(
                path,
                map_location="cpu",
                weights_only=not allow_unsafe_pickle,
            )
        except Exception as exc:
            if not allow_unsafe_pickle:
                raise ValueError(
                    "The checkpoint could not be loaded in weights-only mode. "
                    "Only set allow_unsafe_pickle=True for a checkpoint you trust."
                ) from exc
            raise

    if checkpoint_key is not None and isinstance(values, dict) and checkpoint_key in values:
        values = values[checkpoint_key]
    if not isinstance(values, dict):
        raise ValueError(f"Expected a state dictionary in {path}")
    if not all(isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError(
            f"Checkpoint selection '{checkpoint_key}' in {path} is not a tensor state dictionary"
        )
    return {
        _strip_training_prefixes(str(key)): value
        for key, value in values.items()
    }


class RadarDINO:
    """Feature and field-attention inference for one versioned Radar-DINO model."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: RadarDINOConfig,
        *,
        device: str | torch.device = "auto",
        catalog: ReferenceCatalog | None = None,
    ) -> None:
        if device == "auto":
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            selected_device = torch.device(device)
        self.config = config
        self.catalog = catalog
        self.device = selected_device
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        *,
        device: str | torch.device = "auto",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        allow_unsafe_pickle: bool = False,
    ) -> "RadarDINO":
        """Load a local artifact directory or a Hugging Face model repository."""

        artifact_dir = _artifact_directory(
            model,
            revision=revision,
            cache_dir=cache_dir,
        )
        config = RadarDINOConfig.from_json(artifact_dir / "manifest.json")
        factory = getattr(vision_transformer, config.architecture, None)
        if factory is None or not callable(factory):
            raise ValueError(f"Unsupported architecture '{config.architecture}'")
        network = factory(
            patch_size=config.patch_size,
            num_classes=0,
            in_chans=len(config.fields),
            img_size=[config.input_size[0]],
        )
        if config.embedding_dimension is not None:
            actual_dimension = int(network.embed_dim)
            if actual_dimension != config.embedding_dimension:
                raise ValueError(
                    f"Manifest embedding_dimension={config.embedding_dimension}, "
                    f"but {config.architecture} produces {actual_dimension}"
                )

        weights_path = artifact_dir / config.weights_file
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
        checkpoint = _load_checkpoint(
            weights_path,
            checkpoint_key=config.checkpoint_key,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )
        expected_keys = set(network.state_dict())
        backbone = {
            key: value
            for key, value in checkpoint.items()
            if key in expected_keys
        }
        if not backbone:
            raise ValueError(
                f"No {config.architecture} backbone weights were found in {weights_path}"
            )
        network.load_state_dict(backbone, strict=True)
        catalog = None
        if (artifact_dir / "reference_features.npy").is_file():
            catalog = ReferenceCatalog.from_directory(artifact_dir)
            if catalog.feature_dimension != int(network.embed_dim):
                raise ValueError(
                    f"Reference feature dimension {catalog.feature_dimension} does not "
                    f"match model dimension {network.embed_dim}"
                )
        return cls(network, config, device=device, catalog=catalog)

    def load(
        self,
        path: str | Path,
        *,
        fields: Sequence[str] | None = None,
    ) -> RadarSample:
        return load_radar_netcdf(path, self.config, fields=fields)

    def _load_many(
        self,
        paths: Iterable[str | Path],
        *,
        fields: Sequence[str] | None,
    ) -> list[RadarSample]:
        samples = [self.load(path, fields=fields) for path in paths]
        if not samples:
            raise ValueError("At least one NetCDF path is required")
        return samples

    @torch.inference_mode()
    def embed(
        self,
        paths: str | Path | Iterable[str | Path],
        *,
        fields: Sequence[str] | None = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Return L2-normalized CLS features for one file or an iterable of files."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        single = isinstance(paths, (str, Path))
        path_list = [paths] if single else list(paths)
        if not path_list:
            raise ValueError("At least one NetCDF path is required")
        output = []
        for start in range(0, len(path_list), batch_size):
            samples = self._load_many(
                path_list[start : start + batch_size],
                fields=fields,
            )
            inputs = torch.stack([sample.tensor for sample in samples]).to(self.device)
            features = self.model(inputs)
            features = F.normalize(features, dim=1, p=2)
            output.append(features.cpu().numpy())
        array = np.concatenate(output, axis=0)
        return array[0] if single else array

    @torch.inference_mode()
    def attention(
        self,
        path: str | Path,
        *,
        fields: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return last-layer CLS attention as heads x fields x height x width."""

        sample = self.load(path, fields=fields)
        inputs = sample.tensor.unsqueeze(0).to(self.device)
        raw_attention = self.model.get_last_selfattention(inputs)
        height, width = sample.tensor.shape[-2:]
        maps = field_attention_maps(
            raw_attention,
            num_fields=len(self.config.fields),
            patch_grid=(
                height // self.config.patch_size,
                width // self.config.patch_size,
            ),
            output_size=(height, width),
        )
        return maps[0].cpu().numpy()

    def analyze(
        self,
        path: str | Path,
        *,
        fields: Sequence[str] | None = None,
        include_attention: bool = True,
    ) -> RadarDINOResult:
        """Extract a feature and, by default, field-aware attention for one scan."""

        sample = self.load(path, fields=fields)
        feature = self.embed(sample.path, fields=sample.fields)
        attention = (
            self.attention(sample.path, fields=sample.fields)
            if include_attention
            else None
        )
        umap_coordinate = None
        cluster_label = None
        cluster_probability = None
        neighbors: tuple[dict, ...] = ()
        if self.catalog is not None:
            neighbors = tuple(self.catalog.similar(feature, k=5))
            if self.catalog.umap is not None:
                umap_coordinate = self.catalog.project_umap(feature)
            if self.catalog.clusterer is not None:
                cluster_label, cluster_probability = self.catalog.predict_cluster(feature)
        return RadarDINOResult(
            path=sample.path,
            fields=sample.fields,
            feature=feature,
            attention=attention,
            umap=umap_coordinate,
            cluster=cluster_label,
            cluster_probability=cluster_probability,
            neighbors=neighbors,
        )

    def similar(
        self,
        feature: np.ndarray,
        *,
        k: int = 5,
        exclude_scan_id: str | None = None,
    ) -> list[dict]:
        if self.catalog is None:
            raise RuntimeError("This model artifact does not include a reference catalog")
        return self.catalog.similar(
            feature,
            k=k,
            exclude_scan_id=exclude_scan_id,
        )

    def project_umap(self, feature: np.ndarray) -> np.ndarray:
        if self.catalog is None:
            raise RuntimeError("This model artifact does not include a reference catalog")
        return self.catalog.project_umap(feature)

    def predict_cluster(self, feature: np.ndarray) -> tuple[int, float]:
        if self.catalog is None:
            raise RuntimeError("This model artifact does not include a reference catalog")
        return self.catalog.predict_cluster(feature)
