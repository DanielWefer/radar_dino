from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from radar_dino import (
    RadarDINO,
    RadarDINOConfig,
    ReferenceCatalog,
    export_checkpoint,
)
from radar_dino.attention import field_attention_maps
from radar_dino import vision_transformer
from radar_dino.catalog_builder import save_reference_catalog


pytestmark = pytest.mark.unit


@pytest.fixture
def packaged_tiny_model(tmp_path):
    config = RadarDINOConfig(
        model_id="test-field-token-v1",
        architecture="vit_tiny",
        embedding_dimension=192,
        patch_size=1,
        fields=("reflectivity", "differential_reflectivity"),
        input_size=(2, 3),
        z_level_m=2000.0,
        nan_fill=-1.0,
        normalization={
            "reflectivity": (10.0, 75.0),
            "differential_reflectivity": (-8.0, 12.0),
        },
        weights_file="model.pth",
    )
    config.to_json(tmp_path / "manifest.json")
    network = vision_transformer.vit_tiny(
        patch_size=1,
        num_classes=0,
        in_chans=2,
        img_size=[2],
    )
    prefixed = {
        f"module.backbone.{key}": value
        for key, value in network.state_dict().items()
    }
    prefixed["module.head.test_weight"] = torch.ones(1)
    torch.save({"teacher": prefixed}, tmp_path / "model.pth")
    return tmp_path


def test_config_requires_fixed_normalization_for_every_field():
    with pytest.raises(ValueError, match="Missing fixed normalization"):
        RadarDINOConfig(
            model_id="broken",
            architecture="vit_tiny",
            fields=("reflectivity", "KDP"),
            normalization={"reflectivity": (10.0, 75.0)},
        )


def test_from_pretrained_embeds_and_returns_field_attention(
    packaged_tiny_model,
    radar_netcdf,
):
    dino = RadarDINO.from_pretrained(packaged_tiny_model, device="cpu")

    result = dino.analyze(radar_netcdf)

    assert result.fields == ("reflectivity", "differential_reflectivity")
    assert result.feature.shape == (192,)
    assert np.linalg.norm(result.feature) == pytest.approx(1.0)
    assert result.attention.shape == (3, 2, 2, 3)
    assert result.attention_for("reflectivity", head="mean").shape == (2, 3)
    assert np.isfinite(result.feature).all()
    assert np.isfinite(result.attention).all()


def test_requested_fields_must_match_but_are_reordered_to_model_contract(
    packaged_tiny_model,
    radar_netcdf,
):
    dino = RadarDINO.from_pretrained(packaged_tiny_model, device="cpu")

    sample = dino.load(
        radar_netcdf,
        fields=("differential_reflectivity", "reflectivity"),
    )

    assert sample.fields == ("reflectivity", "differential_reflectivity")
    with pytest.raises(ValueError, match="missing"):
        dino.load(radar_netcdf, fields=("reflectivity",))


def test_field_attention_maps_preserve_batch_head_and_field_axes():
    attention = torch.zeros((2, 3, 13, 13))
    attention[:, :, 0, 1:] = torch.arange(12, dtype=torch.float32)

    maps = field_attention_maps(
        attention,
        num_fields=2,
        patch_grid=(2, 3),
        output_size=(4, 6),
    )

    assert maps.shape == (2, 3, 2, 4, 6)
    torch.testing.assert_close(maps[0], maps[1])


def test_reference_catalog_returns_cosine_ranked_neighbors():
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = [{"scan_id": f"scan-{index}"} for index in range(4)]
    catalog = ReferenceCatalog(features, metadata)

    neighbors = catalog.similar(
        np.array([1.0, 0.0, 0.0]),
        k=2,
        exclude_scan_id="scan-0",
    )

    assert [row["scan_id"] for row in neighbors] == ["scan-1", "scan-2"]
    assert neighbors[0]["similarity"] > neighbors[1]["similarity"]


def test_manifest_json_is_plain_versioned_metadata(tmp_path):
    manifest = RadarDINOConfig(
        model_id="metadata-test",
        architecture="vit_small",
        fields=("reflectivity",),
        normalization={"reflectivity": (10.0, 75.0)},
    )
    path = tmp_path / "manifest.json"

    manifest.to_json(path)

    values = json.loads(path.read_text())
    assert values["schema_version"] == 1
    assert values["fields"] == ["reflectivity"]
    assert RadarDINOConfig.from_json(path) == manifest


def test_reference_catalog_round_trip_uses_jsonl_not_csv(tmp_path):
    catalog = ReferenceCatalog(
        np.eye(3, dtype=np.float32),
        [{"scan_id": f"scan-{index}"} for index in range(3)],
    )

    save_reference_catalog(catalog, tmp_path)
    restored = ReferenceCatalog.from_directory(tmp_path)

    assert not (tmp_path / "reference_metadata.csv").exists()
    assert (tmp_path / "reference_metadata.jsonl").is_file()
    assert restored.metadata == catalog.metadata
    np.testing.assert_allclose(restored.features, catalog.features)


def test_trusted_checkpoint_exports_to_inference_only_safetensors(
    packaged_tiny_model,
    radar_netcdf,
    tmp_path,
):
    config = RadarDINOConfig.from_json(packaged_tiny_model / "manifest.json")
    output = tmp_path / "exported"

    export_checkpoint(
        packaged_tiny_model / "model.pth",
        output,
        config,
    )
    dino = RadarDINO.from_pretrained(output, device="cpu")
    feature = dino.embed(radar_netcdf)

    assert (output / "model.safetensors").is_file()
    assert RadarDINOConfig.from_json(output / "manifest.json").checkpoint_key is None
    assert feature.shape == (192,)
