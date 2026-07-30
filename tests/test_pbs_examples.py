from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_DIR = REPO_ROOT / "examples" / "pbs"
FIELD_ORDER = (
    "reflectivity specific_differential_phase differential_reflectivity "
    "cross_correlation_ratio spectrum_width"
)
JOBS = {
    "train_radar_dino_fieldtoken.pbs": "radar_dino_training.py",
    "infer_radar_dino_fieldtoken.pbs": "radar_dino_inference.py",
    "assoc_infer_radar_dino_fieldtoken.pbs": "radar_dino_associative_inference.py",
}
TORCHRUN_FLAGS = {"--standalone", "--nnodes", "--nproc_per_node"}


def _argument_flags(script: Path) -> set[str]:
    tree = ast.parse(script.read_text())
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ):
                flags.add(argument.value)
    return flags


@pytest.mark.parametrize(("pbs_name", "entrypoint"), JOBS.items())
def test_pbs_job_is_valid_main_fieldtoken_submission(pbs_name, entrypoint):
    path = PBS_DIR / pbs_name
    text = path.read_text()

    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "wip" not in text.lower()
    assert '#PBS -A SSL-SULI2026' in text
    assert '#PBS -l select=1:system=polaris' in text
    assert '#PBS -l filesystems=home:eagle' in text
    assert '#PBS -k doe' in text
    assert 'RADAR_DINO_BRANCH:-main' in text
    assert 'RADAR_DINO_REPO_DIR:-$HOME/radar_dino' in text
    assert 'git grep -q "field_embed"' in text
    assert FIELD_ORDER in text
    assert (REPO_ROOT / entrypoint).is_file()
    assert f'$REPO_DIR/{entrypoint}' in text

    torchrun_block = text.split("\ntorchrun \\\n", 1)[1].split("\necho \"Job finished", 1)[0]
    used_flags = set(re.findall(r"--[a-zA-Z0-9_-]+", torchrun_block)) - TORCHRUN_FLAGS
    assert used_flags <= _argument_flags(REPO_ROOT / entrypoint)


def test_pbs_jobs_share_training_checkpoint_contract():
    training = (PBS_DIR / "train_radar_dino_fieldtoken.pbs").read_text()
    inference = (PBS_DIR / "infer_radar_dino_fieldtoken.pbs").read_text()
    associative = (PBS_DIR / "assoc_infer_radar_dino_fieldtoken.pbs").read_text()

    assert 'radar_train_fieldtoken' in training
    expected_checkpoint = 'radar_train_fieldtoken/model/checkpoint.pth'
    assert expected_checkpoint in inference
    assert expected_checkpoint in associative
    assert '#PBS -q capacity' in training
    assert '#PBS -l walltime=40:00:00' in training
    for text in (inference, associative):
        assert '#PBS -q preemptable' in text
        assert '#PBS -l walltime=2:00:00' in text
        assert '#PBS -r y' in text


def test_readme_links_only_current_pbs_job_names():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "wip" not in readme.lower()
    for name in JOBS:
        assert f"examples/pbs/{name}" in readme
