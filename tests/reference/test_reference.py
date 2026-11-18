"""Integrity checks for the original CellProfiler reference dataset."""

# Literal feature and fixture counts are the frozen M1 contract.
# ruff: noqa: ANN201, PLR2004

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "fixtures" / "reference"


def test_reference_files_and_checksums():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["cellprofiler_version"] == "4.2.8"
    assert manifest["source_commit"] == "2df090ff281cff80943fe679a1c3192fbda9cd6d"
    assert manifest["fixtures"]
    assert manifest["feature_count"] == 119
    assert len(manifest["fixtures"]) == 13
    assert (
        hashlib.sha256((ROOT / manifest["pipeline"]).read_bytes()).hexdigest()
        == manifest["pipeline_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / "reference/requirements.lock").read_bytes()).hexdigest()
        == manifest["requirements_sha256"]
    )
    for fixture in manifest["fixtures"]:
        assert fixture["environment_lock"] == manifest["environment_lock"]
        assert any(path.endswith("nuclei_labels.npy") for path in fixture["sha256"])
        for path, digest in fixture["sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        result = json.loads((ROOT / fixture["result"]).read_text())
        assert result["image_id"] == fixture["name"]
        assert result["features"]
        assert isinstance(result["relationships"], list)


def test_feature_inventory_matches_observed_columns():
    inventory = yaml.safe_load((ROOT / "spec" / "features.yaml").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    expected = {(f["object_scope"], f["name"]) for f in inventory["features"]}
    observed = set()
    for fixture in manifest["fixtures"]:
        result = json.loads((ROOT / fixture["result"]).read_text())
        observed.update(
            (scope, name)
            for scope, columns in result["features"].items()
            for name in columns
        )
    assert expected == observed
    assert len(expected) == manifest["feature_count"]
    assert inventory["relationships"]


def test_reference_replay_when_environment_is_configured():
    python = os.environ.get("CP_ANYWHERE_REFERENCE_PYTHON")
    if python is None:
        pytest.skip(
            "Set CP_ANYWHERE_REFERENCE_PYTHON and PYTHONPATH "
            "for original CellProfiler replay"
        )
    subprocess.run(
        [python, str(ROOT / "scripts" / "reference" / "generate.py"), "--verify"],
        check=True,
        cwd=ROOT,
        timeout=180,
    )
