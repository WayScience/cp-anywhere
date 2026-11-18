"""Integrity checks for the segmentation reference fixtures."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "fixtures" / "segmentation"


def test_segmentation_manifest_and_checksums() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["cellprofiler_version"] == "4.2.8"
    assert manifest["source_commit"] == "2df090ff281cff80943fe679a1c3192fbda9cd6d"
    assert manifest["cases"]
    for case in manifest["cases"]:
        result = json.loads((ROOT / case["result"]).read_text())
        for path, digest in result["sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_segmentation_reference_replay_when_environment_is_configured() -> None:
    python = os.environ.get("CP_ANYWHERE_REFERENCE_PYTHON")
    if python is None:
        pytest.skip(
            "Set CP_ANYWHERE_REFERENCE_PYTHON and PYTHONPATH "
            "for original CellProfiler segmentation replay"
        )
    subprocess.run(
        [
            python,
            str(ROOT / "scripts" / "segmentation" / "generate_reference.py"),
            "--verify",
        ],
        check=True,
        cwd=ROOT,
        timeout=180,
    )
