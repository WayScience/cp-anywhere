"""Input reproducibility checks that do not need the CellProfiler install."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/reference/generate_illumination_calculate.py"
FIXTURES = ROOT / "fixtures/illumination_calculate"
CASE_COUNT = 252


def test_case_inputs_are_independent_of_python_hash_seed() -> None:
    # Exercise the actual generator in fresh interpreters: changing the variable
    # in this process would not change Python's already-initialized hash secret.
    probe = """
import hashlib
import json
import runpy
import sys

cases = runpy.run_path(sys.argv[1])["cases"]()
serialized = [
    {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in case.items()
    }
    for case in cases
]
print(json.dumps({
    case["name"]: hashlib.sha256(
        json.dumps(case, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    for case in serialized
}, sort_keys=True))
"""
    fingerprints = []
    for hash_seed in ("1", "987654321"):
        result = subprocess.run(
            [sys.executable, "-c", probe, str(GENERATOR)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        fingerprints.append(json.loads(result.stdout))
    assert len(fingerprints[0]) == CASE_COUNT
    assert fingerprints[0] == fingerprints[1]


def test_checked_in_fixture_inputs_match_their_explicit_seeds() -> None:
    generator = runpy.run_path(str(GENERATOR))
    cases = generator["cases"]()
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert len(cases) == CASE_COUNT
    assert generator["INPUT_SEEDS"] == {
        ("small", "none"): 101,
        ("small", "partial"): 102,
        ("tall", "none"): 201,
        ("tall", "partial"): 202,
    }
    assert manifest["cases"] == [
        {
            "name": case["name"],
            "shape": list(case["pixels"].shape),
            "input_seed": case["input_seed"],
        }
        for case in cases
    ]
    assert {path.stem for path in FIXTURES.glob("*.json")} == {
        "manifest",
        *(case["name"] for case in cases),
    }
    for case in cases:
        fixture = json.loads((FIXTURES / f"{case['name']}.json").read_text())
        assert fixture["input_seed"] == case["input_seed"]
        assert fixture["pixels"] == case["pixels"].tolist()
        assert fixture["mask"] == (
            None if case["mask"] is None else case["mask"].tolist()
        )
        assert fixture["settings"] == {
            key: value
            for key, value in case.items()
            if key not in ("name", "input_seed", "pixels", "mask")
        }
        for key in ("reference", "module_sha256", "input_generation"):
            assert fixture[key] == manifest[key]
