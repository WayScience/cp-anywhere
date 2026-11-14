"""Keep the pre-implementation timing baseline reproducible and complete."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASELINE_FEATURES = 91


def test_initial_baseline_matches_predefined_workload_contract() -> None:
    """Every predefined workload keeps all untuned timing samples."""
    snapshot = json.loads(
        (ROOT / "benchmarks/baseline/initial-cp-measure-timing.json").read_text()
    )
    report = snapshot["timing_report"]
    contract = yaml.safe_load((ROOT / "spec/benchmark-contract.yaml").read_text())
    names = [item["fixture"] for item in contract["representative_workloads"]]
    assert [item["fixture"] for item in report["workloads"]] == names
    assert report["source_commit"] == contract["native_baseline"]["commit"]
    assert report["warmups"] == contract["measurement"]["warmups"]
    assert report["timed_runs"] == contract["measurement"]["timed_runs"]
    manifest = ROOT / "fixtures/reference/manifest.json"
    assert (
        snapshot["scientific_manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    for workload in report["workloads"]:
        assert workload["equivalent_feature_count"] == BASELINE_FEATURES
        assert workload["actual_output_feature_count"] == BASELINE_FEATURES
        for phase in ("conversion", "object_intensity", "neighbors", "end_to_end"):
            samples = workload[phase]["raw_ms"]
            assert len(samples) == report["timed_runs"]
            assert all(value >= 0 for value in samples)
            assert workload[phase]["min_ms"] <= workload[phase]["median_ms"]
            assert workload[phase]["median_ms"] <= workload[phase]["max_ms"]
