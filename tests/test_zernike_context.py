"""Check Zernike evidence integrity and isolated-object oracle parity.

Full-field parity remains a separate, deliberately failing diagnostic command:
``uv run --frozen python scripts/zernike/context_evidence.py --native-check``.
These tests do not assert or waive full-field parity.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import measure

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/size_shape/zernike_context"
TARGETS = [("Nuclei", 6), ("Nuclei", 34), ("Cells", 6)]


@pytest.mark.parametrize("scope,oid", TARGETS)
def test_isolated_real_object_matches_actual_cp(scope: str, oid: int) -> None:
    """Preserve the verified local parity without claiming full-field parity."""
    oracle = json.loads((FIXTURE / "oracle.json").read_text())
    with np.load(FIXTURE / "labels.npz", allow_pickle=False) as archive:
        labels = (archive[f"{scope}_labels"] == oid).astype(np.uint32)
    actual = measure(
        {"DNA": np.zeros(labels.shape)},
        {scope: labels},
        families=["object_size_shape"],
        size_shape_calculate_zernikes=True,
        size_shape_calculate_advanced=False,
    ).features[scope]
    expected = oracle["scopes"][scope]["objects"][str(oid)]["isolated"]
    assert {name for name in actual if "Zernike" in name} == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], [value], rtol=1e-10, atol=1e-10)


def test_full_context_labels_and_oracle_evidence_are_intact() -> None:
    """Keep every label: cropping destroys the upstream tie-breaking context."""
    oracle = json.loads((FIXTURE / "oracle.json").read_text())
    assert oracle["source"]["pilot_oracle_verified"] is True
    with np.load(FIXTURE / "labels.npz", allow_pickle=False) as archive:
        for scope, reference in oracle["scopes"].items():
            labels = archive[f"{scope}_labels"]
            assert list(labels.shape) == reference["shape"]
            assert np.array_equal(
                np.unique(labels), np.arange(reference["label_count"] + 1)
            )
            assert (
                hashlib.sha256(labels.astype("<u4").tobytes()).hexdigest()
                == reference["labels_sha256"]
            )
            for obj in reference["objects"].values():
                # Observed upstream limitation, not a tolerance exemption.
                assert obj["full_circle"]["center"] == obj["isolated_circle"]["center"]
                assert (
                    obj["full_circle"]["radius_hex"]
                    != obj["isolated_circle"]["radius_hex"]
                )
                assert (
                    obj["full_circle"]["included_pixels"]
                    != obj["isolated_circle"]["included_pixels"]
                )
                assert set(obj["full_field"]) == set(obj["isolated"])
