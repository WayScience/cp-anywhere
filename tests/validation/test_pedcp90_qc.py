"""Tests for the bounded PedCP90 decision validator."""

from pathlib import Path

import pytest

from scripts.validation.pedcp90_qc import (
    CRITERIA,
    Decision,
    Observation,
    compare_observations,
    decision_differences,
    local_asset,
    qc_pipeline_text,
    select_rows,
)


def test_exact_decision_requires_flag_skip_and_downstream_stage() -> None:
    """A matching numeric flag alone must not hide a control-flow mismatch."""

    skipped = Decision(flag=1, skipped=True, reached_after_qc=False)
    assert decision_differences(skipped, skipped) == []
    assert decision_differences(skipped, Decision(1, False, True)) == [
        "skipped",
        "reached_after_qc",
    ]
    assert decision_differences(skipped, Decision(0, True, False)) == ["flag"]


def test_select_rows_rejects_duplicates_and_missing_fields(tmp_path: Path) -> None:
    """Each requested field must bind to one and only one published row."""

    manifest = tmp_path / "source.csv"
    manifest.write_text(
        "Metadata_Plate,Metadata_Well,Metadata_Site,FileName_OrigDNA\n"
        "plate,A01,1,first.tiff\nplate,A01,2,second.tiff\nplate,A01,2,other.tiff\n",
        encoding="utf-8",
    )
    assert select_rows(manifest, [("plate", "A01", "1")])[0]["FileName_OrigDNA"] == (
        "first.tiff"
    )
    for site in ("2", "3"):
        with pytest.raises(ValueError, match="exactly one"):
            select_rows(manifest, [("plate", "A01", site)])
    with pytest.raises(ValueError, match="unique"):
        select_rows(manifest, [("plate", "A01", "1")] * 2)


def test_comparison_rejects_missing_criterion_even_when_both_omit_it() -> None:
    """Exact agreement is not enough when both runtimes omit the same channel."""
    observation: Observation = {
        "decision": Decision(0, False, True).record(),
        "criteria_failed": dict.fromkeys(CRITERIA, False),
        "metrics": dict.fromkeys(CRITERIA, 0.0),
    }
    assert compare_observations(observation, observation)["criterion_mismatches"] == []
    omitted = next(iter(CRITERIA))
    del observation["criteria_failed"][omitted]
    result = compare_observations(observation, observation)
    assert result["criterion_mismatches"] == [omitted]


def test_pin_rejects_unrelated_pipeline() -> None:
    """A different pipeline cannot serve as the pinned reference."""
    with pytest.raises(ValueError, match="Pipeline SHA-256"):
        qc_pipeline_text(b"changed settings", full_input_set=True)


def test_staged_asset_verifies_original_file_bytes(tmp_path: Path) -> None:
    """Reject renamed or modified pixels even when the TIFF basename matches."""
    path = tmp_path / "raw.tiff"
    path.write_bytes(b"abc")
    row = {"URL_OrigDNA": "s3://public/plate/raw.tiff"}
    digests = {"raw.tiff": "900150983cd24fb0d6963f7d28e17f72"}
    assert local_asset(row, "OrigDNA", tmp_path, digests)["path"] == str(path)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        local_asset(row, "OrigDNA", tmp_path, digests)
