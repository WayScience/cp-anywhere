"""Check the CPJUMP1 replay's measurement selection and strict exit criteria."""

import pytest

from scripts.validation.compare_jump import (
    _failed,
    _measurement_scope,
    _selected_fields,
)


@pytest.mark.parametrize(
    ("family", "block", "step_list"),
    [
        ("correlation", "19", "colocalization_steps"),
        ("granularity", "20", "granularity_steps"),
        ("radial", "25", "intensity_distribution_steps"),
        ("texture", "27", "texture_steps"),
    ],
)
def test_measurement_scope_picks_the_same_stage_in_both_runs(
    family: str, block: str, step_list: str
) -> None:
    blocks, overrides = _measurement_scope(family)
    assert blocks == (block,)
    assert step_list not in overrides
    assert set(overrides) == {
        "colocalization_steps",
        "granularity_steps",
        "intensity_distribution_steps",
        "measurement_steps",
        "texture_steps",
    } - {step_list}
    assert all(value == [] for value in overrides.values())


def test_extra_columns_fail_strict_replay() -> None:
    report = {
        "mismatches": [],
        "reference_columns_not_realized": [],
        "extra_columns": {"Cells": []},
        "object_ids_match": {"Cells": True},
        "labels": {"Cells/segmented": {"same_ids": True, "different_pixels": 0}},
    }
    assert not _failed(report)
    report["extra_columns"] = {"Cells": ["Intensity_Unexpected"]}
    assert _failed(report)


def test_single_site_replay_keeps_exact_field_identity() -> None:
    assert _selected_fields("1") == ("BR00118045_A01_1",)
    assert _selected_fields("2") == ("BR00118045_A01_2",)
    assert _selected_fields(None) == ("BR00118045_A01_1", "BR00118045_A01_2")
