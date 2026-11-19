"""Tests for the focused, group-final .cppipe runner."""

from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.cppipe import CppipeError
from cp_anywhere.plate_illumination import load_plate_illumination_pipeline

_FACTOR = "Resize by a fraction or multiple of the original size"


def _module(name: str, revision: int, *settings: str, enabled: bool = True) -> str:
    header = f"{name}:[variable_revision_number:{revision}|enabled:{enabled}]"
    return "\n".join((header, *(f"    Setting:{value}" for value in settings)))


def _write(path: Path, *modules: str) -> Path:
    path.write_text(
        f"CellProfiler Pipeline: http://www.cellprofiler.org\nVersion:5\n"
        f"ModuleCount:{len(modules)}\n\n" + "\n\n".join(modules) + "\n",
        encoding="utf-8",
    )
    return path


def _resize(source: str, target: str, factor: str, method: str = "Bilinear") -> str:
    return _module(
        "Resize",
        4,
        source,
        target,
        _FACTOR,
        factor,
        "100",
        "100",
        method,
        "Manual",
        "None",
        "0",
    )


def _across(source: str, target: str, mode: str = "All: Across cycles") -> str:
    return _module(
        "CorrectIlluminationCalculate",
        2,
        source,
        target,
        "Regular",
        "No",
        "1",
        "60",
        "No",
        mode,
        "No smoothing",
        "Manually",
        "10",
        "20",
        "No",
        "Avg",
        "No",
        "Dilated",
        "Yes",
        "auto",
        "5",
        "2",
        "2",
        "40",
        "0.001",
    )


def _save(source: str, cycle: str = "Last cycle") -> str:
    return _module(
        "SaveImages",
        15,
        "Image",
        source,
        "Single name",
        "None",
        r"\g<Plate>_Illum",
        "4",
        "No",
        "",
        "npy",
        "Default Output Folder|",
        "8-bit integer",
        "No",
        cycle,
        "No",
        "No",
        "Elsewhere...|",
        "T (Time)",
    )


def test_ordered_plate_plan_runs_separate_groups(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "illum.cppipe",
        _module("LoadData", 6),
        _resize("OrigDNA", "SmallDNA", "0.5"),
        _across("SmallDNA", "IllumDNA"),
        _resize("IllumDNA", "LargeDNA", "2.0"),
        _module("CreateBatchFiles", 8, enabled=False),
        _save("LargeDNA"),
    )
    plan = load_plate_illumination_pipeline(path)
    assert plan.required_images == ("OrigDNA",)
    assert plan.output_images == ("LargeDNA",)
    first = np.full((5, 7), 257, dtype=np.uint16)
    second = np.full_like(first, 771)
    results = plan.run([{"OrigDNA": first}, {"OrigDNA": second}])
    assert tuple(results) == ("LargeDNA",)
    np.testing.assert_array_equal(
        results["LargeDNA"], np.full((4, 8), 2 / 255, np.float32)
    )
    separate = plan.run([{"OrigDNA": second}])
    np.testing.assert_array_equal(
        separate["LargeDNA"], np.full((4, 8), 3 / 255, np.float32)
    )
    np.testing.assert_array_equal(first, np.full((5, 7), 257, dtype=np.uint16))
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ((_across("DNA", "Illum", "Each"),), "All: Across cycles"),
        (
            (_across("DNA", "Illum").replace("Setting:Regular", "Setting:Background"),),
            "Regular intensity mode",
        ),
        ((_resize("DNA", "ResizeDNA", "0.2", "Bicubic"),), "Bilinear"),
        ((_module("ImageMath", 5),), "ImageMath"),
        ((_across("DNA", "Illum"), _across("Illum", "New")), "group aggregate"),
        ((_across("DNA", "Illum"), _save("Illum", "Every cycle")), "Last cycle"),
        ((_across("DNA", "Illum"), _save("Missing")), "Missing"),
        ((_across("Missing", "Illum"), _resize("DNA", "Missing", "0.2")), "Missing"),
    ],
)
def test_loader_rejects_out_of_scope_plans(
    tmp_path: Path, steps: tuple[str, ...], message: str
) -> None:
    with pytest.raises(CppipeError, match=message):
        load_plate_illumination_pipeline(_write(tmp_path / "bad.cppipe", *steps))


def test_group_input_errors_name_the_location(tmp_path: Path) -> None:
    plan = load_plate_illumination_pipeline(
        _write(tmp_path / "illum.cppipe", _across("DNA", "Illum"))
    )
    with pytest.raises(ValueError, match="nonempty group"):
        plan.run([])
    with pytest.raises(ValueError, match=r"cycle 2.*DNA"):
        plan.run([{"DNA": np.ones((3, 5))}, {}])
    with pytest.raises(ValueError, match=r"cycle 1.*DNA.*nonempty 2D"):
        plan.run([{"DNA": np.ones((2, 3, 5))}])
    with pytest.raises(ValueError, match=r"DNA.*same shape"):
        plan.run([{"DNA": np.ones((3, 5))}, {"DNA": np.ones((4, 5))}])


def test_across_matches_cellprofiler_float32_accumulation(tmp_path: Path) -> None:
    plan = load_plate_illumination_pipeline(
        _write(tmp_path / "illum.cppipe", _across("DNA", "Illum"))
    )
    # Captured from the pinned CellProfiler provider, not a float64 mean.
    levels = (0.8076187968254089, 0.31195539236068726, 0.3331589102745056)
    cycles = [{"DNA": np.array([[level]], dtype=np.float32)} for level in levels]
    np.testing.assert_array_equal(
        plan.run(cycles)["Illum"], np.array([[0.48424434661865234]], dtype=np.float32)
    )


def test_resize_matches_pinned_reference_with_antialiasing(tmp_path: Path) -> None:
    plan = load_plate_illumination_pipeline(
        _write(
            tmp_path / "illum.cppipe",
            _resize("DNA", "Small", "0.5"),
            _across("Small", "Illum"),
        )
    )
    pixels = np.array(
        [[0, 0.7, 0.1, 0.6, 0.1], [0.5, 0.2, 0.9, 1, 0.9], [0.2, 0.3, 1, 0.9, 0.1]],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [0.3668755292892456, 0.4862094521522522],
            [0.39993369579315186, 0.6989682912826538],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(plan.run([{"DNA": pixels}])["Illum"], expected)


def test_resize_revision_five_uses_separate_xy_factors(tmp_path: Path) -> None:
    resize = _module(
        "Resize",
        5,
        "DNA",
        "Small",
        _FACTOR,
        "0.4",
        "1.6",
        "1",
        "100",
        "100",
        "10",
        "Bilinear",
        "Manual",
        "None",
        "0",
    )
    plan = load_plate_illumination_pipeline(
        _write(tmp_path / "illum.cppipe", resize, _across("Small", "Illum"))
    )
    pixels = np.array(
        [[0, 0.7, 0.1, 0.6, 0.1], [0.5, 0.2, 0.9, 1, 0.9], [0.2, 0.3, 1, 0.9, 0.1]],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [0.3439343273639679, 0.33561643958091736],
            [0.3806220293045044, 0.5764464139938354],
            [0.4356536269187927, 0.9376913905143738],
            [0.407077431678772, 0.7467105984687805],
            [0.3880266547203064, 0.619390070438385],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(plan.run([{"DNA": pixels}])["Illum"], expected)
