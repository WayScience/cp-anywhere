"""Check all supported-setting and unsupported-module diagnostics in file order."""

from pathlib import Path

from cp_anywhere import inspect_pipeline, load_pipeline

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/cppipe/manual.cppipe"


def test_preflight_reports_each_block_without_running(tmp_path: Path) -> None:
    text = FIXTURE.read_text().replace("ModuleCount:4", "ModuleCount:7")
    text = text.replace("variable_revision_number:15", "variable_revision_number:14")
    text = text.replace("variable_revision_number:4", "variable_revision_number:99")
    intensity_header = next(
        line for line in text.splitlines() if line.startswith("MeasureObjectIntensity:")
    )
    text = text.replace(
        intensity_header, intensity_header.replace("enabled:True", "enabled:False")
    )
    text += "\n\nLoadData:[variable_revision_number:6|enabled:True]\n"
    text += "\nImageMathNotImplemented:[variable_revision_number:5|enabled:True]\n"
    text += "\nExportToSpreadsheet:[variable_revision_number:13|enabled:True]\n"
    path = tmp_path / "assessment.cppipe"
    path.write_text(text)
    entries = inspect_pipeline(path)
    assert [(e.position, e.name, e.status) for e in entries] == [
        (1, "IdentifyPrimaryObjects", "unsupported_settings"),
        (2, "MeasureObjectIntensity", "disabled"),
        (3, "MeasureImageIntensity", "unsupported_settings"),
        (4, "MeasureObjectNeighbors", "supported"),
        (5, "LoadData", "caller_managed"),
        (6, "ImageMathNotImplemented", "unsupported_module"),
        (7, "ExportToSpreadsheet", "caller_managed"),
    ]
    assert "14" in entries[0].detail
    assert "99" in entries[2].detail


def test_disabled_supported_module_is_not_translated(tmp_path: Path) -> None:
    path = tmp_path / "disabled.cppipe"
    path.write_text(
        FIXTURE.read_text()
        .replace("enabled:True", "enabled:False")
        .replace("variable_revision_number:15", "variable_revision_number:99")
    )
    pipeline = load_pipeline(path)
    assert not pipeline.segmentation_steps
    assert not pipeline.measurement_steps
    assert pipeline.skipped_modules == [e.name for e in inspect_pipeline(path)]
