# ty: ignore[import-not-found]
"""Exercise ExportToDatabase settings in a real multi-module pipeline."""

# ruff: noqa: E402

from pathlib import Path

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.exporttodatabase import ExportToDatabase
from cellprofiler_core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/cppipe/export_to_database"

# One case per objects_choice/allow_overwrite combination this
# project's translator distinguishes.
_CASES = {
    "all_objects_overwrite_schema": {
        "objects_choice": "All",
        "objects_list": "",
        "allow_overwrite": "Data and schema",
    },
    "select_objects_never_overwrite": {
        "objects_choice": "Select...",
        "objects_list": "Cells",
        "allow_overwrite": "Never",
    },
}


def generate() -> None:
    """Chain a real segmentation pipeline with ExportToDatabase."""
    OUT.mkdir(parents=True, exist_ok=True)
    for case, settings in _CASES.items():
        pipeline = Pipeline()
        pipeline.load(str(ROOT / "fixtures/cppipe/tertiary/shrink.cppipe"))
        module = ExportToDatabase()
        module.set_module_num(len(pipeline.modules()) + 1)
        module.db_type.value = "SQLite"
        module.sqlite_file.value = "MyExpt.db"
        module.want_table_prefix.value = False
        module.separate_object_tables.value = "One table per object type"
        module.objects_choice.value = settings["objects_choice"]
        module.objects_list.value = settings["objects_list"]
        module.allow_overwrite.value = settings["allow_overwrite"]
        pipeline.add_module(module)
        with (OUT / f"{case}.cppipe").open("w") as stream:
            pipeline.dump(stream, save_image_plane_details=False)
        print(case, "written")


if __name__ == "__main__":
    generate()
